"""
Script CLI: valida la **calidad de recuperación** de cada encoder
persistido en MongoDB (Sección 4, criterio "benchmark MTEB/BEIR"),
ejecutando un mini-benchmark de recuperación densa estilo BEIR pero sobre
el propio corpus de este reto (``GoldQuery`` en ``--queries-file``), ya
que los checkpoints BERT crudos usados como encoders no tienen score
MTEB-Retrieval oficial (ver ``src/encoders/base.py``).

Para cada encoder solicitado:
    1. Carga sus vectores ya persistidos en Mongo (``embeddings``), no
       recodifica el corpus.
    2. Codifica las consultas de ``--queries-file`` con
       ``estrategia.encode(..., is_query=True)``.
    3. Rankea el corpus por similitud coseno (producto punto; los
       vectores ya están normalizados) y calcula Precision@k, Recall@k,
       MRR y nDCG@k contra los juicios de relevancia declarados.
    4. Imprime una tabla comparativa y escribe un reporte JSON.

Uso::

    python -m src.embeddings.run_retrieval_benchmark --k 10
    python -m src.embeddings.run_retrieval_benchmark --encoders bert-multilingual,bert-tiny \
        --queries-file data/benchmark_queries.json --output logs/retrieval_benchmark.json

Formato de ``--queries-file`` (ver plantilla ``data/benchmark_queries.example.json``)::

    [
      {"query": "¿qué medidas de ciberseguridad...?", "relevant_chunk_ids": ["..."]},
      {"query": "riesgos de IA en defensa",            "relevant_doc_ids": ["..."]}
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Registra cada estrategia concreta vía @EncoderFactory.register (efecto
# secundario necesario antes de EncoderFactory.create()).
from src.encoders import (  # noqa: F401
    bert_strategy,
    bert_large_strategy,
    bert_language_strategy,
    bert_multilingual_uncased_strategy,
    bert_tiny_strategy,
)
from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.retrieval_quality import GoldQuery, evaluar_encoder
from src.encoders.base import EncoderConfig
from src.encoders.factory import EncoderFactory
from src.vectorstore.vector_repository import MongoVectorRepository

logger = logging.getLogger("run_retrieval_benchmark")


def _configurar_logging(verbose: bool) -> None:
    nivel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _cargar_golds(ruta: Path) -> List[GoldQuery]:
    """Lee el JSON de consultas de validación con sus juicios de relevancia."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo de consultas de validación: {ruta}. "
            "Copia y edita la plantilla 'data/benchmark_queries.example.json'."
        )
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    golds = [
        GoldQuery(
            query=item["query"],
            relevant_chunk_ids=item.get("relevant_chunk_ids", []),
            relevant_doc_ids=item.get("relevant_doc_ids", []),
        )
        for item in datos
    ]
    if not golds:
        raise ValueError(f"'{ruta}' no contiene ninguna consulta de validación")
    return golds


def _cargar_vectores_encoder(repositorio: MongoVectorRepository, nombre_encoder: str):
    """Trae en memoria todos los vectores persistidos de un encoder.

    El corpus del reto es de tamaño moderado (Sección 2), así que cargarlo
    completo en memoria para el benchmark es aceptable; para recuperación
    en producción se usa el índice FAISS (Sección 5), no esta ruta.
    """
    chunk_ids: List[str] = []
    chunk_id_a_doc_id: Dict[str, str] = {}
    vectores: List[np.ndarray] = []
    for registro in repositorio.find_by_encoder(nombre_encoder):
        chunk_ids.append(registro.chunk_id)
        chunk_id_a_doc_id[registro.chunk_id] = registro.doc_id
        vectores.append(registro.vector)
    if not vectores:
        return None, None, None
    return np.stack(vectores), chunk_ids, chunk_id_a_doc_id


def _evaluar_un_encoder(
    nombre_encoder: str,
    golds: List[GoldQuery],
    repositorio: MongoVectorRepository,
    k: int,
) -> Optional[Dict[str, float]]:
    vectores_corpus, chunk_ids_corpus, chunk_id_a_doc_id = _cargar_vectores_encoder(repositorio, nombre_encoder)
    if vectores_corpus is None:
        logger.warning(
            "Encoder '%s': no hay vectores persistidos en Mongo (corre antes run_embedding.py). Se omite.",
            nombre_encoder,
        )
        return None

    estrategia = EncoderFactory.create(nombre_encoder, EncoderConfig())
    estrategia.load()
    vectores_query = estrategia.encode([g.query for g in golds], is_query=True)

    metricas = evaluar_encoder(
        golds=golds,
        vectores_query=vectores_query,
        vectores_corpus=vectores_corpus,
        chunk_ids_corpus=chunk_ids_corpus,
        chunk_id_a_doc_id=chunk_id_a_doc_id,
        k=k,
    )
    if metricas is None:
        logger.warning(
            "Encoder '%s': ninguna consulta de validación tiene relevantes resolubles en su corpus persistido.",
            nombre_encoder,
        )
        return None

    metricas["n_chunks_indexados"] = len(chunk_ids_corpus)
    return metricas


def _imprimir_tabla(resultados: Dict[str, Dict[str, float]], k: int) -> None:
    columnas = ["encoder", f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "n_queries", "n_chunks"]
    filas = [columnas] + [
        [
            nombre,
            f"{m['precision_at_k']:.3f}",
            f"{m['recall_at_k']:.3f}",
            f"{m['reciprocal_rank']:.3f}",
            f"{m['ndcg_at_k']:.3f}",
            str(int(m["n_queries"])),
            str(int(m["n_chunks_indexados"])),
        ]
        for nombre, m in resultados.items()
    ]
    anchos = [max(len(fila[i]) for fila in filas) for i in range(len(columnas))]
    for fila in filas:
        print(" | ".join(valor.ljust(anchos[i]) for i, valor in enumerate(fila)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries-file", default="data/benchmark_queries.json", help="JSON de consultas de validación con juicios de relevancia")
    parser.add_argument("--encoders", default=None, help="Nombres de encoders separados por coma (por defecto: ACTIVE_ENCODERS del .env)")
    parser.add_argument("--k", type=int, default=10, help="Corte k para Precision/Recall/nDCG (por defecto: 10)")
    parser.add_argument("--output", default="logs/retrieval_benchmark.json", help="Ruta del reporte JSON de salida")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configurar_logging(args.verbose)

    config = EmbeddingConfig()
    nombres_encoders = (
        [n.strip().lower() for n in args.encoders.split(",") if n.strip()]
        if args.encoders
        else config.encoder_names
    )

    golds = _cargar_golds(Path(args.queries_file))
    logger.info("Consultas de validación cargadas: %d", len(golds))

    repositorio = MongoVectorRepository(
        uri=config.mongo_uri,
        db_name=config.mongo_db,
        collection_name=config.mongo_collection_embeddings,
        username=config.mongo_user,
        password=config.mongo_password,
        auth_source=config.mongo_auth_source,
    )

    resultados: Dict[str, Dict[str, float]] = {}
    try:
        for nombre_encoder in nombres_encoders:
            metricas = _evaluar_un_encoder(nombre_encoder, golds, repositorio, args.k)
            if metricas is not None:
                resultados[nombre_encoder] = metricas
    finally:
        repositorio.close()

    if not resultados:
        logger.error("Ningún encoder pudo evaluarse (sin vectores persistidos o sin relevancia resoluble)")
        return 1

    _imprimir_tabla(resultados, args.k)

    ruta_salida = Path(args.output)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(json.dumps({"k": args.k, "resultados": resultados}, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Reporte escrito en: %s", ruta_salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
