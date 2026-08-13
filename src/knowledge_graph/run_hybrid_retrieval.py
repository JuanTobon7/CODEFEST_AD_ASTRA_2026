"""CLI: recuperación híbrida FAISS + grafo de conocimiento (Sección 8.5).

Carga los artefactos de entrega de ``base_vectorial/encoder_<n>``
(``index.faiss`` + ``metadata.jsonl``, el MISMO origen de los chunks), le
aplica NER/RE a esos chunks para construir el grafo en memoria y fusiona
los dos canales (:class:`VectorIndexAdapter` por encoder + el
:class:`GraphIndexAdapter`) con RRF — el grafo actúa como "un índice más",
sin modelos generativos en ninguna etapa.

Con ``--output-caminos`` guarda un JSON con, por consulta, el top-k elegido
y el CAMINO por el grafo que sustenta la respuesta (entidades de la
consulta → nodos del grafo → vecinos → tripletas → chunks de evidencia),
para auditar la Sección 8.5 pregunta por pregunta.

Uso:
    python -m src.knowledge_graph.run_hybrid_retrieval "¿Qué organización coopera con la NASA?"
    python -m src.knowledge_graph.run_hybrid_retrieval --queries consultas.jsonl --k 10
    python -m src.knowledge_graph.run_hybrid_retrieval --queries consultas.jsonl \
        --k 3 --output-caminos logs/caminos_grafo.json
    python -m src.knowledge_graph.run_hybrid_retrieval --encoders-dir base_vectorial --k0 60 --fusion rrf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from src.embeddings.embedding_config import EmbeddingConfig
from src.encoders.base import EncoderConfig
from src.encoders.factory import EncoderFactory
from src.knowledge_graph.models import Query
from src.knowledge_graph.retrieval.adapters import (
    GraphIndexAdapter,
    VectorIndexAdapter,
)
from src.knowledge_graph.retrieval.base import FusionStrategy, Retriever
from src.knowledge_graph.retrieval.fusion import (
    CombMNZFusionStrategy,
    CombSUMFusionStrategy,
    RRFusionStrategy,
)
from src.knowledge_graph.retrieval.orchestrator import RetrievalOrchestrator
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.re_cli_options import agregar_opciones_re, kwargs_re
from src.knowledge_graph.service import KnowledgeGraphService
from src.knowledge_graph.run_build_graph import ChunkMetadata
from src.retrieval.index_loader import ArtifactosFaltantesError, load_encoder_artifacts

# Importar la estrategia RE registra su decorador en RelationExtractorFactory
# (nli-zero-shot es el default del CLI; el service por defecto sigue siendo
# coocurrencia-oracional, puramente simbólico).
import src.knowledge_graph.extract.nli_relation_strategy  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_hybrid_retrieval")

FUSIONES: dict[str, type[FusionStrategy]] = {
    "rrf": RRFusionStrategy,
    "combsum": CombSUMFusionStrategy,
    "combmnz": CombMNZFusionStrategy,
}


def _importar_estrategias_encoders() -> None:
    """Importa las estrategias para disparar su registro por decorador."""
    from src.encoders import (  # noqa: F401 - registro por decorador (efecto secundario)
        bert_language_strategy,
        bert_large_strategy,
        bert_multilingual_uncased_strategy,
        bert_strategy,
        bert_tiny_strategy,
        e5_multilingual_base_strategy,
        e5_multilingual_small_strategy,
    )


def _cargar_canales_vectoriales(base_dir: Path) -> List[Retriever]:
    """Un :class:`VectorIndexAdapter` por encoder activo con artefacto de entrega."""
    _importar_estrategias_encoders()
    config = EmbeddingConfig()
    device = None if config.embedding_device == "auto" else config.embedding_device
    canales: List[Retriever] = []
    for nombre in config.encoder_names:
        try:
            indice, metadata = load_encoder_artifacts(base_dir, nombre)
        except ArtifactosFaltantesError as exc:
            logger.warning("Se omite encoder '%s': %s", nombre, exc)
            continue
        estrategia = EncoderFactory.create(
            nombre,
            EncoderConfig(
                batch_size=config.batch_size_para(nombre), device_preference=device
            ),
        )
        canales.append(
            VectorIndexAdapter(
                encoder=estrategia, index=indice, metadata=metadata, encoder_name=nombre
            )
        )
    if not canales:
        raise ArtifactosFaltantesError(
            f"No hay artefactos de entrega en '{base_dir}' para ningún encoder activo."
        )
    return canales


def _cargar_canal_grafo(
    base_dir: Path,
    limite: Optional[int],
    grafo_graphml: Optional[Path] = None,
    relation_extractor=None,
) -> GraphIndexAdapter:
    """Construye/carga el grafo alineado con los chunks de los artefactos.

    Por defecto se construye con la metadata del primer encoder activo
    disponible (todos comparten los mismos ``chunk_id``; solo difieren los
    vectores), de modo que el grafo queda alineado con el índice FAISS por
    construcción. Si se pasa ``grafo_graphml``, se carga el archivo ya
    exportado (más rápido, misma alineación). ``relation_extractor`` solo
    se usa al CONSTRUIR (no al cargar desde GraphML).
    """
    servicio = KnowledgeGraphService(relation_extractor=relation_extractor)
    if grafo_graphml is not None and grafo_graphml.exists():
        logger.info("Cargando grafo desde '%s'", grafo_graphml)
        from src.knowledge_graph.graph.inmemory_repository import GraphMLFileRepository
        from src.knowledge_graph.graph.repository import KnowledgeGraphQuery

        grafo = GraphMLFileRepository().cargar(grafo_graphml)
        logger.info(
            "Grafo cargado: %d entidades, %d tripletas",
            grafo.num_entidades,
            grafo.num_tripletas,
        )
        adapter = GraphIndexAdapter(
            extractor=servicio.extractor_entidades,
            grafo=KnowledgeGraphQuery(grafo),
        )
        return adapter

    config = EmbeddingConfig()
    ruta = None
    for nombre in config.encoder_names:
        candidata = base_dir / f"encoder_{nombre}" / "metadata.jsonl"
        if candidata.exists():
            ruta = candidata
            logger.info("Grafo desde '%s' (%s chunks)", ruta, nombre)
            break
    if ruta is None:
        raise ArtifactosFaltantesError(
            f"No hay metadata.jsonl en '{base_dir}' para construir el grafo."
        )

    contados = 0
    with ruta.open(encoding="utf-8") as fh:
        chunks = []
        for linea in fh:
            if not linea.strip():
                continue
            chunks.append(ChunkMetadata(json.loads(linea)))
            contados += 1
            if limite is not None and contados >= limite:
                break
    grafo = servicio.construir_desde_chunks(chunks)
    logger.info(
        "Grafo: %d entidades, %d tripletas", grafo.num_entidades, grafo.num_tripletas
    )
    return servicio.crear_retriever_grafo()


def _leer_consultas(ruta: Path) -> List[Tuple[str, str]]:
    """Lee ``(query_id, texto)`` de un .jsonl con esquema del reto o genérico.

    Acepta el esquema de ``consultas.jsonl`` (``query_id`` + ``query_text``)
    y variantes genéricas (``id``/``query``/``texto``).
    """
    pares: List[Tuple[str, str]] = []
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            dato = json.loads(linea)
            texto = str(
                dato.get("query_text")
                or dato.get("query")
                or dato.get("texto")
                or ""
            ).strip()
            if not texto:
                continue
            qid = str(
                dato.get("query_id") or dato.get("id") or f"q{len(pares) + 1:03d}"
            )
            pares.append((qid, texto))
    return pares


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recuperación híbrida FAISS + grafo (Sección 8.5)")
    parser.add_argument("query", nargs="?", default=None, help="Consulta en lenguaje natural")
    parser.add_argument("--encoders-dir", default="base_vectorial", help="Directorio de artefactos de entrega")
    parser.add_argument("--queries", default=None, help="Archivo .jsonl de consultas (query_id+query_text, o query/texto)")
    parser.add_argument("--k", type=int, default=10, help="Candidatos finales fusionados")
    parser.add_argument("--k-por-canal", type=int, default=50, help="Top-k consultado a cada canal")
    parser.add_argument("--k0", type=int, default=60, help="Constante RRF (solo fusion rrf)")
    parser.add_argument("--fusion", default="rrf", choices=sorted(FUSIONES), help="Estrategia de fusión")
    parser.add_argument("--limite-grafo", type=int, default=None, help="Máximo de chunks para el grafo (default: todos)")
    agregar_opciones_re(parser)
    parser.add_argument("--cargar-grafo", default=None, help="Ruta de un grafo.graphml ya exportado (evita reconstruir)")
    parser.add_argument("--output-caminos", default=None, help="JSON de salida con el camino por el grafo por consulta")
    parser.add_argument("--max-tripletas", type=int, default=100, help="Tope de tripletas por camino (default: 100)")
    parser.add_argument("--max-chunks-camino", type=int, default=50, help="Tope de chunks de evidencia por camino (default: 50)")
    args = parser.parse_args(argv)

    if args.fusion == "rrf":
        fusion: FusionStrategy = FUSIONES[args.fusion](k0=args.k0)
    else:
        fusion = FUSIONES[args.fusion]()

    # Consultas: argumento o archivo (con query_id opcional).
    consultas: List[Tuple[str, str]] = []
    if args.queries:
        consultas = _leer_consultas(Path(args.queries))
    elif args.query:
        consultas = [("q001", args.query)]
    else:
        parser.error("Indique una consulta o un archivo --queries")
    if not consultas:
        logger.error("No se leyeron consultas válidas de '%s'", args.queries)
        return 1

    try:
        canales = _cargar_canales_vectoriales(Path(args.encoders_dir))
        canal_grafo = _cargar_canal_grafo(
            Path(args.encoders_dir), args.limite_grafo,
            Path(args.cargar_grafo) if args.cargar_grafo else None,
            RelationExtractorFactory.create(args.re, **kwargs_re(args)),
        )
        canales.append(canal_grafo)
    except ArtifactosFaltantesError as exc:
        logger.exception("%s", exc)
        return 1

    orquestador = RetrievalOrchestrator(
        retrievers=canales, fusion=fusion, k_por_canal=args.k_por_canal
    )
    logger.info("Canales: %s | fusión: %s", orquestador.canales, fusion.nombre)

    caminos: List[dict] = []
    for qid, texto in consultas:
        query = Query(texto=texto)
        resultado = orquestador.recuperar(query, k_final=args.k)

        entrada = {
            "query_id": qid,
            "consulta": texto,
            "top_k": [
                {
                    "rank": i,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "score": round(float(c.score), 6),
                    "origen": c.origen,
                }
                for i, c in enumerate(resultado, start=1)
            ],
            "camino_grafo": canal_grafo.explicar_camino(
                query,
                max_tripletas=args.max_tripletas,
                max_chunks=args.max_chunks_camino,
                chunk_ids_objetivo=[c.chunk_id for c in resultado],
            ),
        }
        caminos.append(entrada)

        print(f"\n=== {qid}: {texto}")
        if not resultado:
            print("  (sin candidatos)")
            continue
        for i, candidato in enumerate(resultado, start=1):
            print(
                f"  {i:2d}. {candidato.chunk_id}  score={candidato.score:.4f}  "
                f"origen={candidato.origen}  doc={candidato.doc_id}"
            )

    if args.output_caminos:
        ruta_salida = Path(args.output_caminos)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        ruta_salida.write_text(
            json.dumps(caminos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Caminos guardados en '%s' (%d consultas)", ruta_salida, len(caminos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
