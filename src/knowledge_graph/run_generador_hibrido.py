"""CLI: generador de resultados con recuperación HÍBRIDA (FAISS + grafo).

Equivalente a ``generador.py`` (esquema exacto de la Sección 9.3.1:
50 líneas q001..q050, 3 documents ranks 1-3, 10 fragments ranks 1-10,
text ≤ 250 palabras) pero con el grafo de conocimiento como canal ADICIONAL
de evidencia en la fusión (Sección 8.5): los rankings de los encoders
activos (FAISS) y el ranking del grafo comparten el contrato
:class:`ScoredChunk` y se fusionan con la :class:`FusionStrategy`
(RRF/CombSUM/CombMNZ).

Además guarda ``--output-caminos``: el camino por el grafo de cada
consulta (entidades → nodos → vecinos → tripletas → chunks de evidencia y
camino hasta el top-k fusionado), para auditar por qué el grafo propuso
cada candidato.

Sin modelos generativos: NER/RE simbólicos, codificación por los mismos
encoders de la base vectorial, y el texto de los fragmentos sale de la
metadata de entrega (nunca se genera).

Uso:
    python -m src.knowledge_graph.run_generador_hibrido
    python -m src.knowledge_graph.run_generador_hibrido --queries consultas.jsonl \
        --output resultados_hibridos.jsonl --output-caminos logs/caminos_grafo.json
    python -m src.knowledge_graph.run_generador_hibrido --cargar-grafo grafo.graphml \
        --fusion rrf --k0 60 --k-search 50 --doc-agg max
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.knowledge_graph.models import Query, ScoredChunk
from src.knowledge_graph.retrieval.adapters import GraphIndexAdapter
from src.knowledge_graph.retrieval.base import FusionStrategy, Retriever
from src.knowledge_graph.retrieval.fusion import (
    CombMNZFusionStrategy,
    CombSUMFusionStrategy,
    RRFusionStrategy,
)
from src.knowledge_graph.run_hybrid_retrieval import (
    FUSIONES,
    _cargar_canal_grafo,
    _cargar_canales_vectoriales,
)
from src.queries import QueryLoader, build_result_object, verificar_resultados
from src.retrieval.aggregation import aggregate_to_documents
from src.retrieval.chunk_ops import split_or_merge_fragments
from src.retrieval.filters import apply_filters
from src.retrieval.index_loader import build_siguiente_chunk_lookup
from src.retrieval.models import (
    FusedFragment,
    RetrievalFilters,
    RetrievalResult,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_generador_hibrido")

N_DOCS = 3  # Sección 8.6: top-3 documentos.
K_CHUNK_OUT_DEFAULT = 10


def _texto_por_chunk(base_dir: Path) -> Dict[str, str]:
    """Lookup ``chunk_id -> texto`` desde el metadata de entrega del primer encoder.

    Todos los encoders comparten los mismos ``chunk_id``; basta el primero
    con artefacto. Es el MISMO origen del texto que consume ``generador.py``.
    """
    from src.embeddings.embedding_config import EmbeddingConfig

    config = EmbeddingConfig()
    for nombre in config.encoder_names:
        ruta = base_dir / f"encoder_{nombre}" / "metadata.jsonl"
        if ruta.exists():
            lookup: Dict[str, str] = {}
            with ruta.open(encoding="utf-8") as fh:
                for linea in fh:
                    if not linea.strip():
                        continue
                    dato = json.loads(linea)
                    lookup[str(dato.get("chunk_id", ""))] = str(
                        dato.get("texto") or dato.get("text", "")
                    )
            return lookup
    return {}


def _a_fused_fragments(
    fusionados: List[ScoredChunk], texto_por_chunk: Dict[str, str]
) -> List[FusedFragment]:
    """Convierte el ranking fusionado a :class:`FusedFragment` (post-fusiones).

    El grafo y los vectores ya se fusionaron (contrato común); aquí solo se
    reempaqueta para las etapas siguientes (filtros/split/agregación).
    """
    return [
        FusedFragment(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=texto_por_chunk.get(c.chunk_id, ""),
            rrf_score=c.score,
            cosine_score=c.score,
            encoders=[c.origen],
            metadata=dict(c.metadata),
        )
        for c in fusionados
    ]


def recuperar_hibrido(
    texto: str,
    canales_vectoriales: List[Retriever],
    canal_grafo: GraphIndexAdapter,
    texto_por_chunk: Dict[str, str],
    siguiente_chunk,
    fusion: FusionStrategy,
    k_search: int = 50,
    k_chunk_out: int = K_CHUNK_OUT_DEFAULT,
    doc_agg: str = "max",
) -> Tuple[Dict, List[ScoredChunk]]:
    """Pipeline completo híbrido para una consulta (Sección 8.5 → 8.6 → 8.7).

    Todos los canales devuelven el MISMO contrato (:class:`ScoredChunk`):
    la fusión (RRF/CombSUM/CombMNZ) trata al grafo como "un índice más"
    sin distinguir su origen; luego el ranking fusionado se reempaqueta a
    :class:`FusedFragment` para las etapas puras de la Sección 8.

    Returns:
        ``(resultado_dict, ranking_fusionado)`` donde ``resultado_dict`` es
        el contrato de ``retrieve()`` (``{"documents", "fragments"}``) y
        ``ranking_fusionado`` el top-k fusionado (para el camino por el grafo).
    """
    # Etapa 8.1-8.4: cada canal recupera su ranking (mismo contrato).
    rankings: List[List[ScoredChunk]] = [
        adapter.retrieve(Query(texto=texto), k_search) for adapter in canales_vectoriales
    ]
    rankings.append(canal_grafo.retrieve(Query(texto=texto), k_search))

    # Etapa 8.4: fusión (RRF/CombSUM/CombMNZ) — el grafo es "un índice más".
    fusionados = fusion.fusionar(rankings)
    fragmentos_base = _a_fused_fragments(fusionados, texto_por_chunk)

    # Etapa 8.7: post-filtros (sin filtros por defecto → pass-through).
    fragmentos_base = apply_filters(fragmentos_base, RetrievalFilters())

    # Etapa 9.2.1: split/merge a nivel de chunk respetando oraciones.
    fragmentos_finales = split_or_merge_fragments(
        fragmentos_base[:k_chunk_out],
        splitter=None,
        siguiente_chunk=siguiente_chunk,
    )

    # Etapa 8.6: agregación a documento (top-3).
    documentos = [
        doc_id
        for doc_id, _ in aggregate_to_documents(fragmentos_finales, strategy=doc_agg)[:N_DOCS]
    ]

    resultado = RetrievalResult(
        documents=documentos,
        fragments=[f.as_dict() for f in fragmentos_finales],
    )
    return resultado.as_dict(), fusionados


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generador de resultados con recuperación híbrida FAISS + grafo (Sección 9)")
    parser.add_argument("--queries", default="consultas.jsonl", help="Archivo .jsonl de consultas (q001..q050)")
    parser.add_argument("--output", default="resultados_hibridos.jsonl", help="Archivo de salida (esquema Sección 9.3.1)")
    parser.add_argument("--output-caminos", default="logs/caminos_grafo.json", help="JSON con el camino por el grafo por consulta")
    parser.add_argument("--encoders-dir", default="base_vectorial", help="Directorio de artefactos de entrega")
    parser.add_argument("--cargar-grafo", default=None, help="Ruta de un grafo.graphml ya exportado (evita reconstruir)")
    parser.add_argument("--limite-grafo", type=int, default=None, help="Máximo de chunks para el grafo (default: todos)")
    parser.add_argument("--k-search", type=int, default=50, help="Top-k por canal antes de fusionar")
    parser.add_argument("--k0", type=int, default=60, help="Constante RRF (solo fusion rrf)")
    parser.add_argument("--fusion", default="rrf", choices=sorted(FUSIONES), help="Estrategia de fusión")
    parser.add_argument("--doc-agg", default="max", choices=["max", "sum", "weighted_mean"], help="Agregación a documento")
    parser.add_argument("--max-tripletas", type=int, default=100, help="Tope de tripletas por camino")
    parser.add_argument("--max-chunks-camino", type=int, default=50, help="Tope de chunks de evidencia por camino")
    args = parser.parse_args(argv)

    if args.fusion == "rrf":
        fusion: FusionStrategy = FUSIONES[args.fusion](k0=args.k0)
    else:
        fusion = FUSIONES[args.fusion]()

    try:
        canales = _cargar_canales_vectoriales(Path(args.encoders_dir))
        canal_grafo = _cargar_canal_grafo(
            Path(args.encoders_dir),
            args.limite_grafo,
            Path(args.cargar_grafo) if args.cargar_grafo else None,
        )
    except Exception as exc:  # noqa: BLE001 - error de arranque
        logger.exception("%s", exc)
        return 1

    texto_por_chunk = _texto_por_chunk(Path(args.encoders_dir))

    # Lookup del "siguiente chunk" del mismo doc (para split/merge de la
    # Sección 9.2.1) sobre la metadata del primer encoder con artefacto.
    from src.embeddings.embedding_config import EmbeddingConfig

    siguiente_chunk = None
    config = EmbeddingConfig()
    for nombre in config.encoder_names:
        ruta = Path(args.encoders_dir) / f"encoder_{nombre}" / "metadata.jsonl"
        if ruta.exists():
            lineas = []
            with ruta.open(encoding="utf-8") as fh:
                for linea in fh:
                    if linea.strip():
                        lineas.append(json.loads(linea))
            siguiente_chunk = build_siguiente_chunk_lookup({nombre: lineas})
            break
    if siguiente_chunk is None:
        logger.error("No hay metadata.jsonl en '%s' para el split/merge", args.encoders_dir)
        return 1

    consultas = QueryLoader.cargar(args.queries)
    logger.info("Cargadas %d consultas desde '%s'", len(consultas), args.queries)
    logger.info("Canales: %s + grafo | fusión: %s", [c.origen for c in canales], fusion.nombre)

    lineas: List[str] = []
    caminos: List[dict] = []
    fallidas: List[str] = []

    for i, consulta in enumerate(consultas, start=1):
        query_id, texto = consulta.query_id, consulta.query_text
        logger.info("[%d/%d] %s: %r", i, len(consultas), query_id, texto[:70])
        try:
            resultado, fusionados = recuperar_hibrido(
                texto,
                canales,
                canal_grafo,
                texto_por_chunk,
                siguiente_chunk,
                fusion,
                k_search=args.k_search,
                doc_agg=args.doc_agg,
            )
            objeto = build_result_object(query_id, resultado)
            caminos.append(
                {
                    "query_id": query_id,
                    "consulta": texto,
                    "top_k": [
                        {
                            "rank": r,
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "score": round(float(c.score), 6),
                            "origen": c.origen,
                        }
                        for r, c in enumerate(fusionados, start=1)
                    ],
                    "camino_grafo": canal_grafo.explicar_camino(
                        Query(texto=texto),
                        max_tripletas=args.max_tripletas,
                        max_chunks=args.max_chunks_camino,
                        chunk_ids_objetivo=[c.chunk_id for c in fusionados],
                    ),
                }
            )
        except Exception:  # noqa: BLE001 - una consulta no debe tumbar el resto
            logger.exception("%s FALLÓ", query_id)
            fallidas.append(query_id)
            continue

        lineas.append(json.dumps(objeto, ensure_ascii=False))
        logger.info("%s OK", query_id)

    ruta_salida = Path(args.output)
    with open(ruta_salida, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
        if lineas:
            f.write("\n")
    logger.info("Escritas %d líneas en '%s'", len(lineas), ruta_salida)
    verificar_resultados(ruta_salida, len(lineas), fallidas)

    if args.output_caminos:
        ruta_caminos = Path(args.output_caminos)
        ruta_caminos.parent.mkdir(parents=True, exist_ok=True)
        ruta_caminos.write_text(
            json.dumps(caminos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Caminos guardados en '%s' (%d consultas)", ruta_caminos, len(caminos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
