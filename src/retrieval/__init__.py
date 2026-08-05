"""
Módulo de recuperación (retrieval) — Sección 8 del reto CODEFEST AD ASTRA
2026.

Pipeline sobre vectores/puntuaciones/metadata puros, sin ningún modelo
generativo:

1. Codificación de la consulta por encoder activo (``query_encoder``).
2. Búsqueda top-``k_search`` por índice FAISS (``faiss_search``).
3. Fusión multi-encoder con RRF (``rrf``).
4. Post-filtros por metadata y por umbral de coseno (``filters``).
5. Split/merge a nivel de chunk respetando oraciones (``chunk_ops``).
6. Agregación a nivel de documento (``aggregation``).

Punto de entrada de conveniencia: ``retrieve()`` en ``retriever.py``.
"""

from src.retrieval.aggregation import (
    aggregate_to_documents,
    pool_max,
    pool_sum,
    pool_weighted_mean,
)
from src.retrieval.chunk_ops import split_or_merge_fragments
from src.retrieval.faiss_search import search_faiss
from src.retrieval.filters import apply_filters
from src.retrieval.models import (
    Fragment,
    FusedFragment,
    RetrievalFilters,
    RetrievalResult,
    SearchHit,
)
from src.retrieval.query_encoder import encode_query
from src.retrieval.rrf import rrf_fuse

__all__ = [
    "Fragment",
    "FusedFragment",
    "RetrievalFilters",
    "RetrievalResult",
    "SearchHit",
    "aggregate_to_documents",
    "apply_filters",
    "encode_query",
    "pool_max",
    "pool_sum",
    "pool_weighted_mean",
    "rrf_fuse",
    "search_faiss",
    "split_or_merge_fragments",
]
