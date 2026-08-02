"""
``HNSWIndexStrategy`` — índice de grafo (HNSW), sin entrenamiento, prioriza
latencia de consulta sobre uso de memoria.
"""

from __future__ import annotations

import faiss

from src.vectorstore.index_builder_base import FaissIndexBuilderStrategy, IndexBuildConfig
from src.vectorstore.index_builder_factory import IndexBuilderFactory


@IndexBuilderFactory.register("hnsw")
class HNSWIndexStrategy(FaissIndexBuilderStrategy):
    """``faiss.IndexHNSWFlat``: sin entrenamiento, búsqueda muy rápida."""

    index_type_name = "hnsw"
    requires_training = False

    def build(self, dim: int, config: IndexBuildConfig) -> faiss.Index:
        return faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
