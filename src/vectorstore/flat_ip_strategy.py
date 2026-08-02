"""
``FlatIPIndexStrategy`` — índice exacto (fuerza bruta) por producto interno.

Es el **default recomendado** por la especificación del reto para el
volumen de corpus esperado: sin entrenamiento, resultados exactos (no
aproximados), simple de razonar y reproducir.
"""

from __future__ import annotations

import faiss

from src.vectorstore.index_builder_base import FaissIndexBuilderStrategy, IndexBuildConfig
from src.vectorstore.index_builder_factory import IndexBuilderFactory


@IndexBuilderFactory.register("flat_ip")
class FlatIPIndexStrategy(FaissIndexBuilderStrategy):
    """``faiss.IndexFlatIP``: exacto, sin entrenamiento."""

    index_type_name = "flat_ip"
    requires_training = False

    def build(self, dim: int, config: IndexBuildConfig) -> faiss.Index:
        return faiss.IndexFlatIP(dim)
