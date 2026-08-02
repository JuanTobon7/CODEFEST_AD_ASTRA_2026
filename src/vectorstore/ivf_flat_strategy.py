"""
``IVFFlatIndexStrategy`` — índice aproximado por clustering (IVF), útil
cuando el corpus de un encoder supera el umbral configurado
(``FAISS_IVF_AUTO_THRESHOLD``). Requiere entrenamiento (k-means) antes de
poblarlo con ``add``/``add_with_ids``.
"""

from __future__ import annotations

import faiss
import numpy as np

from src.vectorstore.index_builder_base import FaissIndexBuilderStrategy, IndexBuildConfig
from src.vectorstore.index_builder_factory import IndexBuilderFactory


@IndexBuilderFactory.register("ivf_flat")
class IVFFlatIndexStrategy(FaissIndexBuilderStrategy):
    """``faiss.IndexIVFFlat``: aproximado, requiere entrenamiento previo."""

    index_type_name = "ivf_flat"
    requires_training = True

    def build(self, dim: int, config: IndexBuildConfig) -> faiss.Index:
        cuantizador = faiss.IndexFlatIP(dim)
        indice = faiss.IndexIVFFlat(cuantizador, dim, config.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        indice.nprobe = config.ivf_nprobe
        return indice

    def train_if_needed(self, index: faiss.Index, vectors: np.ndarray) -> None:
        """Entrena el índice (k-means sobre ``nlist`` clusters) si aún no lo está."""
        if not index.is_trained:
            index.train(vectors)
