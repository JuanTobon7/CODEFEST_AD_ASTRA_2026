"""
Estrategia abstracta de construcción de índices FAISS (patrón Strategy,
Sección 5.2): el tipo de índice (exacto vs. aproximado, memoria vs.
velocidad) varía según el volumen del corpus y el objetivo de la búsqueda,
igual que el encoder varía según idioma/licencia/eficiencia.

Todos los builders asumen vectores ya normalizados a norma unitaria
(responsabilidad de ``EncoderStrategy.encode()``), de forma que el producto
interno (``METRIC_INNER_PRODUCT``) equivale a similitud coseno.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import faiss
import numpy as np
from pydantic import BaseModel, Field


class IndexBuildConfig(BaseModel):
    """Parámetros runtime de construcción de índices FAISS."""

    ivf_nlist: int = Field(default=100, ge=1)
    ivf_nprobe: int = Field(default=10, ge=1)
    hnsw_m: int = Field(default=32, ge=4)
    ivf_auto_threshold: int = Field(default=50_000, ge=1)


class FaissIndexBuilderStrategy(ABC):
    """Interfaz común de construcción de índices FAISS intercambiable."""

    index_type_name: str
    requires_training: bool = False

    @abstractmethod
    def build(self, dim: int, config: IndexBuildConfig) -> faiss.Index:
        """Construye un índice vacío (sin entrenar ni poblar), listo para ``add``."""

    def train_if_needed(self, index: faiss.Index, vectors: np.ndarray) -> None:
        """No-op por defecto; los índices que requieren entrenamiento la sobreescriben."""
        return None
