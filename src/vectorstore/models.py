"""
``EmbeddingRecord``: unidad persistida en la colección ``embeddings`` de
MongoDB — un vector para un ``(chunk_id, encoder_name)`` (Sección 5.1).

El vector se mantiene como ``np.ndarray`` en memoria; el empaquetado a
``bson.Binary`` (float32 crudo) es responsabilidad del repositorio
(``MongoVectorRepository``), no de este modelo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> str:
    """Marca temporal UTC en formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


class EmbeddingRecord(BaseModel):
    """Vector de un chunk para un encoder concreto, trazable y reproducible."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str
    doc_id: str
    fenomeno: int
    formato: str
    encoder_name: str
    model_id: str
    embedding_dim: int
    vector: np.ndarray
    vector_dtype: str = "float32"
    normalized: bool = True
    hash_texto: Optional[str] = None
    faiss_internal_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def con_marca_temporal(self) -> "EmbeddingRecord":
        """Copia con ``created_at``/``updated_at`` completados si faltan."""
        datos = self.model_dump()
        datos["vector"] = self.vector
        ahora = _utc_now()
        datos["created_at"] = datos.get("created_at") or ahora
        datos["updated_at"] = ahora
        return EmbeddingRecord(**datos)
