"""
Configuración del módulo de embeddings vía variables de entorno (Sección 5).
"""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Configuración externa de la etapa de codificación semántica."""

    active_encoders: str = os.getenv("ACTIVE_ENCODERS", "e5-base")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "auto")  # auto | cpu | cuda | mps
    embedding_output_dir: str = os.getenv("EMBEDDING_OUTPUT_DIR", "base_vectorial")

    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "rag_corpus")
    mongo_collection_chunks: str = os.getenv("MONGO_COLLECTION_CHUNKS", "chunks")
    mongo_collection_embeddings_cache: str = os.getenv(
        "MONGO_COLLECTION_EMBEDDINGS_CACHE", "embeddings_cache"
    )
    mongo_user: Optional[str] = os.getenv("MONGO_USER", "admin")
    mongo_password: Optional[str] = os.getenv("MONGO_PASSWORD", "admin")
    mongo_auth_source: Optional[str] = os.getenv("MONGO_AUTH_SOURCE", "admin")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def encoder_names(self) -> List[str]:
        """Nombres de encoders activos (``ACTIVE_ENCODERS`` separado por comas)."""
        return [n.strip().lower() for n in self.active_encoders.split(",") if n.strip()]
