"""
Configuración del pipeline vía variables de entorno (pydantic-settings).

Toda la configuración externa (MongoDB, parámetros de chunking, tokenizador,
estrategia, etc.) se lee desde ``.env`` o variables de entorno reales.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repositorio (src/models/ -> raíz) y archivos .env que se leen.
# Se anclan al archivo (no al CWD) para que funcione desde cualquier directorio.
_RUTA_PROYECTO = Path(__file__).resolve().parents[2]
_ENV_FILES = (str(_RUTA_PROYECTO / ".env"), str(_RUTA_PROYECTO / "config" / ".env"))


class ChunkingConfig(BaseModel):
    """Parámetros de la estrategia de fragmentación.

    Atributos:
        chunk_size: Tamaño de la ventana deslizante en tokens.
        overlap_size: Solapamiento entre ventanas en tokens.
        min_chunk_tokens: Umbral mínimo; secciones menores se fusionan con la adyacente.
        max_tokens: Límite duro por fragmento (límite del encoder, p. ej. 512).
    """

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    overlap_size: int = int(os.getenv("OVERLAP_SIZE", "100"))
    min_chunk_tokens: int = int(os.getenv("MIN_CHUNK_TOKENS", "50"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))

    def __init__(self, **data):
        super().__init__(**data)
        # Invariante: la ventana + el solapamiento no pueden superar el límite del encoder.
        if self.overlap_size >= self.chunk_size:
            raise ValueError("overlap_size debe ser menor que chunk_size")
        if self.chunk_size > self.max_tokens:
            raise ValueError("chunk_size no puede superar max_tokens (límite del encoder)")


class Settings(BaseSettings):
    """Configuración global del pipeline (leída de ``.env``)."""

    # MongoDB
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "rag_corpus")
    mongo_collection: str = os.getenv("MONGO_COLLECTION", "chunks")
    mongo_user: str = os.getenv("MONGO_USER", "admin")
    mongo_password: str = os.getenv("MONGO_PASSWORD", "admin")
    mongo_auth_source: str = os.getenv("MONGO_AUTH_SOURCE", "admin")

    # Repositorio de chunks: el JSON es el artefacto portable por defecto.
    # Se puede conservar MongoDB con CHUNK_REPOSITORY=mongo.
    chunk_repository: str = os.getenv("CHUNK_REPOSITORY", "json")
    chunks_json_path: Path = Path(os.getenv("CHUNKS_JSON_PATH", "data/chunks.json"))
    
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    overlap_size: int = int(os.getenv("OVERLAP_SIZE", "80"))
    min_chunk_tokens: int = int(os.getenv("MIN_CHUNK_TOKENS", "50"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "paragraph_overlap")

    # Tokenización y segmentación de oraciones
    tokenizer_model: str = os.getenv("TOKENIZER_MODEL", "google-bert/bert-base-multilingual-cased")
    sentence_model: str = os.getenv("SENTENCE_MODEL", "es_core_news_sm")
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "es")

    # Documento
    default_fuente: str = os.getenv("DEFAULT_FUENTE", "desconocida")

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,  # config/.env gana; .env raíz como respaldo
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chunking_config(self) -> ChunkingConfig:
        """Configuración de chunking derivada de los ajustes globales."""
        return ChunkingConfig(
            chunk_size=self.chunk_size,
            overlap_size=self.overlap_size,
            min_chunk_tokens=self.min_chunk_tokens,
            max_tokens=self.max_tokens,
        )
