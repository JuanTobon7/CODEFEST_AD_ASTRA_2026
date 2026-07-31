"""
Interfaz abstracta del repositorio de fragmentos (patrón Repository).

Define el contrato de persistencia sin atar el pipeline a una tecnología
concreta (MongoDB, SQLite, Parquet...). El orquestador depende de esta
abstracción (inversión de dependencias).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.models.chunk import Chunk


class ChunkRepository(ABC):
    """Contrato de persistencia de fragmentos."""

    @abstractmethod
    def save_many(self, chunks: List[Chunk]) -> None:
        """Guarda (upsert idempotente) los fragmentos dados."""

    @abstractmethod
    def find_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Recupera todos los fragmentos de un documento."""

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """True si el fragmento ya está persistido."""
