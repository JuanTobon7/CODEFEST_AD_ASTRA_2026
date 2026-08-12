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

    def connect(self) -> None:
        """Prepara el repositorio si necesita conexion o inicializacion."""

    def close(self) -> None:
        """Libera recursos del repositorio si aplica."""

    @abstractmethod
    def save_many(self, chunks: List[Chunk]) -> None:
        """Guarda (upsert idempotente) los fragmentos dados."""

    @abstractmethod
    def find_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Recupera todos los fragmentos de un documento."""

    @abstractmethod
    def find_all(self, limite: int = 0) -> List[Chunk]:
        """Recupera todos los fragmentos, opcionalmente con un limite."""

    @abstractmethod
    def find_all_balanceado(self, limite: int, n_fenomenos: int = 3) -> List[Chunk]:
        """Recupera un lote equilibrado de fragmentos para codificacion."""

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """True si el fragmento ya está persistido."""
