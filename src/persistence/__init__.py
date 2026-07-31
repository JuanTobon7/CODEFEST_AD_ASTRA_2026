"""
Persistencia de fragmentos (patrón Repository).
"""

from src.persistence.base_repository import ChunkRepository
from src.persistence.mongo_repository import MongoChunkRepository

__all__ = ["ChunkRepository", "MongoChunkRepository"]
