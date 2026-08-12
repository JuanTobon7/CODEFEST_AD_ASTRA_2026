"""
Persistencia de fragmentos (patrón Repository).
"""

from src.persistence.base_repository import ChunkRepository
from src.persistence.json_repository import JsonChunkRepository, JsonChunkRepositoryError
from src.persistence.mongo_repository import MongoChunkRepository

__all__ = [
    "ChunkRepository",
    "JsonChunkRepository",
    "JsonChunkRepositoryError",
    "MongoChunkRepository",
]
