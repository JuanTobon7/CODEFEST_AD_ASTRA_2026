"""
Repositorio MongoDB de fragmentos (pymongo).

- Colección ``chunks`` con índice único sobre ``chunk_id``.
- Índices sobre ``doc_id`` y ``fenomeno``.
- Upsert idempotente por ``chunk_id``: reprocesar un documento no duplica.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pymongo import ASCENDING, IndexModel, MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from src.models.chunk import Chunk
from src.persistence.base_repository import ChunkRepository

logger = logging.getLogger(__name__)


class MongoChunkRepository(ChunkRepository):
    """Persistencia de fragmentos en MongoDB."""

    def __init__(self, uri: str, db_name: str, collection_name: str = "chunks",
                 username: Optional[str] = None, password: Optional[str] = None,
                 auth_source: Optional[str] = None) -> None:
        self._uri = uri
        self._db_name = db_name
        self._collection_name = collection_name
        self._username = username
        self._password = password
        self._cliente: Optional[MongoClient] = None
        self._conectado = False
        self._auth_source = auth_source

    # Gestión de conexión -------------------------------------------------------

    def connect(self) -> None:
        """Abre la conexión y crea los índices (una sola vez)."""
        if self._conectado:
            return
        try:
            opciones = {"serverSelectionTimeoutMS": 5000}
            if self._username is not None:
                opciones["username"] = self._username
            if self._password is not None:
                opciones["password"] = self._password
            if self._auth_source is not None:
                opciones["authSource"] = self._auth_source
            self._cliente = MongoClient(self._uri, **opciones)
            coleccion = self._cliente[self._db_name][self._collection_name]
            coleccion.create_indexes(
                [
                    IndexModel([("chunk_id", ASCENDING)], unique=True, name="uq_chunk_id"),
                    IndexModel([("doc_id", ASCENDING)], name="idx_doc_id"),
                    IndexModel([("fenomeno", ASCENDING)], name="idx_fenomeno"),
                ]
            )
            self._conectado = True
            logger.info("Conectado a MongoDB: %s.%s", self._db_name, self._collection_name)
        except PyMongoError as exc:
            raise PyMongoError(f"No se pudo conectar a MongoDB ({self._uri}): {exc}") from exc

    def close(self) -> None:
        """Cierra la conexión si está abierta."""
        if self._cliente is not None:
            self._cliente.close()
        self._cliente = None
        self._conectado = False

    def __enter__(self) -> "MongoChunkRepository":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # Contrato del repositorio ----------------------------------------------------

    def save_many(self, chunks: List[Chunk]) -> None:
        """Guarda los fragmentos con upsert idempotente por ``chunk_id``."""
        if not chunks:
            return
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        operaciones = [
            UpdateOne(
                {"chunk_id": chunk.chunk_id},
                {"$set": chunk.como_dict_mongo},
                upsert=True,
            )
            for chunk in chunks
        ]
        try:
            resultado = coleccion.bulk_write(operaciones, ordered=False)
            logger.info(
                "Persistidos %d fragmentos (insertados=%d, actualizados=%d)",
                len(chunks),
                resultado.upserted_count,
                resultado.modified_count,
            )
        except PyMongoError as exc:
            logger.error("Fallo al persistir fragmentos: %s", exc)
            raise

    def find_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Recupera los fragmentos de un documento, ordenados por posición."""
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        docs = coleccion.find({"doc_id": doc_id}).sort("posicion", ASCENDING)
        return [Chunk.model_validate(d) for d in docs]

    def exists(self, chunk_id: str) -> bool:
        """True si el fragmento ya está persistido."""
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        return coleccion.find_one({"chunk_id": chunk_id}, {"_id": 1}) is not None

    def find_all(self, limite: int = 0) -> List[Chunk]:
        """Recupera chunks (opcionalmente limitado), para codificación por lotes."""
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        cursor = coleccion.find({})
        if limite:
            cursor = cursor.limit(limite)
        return [Chunk.model_validate(d) for d in cursor]

    def mark_encoder_procesado(self, chunk_id: str, encoder_name: str) -> None:
        """Añade ``encoder_name`` a ``encoders_procesados`` del chunk (idempotente)."""
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        coleccion.update_one({"chunk_id": chunk_id}, {"$addToSet": {"encoders_procesados": encoder_name}})
