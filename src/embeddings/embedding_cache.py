"""
Cache de embeddings ya calculados (Sección 4): evita recomputar el vector de
un chunk si ya fue codificado con el mismo ``encoder_name`` y su
``hash_texto`` no cambió. Respaldo en MongoDB (colección ``embeddings_cache``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pymongo import ASCENDING, IndexModel, MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from src.support.utils import en_lotes

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Verifica y registra qué ``(chunk_id, encoder_name, hash_texto)`` ya se codificó."""

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str = "embeddings_cache",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: Optional[str] = None,
    ) -> None:
        self._uri = uri
        self._db_name = db_name
        self._collection_name = collection_name
        self._username = username
        self._password = password
        self._auth_source = auth_source
        self._cliente: Optional[MongoClient] = None

    def connect(self) -> None:
        """Abre la conexión y crea el índice único ``(chunk_id, encoder_name)``."""
        if self._cliente is not None:
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
                    IndexModel(
                        [("chunk_id", ASCENDING), ("encoder_name", ASCENDING)],
                        unique=True,
                        name="uq_chunk_encoder",
                    )
                ]
            )
        except PyMongoError as exc:
            raise PyMongoError(f"No se pudo conectar al cache de embeddings ({self._uri}): {exc}") from exc

    def close(self) -> None:
        """Cierra la conexión si está abierta."""
        if self._cliente is not None:
            self._cliente.close()
        self._cliente = None

    def __enter__(self) -> "EmbeddingCache":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def pendientes(self, encoder_name: str, chunk_id_to_hash: Dict[str, str]) -> List[str]:
        """``chunk_id`` que NO están cacheados para ``encoder_name`` (o cambiaron de hash).

        Consulta por lotes: un ``$in`` con decenas de miles de ``chunk_id``
        puede exceder el límite de 16MB por documento BSON.
        """
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        cacheados: Dict[str, str] = {}
        for lote in en_lotes(list(chunk_id_to_hash)):
            for doc in coleccion.find(
                {"encoder_name": encoder_name, "chunk_id": {"$in": lote}},
                {"chunk_id": 1, "hash_texto": 1},
            ):
                cacheados[doc["chunk_id"]] = doc["hash_texto"]
        return self._pendientes_de(chunk_id_to_hash, cacheados)

    @staticmethod
    def _pendientes_de(chunk_id_to_hash: Dict[str, str], cacheados: Dict[str, str]) -> List[str]:
        """Lógica pura: pendiente si no está cacheado o si su hash cambió."""
        return [
            chunk_id
            for chunk_id, hash_texto in chunk_id_to_hash.items()
            if cacheados.get(chunk_id) != hash_texto
        ]

    def marcar_procesados(self, encoder_name: str, chunk_id_to_hash: Dict[str, str]) -> None:
        """Registra que ``encoder_name`` ya codificó estos chunks (por ``hash_texto``).

        Escribe por lotes, igual que :meth:`pendientes` lee por lotes: con el
        corpus completo son cientos de miles de operaciones y construirlas
        todas de una vez es un pico de memoria innecesario.
        """
        if not chunk_id_to_hash:
            return
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        for lote in en_lotes(list(chunk_id_to_hash.items())):
            coleccion.bulk_write(
                [
                    UpdateOne(
                        {"chunk_id": chunk_id, "encoder_name": encoder_name},
                        {"$set": {"hash_texto": hash_texto}},
                        upsert=True,
                    )
                    for chunk_id, hash_texto in lote
                ],
                ordered=False,
            )
        logger.info("Cache actualizado: %d chunks marcados para '%s'", len(chunk_id_to_hash), encoder_name)
