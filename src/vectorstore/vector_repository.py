"""
Repositorio de vectores (Sección 5.2): persiste cada ``EmbeddingRecord`` en
MongoDB como fuente de verdad, separado de la colección ``chunks`` (1
documento por ``chunk_id`` en ``chunks``, N documentos por chunk —uno por
encoder activo— en ``embeddings``).

El vector se guarda empaquetado como ``bson.Binary`` (``float32.tobytes()``),
no como ``list[float]``: con miles de chunks × varios encoders el ahorro de
espacio/ancho de banda es significativo frente a JSON/BSON de arrays.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import numpy as np
from bson import Binary
from pymongo import ASCENDING, IndexModel, MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from src.vectorstore.models import EmbeddingRecord

logger = logging.getLogger(__name__)


class VectorRepository(ABC):
    """Contrato de persistencia de vectores por ``(chunk_id, encoder_name)``."""

    @abstractmethod
    def save_many(self, records: List[EmbeddingRecord]) -> None:
        """Upsert idempotente por ``(chunk_id, encoder_name)``."""

    @abstractmethod
    def find_by_encoder(self, encoder_name: str, batch_size: int = 500) -> Iterator[EmbeddingRecord]:
        """Streaming (cursor) de todos los vectores de ``encoder_name``."""

    @abstractmethod
    def find_by_chunk_ids(self, encoder_name: str, chunk_ids: List[str]) -> List[EmbeddingRecord]:
        """Vectores de ``encoder_name`` restringidos a ``chunk_ids`` puntuales."""

    @abstractmethod
    def find_missing(self, encoder_name: str, chunk_ids: List[str]) -> List[str]:
        """``chunk_id`` de ``chunk_ids`` que aún no tienen vector para ``encoder_name``."""

    @abstractmethod
    def count_by_encoder(self, encoder_name: str) -> int:
        """Número de vectores persistidos para ``encoder_name``."""

    @abstractmethod
    def delete_by_chunk_id(self, chunk_id: str) -> None:
        """Elimina todos los vectores (de cualquier encoder) de ``chunk_id``."""

    @abstractmethod
    def set_faiss_internal_id(self, chunk_id: str, encoder_name: str, faiss_id: int) -> None:
        """Registra el ``faiss_internal_id`` asignado en el modo operativo incremental."""


class MongoVectorRepository(VectorRepository):
    """Persistencia de vectores en MongoDB (colección ``embeddings``)."""

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str = "embeddings",
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
        """Abre la conexión y crea los índices (una sola vez)."""
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
                    ),
                    IndexModel([("encoder_name", ASCENDING)], name="idx_encoder_name"),
                ]
            )
            logger.info("Conectado a MongoDB (vectores): %s.%s", self._db_name, self._collection_name)
        except PyMongoError as exc:
            raise PyMongoError(f"No se pudo conectar al repositorio de vectores ({self._uri}): {exc}") from exc

    def close(self) -> None:
        """Cierra la conexión si está abierta."""
        if self._cliente is not None:
            self._cliente.close()
        self._cliente = None

    def __enter__(self) -> "MongoVectorRepository":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _coleccion(self):
        self.connect()
        return self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]

    @staticmethod
    def _a_documento(record: EmbeddingRecord) -> Dict[str, object]:
        """Serializa un ``EmbeddingRecord`` a documento Mongo (vector empaquetado)."""
        ahora = datetime.now(timezone.utc).isoformat()
        return {
            "chunk_id": record.chunk_id,
            "doc_id": record.doc_id,
            "fenomeno": record.fenomeno,
            "formato": record.formato,
            "encoder_name": record.encoder_name,
            "model_id": record.model_id,
            "embedding_dim": record.embedding_dim,
            "vector": Binary(np.asarray(record.vector, dtype=np.float32).tobytes()),
            "vector_dtype": "float32",
            "normalized": record.normalized,
            "hash_texto": record.hash_texto,
            "updated_at": ahora,
        }

    @staticmethod
    def _desde_documento(doc: Dict[str, object]) -> EmbeddingRecord:
        """Reconstruye un ``EmbeddingRecord`` desde el binario crudo del vector."""
        vector = np.frombuffer(doc["vector"], dtype=np.float32).copy()
        return EmbeddingRecord(
            chunk_id=doc["chunk_id"],
            doc_id=doc["doc_id"],
            fenomeno=doc["fenomeno"],
            formato=doc["formato"],
            encoder_name=doc["encoder_name"],
            model_id=doc["model_id"],
            embedding_dim=doc["embedding_dim"],
            vector=vector,
            vector_dtype=doc.get("vector_dtype", "float32"),
            normalized=doc.get("normalized", True),
            hash_texto=doc.get("hash_texto"),
            faiss_internal_id=doc.get("faiss_internal_id"),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
        )

    def save_many(self, records: List[EmbeddingRecord]) -> None:
        """Upsert idempotente por ``(chunk_id, encoder_name)``."""
        if not records:
            return
        coleccion = self._coleccion()
        operaciones = [
            UpdateOne(
                {"chunk_id": r.chunk_id, "encoder_name": r.encoder_name},
                {
                    "$set": self._a_documento(r),
                    "$setOnInsert": {"created_at": datetime.now(timezone.utc).isoformat()},
                },
                upsert=True,
            )
            for r in records
        ]
        try:
            resultado = coleccion.bulk_write(operaciones, ordered=False)
            logger.info(
                "Vectores persistidos: %d (insertados=%d, actualizados=%d)",
                len(records), resultado.upserted_count, resultado.modified_count,
            )
        except PyMongoError as exc:
            logger.error("Fallo al persistir vectores: %s", exc)
            raise

    def find_by_encoder(self, encoder_name: str, batch_size: int = 500) -> Iterator[EmbeddingRecord]:
        """Cursor en streaming: no carga todo el corpus en memoria."""
        coleccion = self._coleccion()
        cursor = coleccion.find({"encoder_name": encoder_name}, batch_size=batch_size)
        for doc in cursor:
            yield self._desde_documento(doc)

    def find_by_chunk_ids(self, encoder_name: str, chunk_ids: List[str]) -> List[EmbeddingRecord]:
        """Lectura puntual (no streaming) para lotes pequeños, p. ej. modo incremental."""
        if not chunk_ids:
            return []
        coleccion = self._coleccion()
        docs = coleccion.find({"encoder_name": encoder_name, "chunk_id": {"$in": list(chunk_ids)}})
        return [self._desde_documento(d) for d in docs]

    def find_missing(self, encoder_name: str, chunk_ids: List[str]) -> List[str]:
        """``chunk_id`` sin vector para ``encoder_name`` entre los solicitados."""
        if not chunk_ids:
            return []
        coleccion = self._coleccion()
        existentes = {
            doc["chunk_id"]
            for doc in coleccion.find(
                {"encoder_name": encoder_name, "chunk_id": {"$in": list(chunk_ids)}}, {"chunk_id": 1}
            )
        }
        return self._faltantes_de(chunk_ids, existentes)

    @staticmethod
    def _faltantes_de(chunk_ids: List[str], existentes) -> List[str]:
        """Lógica pura: preserva el orden de entrada, sin duplicados."""
        vistos = set()
        faltantes = []
        for chunk_id in chunk_ids:
            if chunk_id not in existentes and chunk_id not in vistos:
                faltantes.append(chunk_id)
                vistos.add(chunk_id)
        return faltantes

    def count_by_encoder(self, encoder_name: str) -> int:
        """Número de vectores persistidos para ``encoder_name``."""
        return self._coleccion().count_documents({"encoder_name": encoder_name})

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        """Elimina todos los vectores (de cualquier encoder) de ``chunk_id``."""
        self._coleccion().delete_many({"chunk_id": chunk_id})

    def set_faiss_internal_id(self, chunk_id: str, encoder_name: str, faiss_id: int) -> None:
        """Registra el ``faiss_internal_id`` asignado por el modo operativo incremental."""
        self._coleccion().update_one(
            {"chunk_id": chunk_id, "encoder_name": encoder_name},
            {"$set": {"faiss_internal_id": faiss_id}},
        )
