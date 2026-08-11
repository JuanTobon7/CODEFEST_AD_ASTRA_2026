"""
Repositorio MongoDB de fragmentos (pymongo).

- Colección ``chunks`` con índice único sobre ``chunk_id``.
- Índices sobre ``doc_id`` y ``fenomeno``.
- Upsert idempotente por ``chunk_id``: reprocesar un documento no duplica.
"""

from __future__ import annotations

import logging
import re

from typing import Dict, List, Optional

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
            logger.exception("Fallo al persistir fragmentos: %s", exc)
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

    @staticmethod
    def _cuotas_por_fenomeno(limite: int, n_fenomenos: int = 3) -> Dict[int, int]:
        """Reparte ``limite`` en cuotas por fenómeno (1..``n_fenomenos``).

        Cuota base = ``limite // n_fenomenos``; el residuo se asigna uno a
        cada uno de los primeros fenómenos (determinista). Con ``limite <= 0``
        devuelve ``{}`` (sin tope: todos los chunks).
        """
        if limite <= 0:
            return {}
        cuota = limite // n_fenomenos
        residuo = limite % n_fenomenos
        return {
            fenomeno: cuota + (1 if fenomeno <= residuo else 0)
            for fenomeno in range(1, n_fenomenos + 1)
        }

    @staticmethod
    def _subcarpeta_de(doc_id: str) -> str:
        """Prefijo ``raiz\\subcarpeta`` del ``doc_id`` (primeros 2 componentes).

        Acepta separadores ``\\`` (Windows) y ``/`` (POSIX). Si el ``doc_id``
        no tiene ruta (p. ej. ``Indice_Datos_Codefest.xlsx``), devuelve el
        ``doc_id`` completo: cada documento es su propia "subcarpeta".
        """
        for sep in ("\\", "/"):
            if sep in doc_id:
                partes = doc_id.split(sep)
                return sep.join(partes[:2])
        return doc_id

    @staticmethod
    def _cuotas_por_subcarpeta(limite: int, subcarpetas: List[str]) -> Dict[str, int]:
        """Reparte ``limite`` entre ``subcarpetas`` (determinista).

        Cuota base = ``limite // n``; el residuo se asigna uno a cada una de
        las primeras subcarpetas en orden alfabético. Con ``limite <= 0`` o
        sin subcarpetas devuelve ``{}``.
        """
        if limite <= 0 or not subcarpetas:
            return {}
        n = len(subcarpetas)
        cuota = limite // n
        residuo = limite % n
        return {
            sub: cuota + (1 if i < residuo else 0)
            for i, sub in enumerate(sorted(subcarpetas))
        }

    def find_all_balanceado(self, limite: int, n_fenomenos: int = 3) -> List[Chunk]:
        """Recupera hasta ``limite`` chunks repartidos equitativamente entre los
        ``n_fenomenos`` fenómenos (p. ej. 1000 -> ~333 por fenómeno).

        Dentro de cada fenómeno, la cuota se reparte ENTRE SUS SUBCARPETAS
        (segundo componente del ``doc_id``) para no sesgar el lote hacia la
        primera subcarpeta alfabética (p. ej. ``AI_Index_Stanford`` en F1):
        cada subcarpeta se consulta por separado, ordenada por ``chunk_id``,
        con su cuota (:meth:`_cuotas_por_subcarpeta`). Con ``limite <= 0``
        delega en :meth:`find_all` (todos los chunks).
        """
        cuotas = self._cuotas_por_fenomeno(limite, n_fenomenos)
        if not cuotas:
            return self.find_all()
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        chunks: List[Chunk] = []
        reparto: Dict[int, Dict[str, int]] = {}
        for fenomeno, tope in cuotas.items():
            if tope <= 0:
                continue
            doc_ids = coleccion.distinct("doc_id", {"fenomeno": fenomeno})
            subcarpetas = sorted({self._subcarpeta_de(d) for d in doc_ids})
            sub_cuotas = self._cuotas_por_subcarpeta(tope, subcarpetas)
            reparto[fenomeno] = sub_cuotas
            for sub, sub_tope in sub_cuotas.items():
                if sub_tope <= 0:
                    continue
                docs = (
                    coleccion.find({"fenomeno": fenomeno, "doc_id": {"$regex": "^" + re.escape(sub)}})
                    .sort("chunk_id", ASCENDING)
                    .limit(sub_tope)
                )
                chunks.extend(Chunk.model_validate(d) for d in docs)
        logger.info(
            "Lote balanceado: %d chunks en %d fenómenos con cuotas %s | reparto por subcarpeta %s",
            len(chunks), n_fenomenos, cuotas, reparto,
        )
        return chunks

    def mark_encoder_procesado(self, chunk_id: str, encoder_name: str) -> None:
        """Añade ``encoder_name`` a ``encoders_procesados`` del chunk (idempotente)."""
        self.connect()
        coleccion = self._cliente[self._db_name][self._collection_name]  # type: ignore[union-attr]
        coleccion.update_one({"chunk_id": chunk_id}, {"$addToSet": {"encoders_procesados": encoder_name}})
