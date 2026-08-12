"""Repositorio de chunks respaldado por un unico archivo JSON.

El archivo contiene una lista JSON de objetos. Cada objeto guarda exactamente
la metadata obligatoria de la Tabla 1, por lo que es un artefacto portable y
auditable antes de crear los vectores.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

from src.models.chunk import Chunk
from src.persistence.base_repository import ChunkRepository

logger = logging.getLogger(__name__)


class JsonChunkRepositoryError(RuntimeError):
    """El archivo JSON no pudo leerse o no tiene el esquema esperado."""


class JsonChunkRepository(ChunkRepository):
    """Persistencia idempotente de chunks en un archivo ``.json``.

    ``chunk_id`` es la identidad de cada registro. Las escrituras se realizan
    en un temporal situado junto al JSON final y se reemplazan atomicamente.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> None:
        """Prepara el directorio destino; mantiene paridad con MongoDB."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """No hay una conexion persistente que cerrar para un archivo local."""

    def save_many(self, chunks: List[Chunk]) -> None:
        """Hace upsert por ``chunk_id`` y guarda solo la metadata obligatoria."""
        if not chunks:
            return
        self.connect()
        registros = {chunk.chunk_id: chunk for chunk in self._load_chunks()}
        registros.update({chunk.chunk_id: chunk for chunk in chunks})
        ordenados = sorted(
            registros.values(), key=lambda chunk: (chunk.doc_id, chunk.posicion, chunk.chunk_id)
        )
        self._write_chunks(ordenados)
        logger.info("Persistidos %d fragmentos en %s", len(chunks), self.path)

    def find_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Recupera los fragmentos de ``doc_id`` ordenados por posicion."""
        return [chunk for chunk in self._load_chunks() if chunk.doc_id == doc_id]

    def find_all(self, limite: int = 0) -> List[Chunk]:
        """Recupera todos los chunks en orden estable, con limite opcional."""
        chunks = self._load_chunks()
        return chunks[:limite] if limite > 0 else chunks

    def exists(self, chunk_id: str) -> bool:
        """Indica si ``chunk_id`` ya existe en el archivo JSON."""
        return any(chunk.chunk_id == chunk_id for chunk in self._load_chunks())

    def find_all_balanceado(self, limite: int, n_fenomenos: int = 3) -> List[Chunk]:
        """Recupera un lote equilibrado por fenomeno y subcarpeta del corpus."""
        if limite <= 0:
            return self.find_all()
        cuotas = self._cuotas_por_fenomeno(limite, n_fenomenos)
        por_fenomeno: Dict[int, List[Chunk]] = {}
        for chunk in self._load_chunks():
            por_fenomeno.setdefault(chunk.fenomeno, []).append(chunk)

        seleccionados: List[Chunk] = []
        for fenomeno, cuota_fenomeno in cuotas.items():
            por_subcarpeta: Dict[str, List[Chunk]] = {}
            for chunk in por_fenomeno.get(fenomeno, []):
                por_subcarpeta.setdefault(self._subcarpeta_de(chunk.doc_id), []).append(chunk)
            cuotas_subcarpetas = self._cuotas_por_subcarpeta(
                cuota_fenomeno, list(por_subcarpeta)
            )
            for subcarpeta, cuota_subcarpeta in cuotas_subcarpetas.items():
                seleccionados.extend(por_subcarpeta[subcarpeta][:cuota_subcarpeta])
        return seleccionados

    @staticmethod
    def _cuotas_por_fenomeno(limite: int, n_fenomenos: int = 3) -> Dict[int, int]:
        if limite <= 0:
            return {}
        cuota, residuo = divmod(limite, n_fenomenos)
        return {
            fenomeno: cuota + (1 if fenomeno <= residuo else 0)
            for fenomeno in range(1, n_fenomenos + 1)
        }

    @staticmethod
    def _subcarpeta_de(doc_id: str) -> str:
        for separador in ("\\", "/"):
            if separador in doc_id:
                return separador.join(doc_id.split(separador)[:2])
        return doc_id

    @staticmethod
    def _cuotas_por_subcarpeta(limite: int, subcarpetas: List[str]) -> Dict[str, int]:
        if limite <= 0 or not subcarpetas:
            return {}
        cuota, residuo = divmod(limite, len(subcarpetas))
        return {
            subcarpeta: cuota + (1 if indice < residuo else 0)
            for indice, subcarpeta in enumerate(sorted(subcarpetas))
        }

    def _load_chunks(self) -> List[Chunk]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
        except (OSError, json.JSONDecodeError) as exc:
            raise JsonChunkRepositoryError(
                f"No se pudo leer el repositorio JSON '{self.path}': {exc}"
            ) from exc
        if not isinstance(datos, list):
            raise JsonChunkRepositoryError(
                f"El repositorio JSON '{self.path}' debe contener una lista de chunks"
            )
        try:
            chunks = [Chunk.model_validate(registro) for registro in datos]
        except (TypeError, ValueError) as exc:
            raise JsonChunkRepositoryError(
                f"El repositorio JSON '{self.path}' contiene un chunk invalido: {exc}"
            ) from exc
        return sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.posicion, chunk.chunk_id))

    def _write_chunks(self, chunks: List[Chunk]) -> None:
        temporal: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as archivo:
                temporal = Path(archivo.name)
                json.dump(
                    [chunk.como_dict_json for chunk in chunks],
                    archivo,
                    ensure_ascii=False,
                    indent=2,
                )
                archivo.write("\n")
            os.replace(temporal, self.path)
        except OSError as exc:
            raise JsonChunkRepositoryError(
                f"No se pudo guardar el repositorio JSON '{self.path}': {exc}"
            ) from exc
        finally:
            if temporal is not None and temporal.exists():
                temporal.unlink()
