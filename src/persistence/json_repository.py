"""Repositorio de chunks respaldado por un unico archivo JSON.

El archivo contiene una lista JSON de objetos. Cada objeto guarda la metadata
de la Tabla 1, por lo que es un artefacto portable y auditable antes de crear
los vectores. Con ``solo_obligatorios=False`` tambien conserva los campos
recomendados (``idioma``, ``hash_texto``, ``fecha_publicacion``...), que es el
modo con el que se genera ``metadata.json`` cuando este es la FUENTE DE VERDAD
de los chunks (sustituye a MongoDB).

Lectura y escritura son incrementales (``iter_chunks`` / ``write_all``): el
corpus completo son cientos de miles de fragmentos y ~500 MB de JSON, asi que
nunca se materializa a la vez el texto crudo del archivo y su version parseada.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Iterator, List

from src.models.chunk import Chunk
from src.persistence.base_repository import ChunkRepository

logger = logging.getLogger(__name__)

#: Tamano de bloque de lectura del parser incremental (1 MiB).
_TAMANO_BLOQUE = 1 << 20
#: Por debajo de este residuo la ventana se compacta y se rellena (256 KiB).
_MINIMO_RESIDUO = 1 << 18
#: Caracteres que separan elementos dentro de la lista JSON.
_SEPARADORES = ", \t\r\n"


class JsonChunkRepositoryError(RuntimeError):
    """El archivo JSON no pudo leerse o no tiene el esquema esperado."""


class _VentanaJson:
    """Ventana deslizante sobre un archivo que contiene una lista JSON.

    Colabora con :class:`json.JSONDecoder` para decodificar los elementos de
    la lista uno a uno sin materializar el archivo completo. No es parte de
    la API publica del repositorio.
    """

    __slots__ = ("_archivo", "_buffer", "_eof", "pos")

    def __init__(self, archivo) -> None:
        self._archivo = archivo
        self._buffer = ""
        self._eof = False
        self.pos = 0

    def _rellenar(self) -> bool:
        """Compacta la ventana y lee un bloque mas. ``False`` si ya no hay."""
        if self._eof:
            return False
        bloque = self._archivo.read(_TAMANO_BLOQUE)
        if not bloque:
            self._eof = True
            return False
        self._buffer = self._buffer[self.pos :] + bloque
        self.pos = 0
        return True

    def abrir_lista(self, ruta: Path) -> None:
        """Consume el ``[`` inicial, validando que el JSON sea una lista."""
        while self.pos >= len(self._buffer) or not self._buffer[self.pos :].strip():
            if not self._rellenar():
                return
        while self.pos < len(self._buffer) and self._buffer[self.pos] in _SEPARADORES:
            self.pos += 1
        if self.pos >= len(self._buffer) or self._buffer[self.pos] != "[":
            raise JsonChunkRepositoryError(
                f"El repositorio JSON '{ruta}' debe contener una lista de chunks"
            )
        self.pos += 1

    def saltar_separadores(self) -> bool:
        """Avanza hasta el siguiente elemento. ``False`` si la lista termino."""
        while True:
            while self.pos < len(self._buffer) and self._buffer[self.pos] in _SEPARADORES:
                self.pos += 1
            if self.pos < len(self._buffer):
                return self._buffer[self.pos] != "]"
            if not self._rellenar():
                return False

    def decodificar(self, decoder: json.JSONDecoder) -> tuple:
        """Decodifica el elemento en ``pos``, ampliando la ventana si no cabe.

        Returns:
            ``(objeto, indice_final)`` tal como los devuelve ``raw_decode``.
        """
        while True:
            if len(self._buffer) - self.pos < _MINIMO_RESIDUO:
                self._rellenar()
            try:
                return decoder.raw_decode(self._buffer, self.pos)
            except json.JSONDecodeError:
                if not self._rellenar():
                    raise


class JsonChunkRepository(ChunkRepository):
    """Persistencia idempotente de chunks en un archivo ``.json``.

    ``chunk_id`` es la identidad de cada registro. Las escrituras se realizan
    en un temporal situado junto al JSON final y se reemplazan atomicamente.

    Args:
        path: ruta del archivo ``.json`` con la lista de chunks.
        solo_obligatorios: si ``True`` (por defecto) cada registro guarda
            exclusivamente los ocho campos obligatorios de la Tabla 1; si
            ``False`` guarda tambien los recomendados. La LECTURA acepta
            ambos esquemas indistintamente.
    """

    def __init__(self, path: str | Path, solo_obligatorios: bool = True) -> None:
        self.path = Path(path)
        self.solo_obligatorios = solo_obligatorios

    def connect(self) -> None:
        """Prepara el directorio destino; mantiene paridad con MongoDB."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """No hay una conexion persistente que cerrar para un archivo local."""

    def save_many(self, chunks: List[Chunk]) -> None:
        """Hace upsert por ``chunk_id`` y reescribe el archivo completo."""
        if not chunks:
            return
        self.connect()
        registros = {chunk.chunk_id: chunk for chunk in self._load_chunks()}
        registros.update({chunk.chunk_id: chunk for chunk in chunks})
        ordenados = sorted(registros.values(), key=_orden_estable)
        self._write_chunks(ordenados)
        logger.info("Persistidos %d fragmentos en %s", len(chunks), self.path)

    def write_all(self, chunks: Iterable[Chunk]) -> int:
        """Escribe ``chunks`` (ya ordenados) reemplazando todo el archivo.

        A diferencia de :meth:`save_many` no lee el archivo previo ni acumula
        la lista en memoria: consume el iterable y lo serializa registro a
        registro. Es la via para volcar un corpus completo (p. ej. exportar
        MongoDB a ``metadata.json``) sin coste cuadratico ni picos de RAM.

        Returns:
            Numero de registros escritos.
        """
        self.connect()
        return self._write_chunks(chunks)

    def find_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Recupera los fragmentos de ``doc_id`` ordenados por posicion."""
        coincidencias = [c for c in self.iter_chunks() if c.doc_id == doc_id]
        return sorted(coincidencias, key=_orden_estable)

    def find_all(self, limite: int = 0) -> List[Chunk]:
        """Recupera todos los chunks en orden estable, con limite opcional."""
        chunks = self._load_chunks()
        return chunks[:limite] if limite > 0 else chunks

    def exists(self, chunk_id: str) -> bool:
        """Indica si ``chunk_id`` ya existe en el archivo JSON."""
        return any(chunk.chunk_id == chunk_id for chunk in self.iter_chunks())

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

    @property
    def path_comprimido(self) -> Path:
        """Ruta de la variante ``.gz`` del archivo (la que se versiona en git)."""
        return self.path.with_name(self.path.name + ".gz")

    def _ruta_efectiva(self) -> Path | None:
        """Archivo que se va a leer: el plano, o el ``.gz`` si aquel no esta.

        El JSON crudo del corpus completo son cientos de MB, demasiado para
        versionarlo en git, asi que en el repositorio vive comprimido. Al
        resolver aqui el respaldo, ``CHUNKS_JSON_PATH=metadata.json`` funciona
        igual con un clon recien hecho (que solo trae ``metadata.json.gz``) que
        con una copia ya descomprimida — sin configuracion ni pasos manuales.
        """
        if self.path.exists():
            return self.path
        if self.path_comprimido.exists():
            return self.path_comprimido
        return None

    def _abrir(self, ruta: Path):
        """Abre ``ruta`` en texto UTF-8, descomprimiendo si termina en ``.gz``."""
        if ruta.suffix == ".gz":
            return gzip.open(ruta, "rt", encoding="utf-8")
        return ruta.open("r", encoding="utf-8")

    def iter_chunks(self) -> Iterator[Chunk]:
        """Itera los chunks del archivo sin cargarlo entero en memoria.

        Parsea la lista JSON de forma incremental (un objeto a la vez), de
        modo que el pico de RAM es el de los ``Chunk`` construidos y no el
        del texto crudo del archivo mas su arbol de dicts.
        """
        if self._ruta_efectiva() is None:
            return
        try:
            for registro in self._iter_objetos_json():
                yield Chunk.model_validate(registro)
        except (TypeError, ValueError) as exc:
            raise JsonChunkRepositoryError(
                f"El repositorio JSON '{self.path}' contiene un chunk invalido: {exc}"
            ) from exc

    def _iter_objetos_json(self) -> Iterator[dict]:
        """Emite los objetos de la lista JSON del archivo, uno a uno.

        Mantiene una ventana deslizante sobre el archivo: se compacta cuando
        el residuo baja de :data:`_MINIMO_RESIDUO` y se amplia solo si un
        objeto no cabe entero en ella, asi que ni el texto crudo ni la lista
        completa llegan a estar en memoria.
        """
        decoder = json.JSONDecoder()
        ruta = self._ruta_efectiva()
        if ruta is None:
            return
        try:
            with self._abrir(ruta) as archivo:
                ventana = _VentanaJson(archivo)
                ventana.abrir_lista(ruta)
                while True:
                    if not ventana.saltar_separadores():
                        return
                    objeto, fin = ventana.decodificar(decoder)
                    if not isinstance(objeto, dict):
                        raise JsonChunkRepositoryError(
                            f"El repositorio JSON '{self.path}' debe contener objetos chunk"
                        )
                    ventana.pos = fin
                    yield objeto
        except (OSError, json.JSONDecodeError) as exc:
            raise JsonChunkRepositoryError(
                f"No se pudo leer el repositorio JSON '{self.path}': {exc}"
            ) from exc

    def _load_chunks(self) -> List[Chunk]:
        return sorted(self.iter_chunks(), key=_orden_estable)

    def _write_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Serializa ``chunks`` a un temporal y lo mueve atomicamente."""
        campo = "como_dict_json" if self.solo_obligatorios else "como_dict_metadata"
        escritos = 0
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
                archivo.write("[\n")
                for chunk in chunks:
                    if escritos:
                        archivo.write(",\n")
                    json.dump(getattr(chunk, campo), archivo, ensure_ascii=False)
                    escritos += 1
                archivo.write("\n]\n")
            os.replace(temporal, self.path)
            temporal = None
        except OSError as exc:
            raise JsonChunkRepositoryError(
                f"No se pudo guardar el repositorio JSON '{self.path}': {exc}"
            ) from exc
        finally:
            if temporal is not None and temporal.exists():
                temporal.unlink()
        return escritos


def _orden_estable(chunk: Chunk) -> tuple:
    """Clave de orden determinista y reproducible entre corridas."""
    return (chunk.doc_id, chunk.posicion, chunk.chunk_id)
