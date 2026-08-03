"""
Utilidades varias: hash de texto, temporización y extracción de extensión.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def hash_texto(texto: str) -> str:
    """Hash SHA-256 del texto (para deduplicación de fragmentos)."""
    return hashlib.sha256(texto.encode("utf-8", errors="replace")).hexdigest()


def en_lotes(items: Sequence[T], tam: int = 2000) -> Iterator[List[T]]:
    """Parte ``items`` en lotes de a lo sumo ``tam`` elementos.

    Usado para consultas Mongo con ``$in`` sobre listas grandes de
    ``chunk_id``: un solo ``$in`` con decenas de miles de IDs puede exceder
    el límite de 16MB por documento BSON.
    """
    for inicio in range(0, len(items), tam):
        yield list(items[inicio : inicio + tam])


def extension_de(path: Path) -> str:
    """Extensión en minúsculas y sin punto, p. ej. ``.PDF`` -> ``pdf``."""
    return path.suffix.lower().lstrip(".")


def slugify(texto: str) -> str:
    """Convierte texto en un identificador seguro (minúsculas, sin espacios)."""
    import re

    texto = texto.strip().lower()
    texto = re.sub(r"[^a-z0-9áéíóúñü]+", "_", texto)
    return texto.strip("_")


@contextmanager
def cronometrar() -> Iterator[float]:
    """Mide segundos transcurridos dentro del bloque ``with``."""
    inicio = time.perf_counter()
    yield lambda: time.perf_counter() - inicio
