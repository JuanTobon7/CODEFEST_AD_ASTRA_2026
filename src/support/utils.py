"""
Utilidades varias: hash de texto, temporización y extracción de extensión.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def hash_texto(texto: str) -> str:
    """Hash SHA-256 del texto (para deduplicación de fragmentos)."""
    return hashlib.sha256(texto.encode("utf-8", errors="replace")).hexdigest()


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
