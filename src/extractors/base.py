"""
Clase base abstracta de extractores (patrón Factory Method).

Cada extractor conoce los formatos que soporta y expone ``extract``, que
devuelve un :class:`ExtractedDocument` con el texto ya segmentado en
secciones estructurales cuando el formato lo permite.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List

from src.models.extracted_document import ExtractedDocument

logger = logging.getLogger(__name__)


class ExtractorError(Exception):
    """Error controlado durante la extracción de un archivo."""


class BaseExtractor(ABC):
    """Contrato común de todos los extractores.

    Atributos de clase:
        supported_formats: Extensiones de archivo soportadas (sin punto).
    """

    supported_formats: ClassVar[List[str]] = []

    @abstractmethod
    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el texto estructurado de ``filepath``.

        Responsabilidades de la implementación:
        1. Validar la existencia/legibilidad del archivo.
        2. Extraer el texto crudo respetando el orden de lectura.
        3. Segmentar en :class:`Section` (fase 1 del chunking híbrido)
           cuando el formato lo permita.
        4. Completar metadata de documento (idioma, título, fecha).
        5. Lanzar :class:`ExtractorError` con mensaje detallado (archivo + causa)
           ante cualquier fallo, sin tragar excepciones a ciegas.
        """

    # Utilidades compartidas por las implementaciones ------------------------

    @staticmethod
    def _leer_bytes(filepath: Path) -> bytes:
        """Lee el archivo como bytes con manejo explícito de errores."""
        if not filepath.exists():
            raise ExtractorError(f"El archivo no existe: {filepath}")
        if not filepath.is_file():
            raise ExtractorError(f"La ruta no es un archivo regular: {filepath}")
        try:
            return filepath.read_bytes()
        except OSError as exc:
            raise ExtractorError(f"No se pudo leer {filepath}: {exc}") from exc

    @staticmethod
    def _decodificar(data: bytes, fallback: str = "utf-8") -> str:
        """Decodifica bytes a texto; ante errores de codificación usa UTF-8 con
        sustitución (el cleaner normaliza la codificación después)."""
        for encoding in (fallback, "utf-8", "latin-1"):
            try:
                return data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return data.decode("utf-8", errors="replace")
