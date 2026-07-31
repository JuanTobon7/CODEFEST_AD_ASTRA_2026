"""
Modelo de documento extraído y sus secciones estructurales.

Un :class:`ExtractedDocument` representa un archivo de entrada del corpus ya
procesado por un extractor y limpiado por el cleaner. El texto se entrega
segmentado en :class:`Section` (fase 1 del chunking híbrido) siempre que el
formato lo permita.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Formato(str, Enum):
    """Formatos de archivo soportados por el pipeline (Tabla 1 del reto)."""

    PDF = "pdf"
    HTML = "html"
    MD = "md"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    IMAGE = "image"
    PBF = "pbf"
    TXT = "txt"


class Section(BaseModel):
    """Unidad estructural cohesiva dentro de un documento.

    Atributos:
        titulo: Encabezado o título de la sección (si existe).
        texto: Contenido de la sección, ya limpio.
        orden: Posición ordinal de la sección en el documento (0-based).
        splittable: Indica si la sección puede subdividirse por tamaño.
            ``False`` para unidades atómicas (filas CSV/XLSX, elementos PBF),
            que nunca se parten ni se fusionan en la fase 2.
    """

    titulo: Optional[str] = Field(default=None, description="Encabezado estructural de la sección")
    texto: str = Field(default="", description="Texto de la sección, sin modificaciones")
    orden: int = Field(default=0, ge=0, description="Índice ordinal de la sección (0-based)")
    splittable: bool = Field(default=True, description="Si la sección admite subdivisión por tamaño")

    @field_validator("titulo")
    @classmethod
    def _titulo_no_vacio(cls, valor: Optional[str]) -> Optional[str]:
        """Evita títulos vacíos que ensucien la metadata."""
        if valor is not None and not valor.strip():
            return None
        return valor


class ExtractedDocument(BaseModel):
    """Documento extraído del archivo de origen, listo para fragmentar.

    Atributos:
        doc_id: Identificador único e inmutable del documento.
        fuente: Nombre o URL del archivo original.
        formato: Formato del archivo de origen.
        fenomeno: Fenómeno asociado (1, 2 o 3).
        secciones: Lista ordenada de secciones estructurales.
        idioma: Código de idioma detectado (``es``, ``en``, ``pt``...).
        titulo_documento: Título del documento si se pudo determinar.
        fecha_publicacion: Fecha de publicación si está disponible.
        metadata: Campos descriptivos adicionales del documento.
    """

    doc_id: str = Field(
        default="",
        description="ID único e inmutable del documento (lo asigna el pipeline tras extraer)",
    )
    fuente: str = Field(description="Nombre o URL del archivo original")
    formato: Formato = Field(description="Formato del archivo de origen")
    fenomeno: int = Field(default=1, ge=1, le=3, description="Fenómeno: 1, 2 o 3")
    secciones: List[Section] = Field(default_factory=list)
    idioma: Optional[str] = Field(default=None)
    titulo_documento: Optional[str] = Field(default=None)
    fecha_publicacion: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def texto_completo(self) -> str:
        """Texto completo del documento (todas las secciones, en orden)."""
        secciones_ordenadas = sorted(self.secciones, key=lambda s: s.orden)
        return "\n\n".join(s.texto for s in secciones_ordenadas if s.texto)
