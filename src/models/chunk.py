"""
Modelo de fragmento (chunk) y configuración de fragmentación.

``Chunk`` es la unidad mínima indexable que se persiste en MongoDB con la
metadata obligatoria de la Tabla 1 del reto (CODEFEST AD ASTRA 2026).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, List, Optional

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> str:
    """Marca temporal UTC en formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


class Chunk(BaseModel):
    """Fragmento de texto indexable con metadata obligatoria.

    Campos obligatorios (Tabla 1): ``doc_id``, ``chunk_id``, ``fuente``,
    ``formato``, ``fenomeno``, ``posicion``, ``num_tokens``, ``texto``.
    El resto son recomendados/deseables.
    """

    doc_id: str = Field(description="ID único del documento de origen")
    chunk_id: str = Field(description="ID único del fragmento dentro del documento")
    fuente: str = Field(description="Nombre o URL del archivo original")
    formato: str = Field(description="Formato del archivo: pdf, html, md, json, csv, xlsx, image, pbf")
    fenomeno: int = Field(ge=1, le=3, description="Fenómeno: 1, 2 o 3")
    posicion: int = Field(ge=0, description="Índice ordinal del fragmento (empieza en 0)")
    num_tokens: int = Field(ge=0, description="Conteo de tokens del fragmento")
    texto: str = Field(description="Texto original del fragmento, sin modificaciones")

    # Campos recomendados (deseables)
    idioma: Optional[str] = Field(default=None)
    titulo_documento: Optional[str] = Field(default=None)
    fecha_publicacion: Optional[str] = Field(default=None)
    chunking_strategy: Optional[str] = Field(default=None)
    seccion: Optional[str] = Field(default=None, description="Encabezado estructural de origen")
    overlap_con: Optional[str] = Field(default=None, description="chunk_id anterior con solapamiento")
    hash_texto: Optional[str] = Field(default=None, description="Hash del texto para deduplicación")
    created_at: Optional[str] = Field(default=None)

    # Campos de auditoría interna
    validation_warnings: List[str] = Field(default_factory=list)

    # Esquema de entrega: estos campos deben estar presentes en cada registro
    # persistido en el archivo JSON de chunks.
    CAMPOS_METADATA_OBLIGATORIA: ClassVar[tuple[str, ...]] = (
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    )

    @field_validator("doc_id", "chunk_id", "fuente")
    @classmethod
    def _campos_no_vacios(cls, valor: str, info) -> str:
        """Los identificadores y la fuente no pueden ser vacíos."""
        if not valor.strip():
            raise ValueError(f"{info.field_name} no puede ser vacío")
        return valor

    @property
    def como_dict_mongo(self) -> dict:
        """Representación apta para persistir en MongoDB."""
        datos = self.model_dump()
        if datos.get("created_at") is None:
            datos["created_at"] = _utc_now()
        return datos

    @property
    def como_dict_json(self) -> dict:
        """Metadata canonica para un registro del repositorio JSON.

        El artefacto de chunks conserva exclusivamente los ocho campos
        obligatorios de la Tabla 1 y no incluye campos internos de ejecucion.
        """
        return self.model_dump(include=set(self.CAMPOS_METADATA_OBLIGATORIA))
