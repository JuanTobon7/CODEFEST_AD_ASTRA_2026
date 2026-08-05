"""
Modelo de consulta de evaluación (Sección 9 del reto CODEFEST AD ASTRA 2026).

Una :class:`Query` es una de las 50 preguntas de evaluación identificadas
``q001``..``q050``. El texto se conserva tal cual viene del PDF oficial
(solo se normaliza el espacio en blanco), sin reescritura ni resumen.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

QUERY_ID_PATTERN = re.compile(r"^q\d{3}$")


def _utc_now() -> str:
    """Marca temporal UTC en formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


class Query(BaseModel):
    """Una de las 50 consultas de evaluación (q001..q050)."""

    query_id: str = Field(description="Identificador qNNN de la consulta")
    query_text: str = Field(description="Texto de la pregunta, sin modificaciones")
    created_at: Optional[str] = Field(default=None)

    @field_validator("query_id")
    @classmethod
    def _query_id_valido(cls, valor: str) -> str:
        """El ID sigue el patrón exacto qNNN (3 dígitos con cero a la izquierda)."""
        texto = valor.strip()
        if not QUERY_ID_PATTERN.match(texto):
            raise ValueError(f"query_id debe seguir el patrón qNNN: '{valor}'")
        return texto

    @field_validator("query_text")
    @classmethod
    def _texto_no_vacio(cls, valor: str) -> str:
        """El texto de la consulta no puede estar vacío."""
        if not valor.strip():
            raise ValueError("query_text no puede ser vacío")
        return valor.strip()

    @property
    def como_dict_mongo(self) -> dict:
        """Representación apta para persistir en MongoDB."""
        datos = self.model_dump()
        if datos.get("created_at") is None:
            datos["created_at"] = _utc_now()
        return datos
