"""
Resultado de la ingesta de un archivo, con conteos y advertencias.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestionResult(BaseModel):
    """Resumen del procesamiento de un documento.

    Atributos:
        doc_id: Identificador del documento procesado.
        fuente: Archivo de origen.
        formato: Formato detectado.
        fenomeno: Fenómeno asignado (1, 2 o 3).
        status: ``ok`` si se persistió al menos un fragmento, ``error`` si falló.
        num_chunks: Total de fragmentos generados.
        num_guardados: Fragmentos efectivamente persistidos.
        num_rechazados: Fragmentos rechazados por validación dura.
        errores: Motivos de error del archivo (vacío si no hubo).
        warnings: Advertencias acumuladas (blandas).
        tiempo_seg: Segundos que tomó el procesamiento del archivo.
    """

    doc_id: str = Field(default="")
    fuente: str = Field(default="")
    formato: Optional[str] = Field(default=None)
    fenomeno: Optional[int] = Field(default=None)
    status: str = Field(default="ok", description="ok | error")
    num_chunks: int = Field(default=0)
    num_guardados: int = Field(default=0)
    num_rechazados: int = Field(default=0)
    errores: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    tiempo_seg: float = Field(default=0.0)
