"""
Resumen agregado de un lote de ingesta (múltiples archivos).
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field

from src.models.pipeline_result import IngestionResult

logger = logging.getLogger("ingestion")


class BatchSummary(BaseModel):
    """Resumen del procesamiento de un lote de archivos.

    Se construye a partir de los :class:`IngestionResult` individuales y es
    responsable de su propia presentación en el log (alta cohesión).
    """

    total: int = Field(default=0, description="Archivos procesados")
    ok: int = Field(default=0, description="Archivos con status 'ok'")
    error: int = Field(default=0, description="Archivos con status 'error'")
    chunks_guardados: int = Field(default=0, description="Fragmentos persistidos")
    errores: List[str] = Field(
        default_factory=list, description="Líneas 'fuente | motivo' de cada error"
    )

    @classmethod
    def from_results(cls, resultados: List[IngestionResult]) -> "BatchSummary":
        """Agrega los resultados individuales del lote."""
        ok = [r for r in resultados if r.status == "ok"]
        error = [r for r in resultados if r.status == "error"]
        return cls(
            total=len(resultados),
            ok=len(ok),
            error=len(error),
            chunks_guardados=sum(r.num_guardados for r in ok),
            errores=[
                f"{r.fuente} | {'; '.join(r.errores)}" for r in error if r.errores
            ],
        )

    def log_resumen(self) -> None:
        """Escribe el resumen del lote en el log estructurado."""
        logger.info(
            "RESUMEN | archivos=%d | ok=%d | error=%d | chunks_guardados=%d",
            self.total,
            self.ok,
            self.error,
            self.chunks_guardados,
        )
        for error_linea in self.errores:
            logger.warning("  ERROR  | %s", error_linea)
