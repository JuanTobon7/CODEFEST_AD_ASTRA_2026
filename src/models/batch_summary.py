"""
Resumen agregado de un lote de ingesta (múltiples archivos).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

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
    archivos_por_fenomeno: Dict[int, int] = Field(
        default_factory=dict, description="Archivos procesados por fenómeno (1, 2, 3)"
    )
    chunks_por_fenomeno: Dict[int, int] = Field(
        default_factory=dict, description="Fragmentos persistidos por fenómeno (1, 2, 3)"
    )
    errores: List[str] = Field(
        default_factory=list, description="Líneas 'fuente | motivo' de cada error"
    )
    rechazos_detalle: List[Dict] = Field(
        default_factory=list, description="Detalle de rechazos de todo el lote (modo debug)"
    )

    @classmethod
    def from_results(cls, resultados: List[IngestionResult]) -> "BatchSummary":
        """Agrega los resultados individuales del lote."""
        ok = [r for r in resultados if r.status == "ok"]
        error = [r for r in resultados if r.status == "error"]
        archivos_por_fenomeno: Dict[int, int] = defaultdict(int)
        chunks_por_fenomeno: Dict[int, int] = defaultdict(int)
        for resultado in resultados:
            if resultado.fenomeno is None:
                continue
            archivos_por_fenomeno[resultado.fenomeno] += 1
            if resultado.status == "ok":
                chunks_por_fenomeno[resultado.fenomeno] += resultado.num_guardados
        return cls(
            total=len(resultados),
            ok=len(ok),
            error=len(error),
            chunks_guardados=sum(r.num_guardados for r in ok),
            archivos_por_fenomeno=dict(sorted(archivos_por_fenomeno.items())),
            chunks_por_fenomeno=dict(sorted(chunks_por_fenomeno.items())),
            errores=[
                f"{r.fuente} | {'; '.join(r.errores)}" for r in error if r.errores
            ],
            rechazos_detalle=[
                rechazo for r in resultados for rechazo in r.rechazos_detalle
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
        if self.archivos_por_fenomeno or self.chunks_por_fenomeno:
            archivos = ", ".join(
                f"F{fenomeno}={n}"
                for fenomeno, n in sorted(self.archivos_por_fenomeno.items())
            )
            chunks = ", ".join(
                f"F{fenomeno}={n}"
                for fenomeno, n in sorted(self.chunks_por_fenomeno.items())
            )
            logger.info("FENÓMENOS | archivos: %s | chunks: %s", archivos, chunks)
        for error_linea in self.errores:
            logger.warning("  ERROR  | %s", error_linea)

    # Diagnóstico de rechazos (modo debug) --------------------------------------

    def escribir_log_rechazos(
        self, ruta: Path, umbral_ratio: float = 0.5, max_texto: int = 400
    ) -> None:
        """Vuelca a un archivo el detalle de los chunks rechazados.

        Escribe la causa (regla de negocio) de cada rechazo; para los
        documentos que rechazan la mayoría de sus fragmentos (ratio >=
        ``umbral_ratio``) incluye además el texto del chunk truncado.

        Args:
            ruta: Archivo .txt de salida (se crean los directorios).
            umbral_ratio: Fracción mínima de rechazos para volcar el detalle completo.
            max_texto: Caracteres de texto del chunk a incluir por rechazo.
        """
        if not self.rechazos_detalle:
            Path(ruta).parent.mkdir(parents=True, exist_ok=True)
            Path(ruta).write_text("Sin rechazos de chunks en este lote.\n", encoding="utf-8")
            return

        por_doc: Dict[str, Dict] = defaultdict(
            lambda: {"max_pos": -1, "rechazados": 0}
        )
        for detalle in self.rechazos_detalle:
            clave = detalle["doc_id"]
            por_doc[clave]["rechazados"] += 1
            por_doc[clave]["max_pos"] = max(por_doc[clave]["max_pos"], detalle["posicion"])

        conteo_reglas: Counter = Counter(d["regla"] for d in self.rechazos_detalle)
        lineas: List[str] = [
            "=" * 90,
            "LOG DE RECHAZOS DE CHUNKS (modo debug)",
            f"total rechazos={len(self.rechazos_detalle)}",
            "reglas incumplidas: "
            + ", ".join(f"{regla}={n}" for regla, n in conteo_reglas.most_common()),
            "=" * 90,
            "",
        ]

        # Documentos con rechazos, ordenados por ratio de rechazo (descendente).
        docs = sorted(
            por_doc.items(),
            key=lambda kv: -(kv[1]["rechazados"] / max(kv[1]["max_pos"] + 1, 1)),
        )
        for doc_id, datos in docs:
            total_estimado = max(datos["max_pos"] + 1, datos["rechazados"])
            ratio = datos["rechazados"] / max(total_estimado, 1)
            flag = " <<< MAYORÍA RECHAZADA" if ratio >= umbral_ratio else ""
            lineas.append(
                f"DOC doc_id={doc_id} | rechazados={datos['rechazados']}/{total_estimado} "
                f"ratio={ratio:.0%}{flag}"
            )
            for detalle in self.rechazos_detalle:
                if detalle["doc_id"] != doc_id:
                    continue
                texto = detalle.get("texto") or ""
                if len(texto) > max_texto:
                    texto = texto[:max_texto] + "…"
                texto = " ".join(texto.split())
                lineas.append(
                    f"  {detalle['chunk_id']} | regla={detalle['regla']} | "
                    f"pos={detalle['posicion']} | {detalle['motivo']}"
                )
                if ratio >= umbral_ratio and texto:
                    lineas.append(f"      texto: {texto}")
            lineas.append("")

        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("\n".join(lineas), encoding="utf-8")
        logger.info(
            "Log de rechazos (debug) escrito en %s (%d rechazos)", ruta, len(self.rechazos_detalle)
        )
