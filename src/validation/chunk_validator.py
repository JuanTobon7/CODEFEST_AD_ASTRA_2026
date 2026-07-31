"""
ChunkValidator: validaciones duras y blandas antes de guardar en Mongo.

Validaciones duras (rechazo del fragmento):
- Campos obligatorios presentes y con el tipo correcto (Tabla 1).
- ``num_tokens`` dentro del límite del encoder (512 por defecto).
- ``posicion`` creciente y sin huecos dentro de un mismo ``doc_id``.
- ``chunk_id`` único dentro del documento.

Validaciones blandas (warning, se guarda con ``validation_warnings``):
- El texto termina en puntuación terminal (no corta una oración), salvo
  unidades atómicas justificadas (filas de tabla, elementos PBF).
- El texto empieza en límite de oración (no arranca a mitad).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set

from src.models.chunk import Chunk
from src.support.sentence_splitter import PUNTUACION_TERMINAL

logger = logging.getLogger(__name__)

# Campos obligatorios de la Tabla 1 y sus tipos esperados.
_CAMPOS_OBLIGATORIOS: dict = {
    "doc_id": str,
    "chunk_id": str,
    "fuente": str,
    "formato": str,
    "fenomeno": int,
    "posicion": int,
    "num_tokens": int,
    "texto": str,
}

_RE_FIN_PUNTUACION = re.compile(r"[.!?»›]$")
_RE_INICIO_PUNTUACION = re.compile(r"^[¡¿\"'“‘«(]|[A-ZÁÉÍÓÚÑÜ]")


@dataclass
class ValidationResult:
    """Resultado de la validación de un lote de fragmentos."""

    validos: List[Chunk] = field(default_factory=list)
    rechazados: List[Chunk] = field(default_factory=list)
    motivos: List[str] = field(default_factory=list)
    rechazos_detalle: List[dict] = field(default_factory=list)


class ChunkValidator:
    """Valida fragmentos; los que fallan validación dura se descartan."""

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def validate(self, chunks: List[Chunk]) -> ValidationResult:
        """Valida el lote completo.

        Returns:
            :class:`ValidationResult` con los fragmentos válidos (con sus
            warnings) y los rechazados (con el motivo de cada rechazo).
        """
        resultado = ValidationResult()
        vistos: Set[str] = set()
        posicion_esperada = 0
        doc_id_actual: str | None = None

        for chunk in sorted(chunks, key=lambda c: (c.doc_id, c.posicion)):
            if doc_id_actual != chunk.doc_id:
                doc_id_actual = chunk.doc_id
                posicion_esperada = 0
            regla, motivo = self._validar_duro(chunk)
            if motivo is None:
                # Integridad de posiciones y unicidad de chunk_id por documento.
                if chunk.posicion != posicion_esperada:
                    regla, motivo = (
                        "posicion",
                        f"posicion {chunk.posicion} fuera de secuencia "
                        f"(se esperaba {posicion_esperada})",
                    )
                elif chunk.chunk_id in vistos:
                    regla, motivo = "duplicado", f"chunk_id duplicado: {chunk.chunk_id}"

            if motivo is not None:
                logger.warning(
                    "Chunk rechazado | doc=%s pos=%s fuente=%s | [%s] %s",
                    chunk.doc_id,
                    chunk.posicion,
                    chunk.fuente,
                    regla,
                    motivo,
                )
                resultado.rechazados.append(chunk)
                resultado.motivos.append(f"[{regla}] {motivo}")
                resultado.rechazos_detalle.append(
                    {
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "posicion": chunk.posicion,
                        "fuente": chunk.fuente,
                        "regla": regla,
                        "motivo": motivo,
                        "texto": chunk.texto,
                    }
                )
                # Un rechazo por otra regla no debe romper la secuencia de los
                # fragmentos siguientes válidos: si este era el esperado, se
                # avanza la posición para evitar rechazos en cascada.
                if chunk.posicion == posicion_esperada:
                    posicion_esperada += 1
                continue

            vistos.add(chunk.chunk_id)
            posicion_esperada += 1

            # Validaciones blandas (warnings): no impiden el guardado.
            for advertencia in self._validar_blando(chunk):
                if advertencia not in chunk.validation_warnings:
                    chunk.validation_warnings.append(advertencia)
            resultado.validos.append(chunk)
        return resultado

    # Validaciones duras ----------------------------------------------------------

    def _validar_duro(self, chunk: Chunk) -> tuple[str, str | None]:
        """Devuelve ``(regla, motivo)``; ``motivo`` es ``None`` si pasa.

        ``regla`` identifica la regla de negocio incumplida (útil para logs
        y análisis de rechazos): ``obligatorio``, ``tipo``, ``fenomeno``,
        ``tokens``, ``posicion`` o ``duplicado``.
        """
        for campo, tipo in _CAMPOS_OBLIGATORIOS.items():
            valor = getattr(chunk, campo, None)
            if valor is None or (isinstance(valor, str) and not valor.strip()):
                return "obligatorio", f"campo obligatorio ausente: {campo}"
            if not isinstance(valor, tipo):
                return "tipo", f"campo '{campo}' con tipo incorrecto (esperado {tipo.__name__})"
        if chunk.fenomeno not in (1, 2, 3):
            return "fenomeno", f"fenomeno fuera de rango: {chunk.fenomeno}"
        if chunk.num_tokens > self.max_tokens:
            return (
                "tokens",
                f"num_tokens {chunk.num_tokens} supera el límite del encoder "
                f"({self.max_tokens})",
            )
        if chunk.posicion < 0:
            return "posicion", f"posicion negativa: {chunk.posicion}"
        return "ok", None

    # Validaciones blandas ----------------------------------------------------------

    def _validar_blando(self, chunk: Chunk) -> List[str]:
        """Advertencias que no impiden el guardado."""
        advertencias: List[str] = []
        formato = chunk.formato
        es_unidad_atomica = formato in ("csv", "xlsx", "pbf")

        if not _RE_FIN_PUNTUACION.search(chunk.texto.rstrip()):
            if not es_unidad_atomica:
                advertencias.append("el texto no termina en puntuación terminal (posible oración cortada)")

        primera = chunk.texto.lstrip()
        if primera and not _RE_INICIO_PUNTUACION.match(primera):
            if not es_unidad_atomica:
                advertencias.append("el texto parece empezar a mitad de una oración")
        return advertencias
