"""
Estructuras de datos del módulo de recuperación (retrieval).

El pipeline completo (Sección 8 del reto CODEFEST AD ASTRA 2026) se modela
con tres formas de datos, cada una correspondiente a una etapa:

- :class:`SearchHit`: resultado crudo de UN índice FAISS (un encoder),
  con su similitud coseno original y su rango 1-based dentro de ese índice.
- :class:`FusedFragment`: fragmento tras la fusión multi-encoder con RRF
  (paso 3); conserva ``rrf_score`` (score de fusión) y ``cosine_score``
  (máxima similitud coseno entre los índices donde aparece, usada por el
  filtro por vector del paso 4).
- :class:`Fragment`: unidad final a nivel de chunk tras el split/merge del
  paso 5, lista para serializarse en la salida de ``retrieve()``.

Ninguna de estas estructuras depende de modelos generativos ni de MongoDB:
son puras (``dataclass`` + ``typing``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Rango de fechas opcional para el filtro por fecha_publicacion:
# (inicio, fin), cada extremo puede ser date/datetime/str ISO o None (abierto).
RangoFechas = Tuple[Optional[date], Optional[date]]

# Extremo de rango tal como lo recibe la API pública (date | datetime | str | None).
ExtremoFecha = Optional[Any]


def _parse_fecha(valor: ExtremoFecha) -> Optional[date]:
    """Normaliza un extremo de fecha (``date``/``datetime``/str ISO) a ``date``.

    Devuelve ``None`` si no se puede interpretar (el filtro se omite para
    ese extremo, comportamiento conservador: nunca se descarta por error
    de parsing).
    """
    if valor is None:
        return None
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            return None
        # ISO 8601 con o sin zona horaria (Z = UTC).
        try:
            return datetime.fromisoformat(texto.replace("Z", "+00:00")).date()
        except ValueError:
            pass
        for formato in ("%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(texto[:10], formato).date()
            except ValueError:
                continue
    return None


def normalizar_rango_fechas(rango: Optional[Sequence[Any]]) -> Optional[RangoFechas]:
    """Convierte un rango de fechas libre a ``(date|None, date|None)``.

    Un rango con ambos extremos ``None`` equivale a "sin filtro" (``None``).
    """
    if rango is None:
        return None
    inicio, fin = rango[0], rango[1]
    normalizado = (_parse_fecha(inicio), _parse_fecha(fin))
    if normalizado == (None, None):
        return None
    return normalizado


@dataclass
class SearchHit:
    """Resultado de búsqueda en UN índice FAISS (un encoder activo)."""

    chunk_id: str
    doc_id: str
    encoder_name: str
    rank: int  # rango 1-based dentro del top-k de este índice
    score: float  # similitud coseno original (producto punto, vectores normalizados)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedFragment:
    """Fragmento tras la fusión multi-encoder con RRF (paso 3)."""

    chunk_id: str
    doc_id: str
    text: str
    rrf_score: float
    cosine_score: float  # mejor similitud coseno entre los índices donde aparece
    encoders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fragment:
    """Unidad final a nivel de chunk tras split/merge (paso 5)."""

    chunk_id: str
    doc_id: str
    text: str
    score: float  # score RRF heredado del fragmento padre
    posicion: int = 0
    sub_indice: int = 0  # > 0 si proviene de dividir un fragmento largo
    merged_with: Optional[str] = None  # chunk_id del siguiente chunk fusionado
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Serialización del contrato de salida de ``retrieve()``."""
        datos: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
        }
        for clave in ("fuente", "formato", "fenomeno", "num_tokens", "idioma"):
            if clave in self.metadata:
                datos[clave] = self.metadata[clave]
        if self.sub_indice > 0:
            datos["sub_indice"] = self.sub_indice
        if self.merged_with:
            datos["merged_with"] = self.merged_with
        return datos


@dataclass
class RetrievalResult:
    """Salida final del retrieval (contrato de la función ``retrieve()``)."""

    documents: List[str]
    fragments: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        """Serialización plana exigida por la firma del reto."""
        return {"documents": self.documents, "fragments": self.fragments}


@dataclass
class RetrievalFilters:
    """Post-filtros de la Sección 8.7, aplicados ANTES del recorte final.

    Todos los filtros operan sobre metadata/vectores, nunca sobre texto
    generado. Un filtro cuyo valor es ``None`` no restringe nada.
    """

    fenomeno: Optional[int] = None
    formato: Optional[str] = None
    idioma: Optional[str] = None
    date_range: Optional[RangoFechas] = None
    theta: float = 0.0  # umbral de similitud coseno original (antes de RRF)

    def __post_init__(self) -> None:
        self.date_range = normalizar_rango_fechas(self.date_range)
