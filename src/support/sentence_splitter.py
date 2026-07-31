"""
Segmentador de oraciones multilingüe (ES/EN/PT) con respaldo determinista.

Usa spacy cuando está disponible (modelo configurable); en entornos sin
spacy se cae a un segmentador por regex que respeta abreviaturas comunes,
para que los tests y el pipeline mínimo no dependan de modelos grandes.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

# Puntuación terminal que cierra una oración (válida para ES/EN/PT).
PUNTUACION_TERMINAL = set(".?!»›") | {"¡"}

# Abreviaturas que no terminan oración.
_ABREVIATURAS = {
    "sr", "sra", "srta", "dr", "dra", "prof", "gral", "pág", "pag", "ej",
    "etc", "vs", "aprox", "p.ej", "s.n", "núm", "num", "art", "fig", "figs",
    "ed", "eds", "vol", "pp", "mt", "km", "kg", "ha", "cm", "mm", "hz", "khz",
    "no", "nos", "inc", "ltd", "co", "corp", "st", "mr", "mrs", "ms", "dr",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "ene", "feb", "abr", "ago", "dic", "set", "oct", "nov",
    "al", "cap", "sec", "secc", "párr", "parr", "a.c", "d.c", "a.m", "p.m",
}

_REGEX_DIVISOR = re.compile(r"(?<=[.!?»›])\s+|(?<=[.!?»›])\n+")


class SentenceSplitter:
    """Divide texto en oraciones completas sin cortarlas a la mitad."""

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or "es_core_news_sm"
        self._nlp = self._cargar_spacy()

    def _cargar_spacy(self):
        """Carga el modelo spacy si está disponible (opcional)."""
        try:
            import spacy  # type: ignore

            return spacy.load(self._model, disable=["ner", "tagger", "lemmatizer"])
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "spacy (%s) no disponible: %s. Se usará segmentación por regex.",
                self._model,
                exc,
            )
            return None

    def split(self, texto: str) -> List[str]:
        """Segmenta ``texto`` en oraciones completas (no vacías)."""
        if not texto or not texto.strip():
            return []
        if self._nlp is not None:
            return [s.text.strip() for s in self._nlp(texto).sents if s.text.strip()]
        return self._split_regex(texto)

    def _split_regex(self, texto: str) -> List[str]:
        """Segmentación determinista de respaldo (ES/EN/PT)."""
        oraciones: List[str] = []
        buffer = ""
        for token in _REGEX_DIVISOR.split(texto):
            if not token:
                continue
            buffer = (buffer + " " + token).strip()
            if self._termina_oracion(buffer):
                oraciones.append(buffer)
                buffer = ""
        if buffer.strip():
            oraciones.append(buffer.strip())
        return oraciones

    @staticmethod
    def _termina_oracion(texto: str) -> bool:
        """True si el texto termina en puntuación terminal y no en abreviatura."""
        if not texto:
            return False
        if texto[-1] not in PUNTUACION_TERMINAL:
            return False
        ultima_palabra = texto.rstrip(" \t\n.").rsplit(" ", 1)[-1].lower()
        base = re.sub(r"[.,;:!?»›]+$", "", ultima_palabra).lower()
        if base in _ABREVIATURAS:
            return False
        # "etc." / "p.ej." tampoco cierran oración
        if re.fullmatch(r"[a-z]+(\.\s?[a-z]+)*\.", ultima_palabra) and (
            "etc" in ultima_palabra or ultima_palabra.count(".") >= 2
        ):
            return False
        return True


@lru_cache(maxsize=4)
def _get_splitter(model: str) -> SentenceSplitter:
    """Singleton por modelo."""
    return SentenceSplitter(model)


class SentenceSplitterFactory:
    """Factory simple de segmentadores (cacheados por modelo)."""

    def __init__(self, model: str = "es_core_news_sm") -> None:
        self.model = model

    def create(self) -> SentenceSplitter:
        return _get_splitter(self.model)
