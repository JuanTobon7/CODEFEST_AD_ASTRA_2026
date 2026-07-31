"""
Contador de tokens usando el tokenizador del encoder elegido.

No se usa ``len(texto.split())``: el conteo debe ser fiel al límite del
encoder (512 tokens). Por defecto se emplea ``tiktoken``; si no está
instalado se cae a un tokenizador aproximado (regex) para que el pipeline
siga siendo ejecutable en entornos mínimos.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+|[.,;:!?»«\"'()\[\]{}]")


class Tokenizer:
    """Contador de tokens con API uniforme independiente del backend."""

    def __init__(self, model: str = "cl100k_base") -> None:
        self.model = model
        self._enc = self._cargar_encoder(model)

    @staticmethod
    def _cargar_encoder(model: str):
        """Carga el encoder de tiktoken o devuelve ``None`` (modo aproximado)."""
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding(model)
            logger.info("Tokenizer tiktoken cargado: %s", model)
            return enc
        except Exception as exc:  # pragma: no cover - entorno sin tiktoken
            logger.warning(
                "tiktoken no disponible (%s); usando tokenizador aproximado (regex).", exc
            )
            return None

    def count_tokens(self, texto: str) -> int:
        """Cuenta tokens de un texto con el tokenizador configurado."""
        if not texto:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(texto))
        return len(_WORD_RE.findall(texto))

    def truncate(self, texto: str, max_tokens: int) -> str:
        """Trunca el texto hasta ``max_tokens`` respetando el tokenizador."""
        if self.count_tokens(texto) <= max_tokens:
            return texto
        if self._enc is not None:
            tokens = self._enc.encode(texto)[:max_tokens]
            return self._enc.decode(tokens)
        return " ".join(_WORD_RE.findall(texto)[:max_tokens])


@lru_cache(maxsize=8)
def _get_tokenizer(model: str) -> Tokenizer:
    """Singleton por modelo de tokenizador."""
    return Tokenizer(model)


class TokenizerFactory:
    """Factory simple de tokenizadores (cacheados por modelo)."""

    def __init__(self, model: str = "cl100k_base") -> None:
        self.model = model

    def create(self) -> Tokenizer:
        return _get_tokenizer(self.model)
