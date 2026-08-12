"""Tests del corte duro por tokens en ``ajustar_a_limite`` (sin margen).

Un chunk con una oración gigante (> max_input_tokens) ya no se EXCLUYE:
se recorta a ``max_input_tokens`` tokens exactos (el encode de
sentence-transformers trunca internamente los special tokens sobrantes) y
se codifica la porción que se pudo tomar.

Prefijo ``enc_``: evita colisión de basename con tests existentes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.encoders.base import EncoderStrategy
from src.encoders.orchestrator import EncoderOrchestrator
from src.models.chunk import Chunk


class _FakeTokenizador:
    """Tokenizador fake: 1 token por palabra (+2 specials si se piden)."""

    def encode(self, texto, add_special_tokens=True):
        n = len(texto.split()) + (2 if add_special_tokens else 0)
        return list(range(n))

    def decode(self, ids, skip_special_tokens=True):
        return f"texto-{len(ids)}tokens"


class _FakeModeloConTokenizador:
    def __init__(self, dim):
        self.dim = dim
        self.tokenizer = _FakeTokenizador()

    def encode(self, textos, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        vectores = np.array([[float(len(t) % 7 + 1)] * self.dim for t in textos], dtype=np.float32)
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        return vectores / normas


def _estrategia(max_tokens: int = 10) -> EncoderStrategy:
    class _Estrategia(EncoderStrategy):
        model_id = "fake/con-tokenizador"
        embedding_dim = 4
        max_input_tokens = max_tokens
        supported_languages = ["es", "en", "pt"]
        license = "mit"

        def _cargar_modelo(self, device: str):
            return _FakeModeloConTokenizador(self.embedding_dim)

    estrategia = _Estrategia()
    estrategia._registry_name = "fake-con-tokenizador"  # type: ignore[attr-defined]
    return estrategia


def _chunk(chunk_id: str, texto: str) -> Chunk:
    return Chunk(
        doc_id="doc-1", chunk_id=chunk_id, fuente="test.md", formato="md",
        fenomeno=1, posicion=0, num_tokens=len(texto.split()), texto=texto,
    )


def test_corte_duro_recorta_a_max_input_tokens_exactos():
    """Sin margen: el recorte usa exactamente ``max_input_tokens`` tokens."""
    estrategia = _estrategia(max_tokens=10)
    texto = " ".join(f"palabra_{i}" for i in range(12))  # 12 tokens, sin puntuación
    ajustado = estrategia.ajustar_a_limite(texto)
    assert ajustado == "texto-10tokens"  # 10 = max_input_tokens, sin margen


def test_ajustar_a_limite_no_devuelve_none_con_oracion_gigante():
    """Una oración > límite ya no excluye: se toma la porción que se pudo."""
    estrategia = _estrategia(max_tokens=10)
    texto = " ".join(f"palabra_{i}" for i in range(30)) + "."
    ajustado = estrategia.ajustar_a_limite(texto)
    assert ajustado is not None
    assert ajustado == "texto-10tokens"


def test_texto_que_cabe_no_se_toca():
    estrategia = _estrategia(max_tokens=10)
    texto = "hola mundo"  # 2 + 2 specials = 4 <= 10
    assert estrategia.ajustar_a_limite(texto) == texto


def test_truncado_por_oraciones_cuando_es_posible():
    """Si hay puntuación, se preservan oraciones completas (no corte duro)."""
    estrategia = _estrategia(max_tokens=10)
    texto = "Primera oración corta. " + " ".join(f"palabra_{i}" for i in range(20)) + "."
    ajustado = estrategia.ajustar_a_limite(texto)
    assert ajustado == "Primera oración corta."


def test_orquestador_no_excluye_chunk_con_tokenizador():
    """End-to-end: el chunk gigante se codifica (truncado), no se excluye."""
    estrategia = _estrategia(max_tokens=10)
    orquestador = EncoderOrchestrator([estrategia])
    chunk_largo = _chunk("c1", " ".join(f"palabra_{i}" for i in range(30)) + ".")

    resultado = orquestador.run([chunk_largo])["fake-con-tokenizador"]

    assert "c1" in resultado.chunk_ids
    assert "c1" not in resultado.chunks_excluidos
    assert resultado.vectors.shape == (1, 4)