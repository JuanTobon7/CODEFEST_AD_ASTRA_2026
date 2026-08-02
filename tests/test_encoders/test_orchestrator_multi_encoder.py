"""
Tests: ``EncoderOrchestrator`` debe correr 1..N encoders sobre el mismo lote
de chunks, cada uno produciendo su propio espacio vectorial independiente y
trazable por ``chunk_id`` (Sección 4.4).
"""

from __future__ import annotations

import numpy as np

from src.encoders.base import EncoderStrategy
from src.encoders.orchestrator import EncoderOrchestrator
from src.models.chunk import Chunk


class _FakeModel:
    """Sustituto de ``SentenceTransformer`` con dimensión fija determinista."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.tokenizer = None  # fuerza el fallback de conteo aproximado por caracteres

    def encode(self, textos, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectores = np.array([[float(len(t) % 7 + 1)] * self.dim for t in textos], dtype=np.float32)
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        return vectores / normas


def _crear_estrategia(nombre: str, dim: int, max_tokens: int = 512) -> EncoderStrategy:
    class _Estrategia(EncoderStrategy):
        model_id = f"fake/{nombre}"
        embedding_dim = dim
        max_input_tokens = max_tokens
        supported_languages = ["es", "en", "pt"]
        license = "mit"

        def _cargar_modelo(self, device: str):
            return _FakeModel(dim)

    estrategia = _Estrategia()
    estrategia._registry_name = nombre  # type: ignore[attr-defined]
    return estrategia


def _chunk(chunk_id: str, texto: str) -> Chunk:
    return Chunk(
        doc_id="doc-1", chunk_id=chunk_id, fuente="test.md", formato="md",
        fenomeno=1, posicion=0, num_tokens=len(texto.split()), texto=texto,
    )


def test_orchestrator_corre_un_solo_encoder():
    estrategia = _crear_estrategia("e5-fake", dim=16)
    orquestador = EncoderOrchestrator([estrategia])
    chunks = [_chunk("c1", "hola mundo"), _chunk("c2", "otro fragmento de texto")]

    resultados = orquestador.run(chunks)

    assert set(resultados) == {"e5-fake"}
    resultado = resultados["e5-fake"]
    assert resultado.vectors.shape == (2, 16)
    assert resultado.chunk_ids == ["c1", "c2"]
    assert np.allclose(np.linalg.norm(resultado.vectors, axis=1), 1.0, atol=1e-5)


def test_orchestrator_corre_multiples_encoders_independientes():
    e_grande = _crear_estrategia("e5-large-fake", dim=32)
    e_liviano = _crear_estrategia("minilm-fake", dim=8)
    orquestador = EncoderOrchestrator([e_grande, e_liviano])
    chunks = [_chunk("c1", "texto uno"), _chunk("c2", "texto dos")]

    resultados = orquestador.run(chunks)

    assert set(resultados) == {"e5-large-fake", "minilm-fake"}
    assert resultados["e5-large-fake"].vectors.shape[1] == 32
    assert resultados["minilm-fake"].vectors.shape[1] == 8
    # Cada encoder mantiene su propio espacio vectorial, misma cobertura de chunk_ids.
    assert resultados["e5-large-fake"].chunk_ids == resultados["minilm-fake"].chunk_ids


def test_orchestrator_excluye_chunk_que_excede_max_tokens_sin_poder_truncar():
    estrategia = _crear_estrategia("e5-fake-corto", dim=4, max_tokens=1)
    orquestador = EncoderOrchestrator([estrategia])
    chunk_largo = _chunk("c1", "Una oración larga sin puntuación que excede el límite de tokens")

    resultados = orquestador.run([chunk_largo])

    resultado = resultados["e5-fake-corto"]
    assert "c1" not in resultado.chunk_ids
    assert "c1" in resultado.chunks_excluidos
    assert resultado.vectors.shape[0] == 0


def test_orchestrator_metadata_criterios_incluye_los_6_criterios():
    estrategia = _crear_estrategia("e5-meta", dim=4)
    orquestador = EncoderOrchestrator([estrategia])

    resultados = orquestador.run([_chunk("c1", "texto corto")])
    metadata = resultados["e5-meta"].metadata_criterios

    for clave in (
        "supported_languages", "embedding_dim", "max_input_tokens",
        "mteb_retrieval_score", "license", "avg_encode_time_ms_per_batch", "device_preference",
    ):
        assert clave in metadata
