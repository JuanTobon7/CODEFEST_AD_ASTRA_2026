"""
Tests: lógica pura de ``MongoVectorRepository`` (sin MongoDB real) — qué
``chunk_id`` faltan para un encoder, dado un conjunto de existentes.
"""

from __future__ import annotations

from src.vectorstore.vector_repository import MongoVectorRepository


def test_todos_faltantes_si_no_hay_nada_persistido():
    faltantes = MongoVectorRepository._faltantes_de(["c1", "c2"], set())
    assert faltantes == ["c1", "c2"]


def test_ningun_faltante_si_todos_existen():
    faltantes = MongoVectorRepository._faltantes_de(["c1", "c2"], {"c1", "c2"})
    assert faltantes == []


def test_preserva_orden_y_evita_duplicados():
    faltantes = MongoVectorRepository._faltantes_de(["c1", "c2", "c1", "c3"], {"c2"})
    assert faltantes == ["c1", "c3"]


def test_serializacion_a_documento_empaqueta_vector_como_binario():
    import numpy as np

    from src.vectorstore.models import EmbeddingRecord

    registro = EmbeddingRecord(
        chunk_id="c1", doc_id="d1", fenomeno=1, formato="md",
        encoder_name="e5-base", model_id="intfloat/multilingual-e5-base",
        embedding_dim=4, vector=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )
    documento = MongoVectorRepository._a_documento(registro)

    assert documento["chunk_id"] == "c1"
    assert documento["embedding_dim"] == 4
    reconstruido = MongoVectorRepository._desde_documento({**documento, "chunk_id": "c1", "doc_id": "d1"})
    assert np.allclose(reconstruido.vector, registro.vector)
