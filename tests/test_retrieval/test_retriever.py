"""
Tests end-to-end del orquestador ``Retriever`` con encoders sintéticos e
índices FAISS reales (determinístico, sin red ni modelos HuggingFace), y
verificación de la firma pública de ``retrieve()``.
"""

from __future__ import annotations

import inspect

import faiss
import numpy as np

from src.encoders.base import EncoderStrategy
from src.retrieval.retriever import Retriever, retrieve

DIM = 4


class _FakeEncoder(EncoderStrategy):
    """Encoder sintético: devuelve siempre un vector fijo (sin cargar modelo)."""

    model_id = "fake-model"
    embedding_dim = DIM
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "mit"

    def __init__(self, vector: np.ndarray, name: str) -> None:
        super().__init__()
        self._vector = np.asarray(vector, dtype=np.float32)
        self._registry_name = name

    def _cargar_modelo(self, device: str):
        return None

    def encode(self, texts, is_query: bool = False, batch_size=None) -> np.ndarray:
        return np.tile(self._vector, (len(texts), 1))


def _corpus():
    """3 chunks: d1 tiene c1 y c2 (cortos), d2 tiene c3 (corto)."""
    texto_c1 = " ".join(["palabra_c1"] * 50)
    texto_c2 = " ".join(["palabra_c2"] * 50)
    texto_c3 = " ".join(["palabra_c3"] * 50)
    metadata = [
        {"chunk_id": "c1", "doc_id": "d1", "posicion": 0, "texto": texto_c1, "formato": "md", "fenomeno": 1},
        {"chunk_id": "c2", "doc_id": "d1", "posicion": 1, "texto": texto_c2, "formato": "md", "fenomeno": 1},
        {"chunk_id": "c3", "doc_id": "d2", "posicion": 0, "texto": texto_c3, "formato": "pdf", "fenomeno": 2},
    ]
    return metadata


def _escenario():
    """Dos índices FlatIP con el mismo corpus; c1 alineado con la consulta."""
    metadata = _corpus()
    vectores = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # c1
            [0.0, 1.0, 0.0, 0.0],  # c2
            [0.0, 0.0, 1.0, 0.0],  # c3
        ],
        dtype=np.float32,
    )
    indice_a = faiss.IndexFlatIP(DIM)
    indice_b = faiss.IndexFlatIP(DIM)
    indice_a.add(vectores)
    indice_b.add(vectores)

    enc_a = _FakeEncoder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "enc-a")
    enc_b = _FakeEncoder(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "enc-b")

    retriever = Retriever(
        indices={"enc-a": indice_a, "enc-b": indice_b},
        metadata={"enc-a": metadata, "enc-b": metadata},
        encoders=[enc_a, enc_b],
    )
    return retriever


def test_retriever_pipeline_completo():
    retriever = _escenario()
    resultado = retriever.retrieve("consulta de prueba", k_search=5, k_chunk_out=10, doc_agg="max")

    # Etapa 6: d1 tiene el mejor fragmento (c1 rank 1 en ambos índices).
    assert resultado.documents == ["d1", "d2"]

    # Etapa 5: c1 (corto) se fusionó con c2, su siguiente chunk de d1.
    fragmentos = resultado.fragments
    assert len(fragmentos) <= 10
    primero = fragmentos[0]
    assert primero["chunk_id"] == "c1"
    assert primero["doc_id"] == "d1"
    assert "palabra_c1" in primero["text"] and "palabra_c2" in primero["text"]
    assert "merged_with" in primero
    # El fragmento fusionado no supera las 250 palabras.
    assert len(primero["text"].split()) <= 250


def test_retriever_respeta_filtros_de_metadata():
    retriever = _escenario()
    resultado = retriever.retrieve(
        "consulta", k_search=5, k_chunk_out=10, phenomenon_filter=2, doc_agg="max"
    )
    # Solo queda c3 (fenomeno 2, doc d2).
    assert resultado.documents == ["d2"]
    assert all(f["fenomeno"] == 2 for f in resultado.fragments)
    assert all(f["doc_id"] == "d2" for f in resultado.fragments)


def test_retriever_theta_descarta_fragmentos_no_relacionados():
    retriever = _escenario()
    resultado = retriever.retrieve("consulta", k_search=5, k_chunk_out=10, theta=0.95)
    # Solo c1 tiene coseno 1.0 con la consulta (c2 y c3 son ortogonales).
    assert resultado.documents == ["d1"]
    assert [f["chunk_id"] for f in resultado.fragments] == ["c1"]


def test_retriever_sin_resultados_devuelve_vacio():
    retriever = Retriever(indices={}, metadata={}, encoders=[])
    resultado = retriever.retrieve("consulta")
    assert resultado.as_dict() == {"documents": [], "fragments": []}


def test_firma_publica_retrieve_exacta():
    """La función pública respeta la firma del reto, con esos defaults."""
    sig = inspect.signature(retrieve)
    nombres = list(sig.parameters)
    assert nombres == [
        "query",
        "phenomenon_filter",
        "format_filter",
        "lang_filter",
        "date_range",
        "theta",
        "k_search",
        "k_chunk_out",
        "doc_agg",
    ]
    assert sig.parameters["theta"].default == 0.0
    assert sig.parameters["k_search"].default == 50
    assert sig.parameters["k_chunk_out"].default == 10
    assert sig.parameters["doc_agg"].default == "max"
