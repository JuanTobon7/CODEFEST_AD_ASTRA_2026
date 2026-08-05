"""
Tests de la búsqueda en índice FAISS real (determinística, sin red) y del
mapeo de IDs internos a metadata de entrega.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from src.retrieval.faiss_search import search_faiss

DIM = 4


def _indice_y_metadata():
    """Índice FlatIP con 3 vectores ortogonales normalizados + metadata."""
    indice = faiss.IndexFlatIP(DIM)
    vectores = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # c1 -> eje x
            [0.0, 1.0, 0.0, 0.0],  # c2 -> eje y
            [0.0, 0.0, 1.0, 0.0],  # c3 -> eje z
        ],
        dtype=np.float32,
    )
    indice.add(vectores)
    metadata = [
        {"chunk_id": "c1", "doc_id": "d1", "posicion": 0, "texto": "uno", "fenomeno": 1, "formato": "md"},
        {"chunk_id": "c2", "doc_id": "d1", "posicion": 1, "texto": "dos", "fenomeno": 2, "formato": "md"},
        {"chunk_id": "c3", "doc_id": "d2", "posicion": 0, "texto": "tres", "fenomeno": 3, "formato": "pdf"},
    ]
    return indice, metadata


def test_search_faiss_ranking_coseno_y_rangos():
    indice, metadata = _indice_y_metadata()
    vector_crudo = np.array([[0.9, 0.3, 0.2, 0.0]], dtype=np.float32)
    norma = float(np.linalg.norm(vector_crudo))
    query = vector_crudo / norma

    hits = search_faiss(indice, query, metadata, encoder_name="bert-multilingual", k=2)

    assert len(hits) == 2
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].encoder_name == "bert-multilingual"
    assert hits[0].doc_id == "d1"
    assert hits[0].metadata["fenomeno"] == 1
    # Producto punto = similitud coseno (vectores normalizados): la consulta
    # se normalizó, así que el score contra c1 es 0.9 / |query|.
    assert hits[0].score == pytest.approx(0.9 / norma, abs=1e-4)
    assert hits[1].score == pytest.approx(0.3 / norma, abs=1e-4)


def test_search_faiss_k_mayor_que_ntotal():
    indice, metadata = _indice_y_metadata()
    query = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    hits = search_faiss(indice, query, metadata, encoder_name="A", k=100)
    assert len(hits) == indice.ntotal == 3
    assert hits[0].chunk_id == "c2"


def test_search_faiss_indice_vacio_devuelve_vacio():
    indice = faiss.IndexFlatIP(DIM)
    query = np.zeros((1, DIM), dtype=np.float32)
    assert search_faiss(indice, query, [], encoder_name="A", k=5) == []


def test_search_faiss_desalineacion_de_dimension_lanza():
    indice, metadata = _indice_y_metadata()
    query = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        search_faiss(indice, query, metadata, encoder_name="A", k=5)
