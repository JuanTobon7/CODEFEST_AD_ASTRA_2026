"""
Tests de la agregación a nivel de documento (Sección 8.6) con las tres
estrategias intercambiables: max pooling, suma y media ponderada por rango.
"""

from __future__ import annotations

import pytest

from src.retrieval.aggregation import (
    aggregate_to_documents,
    pool_max,
    pool_sum,
    pool_weighted_mean,
)
from src.retrieval.models import Fragment


def _fragmento(doc_id, chunk_id, score):
    return Fragment(chunk_id=chunk_id, doc_id=doc_id, text=f"texto {chunk_id}", score=score, posicion=0)


def test_pooling_individuales():
    assert pool_max([0.1, 0.7, 0.5]) == pytest.approx(0.7)
    assert pool_max([]) == 0.0
    assert pool_sum([0.1, 0.7, 0.5]) == pytest.approx(1.3)
    # media ponderada por 1/rank: (0.1*1 + 0.7*0.5 + 0.5*1/3) / (1 + 0.5 + 1/3)
    esperado = (0.1 + 0.35 + 0.5 / 3) / (1 + 0.5 + 1 / 3)
    assert pool_weighted_mean([0.1, 0.7, 0.5], [1, 2, 3]) == pytest.approx(esperado)


def test_aggregate_max_el_mejor_fragmento_manda():
    fragmentos = [
        _fragmento("d1", "c1", score=0.8),
        _fragmento("d1", "c2", score=0.6),
        _fragmento("d2", "c3", score=0.9),
    ]
    ranking = aggregate_to_documents(fragmentos, strategy="max")
    assert ranking == [("d2", 0.9), ("d1", 0.8)]


def test_aggregate_sum_premia_mas_fragmentos():
    fragmentos = [
        _fragmento("d1", "c1", score=0.5),
        _fragmento("d1", "c2", score=0.4),
        _fragmento("d2", "c3", score=0.8),
    ]
    ranking = aggregate_to_documents(fragmentos, strategy="sum")
    # d1 suma 0.9 > d2 con 0.8 aunque d2 tenga el mejor fragmento.
    assert ranking == [("d1", 0.9), ("d2", 0.8)]


def test_aggregate_weighted_mean_por_rango():
    fragmentos = [
        _fragmento("d1", "c1", score=1.0),  # rank 1
        _fragmento("d1", "c2", score=0.0),  # rank 2
        _fragmento("d2", "c3", score=0.9),  # rank 3
    ]
    ranking = aggregate_to_documents(fragmentos, strategy="weighted_mean")
    por_doc = dict(ranking)
    # d1 = (1.0*1 + 0.0*0.5)/(1.5) = 2/3; d2 = 0.9*1/3 / (1/3) = 0.9
    assert por_doc["d1"] == pytest.approx(2 / 3)
    assert por_doc["d2"] == pytest.approx(0.9)
    assert ranking[0][0] == "d2"


def test_aggregate_estrategia_invalida():
    with pytest.raises(ValueError):
        aggregate_to_documents([], strategy="avg")


def test_aggregate_vacio_devuelve_vacio():
    assert aggregate_to_documents([]) == []
