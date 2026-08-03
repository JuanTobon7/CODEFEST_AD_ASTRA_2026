"""
Tests de las métricas puras de calidad de recuperación (Precision@k,
Recall@k, MRR, nDCG@k) y de ``evaluar_encoder`` con vectores sintéticos
(no requieren MongoDB ni un encoder real).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.retrieval_quality import (
    GoldQuery,
    evaluar_encoder,
    evaluar_ranking,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)


def test_precision_recall_ranking_perfecto():
    retrieved = ["a", "b", "c", "d"]
    relevantes = {"a", "b"}
    assert precision_at_k(retrieved, relevantes, k=2) == 1.0
    assert recall_at_k(retrieved, relevantes, k=2) == 1.0


def test_precision_recall_ranking_sin_relevantes_en_top_k():
    retrieved = ["x", "y", "a", "b"]
    relevantes = {"a", "b"}
    assert precision_at_k(retrieved, relevantes, k=2) == 0.0
    assert recall_at_k(retrieved, relevantes, k=2) == 0.0
    assert recall_at_k(retrieved, relevantes, k=4) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "x"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_ranking_ideal_es_uno():
    retrieved = ["a", "b", "x", "y"]
    relevantes = {"a", "b"}
    assert ndcg_at_k(retrieved, relevantes, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_penaliza_orden_invertido():
    ideal = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
    invertido = ndcg_at_k(["x", "b", "a"], {"a", "b"}, k=3)
    assert invertido < ideal


def test_gold_query_relevancia_por_doc_id():
    gold = GoldQuery(query="q", relevant_doc_ids=["doc-1"])
    mapping = {"c1": "doc-1", "c2": "doc-2"}
    assert gold.chunks_relevantes(mapping) == {"c1"}


def test_evaluar_ranking_lanza_si_no_hay_relevantes_declarados():
    gold = GoldQuery(query="q")  # sin relevant_chunk_ids ni relevant_doc_ids
    with pytest.raises(ValueError):
        evaluar_ranking(gold, retrieved=["a", "b"], chunk_id_a_doc_id={}, k=2)
    # 'inexistente' sí se resuelve como relevante conocido (no requiere estar en el corpus)
    assert evaluar_ranking(
        GoldQuery(query="q", relevant_chunk_ids=["a"]), retrieved=["a", "b"], chunk_id_a_doc_id={}, k=2
    )["precision_at_k"] == pytest.approx(0.5)


def test_evaluar_encoder_prefiere_vectores_mas_similares():
    # Corpus de 3 vectores unitarios en ejes distintos; el chunk relevante es 'a'.
    vectores_corpus = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    chunk_ids = ["a", "b", "c"]
    chunk_id_a_doc_id = {"a": "doc-1", "b": "doc-2", "c": "doc-3"}
    golds = [GoldQuery(query="busca a", relevant_chunk_ids=["a"])]
    # La consulta está alineada con el vector de 'a'.
    vectores_query = np.array([[1.0, 0.0]], dtype=np.float32)

    metricas = evaluar_encoder(golds, vectores_query, vectores_corpus, chunk_ids, chunk_id_a_doc_id, k=1)

    assert metricas is not None
    assert metricas["precision_at_k"] == pytest.approx(1.0)
    assert metricas["reciprocal_rank"] == pytest.approx(1.0)
    assert metricas["n_queries"] == 1


def test_evaluar_encoder_ignora_golds_sin_relevantes_resolubles():
    vectores_corpus = np.array([[1.0, 0.0]], dtype=np.float32)
    chunk_ids = ["a"]
    golds = [GoldQuery(query="sin relevantes conocidos")]
    vectores_query = np.array([[1.0, 0.0]], dtype=np.float32)

    metricas = evaluar_encoder(golds, vectores_query, vectores_corpus, chunk_ids, {}, k=1)

    assert metricas is None
