"""
Métricas de calidad de recuperación densa (Precision@k, Recall@k, MRR,
nDCG@k) para validar objetivamente qué tan bien recupera cada encoder
registrado, en el espíritu de MTEB/BEIR:

    "Benchmarks públicos como MTEB y BEIR evalúan la calidad de los
    embeddings en tareas de recuperación de información. Un encoder con
    buen desempeño en recuperación densa es preferible a uno optimizado
    para otras tareas (clasificación, similitud de pares)."

Como los encoders de este proyecto son checkpoints BERT crudos (sin
fine-tuning de *sentence embeddings*, ver ``src/encoders/base.py``), no
reportan score MTEB-Retrieval oficial — este módulo mide su desempeño de
recuperación **sobre el propio corpus**, con un conjunto de consultas y
juicios de relevancia curados a mano (``GoldQuery``), en vez de asumir que
un buen score en BEIR (dominios genéricos en inglés) se traslada 1:1 al
corpus multilingüe/de dominio específico de este reto.

Toda la lógica aquí es pura (numpy), sin dependencia de MongoDB ni de los
encoders reales: se puede testear con vectores sintéticos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np


@dataclass
class GoldQuery:
    """Una consulta de validación con sus juicios de relevancia conocidos.

    La relevancia se puede declarar a nivel de chunk (más preciso) o de
    documento (más fácil de curar a mano: "cualquier chunk de este
    documento cuenta como relevante").
    """

    query: str
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevant_doc_ids: List[str] = field(default_factory=list)

    def chunks_relevantes(self, chunk_id_a_doc_id: Dict[str, str]) -> Set[str]:
        """Resuelve el conjunto final de ``chunk_id`` relevantes para esta consulta."""
        relevantes = set(self.relevant_chunk_ids)
        if self.relevant_doc_ids:
            docs = set(self.relevant_doc_ids)
            relevantes |= {cid for cid, did in chunk_id_a_doc_id.items() if did in docs}
        return relevantes


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fracción de los ``k`` primeros resultados que son relevantes."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for cid in top_k if cid in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fracción de los relevantes totales que aparecen en los ``k`` primeros resultados."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """``1/posición`` del primer resultado relevante (0.0 si ninguno lo es)."""
    for posicion, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / posicion
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """nDCG@k con relevancia binaria (1 si es relevante, 0 si no)."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(posicion + 1)
        for posicion, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(posicion + 1) for posicion in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluar_ranking(gold: GoldQuery, retrieved: Sequence[str], chunk_id_a_doc_id: Dict[str, str], k: int) -> Dict[str, float]:
    """Métricas de una sola consulta contra su ranking de resultados recuperado."""
    relevantes = gold.chunks_relevantes(chunk_id_a_doc_id)
    if not relevantes:
        raise ValueError(f"GoldQuery sin relevantes resolubles: '{gold.query}'")
    return {
        "precision_at_k": precision_at_k(retrieved, relevantes, k),
        "recall_at_k": recall_at_k(retrieved, relevantes, k),
        "reciprocal_rank": reciprocal_rank(retrieved, relevantes),
        "ndcg_at_k": ndcg_at_k(retrieved, relevantes, k),
    }


def rankear_top_k(vector_query: np.ndarray, vectores_corpus: np.ndarray, chunk_ids_corpus: Sequence[str], k: int) -> List[str]:
    """``chunk_id`` ordenados por similitud coseno descendente (producto punto:
    los vectores del corpus y de la consulta ya están normalizados a norma
    unitaria por ``EncoderStrategy.encode()``).
    """
    similitudes = vectores_corpus @ vector_query
    top_k = min(k, len(chunk_ids_corpus))
    indices_top = np.argsort(-similitudes)[:top_k]
    return [chunk_ids_corpus[i] for i in indices_top]


def evaluar_encoder(
    golds: Sequence[GoldQuery],
    vectores_query: np.ndarray,
    vectores_corpus: np.ndarray,
    chunk_ids_corpus: Sequence[str],
    chunk_id_a_doc_id: Dict[str, str],
    k: int = 10,
    ignorar_gold_sin_relevantes: bool = True,
) -> Optional[Dict[str, float]]:
    """Promedia las métricas de ``evaluar_ranking`` sobre todas las ``golds``.

    Returns:
        Métricas promediadas, o ``None`` si ninguna ``GoldQuery`` tuvo
        relevantes resolubles en este corpus (nada que evaluar).
    """
    acumulado: Dict[str, List[float]] = {"precision_at_k": [], "recall_at_k": [], "reciprocal_rank": [], "ndcg_at_k": []}
    for gold, vector_query in zip(golds, vectores_query):
        if not gold.chunks_relevantes(chunk_id_a_doc_id):
            if ignorar_gold_sin_relevantes:
                continue
            raise ValueError(f"GoldQuery sin relevantes resolubles: '{gold.query}'")
        retrieved = rankear_top_k(vector_query, vectores_corpus, chunk_ids_corpus, k)
        metricas = evaluar_ranking(gold, retrieved, chunk_id_a_doc_id, k)
        for clave, valor in metricas.items():
            acumulado[clave].append(valor)

    if not acumulado["precision_at_k"]:
        return None
    return {clave: float(np.mean(valores)) for clave, valores in acumulado.items()} | {"n_queries": len(acumulado["precision_at_k"])}
