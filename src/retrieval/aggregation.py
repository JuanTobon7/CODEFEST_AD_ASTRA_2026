"""
Etapa 6 del pipeline: agregación a nivel de documento (Sección 8.6).

Agrupa los fragmentos finales por ``doc_id`` y calcula un score agregado por
documento con una estrategia configurable. Las tres estrategias son
funciones intercambiables (misma firma) sobre la secuencia de scores de los
fragmentos de un documento:

- ``pool_max``: score del mejor fragmento (max pooling).
- ``pool_sum``: suma de los scores de todos sus fragmentos.
- ``pool_weighted_mean``: media ponderada por rango (1/rank) del fragmento
  dentro de la lista fusionada; privilegia a los fragmentos mejor rankeados
  sin que un documento con muchos fragmentos marginales domine la suma.

El orden final de documentos es por score agregado descendente (desempate
por ``doc_id`` para determinismo).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from src.retrieval.models import Fragment


def pool_max(scores: Sequence[float]) -> float:
    """Max pooling: el mejor fragmento representa al documento."""
    return max(scores, default=0.0)


def pool_sum(scores: Sequence[float]) -> float:
    """Suma de scores: un documento con más fragmentos relevantes gana."""
    return float(sum(scores))


def pool_weighted_mean(scores: Sequence[float], ranks: Sequence[int]) -> float:
    """Media ponderada por ``1/rank`` (rank 1-based dentro de la lista final)."""
    if not scores:
        return 0.0
    pesos = [1.0 / max(1, r) for r in ranks]
    return float(sum(s * w for s, w in zip(scores, pesos)) / sum(pesos))


def aggregate_to_documents(
    fragmentos: Sequence[Fragment],
    strategy: str = "max",
) -> List[Tuple[str, float]]:
    """Agrupa ``fragmentos`` por ``doc_id`` y calcula el score agregado.

    Args:
        fragmentos: fragmentos finales (tras split/merge), en orden de
            relevancia (el rango 1-based es su posición en esta lista).
        strategy: ``"max"`` (default), ``"sum"`` o ``"weighted_mean"``.

    Returns:
        Lista de ``(doc_id, score_agregado)`` ordenada por score descendente
        (desempate: ``doc_id`` ascendente para determinismo).

    Raises:
        ValueError: si ``strategy`` no es una de las tres soportadas.
    """
    if strategy not in {"max", "sum", "weighted_mean"}:
        raise ValueError(
            f"Estrategia de agregación desconocida: '{strategy}'. "
            "Usar 'max', 'sum' o 'weighted_mean'."
        )

    por_doc: Dict[str, List[float]] = defaultdict(list)
    rangos: Dict[str, List[int]] = defaultdict(list)
    for rango, fragmento in enumerate(fragmentos, start=1):
        por_doc[fragmento.doc_id].append(fragmento.score)
        rangos[fragmento.doc_id].append(rango)

    if strategy == "max":
        scores = {doc: pool_max(sc) for doc, sc in por_doc.items()}
    elif strategy == "sum":
        scores = {doc: pool_sum(sc) for doc, sc in por_doc.items()}
    else:
        scores = {
            doc: pool_weighted_mean(sc, rangos[doc])
            for doc, sc in por_doc.items()
        }

    return sorted(scores.items(), key=lambda par: (-par[1], par[0]))
