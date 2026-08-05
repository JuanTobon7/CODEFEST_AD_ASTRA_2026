"""
Etapa 3 del pipeline: fusión multi-encoder con Reciprocal Rank Fusion
(RRF, Sección 8.3).

    s_RRF(c) = sum_j [ 1 / (k0 + r_j(c)) ]

donde ``r_j(c)`` es el rango de ``c`` en el índice ``j`` (empieza en 1),
``k0`` es una constante de suavizado (60 por defecto, parametrizable) y
``j`` recorre SOLO los índices donde ``c`` aparece en el top-``k_search``.
Los fragmentos ausentes de un índice no aportan término (no se les asigna
rango artificial), de modo que la suma nunca incluye ceros ficticios.

El resultado se ordena por ``s_RRF`` descendente. Como subproducto, cada
fragmento fusionado conserva ``cosine_score`` = máximo de sus similitudes
coseno originales por índice (necesario para el filtro por vector del
paso 4) y la lista de encoders que lo recuperaron.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from src.retrieval.models import FusedFragment, SearchHit


class RRFError(ValueError):
    """``k0`` inválido para la fusión RRF."""


def rrf_fuse(rankings: Sequence[Sequence[SearchHit]], k0: int = 60) -> List[FusedFragment]:
    """Fusiona los rankings por índice (uno por encoder) con RRF.

    Args:
        rankings: un ranking por índice activo; cada ranking es una lista de
            :class:`SearchHit` ordenada por rango (1-based, el orden que
            devuelve ``search_faiss``).
        k0: constante de suavizado de RRF (> 0). El rango 1 aporta
            ``1/(k0+1)``, el rango 50 aporta ``1/(k0+50)``, etc.

    Returns:
        Fragmentos fusionados, ordenados por ``rrf_score`` descendente
        (desempate: ``cosine_score`` descendente y luego ``chunk_id`` para
        que el orden sea determinista).

    Raises:
        RRFError: si ``k0 <= 0``.
    """
    if k0 <= 0:
        raise RRFError(f"k0 debe ser > 0, se recibió {k0}")

    # acumula por chunk_id: s_RRF, mejor coseno y encoders que lo recuperaron
    puntuaciones: Dict[str, float] = defaultdict(float)
    mejor_coseno: Dict[str, float] = {}
    encoders: Dict[str, List[str]] = defaultdict(list)
    primero: Dict[str, SearchHit] = {}

    for ranking in rankings:
        for hit in ranking:
            cid = hit.chunk_id
            puntuaciones[cid] += 1.0 / (k0 + hit.rank)
            mejor_coseno[cid] = max(mejor_coseno.get(cid, float("-inf")), hit.score)
            encoders[cid].append(hit.encoder_name)
            primero.setdefault(cid, hit)

    fusionados: List[FusedFragment] = []
    for cid, rrf in puntuaciones.items():
        hit = primero[cid]
        fusionados.append(
            FusedFragment(
                chunk_id=cid,
                doc_id=hit.doc_id,
                # La metadata de entrega usa la clave ``texto`` (campo del
                # modelo ``Chunk``); se acepta también ``text`` por comodidad.
                text=str(hit.metadata.get("texto") or hit.metadata.get("text", "")),
                rrf_score=rrf,
                cosine_score=mejor_coseno[cid],
                encoders=encoders[cid],
                metadata=dict(hit.metadata),
            )
        )

    fusionados.sort(key=lambda f: (-f.rrf_score, -f.cosine_score, f.chunk_id))
    return fusionados
