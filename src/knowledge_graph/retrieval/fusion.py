"""Motor de fusión de rankings (Pure Fabrication, Sección 8.4/8.5).

:class:`ScoreNormalizer` y las estrategias de fusión no representan un
concepto del dominio del reto: son clases técnicas (fabricación pura) que
resuelven el problema de combinar listas de :class:`ScoredChunk` de
orígenes distintos (FAISS + grafo) tratándolas con el mismo contrato.

Fusiones soportadas (todas deterministas, sin LLM):

- RRF (Reciprocal Rank Fusion):  s_RRF(c) = Σ_j 1/(k0 + r_j(c))
- CombSUM:                          s(c) = Σ_j s'_j(c)   (s' normalizado)
- CombMNZ:                          s(c) = m(c) · Σ_j s'_j(c), con m(c) =
  número de listas donde c aparece (refuerza consenso entre canales).

Los scores de cada lista se normalizan por min-max ANTES de CombSUM/MNZ
(las escalas de coseno y de evidencia de grafo no son comparables).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from src.knowledge_graph.models import ScoredChunk
from src.knowledge_graph.retrieval.base import FusionStrategy


class ScoreNormalizer:
    """Normaliza una lista de scores a [0, 1] por min-max (Pure Fabrication)."""

    @classmethod
    def min_max(cls, ranking: Sequence[ScoredChunk]) -> List[ScoredChunk]:
        """Devuelve copias de ``ranking`` con ``score`` en [0, 1].

        Una lista con un único elemento (o scores idénticos) se normaliza
        a 1.0 (no hay dispersión que explotar; el orden se conserva).
        """
        if not ranking:
            return []
        scores = [c.score for c in ranking]
        minimo, maximo = min(scores), max(scores)
        rango = maximo - minimo
        if rango <= 0:
            return [c.model_copy(update={"score": 1.0}) for c in ranking]
        return [
            c.model_copy(update={"score": (c.score - minimo) / rango}) for c in ranking
        ]


class RRFusionStrategy(FusionStrategy):
    """Fusión por ranks (k0 parametrizable; 60 como en el resto del pipeline)."""

    nombre = "rrf"

    def __init__(self, k0: int = 60) -> None:
        if k0 <= 0:
            raise ValueError(f"k0 debe ser > 0, se recibió {k0}")
        self.k0 = k0

    def fusionar(self, rankings: Sequence[Sequence[ScoredChunk]]) -> List[ScoredChunk]:
        acumulado: Dict[str, float] = defaultdict(float)
        primero: Dict[str, ScoredChunk] = {}
        for ranking in rankings:
            for rango, chunk in enumerate(ranking, start=1):
                acumulado[chunk.chunk_id] += 1.0 / (self.k0 + rango)
                primero.setdefault(chunk.chunk_id, chunk)
        fusion: List[ScoredChunk] = [
            primero[cid].model_copy(update={"score": score, "origen": "rrf"})
            for cid, score in acumulado.items()
        ]
        return _ordenar(fusion)


class CombSUMFusionStrategy(FusionStrategy):
    """Suma de scores normalizados por lista (cada canal pesa igual)."""

    nombre = "combsum"

    def fusionar(self, rankings: Sequence[Sequence[ScoredChunk]]) -> List[ScoredChunk]:
        acumulado: Dict[str, float] = defaultdict(float)
        primero: Dict[str, ScoredChunk] = {}
        for ranking in rankings:
            for chunk in ScoreNormalizer.min_max(ranking):
                acumulado[chunk.chunk_id] += chunk.score
                primero.setdefault(chunk.chunk_id, chunk)
        fusion: List[ScoredChunk] = [
            primero[cid].model_copy(update={"score": score, "origen": "combsum"})
            for cid, score in acumulado.items()
        ]
        return _ordenar(fusion)


class CombMNZFusionStrategy(FusionStrategy):
    """CombMNZ: suma normalizada ponderada por el número de canales de consenso."""

    nombre = "combmnz"

    def fusionar(self, rankings: Sequence[Sequence[ScoredChunk]]) -> List[ScoredChunk]:
        suma: Dict[str, float] = defaultdict(float)
        conteo: Dict[str, int] = defaultdict(int)
        primero: Dict[str, ScoredChunk] = {}
        for ranking in rankings:
            vistos: set[str] = set()
            for chunk in ScoreNormalizer.min_max(ranking):
                suma[chunk.chunk_id] += chunk.score
                vistos.add(chunk.chunk_id)
                primero.setdefault(chunk.chunk_id, chunk)
            for cid in vistos:
                conteo[cid] += 1
        fusion: List[ScoredChunk] = [
            primero[cid].model_copy(
                update={"score": suma[cid] * conteo[cid], "origen": "combmnz"}
            )
            for cid in suma
        ]
        return _ordenar(fusion)


def _ordenar(fusion: List[ScoredChunk]) -> List[ScoredChunk]:
    """Orden determinista: score desc, luego chunk_id asc."""
    fusion.sort(key=lambda c: (-c.score, c.chunk_id))
    return fusion
