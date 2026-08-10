"""Orquestador de recuperación híbrida (Controller, Sección 8.5).

Coordina los canales de evidencia (vectorial y/o grafo) y delega la fusión
en la :class:`FusionStrategy` inyectada. Si solo hay un canal (p. ej. el
grafo es un componente bonus no implementado), opera sin cambios
estructurales: devuelve el ranking de ese único canal — esa es la ventaja
de haber diseñado los canales como :class:`Retriever` intercambiables.

El orquestador NO conoce FAISS, NetworkX, encoders ni NER (DIP): solo
conoce el contrato ``retrieve(q, k) -> List[ScoredChunk]``.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from src.knowledge_graph.models import Query, ScoredChunk
from src.knowledge_graph.retrieval.base import FusionStrategy, Retriever

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Controlador delgado: consulta cada canal y fusiona sus rankings."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fusion: FusionStrategy,
        k_por_canal: int = 50,
    ) -> None:
        """Inicializa con los canales y la estrategia de fusión.

        Args:
            retrievers: canales de evidencia (0..n): vectorial, grafo, o
                cualquiera futuro que implemente :class:`Retriever`.
            fusion: estrategia RRF/CombSUM/CombMNZ (intercambiable).
            k_por_canal: top-k consultado a cada canal antes de fusionar.
        """
        self._retrievers = list(retrievers)
        self._fusion = fusion
        self._k_por_canal = k_por_canal

    @property
    def canales(self) -> List[str]:
        """Etiquetas de los canales registrados (auditoría)."""
        return [r.origen for r in self._retrievers]

    def recuperar(self, query: Query, k_final: int = 10) -> List[ScoredChunk]:
        """Recupera y fusiona los ``k_final`` mejores chunks para ``query``.

        Pasos (Sección 8.5): 1) cada canal recupera su ranking en paralelo
        lógico; 2) los canales sin resultados ("mudos") no aportan; 3) si
        hay 2+ rankings se fusionan con la estrategia configurada; con 1
        ranking se devuelve tal cual (el grafo es un índice opcional).
        """
        if not self._retrievers:
            logger.warning("RetrievalOrchestrator sin canales: devuelve vacío")
            return []

        rankings: List[List[ScoredChunk]] = []
        for retriever in self._retrievers:
            try:
                ranking = retriever.retrieve(query, self._k_por_canal)
            except Exception as exc:  # noqa: BLE001 - un canal no debe tumbar el resto
                logger.warning("Canal '%s' falló (%s); se omite", retriever.origen, exc)
                continue
            if ranking:
                rankings.append(ranking)
            else:
                logger.info("Canal '%s' no aportó resultados; se omite", retriever.origen)

        if not rankings:
            return []
        if len(rankings) == 1:
            return rankings[0][:k_final]

        fusionados = self._fusion.fusionar(rankings)
        return fusionados[:k_final]
