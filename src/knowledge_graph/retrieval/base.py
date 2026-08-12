"""Interfaz común de recuperación (Sección 8.5): el grafo como "un índice más".

El punto de integración crítico con embeddings/FAISS: TANTO el índice
vectorial como el grafo de conocimiento implementan :class:`Retriever` con
el MISMO contrato de salida (:class:`ScoredChunk`). Así el motor de fusión
(:class:`FusionStrategy`) combina ambas listas sin conocer su origen (DIP,
OCP, Adapter): si mañana el grafo pasa de NetworkX a Neo4j, o se añade un
segundo encoder, nada de lo que está aguas abajo cambia.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from src.knowledge_graph.models import Query, ScoredChunk


class Retriever(ABC):
    """Fuente de evidencia recuperable con contrato común.

    Implementaciones: :class:`VectorIndexAdapter` (FAISS) y
    :class:`GraphIndexAdapter` (grafo de conocimiento). NINGUNA implementación
    puede invocar un LLM/decoder en indexación ni recuperación.
    """

    @abstractmethod
    def retrieve(self, q: Query, k: int) -> List[ScoredChunk]:
        """Recupera los ``k`` mejores chunks para la consulta ``q``.

        Returns:
            Lista de :class:`ScoredChunk` ordenada por ``score``
            descendente (el fusionador espera este orden para el RRF).
        """

    @property
    @abstractmethod
    def origen(self) -> str:
        """Etiqueta de trazabilidad del canal (encoder, "grafo", ...)."""


class FusionStrategy(ABC):
    """Estrategia de fusión de rankings (Strategy, Sección 8.4).

    Implementaciones: :class:`RRFusionStrategy`, :class:`CombSUMFusionStrategy`,
    :class:`CombMNZFusionStrategy` — intercambiables en runtime mediante
    inyección, sin tocar el orquestador (OCP).
    """

    nombre: str = "fusion"

    @abstractmethod
    def fusionar(self, rankings: Sequence[Sequence[ScoredChunk]]) -> List[ScoredChunk]:
        """Combina las listas de :class:`ScoredChunk` (mismo contrato).

        Args:
            rankings: una lista por canal de evidencia (vectorial, grafo,
                o cualquier Retriever futuro), ya ordenada por score.

        Returns:
            Lista fusionada ordenada por score descendente. Los chunks que
            solo aparecen en un canal reciben la puntuación de ese canal.
        """
