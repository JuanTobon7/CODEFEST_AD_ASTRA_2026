"""Construcción y consulta del grafo: interfaces separadas (ISP) + Builder.

ISP (Interface Segregation): la escritura (:class:`GraphBuilder`) y la
lectura (:class:`GraphQuery`) son interfaces DISTINTAS porque sus
consumidores son distintos:

- ``GraphBuilder`` lo consume el pipeline de indexación (caso de uso
  ``IndexKnowledgeGraphUseCase``): solo agrega entidades y tripletas.
- ``GraphQuery`` lo consume la recuperación (``GraphIndexAdapter``): solo
  resuelve vecinos y chunks vinculados a entidades.

Builder (patrón): ``KnowledgeGraphBuilder`` agrega fragmento a fragmento
(chunk a chunk) y al final produce un :class:`KnowledgeGraph` inmutable
para exportar. DIP: el orquestador depende de las abstracciones, no de
NetworkX ni de ningún backend concreto.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Union

from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.models import Entity, Tripleta


class GraphRepository(ABC):
    """Persistencia del grafo: ``guardar``/``cargar`` un KnowledgeGraph.

    Se consume desde el caso de uso de indexación y desde la fachada;
    las implementaciones encapsulan cada backend (GraphML, memoria, Neo4j).
    """

    @abstractmethod
    def guardar(self, grafo: KnowledgeGraph, ruta: Union[str, Path]) -> Path:
        """Persiste ``grafo`` y devuelve la ruta escrita."""

    @abstractmethod
    def cargar(self, ruta: Union[str, Path]) -> KnowledgeGraph:
        """Reconstruye el :class:`KnowledgeGraph` desde ``ruta``."""


class GraphBuilder(ABC):
    """Interfaz de ESCRITURA del grafo (la consume la indexación)."""

    @abstractmethod
    def agregar_entidad(self, entidad: Entity) -> None:
        """Registra una entidad (idempotente por id canónico)."""

    @abstractmethod
    def agregar_tripleta(self, tripleta: Tripleta) -> None:
        """Registra una tripleta trazable (doc_id + chunk_id obligatorios)."""

    @abstractmethod
    def construir(self) -> KnowledgeGraph:
        """Devuelve el grafo completo construido hasta el momento.

        El producto resultante es la representación inmutable que después se
        persiste (GraphML) y se consulta (GraphQuery).
        """


class GraphQuery(ABC):
    """Interfaz de LECTURA del grafo (la consume la recuperación).

    El :class:`GraphIndexAdapter` depende SOLO de esta abstracción: si el
    backend cambia (NetworkX → Neo4j), el adapter no cambia (LSP/OCP).
    """

    @abstractmethod
    def tiene(self, nombre: str) -> bool:
        """True si la entidad normalizada ``nombre`` existe en el grafo."""

    @abstractmethod
    def vecinos_primer_orden(self, nombre: str) -> Set[str]:
        """Ids de las entidades a un salto de ``nombre`` (Sección 8.5)."""

    @abstractmethod
    def chunk_ids_de(self, nombre: str) -> Set[str]:
        """Chunk_ids vinculados a las tripletas de la entidad ``nombre``."""

    @abstractmethod
    def chunk_ids_de_vecindario(self, nombre: str, grado: int = 1) -> Set[str]:
        """Chunk_ids de la entidad y de sus vecinos hasta ``grado``."""

    @abstractmethod
    def tripletas_de(self, nombre: str) -> List[Tripleta]:
        """Tripletas que involucran a ``nombre`` (para scoring por evidencia)."""

    @abstractmethod
    def tripletas_de_chunk(self, chunk_id: str) -> List[Tripleta]:
        """Tripletas cuya evidencia es el chunk ``chunk_id`` (auditoría del camino)."""


class KnowledgeGraphBuilder(GraphBuilder):
    """Builder concreto en memoria (producto: :class:`KnowledgeGraph`)."""

    def __init__(self) -> None:
        self._grafo = KnowledgeGraph()

    def agregar_entidad(self, entidad: Entity) -> None:
        self._grafo.agregar_entidad(entidad)

    def agregar_tripleta(self, tripleta: Tripleta) -> None:
        self._grafo.agregar_tripleta(tripleta)

    def construir(self) -> KnowledgeGraph:
        """Devuelve el grafo acumulado (sin copia: el builder es de un solo uso)."""
        return self._grafo


class KnowledgeGraphQuery(GraphQuery):
    """Consulta sobre un :class:`KnowledgeGraph` en memoria (backend puro).

    Information Expert delegado: el grafo conoce su propia estructura; esta
    clase solo expone el subconjunto de lectura que la recuperación
    necesita (segregación de interfaces).
    """

    def __init__(self, grafo: KnowledgeGraph) -> None:
        self._grafo = grafo

    def tiene(self, nombre: str) -> bool:
        return self._grafo.tiene(nombre)

    def vecinos_primer_orden(self, nombre: str) -> Set[str]:
        return self._grafo.vecinos(nombre, grado=1)

    def chunk_ids_de(self, nombre: str) -> Set[str]:
        return self._grafo.chunk_ids_de(nombre)

    def chunk_ids_de_vecindario(self, nombre: str, grado: int = 1) -> Set[str]:
        return self._grafo.chunk_ids_de_vecindario(nombre, grado=grado)

    def tripletas_de(self, nombre: str) -> List[Tripleta]:
        return self._grafo.tripletas_de(nombre)

    def tripletas_de_chunk(self, chunk_id: str) -> List[Tripleta]:
        return self._grafo.tripletas_de_chunk(chunk_id)
