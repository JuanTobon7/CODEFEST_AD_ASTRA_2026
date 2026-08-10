"""Repositorios del grafo (patrón Repository, Sección 7.3).

``GraphRepository`` es la abstracción de persistencia; los consumidores
(caso de uso de indexación, servicio) dependen SOLO de ella (DIP). Cada
implementación encapsula un backend distinto (Variación protegida):

- :class:`GraphMLFileRepository`: archivo ``grafo.graphml`` en disco
  (salida oficial de la Sección 7.3), sin dependencias adicionales.
- :class:`NetworkXInMemoryRepository`: grafo en memoria de la sesión
  (backend puro, útil en pruebas y en el caso de uso sin I/O).
- :class:`Neo4jRepository`: stub documentado que demuestra cómo se
  agregaría un backend nuevo SIN tocar el código existente (OCP); la
  implementación real exigiría el driver ``neo4j``, no instalado aquí.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import GraphRepository
from src.knowledge_graph.graph.serializer import GraphMLSerializer

logger = logging.getLogger(__name__)


class GraphMLFileRepository(GraphRepository):
    """Repositorio de archivo: GraphML válido cargable por NetworkX."""

    def __init__(self, serializer: GraphMLSerializer | None = None) -> None:
        self._serializer = serializer or GraphMLSerializer()

    def guardar(self, grafo: KnowledgeGraph, ruta: Union[str, Path]) -> Path:
        """Escribe ``grafo.graphml`` en ``ruta`` (requisito de la Sección 7.3)."""
        return self._serializer.escribir(grafo, ruta)

    def cargar(self, ruta: Union[str, Path]) -> KnowledgeGraph:
        """Reconstruye el grafo leyendo el GraphML con NetworkX.

        Usa ``networkx.read_graphml`` (fallback a la stdlib, sin lxml) y
        reconstruye nodos, aristas y los atributos de trazabilidad de cada
        tripleta. Es una lectura de validación/inspección: el pipeline de
        indexación usa el builder en memoria y solo exporta GraphML.
        """
        import networkx as nx

        nx_grafo = nx.read_graphml(str(ruta))
        grafo = KnowledgeGraph()
        for nid, datos in nx_grafo.nodes(data=True):
            from src.knowledge_graph.models import Entity, EntityType

            grafo.agregar_entidad(
                Entity(
                    id=nid,
                    nombre=str(datos.get("nombre", nid)),
                    tipo=EntityType(datos.get("tipo", "OTRO")),
                )
            )
        # read_graphml devuelve Graph (3-tuplas) o MultiGraph (4-tuplas con
        # clave) según haya aristas paralelas; se aceptan ambas formas.
        for arista in nx_grafo.edges(data=True):
            if len(arista) == 4:
                sujeto, objeto, _clave, datos = arista
            else:
                sujeto, objeto, datos = arista
            from src.knowledge_graph.models import RelationType, Tripleta

            # La orientación canónica viaja en los atributos de la arista
            # (NetworkX normaliza el orden de las aristas no dirigidas).
            sujeto = str(datos.get("sujeto", sujeto))
            objeto = str(datos.get("objeto", objeto))
            grafo.agregar_tripleta(
                Tripleta(
                    sujeto=sujeto,
                    relacion=RelationType(datos.get("relacion", "COOCURRENCIA")),
                    objeto=objeto,
                    doc_id=str(datos.get("doc_id", "")),
                    chunk_id=str(datos.get("chunk_id", "")),
                    evidencia=str(datos.get("evidencia", "")),
                    confianza=float(datos.get("confianza", 1.0)),
                )
            )
        return grafo


class NetworkXInMemoryRepository(GraphRepository):
    """Backend en memoria de la sesión (sin I/O en el camino caliente).

    ``guardar``/``cargar`` operan sobre el grafo en memoria que se le
    inyecta; ``cargar`` con una ruta inexistente devuelve un grafo vacío
    (semántica de "índice no construido aún", útil para arrancar el
    servicio sin archivo).
    """

    def __init__(self, grafo: KnowledgeGraph | None = None) -> None:
        self._grafo = grafo or KnowledgeGraph()

    def guardar(self, grafo: KnowledgeGraph, ruta: Union[str, Path]) -> Path:
        self._grafo = grafo
        return Path(str(ruta))

    def cargar(self, ruta: Union[str, Path]) -> KnowledgeGraph:
        ruta = Path(ruta)
        if not ruta.exists():
            return self._grafo
        logger.warning("NetworkXInMemoryRepository ignora la ruta '%s' (backend en memoria)", ruta)
        return self._grafo

    @property
    def grafo(self) -> KnowledgeGraph:
        """Grafo vivo del repositorio (inspección/test)."""
        return self._grafo


class Neo4jRepository(GraphRepository):
    """Backend Neo4j (plantilla de integración, OCP).

    No se implementa aquí porque el driver ``neo4j`` no está instalado en el
    entorno y el reto exige que la salida oficial sea GraphML. Su existencia
    demuestra el patrón: el :class:`KnowledgeGraphService` cambiaría el
    repositorio inyectado y nada más (ni el builder, ni el adapter, ni la
    fusión) se tocaría.
    """

    def guardar(self, grafo: KnowledgeGraph, ruta: Union[str, Path]) -> Path:
        raise NotImplementedError(
            "Neo4jRepository requiere el paquete 'neo4j' y un servidor Neo4j; "
            "la salida oficial del reto es GraphML (GraphMLFileRepository)."
        )

    def cargar(self, ruta: Union[str, Path]) -> KnowledgeGraph:
        raise NotImplementedError(
            "Neo4jRepository requiere el paquete 'neo4j' y un servidor Neo4j; "
            "use GraphMLFileRepository para la salida oficial."
        )
