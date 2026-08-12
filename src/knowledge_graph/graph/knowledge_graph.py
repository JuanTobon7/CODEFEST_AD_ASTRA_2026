"""Grafo de conocimiento en memoria: G = (E, R, T) con evidencia textual.

:class:`KnowledgeGraph` es el agregado de dominio (Information Expert):
conoce sus entidades, sus tripletas y las operaciones estructurales que la
recuperación necesita (vecinos de primer orden, chunks vinculados), sin
depender de ningún backend concreto. Es el "producto" que produce el
:class:`GraphBuilder` y el que serializa el :class:`GraphMLSerializer`.

NetworkX solo se toca de forma perezosa en :meth:`to_networkx` (exportación
opcional); el modelo interno usa dicts puros de Python, deterministas y sin
dependencias pesadas.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

from src.knowledge_graph.models import Entity, Tripleta

#: (sujeto, relacion, objeto) compacto para aristas.
Arista = Tuple[str, str, str]


class KnowledgeGraph:
    """Grafo dirigido conceptualmente (las aristas se indexan en ambos sentidos).

    El reto define el grafo con tripletas T ⊆ E × R × E; para la
    recuperación por vecindario los enlaces se tratan como no dirigidos
    (si A "coopera con" B, B está a un salto de A).
    """

    def __init__(self) -> None:
        self._entidades: Dict[str, Entity] = {}
        self._tripletas: List[Tripleta] = []
        self._adyacencia: Dict[str, Set[str]] = defaultdict(set)
        self._tripletas_por_entidad: Dict[str, List[Tripleta]] = defaultdict(list)
        self._tripletas_por_chunk: Dict[str, List[Tripleta]] = defaultdict(list)

    # -- Escritura (invocada por el GraphBuilder) ---------------------------

    def agregar_entidad(self, entidad: Entity) -> None:
        """Registra ``entidad`` (idempotente: actualiza la existente)."""
        self._entidades[entidad.id] = entidad

    def agregar_tripleta(self, tripleta: Tripleta) -> None:
        """Registra ``tripleta`` exigiendo que sus extremos existan (fail-fast).

        Raises:
            ValueError: si ``sujeto`` u ``objeto`` no es una entidad del grafo.
        """
        if tripleta.sujeto not in self._entidades:
            raise ValueError(
                f"Tripleta con sujeto desconocido '{tripleta.sujeto}' (agregue la entidad primero)"
            )
        if tripleta.objeto not in self._entidades:
            raise ValueError(
                f"Tripleta con objeto desconocido '{tripleta.objeto}' (agregue la entidad primero)"
            )
        self._tripletas.append(tripleta)
        self._adyacencia[tripleta.sujeto].add(tripleta.objeto)
        self._adyacencia[tripleta.objeto].add(tripleta.sujeto)
        self._tripletas_por_entidad[tripleta.sujeto].append(tripleta)
        self._tripletas_por_entidad[tripleta.objeto].append(tripleta)
        self._tripletas_por_chunk[tripleta.chunk_id].append(tripleta)

    # -- Lectura estructural (Information Expert sobre sus propios datos) ---

    @property
    def num_entidades(self) -> int:
        """Nodos del grafo."""
        return len(self._entidades)

    @property
    def num_tripletas(self) -> int:
        """Aristas del grafo."""
        return len(self._tripletas)

    def tiene(self, nombre: str) -> bool:
        """True si la entidad normalizada ``nombre`` es un nodo del grafo."""
        return nombre in self._entidades

    def entidad(self, nombre: str) -> Optional[Entity]:
        """Devuelve la :class:`Entity` del nodo ``nombre`` (o ``None``)."""
        return self._entidades.get(nombre)

    def entidades(self) -> Iterator[Entity]:
        """Itera las entidades en orden canónico (id)."""
        yield from (self._entidades[eid] for eid in sorted(self._entidades))

    def tripletas(self) -> Iterator[Tripleta]:
        """Itera las tripletas en orden de inserción (estable)."""
        yield from self._tripletas

    def tripletas_de(self, nombre: str) -> List[Tripleta]:
        """Tripletas que involucran a la entidad ``nombre`` (Information Expert)."""
        return list(self._tripletas_por_entidad.get(nombre, []))

    def tripletas_de_chunk(self, chunk_id: str) -> List[Tripleta]:
        """Tripletas cuya evidencia textual es el chunk ``chunk_id``.

        Se usa para auditar el camino por el grafo hacia un chunk concreto
        del resultado fusionado (Sección 8.5).
        """
        return list(self._tripletas_por_chunk.get(chunk_id, []))

    def vecinos(self, nombre: str, grado: int = 1) -> Set[str]:
        """Ids de entidades a ``grado`` saltos de ``nombre`` (sin incluirla).

        El ``GraphIndexAdapter`` expande a los vecinos de primer orden
        (grado=1) tal como exige la Sección 8.5; el grado es parametrizable.
        BFS nivel a nivel: los nodos ya alcanzados no se re-exploran.
        """
        if not self.tiene(nombre):
            return set()
        frontera = {nombre}
        alcanzados: Set[str] = set()
        for _ in range(grado):
            siguiente: Set[str] = set()
            for nodo in frontera:
                siguiente.update(self._adyacencia.get(nodo, ()))
            siguiente -= alcanzados
            siguiente.discard(nombre)
            if not siguiente:
                break
            alcanzados.update(siguiente)
            frontera = siguiente
        return alcanzados

    def chunk_ids_de(self, nombre: str) -> Set[str]:
        """Chunks que evidencian tripletas de la entidad ``nombre``."""
        return {t.chunk_id for t in self._tripletas_por_entidad.get(nombre, [])}

    def chunk_ids_de_vecindario(self, nombre: str, grado: int = 1) -> Set[str]:
        """Chunks de la entidad ``nombre`` y de sus vecinos hasta ``grado``."""
        ids = set(self.chunk_ids_de(nombre))
        for vecino in self.vecinos(nombre, grado=grado):
            ids.update(self.chunk_ids_de(vecino))
        return ids

    def aristas(self) -> Iterator[Arista]:
        """Aristas (sujeto, relacion, objeto) de cada tripleta (para serializar)."""
        for t in self._tripletas:
            yield (t.sujeto, t.relacion.value, t.objeto)

    # -- Exportación opcional a NetworkX (perezosa) -------------------------

    def to_networkx(self):
        """Convierte el grafo a ``networkx.MultiGraph`` (import perezoso).

        Útil para inspección/visualización y para validar que el GraphML
        generado por el serializador es cargable por NetworkX sin
        dependencias adicionales.
        """
        import networkx as nx

        grafo = nx.MultiGraph()
        for e in self.entidades():
            grafo.add_node(e.id, tipo=e.tipo.value, nombre=e.nombre)
        for i, t in enumerate(self._tripletas):
            grafo.add_edge(
                t.sujeto,
                t.objeto,
                key=i,
                relacion=t.relacion.value,
                doc_id=t.doc_id,
                chunk_id=t.chunk_id,
                confianza=t.confianza,
            )
        return grafo
