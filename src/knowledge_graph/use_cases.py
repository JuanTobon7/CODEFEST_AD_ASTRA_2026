"""Casos de uso del grafo de conocimiento (Controller, GRASP).

Un caso de uso por acción (Sección 7 del reto):

- :class:`IndexKnowledgeGraphUseCase`: construye y persiste el grafo a
  partir de los chunks de la base vectorial (indexación, sin LLM).
- :class:`RetrieveViaGraphUseCase`: recupera candidatos vía el grafo
  (participa en la fusión híbrida de la Sección 8.5).

Ambos orquestan abstracciones (pipeline, builder, repositorio, adapter)
SIN contener lógica de dominio: no saben hacer NER, no saben serializar
GraphML, no saben puntuar — solo coordinan (bajo acoplamiento / alta
cohesión, DIP).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Protocol, Union

from src.knowledge_graph.extract.pipeline import ExtractionPipeline
from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import GraphBuilder, GraphRepository
from src.knowledge_graph.models import Query, ScoredChunk
from src.knowledge_graph.retrieval.adapters import GraphIndexAdapter

logger = logging.getLogger(__name__)

#: Cada cuántos chunks se registra el progreso de la indexación.
_PROGRESO_CADA = 100


class ChunkFuente(Protocol):
    """Contrato mínimo de un chunk de entrada (duck typing).

    Cualquier objeto con ``doc_id``/``chunk_id``/``texto`` sirve — por
    ejemplo el modelo :class:`Chunk` del pipeline de ingesta —, de modo que
    este módulo no depende de la persistencia (Mongo) ni del ingestador.
    """

    doc_id: str
    chunk_id: str
    texto: str


class IndexKnowledgeGraphUseCase:
    """Construye el grafo (NER → RE → tripletas → grafo) y lo persiste.

    Controller: itera los chunks, delega cada etapa al pipeline, agrega
    entidades/tripletas al :class:`GraphBuilder` y pide al repositorio que
    guarde el producto final (``grafo.graphml`` por defecto).
    """

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        builder: GraphBuilder,
        repositorio: GraphRepository | None = None,
        ruta_salida: Union[str, Path] = "grafo.graphml",
    ) -> None:
        self._pipeline = pipeline
        self._builder = builder
        self._repositorio = repositorio
        self._ruta_salida = Path(ruta_salida)

    def ejecutar(self, chunks: Iterable[ChunkFuente]) -> KnowledgeGraph:
        """Procesa ``chunks`` y devuelve el grafo construido (y persistido).

        Args:
            chunks: iterable de chunks (doc_id, chunk_id, texto). Cada
                tripleta conserva esa procedencia (trazabilidad Sección 7.3).

        Returns:
            El :class:`KnowledgeGraph` resultante, listo para exportar.
        """
        inicio = time.perf_counter()
        procesados = 0
        for chunk in chunks:
            resultado = self._pipeline.procesar_chunk(
                chunk.doc_id, chunk.chunk_id, chunk.texto
            )
            for entidad in resultado.entidades:
                self._builder.agregar_entidad(entidad)
            for tripleta in resultado.tripletas():
                self._builder.agregar_tripleta(tripleta)
            procesados += 1
            if procesados % _PROGRESO_CADA == 0:
                self._log_progreso(procesados, inicio)

        grafo = self._builder.construir()
        if self._repositorio is not None:
            self._repositorio.guardar(grafo, self._ruta_salida)
        return grafo

    @staticmethod
    def _log_progreso(procesados: int, inicio: float) -> None:
        """Registra ritmo y ETA estimada (el total se desconoce: es un iterable)."""
        transcurrido = time.perf_counter() - inicio
        ritmo = procesados / transcurrido
        logger.info(
            "Grafo: %d chunks procesados | %.2f chunks/s | %.1f min transcurridos",
            procesados, ritmo, transcurrido / 60,
        )


class RetrieveViaGraphUseCase:
    """Recupera candidatos a través del grafo (canal simbólico de la 8.5).

    Controller: traduce la consulta a :class:`Query` y delega en el
    :class:`GraphIndexAdapter` (que sí sabe resolver entidades → vecinos →
    chunks). El orquestador de fusión consumirá el resultado como si viniera
    de cualquier otro canal.
    """

    def __init__(self, adapter: GraphIndexAdapter) -> None:
        self._adapter = adapter

    def ejecutar(self, texto_consulta: str, k: int = 10) -> List[ScoredChunk]:
        """Chunks candidatos del grafo para ``texto_consulta`` (hasta ``k``)."""
        return self._adapter.retrieve(Query(texto=texto_consulta), k)
