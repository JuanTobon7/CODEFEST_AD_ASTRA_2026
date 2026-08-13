"""Fachada del módulo de Grafo de Conocimiento (Facade, Sección 7).

:class:`KnowledgeGraphService` es el punto de entrada ÚNICO del resto del
sistema: oculta la composición de NER + RE + construcción + persistencia
bajo una API pequeña (``construir_desde_chunks``, ``exportar_graphml``,
``crear_retriever_grafo``). El resto del sistema NUNCA instancia NER, RE,
builder ni repositorio directamente (bajo acoplamiento, DIP).

La fachada también ensambla la integración con la recuperación híbrida:
con el mismo :class:`EntityExtractor` (el usado al indexar) se crea el
:class:`GraphIndexAdapter` que participa en la fusión con FAISS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Union

from src.knowledge_graph.extract.base import EntityExtractor, RelationExtractor
from src.knowledge_graph.extract.cooccurrence_relation_strategy import (
    CooccurrenceRelationExtractor,
)
from src.knowledge_graph.extract.pipeline import ExtractionPipeline
from src.knowledge_graph.extract.regex_entity_strategy import RegexEntityExtractor
from src.knowledge_graph.graph.inmemory_repository import GraphMLFileRepository
from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import (
    GraphBuilder,
    GraphQuery,
    GraphRepository,
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.models import Query, ScoredChunk
from src.knowledge_graph.retrieval.adapters import GraphIndexAdapter
from src.knowledge_graph.use_cases import ChunkFuente, IndexKnowledgeGraphUseCase

logger = logging.getLogger(__name__)

#: Ruta por defecto del entregable de la Sección 7.3 (grafo.graphml).
RUTA_GRAPHML_DEFAULT = "grafo.graphml"


class KnowledgeGraphService:
    """Fachada: NER + RE + builder + persistencia + retriever de grafo.

    Args:
        entity_extractor: estrategia NER (default: regex-gazetteer, sin LLM).
        relation_extractor: estrategia RE (default: co-ocurrencia, sin LLM).
        builder: constructor del grafo (default: en memoria).
        repositorio: persistencia (default: GraphML en ``grafo.graphml``).
    """

    def __init__(
        self,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        builder: GraphBuilder | None = None,
        repositorio: GraphRepository | None = None,
    ) -> None:
        self._entity_extractor = entity_extractor or RegexEntityExtractor()
        self._relation_extractor = relation_extractor or CooccurrenceRelationExtractor()
        self._builder = builder or KnowledgeGraphBuilder()
        self._repositorio = repositorio or GraphMLFileRepository()
        self._pipeline = ExtractionPipeline(
            ner=self._entity_extractor, re=self._relation_extractor
        )
        self._grafo: KnowledgeGraph | None = None

    # -- Indexación (Sección 7.1-7.3) --------------------------------------

    def construir_desde_chunks(self, chunks: Iterable[ChunkFuente]) -> KnowledgeGraph:
        """Construye el grafo a partir de los chunks de la base vectorial.

        Cada tripleta conserva ``doc_id``/``chunk_id`` de origen y el grafo
        queda disponible en memoria (``grafo_actual``) para la recuperación.
        """
        caso = IndexKnowledgeGraphUseCase(
            pipeline=self._pipeline, builder=self._builder
        )
        self._grafo = caso.ejecutar(chunks)
        logger.info(
            "Grafo construido: %d entidades, %d tripletas",
            self._grafo.num_entidades,
            self._grafo.num_tripletas,
        )
        return self._grafo

    def exportar_graphml(self, ruta: Union[str, Path] = RUTA_GRAPHML_DEFAULT) -> Path:
        """Persiste el grafo construido como ``grafo.graphml`` (Sección 7.3).

        Raises:
            ValueError: si aún no se construyó ningún grafo.
        """
        if self._grafo is None:
            raise ValueError(
                "No hay grafo que exportar: ejecute 'construir_desde_chunks' primero."
            )
        return self._repositorio.guardar(self._grafo, Path(ruta))

    @property
    def grafo_actual(self) -> KnowledgeGraph | None:
        """Grafo construido en la sesión (``None`` si aún no se indexó)."""
        return self._grafo

    # -- Recuperación híbrida (Sección 8.5) ---------------------------------

    def crear_retriever_grafo(
        self, peso_directo: float = 1.0, peso_vecino: float = 0.5
    ) -> GraphIndexAdapter:
        """Canal simbólico listo para la fusión con FAISS.

        Usa el MISMO extractor de entidades con que se indexó el grafo
        (consistencia del vocabulario consulta ↔ índice) y el :class:`GraphQuery`
        del grafo actual.

        Raises:
            ValueError: si aún no se construyó ningún grafo.
        """
        if self._grafo is None:
            raise ValueError(
                "No hay grafo que consultar: ejecute 'construir_desde_chunks' primero."
            )
        query: GraphQuery = KnowledgeGraphQuery(self._grafo)
        return GraphIndexAdapter(
            extractor=self._entity_extractor,
            grafo=query,
            peso_directo=peso_directo,
            peso_vecino=peso_vecino,
        )

    def recuperar_via_grafo(self, texto_consulta: str, k: int = 10) -> List[ScoredChunk]:
        """Conveniencia: candidatos del grafo para ``texto_consulta``."""
        retriever = self.crear_retriever_grafo()
        return retriever.retrieve(Query(texto=texto_consulta), k)

    # -- Inspección ---------------------------------------------------------

    @property
    def extractor_entidades(self) -> EntityExtractor:
        """Extractor NER activo (para auditoría e informes)."""
        return self._entity_extractor

    def resumen(self) -> dict:
        """Resumen del estado del servicio (informe técnico).

        Incluye la ficha del modelo de relaciones cuando la estrategia activa
        usa uno, para que la auditoría de licencias del reto (permisiva y no
        generativa) quede trazada en el log de cada construcción del grafo.
        """
        resumen = {
            "ner": self._entity_extractor.name,
            "re": self._relation_extractor.name,
            "entidades": self._grafo.num_entidades if self._grafo else 0,
            "tripletas": self._grafo.num_tripletas if self._grafo else 0,
            "repositorio": type(self._repositorio).__name__,
        }
        if self._relation_extractor.name == "nli-zero-shot":
            from src.knowledge_graph.extract.nli_config import FICHA_MODELO_NLI

            resumen["modelo_re"] = dict(FICHA_MODELO_NLI)
        return resumen
