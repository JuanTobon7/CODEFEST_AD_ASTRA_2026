"""Módulo de Grafo de Conocimiento (Sección 7, componente bonus).

G = (E, R, T) con T ⊆ E × R × E, construido en 3 etapas (NER → RE →
construcción/persistencia) SIN modelos generativos en ninguna etapa de
indexación ni recuperación. Cada tripleta conserva ``doc_id``/``chunk_id``
de origen y la salida final es ``grafo.graphml`` cargable con NetworkX.

Puntos de entrada públicos:
- :class:`KnowledgeGraphService` (Facade): indexa chunks, exporta GraphML
  y crea el canal simbólico de recuperación.
- :class:`RetrievalOrchestrator` (Controller): fusiona canales (FAISS +
  grafo) con la :class:`FusionStrategy` configurada (RRF/CombSUM/CombMNZ).
"""

from src.knowledge_graph.extract.base import (
    EntityExtractor,
    RelationExtractor,
    normalizar_id_entidad,
)
from src.knowledge_graph.extract.cooccurrence_relation_strategy import (
    CooccurrenceRelationExtractor,
)
from src.knowledge_graph.extract.factory import (
    EntityExtractorFactory,
    RelationExtractorFactory,
)
from src.knowledge_graph.extract.mrebel_relation_strategy import MrebelRelationExtractor
from src.knowledge_graph.extract.nli_relation_strategy import NLIRelationExtractor
from src.knowledge_graph.extract.pipeline import ExtractionPipeline
from src.knowledge_graph.extract.regex_entity_strategy import RegexEntityExtractor
from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import (
    GraphBuilder,
    GraphQuery,
    GraphRepository,
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.models import (
    Entity,
    EntityType,
    ExtractionResult,
    Query,
    Relation,
    RelationType,
    ScoredChunk,
    Tripleta,
)
from src.knowledge_graph.retrieval.adapters import (
    GraphIndexAdapter,
    VectorIndexAdapter,
)
from src.knowledge_graph.retrieval.base import FusionStrategy, Retriever
from src.knowledge_graph.retrieval.fusion import (
    CombMNZFusionStrategy,
    CombSUMFusionStrategy,
    RRFusionStrategy,
)
from src.knowledge_graph.retrieval.orchestrator import RetrievalOrchestrator
from src.knowledge_graph.service import KnowledgeGraphService
from src.knowledge_graph.use_cases import (
    IndexKnowledgeGraphUseCase,
    RetrieveViaGraphUseCase,
)

__all__ = [
    "CombMNZFusionStrategy",
    "CombSUMFusionStrategy",
    "CooccurrenceRelationExtractor",
    "Entity",
    "EntityExtractor",
    "EntityExtractorFactory",
    "EntityType",
    "ExtractionPipeline",
    "ExtractionResult",
    "FusionStrategy",
    "GraphBuilder",
    "GraphIndexAdapter",
    "GraphQuery",
    "GraphRepository",
    "IndexKnowledgeGraphUseCase",
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphQuery",
    "KnowledgeGraphService",
    "MrebelRelationExtractor",
    "NLIRelationExtractor",
    "Query",
    "RRFusionStrategy",
    "RegexEntityExtractor",
    "Relation",
    "RelationExtractor",
    "RelationExtractorFactory",
    "RelationType",
    "RetrieveViaGraphUseCase",
    "Retriever",
    "RetrievalOrchestrator",
    "ScoredChunk",
    "Tripleta",
    "VectorIndexAdapter",
    "normalizar_id_entidad",
]
