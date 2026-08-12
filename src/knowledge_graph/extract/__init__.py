"""Extracción NER/RE del grafo de conocimiento (Sección 7.1-7.2)."""

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

__all__ = [
    "CooccurrenceRelationExtractor",
    "EntityExtractor",
    "EntityExtractorFactory",
    "ExtractionPipeline",
    "MrebelRelationExtractor",
    "NLIRelationExtractor",
    "RegexEntityExtractor",
    "RelationExtractor",
    "RelationExtractorFactory",
    "normalizar_id_entidad",
]
