"""Estructura y persistencia del grafo (Sección 7.3)."""

from src.knowledge_graph.graph.inmemory_repository import (
    GraphMLFileRepository,
    Neo4jRepository,
    NetworkXInMemoryRepository,
)
from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import (
    GraphBuilder,
    GraphQuery,
    GraphRepository,
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.graph.serializer import GraphMLSerializer

__all__ = [
    "GraphBuilder",
    "GraphMLFileRepository",
    "GraphMLSerializer",
    "GraphQuery",
    "GraphRepository",
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphQuery",
    "Neo4jRepository",
    "NetworkXInMemoryRepository",
]
