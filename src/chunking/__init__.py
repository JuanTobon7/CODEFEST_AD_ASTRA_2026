"""
Estrategias de fragmentación de texto (patrón Strategy).
"""

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.chunking.hybrid_strategy import HybridChunkingStrategy
from src.chunking.paragraph_overlap_strategy import ParagraphOverlapChunkingStrategy
from src.chunking.paragraph_strategy import ParagraphChunkingStrategy
from src.chunking.semantic_overlap_strategy import SemanticOverlapChunkingStrategy
from src.chunking.structural_strategy import StructuralChunkingStrategy

__all__ = [
    "ChunkingStrategy",
    "ChunkingStrategyFactory",
    "HybridChunkingStrategy",
    "ParagraphChunkingStrategy",
    "ParagraphOverlapChunkingStrategy",
    "SemanticOverlapChunkingStrategy",
    "StructuralChunkingStrategy",
    "TextSegmenter",
]
