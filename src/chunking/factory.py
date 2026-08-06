"""
Factory de estrategias de chunking por nombre (configurable en runtime).

Permite cambiar la estrategia desde ``.env`` (``CHUNKING_STRATEGY=hybrid``)
sin tocar el código del pipeline::

    strategy = ChunkingStrategyFactory.create("hybrid", config=chunking_config)
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.chunking.hybrid_strategy import HybridChunkingStrategy
from src.chunking.paragraph_overlap_strategy import ParagraphOverlapChunkingStrategy
from src.chunking.paragraph_strategy import ParagraphChunkingStrategy
from src.chunking.semantic_overlap_strategy import SemanticOverlapChunkingStrategy
from src.chunking.structural_strategy import StructuralChunkingStrategy
from src.models.config import ChunkingConfig

logger = logging.getLogger(__name__)

_REGISTRO: Dict[str, Type[ChunkingStrategy]] = {
    "structural": StructuralChunkingStrategy,
    "semantic": SemanticOverlapChunkingStrategy,
    "hybrid": HybridChunkingStrategy,
    "paragraph": ParagraphChunkingStrategy,
    "paragraph_overlap": ParagraphOverlapChunkingStrategy,
}


class ChunkingStrategyFactory:
    """Crea la estrategia de chunking indicada por nombre."""

    def __init__(self, segmenter: TextSegmenter) -> None:
        self.segmenter = segmenter

    def create(self, nombre: str, config: ChunkingConfig) -> ChunkingStrategy:
        """Instancia la estrategia ``nombre`` inyectando el segmentador.

        Args:
            nombre: ``structural``, ``semantic``, ``hybrid``, ``paragraph``
                o ``paragraph_overlap``.
            config: Configuración usada para validar los parámetros.

        Raises:
            ValueError: Si el nombre no está registrado.
        """
        clave = nombre.strip().lower()
        clase = _REGISTRO.get(clave)
        if clase is None:
            raise ValueError(
                f"Estrategia de chunking desconocida: '{nombre}'. "
                f"Disponibles: {', '.join(sorted(_REGISTRO))}"
            )
        return clase(self.segmenter)
