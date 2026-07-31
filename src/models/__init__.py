"""
Modelos de datos del pipeline (Pydantic v2).

Se exponen aquí los tipos públicos del dominio:
- :class:`Section` — unidad estructural dentro de un documento.
- :class:`ExtractedDocument` — documento extraído y limpio.
- :class:`Chunk` — fragmento indexable (unidad mínima).
- :class:`ChunkingConfig` — parámetros de fragmentación.
- :class:`IngestionResult` — resultado de la ingesta de un archivo.
"""

from src.models.batch_summary import BatchSummary
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig, Settings
from src.models.extracted_document import ExtractedDocument, Formato, Section
from src.models.pipeline_result import IngestionResult

__all__ = [
    "BatchSummary",
    "Chunk",
    "ChunkingConfig",
    "ExtractedDocument",
    "Formato",
    "IngestionResult",
    "Section",
    "Settings",
]
