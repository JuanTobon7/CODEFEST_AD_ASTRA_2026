"""
Estrategia estructural pura (Fase 1 únicamente).

Cada sección estructural se convierte en un chunk tal cual: no se divide por
tamaño ni se fusiona. Es la línea base para comparar resultados en el informe
técnico frente a la híbrida y la semántica pura.
"""

from __future__ import annotations

from typing import List

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument


class StructuralChunkingStrategy(ChunkingStrategy):
    """Un chunk por sección estructural, sin subdivisiones."""

    nombre = "structural"

    def __init__(self, segmenter: TextSegmenter) -> None:
        super().__init__(segmenter)

    def chunk(self, extracted_doc: ExtractedDocument, config: ChunkingConfig) -> List[Chunk]:
        chunks: List[Chunk] = []
        for seccion in sorted(extracted_doc.secciones, key=lambda s: s.orden):
            texto = seccion.texto.strip()
            if not texto:
                continue
            # Sección más larga que el límite del encoder: se reparte en
            # oraciones completas para no perder contenido (el validador
            # rechazaría el fragmento entero por num_tokens).
            self._anadir_chunks(
                extracted_doc, texto, config, chunks, seccion=seccion.titulo
            )
        return chunks
