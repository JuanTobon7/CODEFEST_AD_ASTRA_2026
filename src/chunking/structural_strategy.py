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
            if self.segmenter.count_tokens(texto) > config.max_tokens:
                # Sección más larga que el límite del encoder: corte duro por
                # tokens para no perder contenido (el validador rechazaría).
                for pieza in self.segmenter.dividir_por_tokens(texto, config.max_tokens):
                    chunks.append(
                        self._construir_chunk(
                            extracted_doc, pieza, len(chunks), config, seccion=seccion.titulo
                        )
                    )
                continue
            chunks.append(
                self._construir_chunk(
                    extracted_doc, texto, len(chunks), config, seccion=seccion.titulo
                )
            )
        return chunks
