"""
Estrategia semántica con solapamiento (Fase 2 únicamente, ignorando estructura).

Toma el texto completo del documento y aplica una ventana deslizante de
``chunk_size`` tokens con ``overlap_size`` de solapamiento. Tanto el inicio
como el fin de cada ventana caen en límites de oración completa (nunca corta
una oración); el solapamiento se cuantiza por oraciones.
"""

from __future__ import annotations

from typing import List

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument


class SemanticOverlapChunkingStrategy(ChunkingStrategy):
    """Ventana deslizante semántica sobre el documento completo."""

    nombre = "semantic"

    def __init__(self, segmenter: TextSegmenter) -> None:
        super().__init__(segmenter)

    def chunk(self, extracted_doc: ExtractedDocument, config: ChunkingConfig) -> List[Chunk]:
        texto = extracted_doc.texto_completo.strip()
        if not texto:
            return []

        oraciones = self.segmenter.split_oraciones(texto)
        ventanas = self.segmenter.ventanas_deslizantes(
            oraciones, config.chunk_size, config.overlap_size
        )

        chunks: List[Chunk] = []
        for posicion, (ini, fin) in enumerate(ventanas):
            texto_chunk = " ".join(oraciones[ini : fin + 1]).strip()
            overlap_con = None
            if posicion > 0:
                overlap_con = chunks[-1].chunk_id
            if self.segmenter.count_tokens(texto_chunk) > config.max_tokens:
                # Oración(es) más largas que el límite del encoder: corte duro
                # por tokens para no perder contenido (el validador rechazaría).
                for pieza in self.segmenter.dividir_por_tokens(texto_chunk, config.max_tokens):
                    chunks.append(
                        self._construir_chunk(
                            extracted_doc, pieza, len(chunks), config,
                            overlap_con=overlap_con,
                        )
                    )
                continue
            chunks.append(
                self._construir_chunk(
                    extracted_doc, texto_chunk, len(chunks), config, overlap_con=overlap_con
                )
            )
        return chunks
