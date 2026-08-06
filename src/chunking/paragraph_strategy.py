"""
Estrategia por párrafo pura.

Cada párrafo del texto original (separado por línea en blanco, ``\\n\\n``) se
convierte en un chunk tal cual: se respetan los saltos de párrafo del autor
como unidades de segmentación, sin fusionarlos ni partirlos. Es la estrategia
más cercana a la estructura semántica del autor.

Excepciones (para no perder contenido frente al validador):
- Un párrafo que excede ``max_tokens`` (límite del encoder) se parte por
  tokens (corte duro), igual que las demás estrategias del pipeline.
- Las secciones atómicas (``splittable=False``: filas CSV/XLSX, elementos
  PBF) son una única unidad: no se segmentan en párrafos.
"""

from __future__ import annotations

from typing import List

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument


class ParagraphChunkingStrategy(ChunkingStrategy):
    """Un chunk por párrafo original, sin fusiones ni subdivisiones."""

    nombre = "paragraph"

    def __init__(self, segmenter: TextSegmenter) -> None:
        super().__init__(segmenter)

    def chunk(self, extracted_doc: ExtractedDocument, config: ChunkingConfig) -> List[Chunk]:
        chunks: List[Chunk] = []
        for seccion in sorted(extracted_doc.secciones, key=lambda s: s.orden):
            texto = seccion.texto.strip()
            if not texto:
                continue
            if not seccion.splittable:
                # Unidad atómica (fila CSV/XLSX, elemento PBF): un solo chunk.
                parrafos = [texto]
            else:
                parrafos = self.segmenter.split_parrafos(texto)
            for parrafo in parrafos:
                self._anadir_parrafo(
                    extracted_doc, parrafo, config, chunks, seccion=seccion.titulo
                )
        return chunks

    def _anadir_parrafo(
        self,
        extracted_doc: ExtractedDocument,
        parrafo: str,
        config: ChunkingConfig,
        chunks: List[Chunk],
        seccion: str | None = None,
    ) -> None:
        """Añade un párrafo como chunk, o varios si supera ``max_tokens``.

        Párrafo más largo que el límite del encoder: se parte por tokens
        (corte duro) para no perder contenido, ya que el validador rechazaría
        el fragmento completo.
        """
        if self.segmenter.count_tokens(parrafo) > config.max_tokens:
            for pieza in self.segmenter.dividir_por_tokens(parrafo, config.max_tokens):
                chunks.append(
                    self._construir_chunk(
                        extracted_doc, pieza, len(chunks), config, seccion=seccion
                    )
                )
            return
        chunks.append(
            self._construir_chunk(
                extracted_doc, parrafo, len(chunks), config, seccion=seccion
            )
        )
