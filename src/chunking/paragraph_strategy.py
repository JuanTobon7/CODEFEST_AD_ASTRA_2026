"""
Estrategia por párrafo pura.

Cada párrafo del texto original (separado por línea en blanco, ``\\n\\n``) se
convierte en un chunk tal cual: se respetan los saltos de párrafo del autor
como unidades de segmentación, sin fusionarlos ni partirlos. Es la estrategia
más cercana a la estructura semántica del autor.

Excepciones (para no perder contenido frente al validador):
- Un párrafo que excede ``max_tokens`` (límite del encoder) se reparte en
  oraciones completas, igual que las demás estrategias del pipeline: el corte
  retrocede al final de la última oración que cabe en el límite.
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
                self._anadir_chunks(
                    extracted_doc, parrafo, config, chunks, seccion=seccion.titulo
                )
        return chunks
