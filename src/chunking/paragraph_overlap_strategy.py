"""
Estrategia híbrida por párrafo + superposición.

Combina la segmentación por párrafos (se respetan los saltos de párrafo del
texto original como unidades de segmentación) con una ventana deslizante: los
párrafos consecutivos se agrupan en ventanas de ``chunk_size`` tokens con
``overlap_size`` de solapamiento. Tanto el inicio como el fin de cada ventana
caen en límites de párrafo (nunca se parte un párrafo); el solapamiento se
cuantiza por párrafos. Dentro de cada chunk los párrafos se unen con
``\\n\\n`` para preservar la estructura original.

Reglas:
- Sección completa por debajo de ``chunk_size`` tokens → un solo chunk.
- Sección atómica (``splittable=False``) → un solo chunk (no se ventanea).
- Párrafo/ventana que excede ``max_tokens`` → se reparte en oraciones
  completas (``_anadir_chunks`` de la clase base), nunca por corte ciego de
  tokens: el corte retrocede al final de la última oración que cabe.
"""

from __future__ import annotations

from typing import List

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument, Section


class ParagraphOverlapChunkingStrategy(ChunkingStrategy):
    """Ventana deslizante por párrafos con solapamiento entre ventanas."""

    nombre = "paragraph_overlap"

    def __init__(self, segmenter: TextSegmenter) -> None:
        super().__init__(segmenter)

    def chunk(self, extracted_doc: ExtractedDocument, config: ChunkingConfig) -> List[Chunk]:
        chunks: List[Chunk] = []
        for seccion in sorted(extracted_doc.secciones, key=lambda s: s.orden):
            texto = seccion.texto.strip()
            if not texto:
                continue
            if not seccion.splittable or self.segmenter.count_tokens(texto) <= config.chunk_size:
                self._anadir_chunks(
                    extracted_doc, texto, config, chunks, seccion=seccion.titulo
                )
                continue
            self._dividir_seccion(extracted_doc, seccion, config, chunks)
        return chunks

    def _dividir_seccion(
        self,
        extracted_doc: ExtractedDocument,
        seccion: Section,
        config: ChunkingConfig,
        chunks: List[Chunk],
    ) -> None:
        """Aplica la ventana deslizante sobre los párrafos de la sección.

        Reutiliza :meth:`TextSegmenter.ventanas_deslizantes`, que acepta
        cualquier lista de unidades atómicas contables por tokens: aquí las
        unidades son los párrafos, de modo que los cortes y el solapamiento
        caen siempre en límites de párrafo. Los chunks se añaden a la lista
        compartida del documento para que las posiciones sean globales.
        """
        parrafos = self.segmenter.split_parrafos(seccion.texto)
        ventanas = self.segmenter.ventanas_deslizantes(
            parrafos, config.chunk_size, config.overlap_size
        )
        for posicion_local, (ini, fin) in enumerate(ventanas):
            texto_chunk = "\n\n".join(parrafos[ini : fin + 1]).strip()
            overlap_con = None
            if posicion_local > 0 and chunks:
                overlap_con = chunks[-1].chunk_id
            self._anadir_chunks(
                extracted_doc, texto_chunk, config, chunks,
                seccion=seccion.titulo, overlap_con=overlap_con,
            )
