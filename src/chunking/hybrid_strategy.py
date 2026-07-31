"""
Estrategia híbrida: estructural + semántica con solapamiento.

Fase 1 (estructural): usa las secciones del documento extraído como unidades
cohesivas; las secciones menores a ``min_chunk_tokens`` se concatenan con la
adyacente (anterior o siguiente) respetando ``max_tokens``.

Fase 2 (semántica): dentro de cada sección, si el texto supera ``chunk_size``
tokens se aplica una ventana deslizante con ``overlap_size`` de solapamiento.
Los cortes retroceden siempre al final de la última oración completa; las
unidades atómicas (``splittable=False``: filas CSV/XLSX, elementos PBF) nunca
se parten ni se fusionan.

Esta estrategia compone internamente las fases de ``StructuralChunkingStrategy``
y ``SemanticOverlapChunkingStrategy``.
"""

from __future__ import annotations

from typing import List

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument, Section


class HybridChunkingStrategy(ChunkingStrategy):
    """Fragmentación híbrida estructural + semántica por superposición."""

    nombre = "hybrid"

    def __init__(self, segmenter: TextSegmenter) -> None:
        super().__init__(segmenter)

    def chunk(self, extracted_doc: ExtractedDocument, config: ChunkingConfig) -> List[Chunk]:
        # Fase 1: segmentación estructural (y fusión de secciones pequeñas).
        secciones = self._fusionar_secciones_pequenas(
            sorted(extracted_doc.secciones, key=lambda s: s.orden), config
        )

        # Fase 2: ventana deslizante semántica dentro de cada sección.
        chunks: List[Chunk] = []
        for seccion in secciones:
            texto = seccion.texto.strip()
            if not texto:
                continue
            if not seccion.splittable or self.segmenter.count_tokens(texto) <= config.chunk_size:
                self._anadir_chunks(extracted_doc, texto, config, chunks, seccion=seccion.titulo)
                continue
            chunks.extend(self._dividir_seccion(extracted_doc, seccion, config))
        return chunks

    def _anadir_chunks(
        self,
        extracted_doc: ExtractedDocument,
        texto: str,
        config: ChunkingConfig,
        chunks: List[Chunk],
        seccion: str | None = None,
        overlap_con: str | None = None,
    ) -> None:
        """Añade ``texto`` como un chunk, o varios si supera ``max_tokens``.

        Unidad atómica u oración única más larga que el límite del encoder:
        se parte por tokens (corte duro) para no perder contenido, ya que el
        validador rechazaría el fragmento completo.
        """
        if self.segmenter.count_tokens(texto) > config.max_tokens:
            for pieza in self.segmenter.dividir_por_tokens(texto, config.max_tokens):
                chunks.append(
                    self._construir_chunk(
                        extracted_doc, pieza, len(chunks), config,
                        seccion=seccion, overlap_con=overlap_con,
                    )
                )
            return
        chunks.append(
            self._construir_chunk(
                extracted_doc, texto, len(chunks), config,
                seccion=seccion, overlap_con=overlap_con,
            )
        )

    # Fase 1: fusión de secciones pequeñas ------------------------------------

    def _fusionar_secciones_pequenas(
        self, secciones: List[Section], config: ChunkingConfig
    ) -> List[Section]:
        """Une secciones por debajo de ``min_chunk_tokens`` con la adyacente.

        Solo se fusionan secciones partibles (``splittable=True``) y solo si
        el resultado no supera ``max_tokens``. Se itera hasta el punto fijo
        para encadenar varias secciones pequeñas consecutivas.
        """
        resultado = list(secciones)
        for _ in range(len(resultado)):  # a lo sumo n pasadas
            cambio = False
            nuevo: List[Section] = []
            i = 0
            n = len(resultado)
            while i < n:
                seccion = resultado[i]
                if self._tokens(seccion) < config.min_chunk_tokens and seccion.splittable:
                    # Fusionar con la sección anterior ya procesada.
                    if nuevo and nuevo[-1].splittable:
                        previa = nuevo[-1]
                        if self._tokens(previa) + self._tokens(seccion) <= config.max_tokens:
                            nuevo[-1] = self._concatenar(previa, seccion)
                            cambio = True
                            i += 1
                            continue
                    # Fusionar con la sección siguiente.
                    if i + 1 < n and resultado[i + 1].splittable:
                        siguiente = resultado[i + 1]
                        if self._tokens(seccion) + self._tokens(siguiente) <= config.max_tokens:
                            nuevo.append(self._concatenar(seccion, siguiente))
                            cambio = True
                            i += 2
                            continue
                nuevo.append(seccion)
                i += 1
            resultado = nuevo
            if not cambio:
                break
        return resultado

    # Fase 2: ventana deslizante dentro de la sección --------------------------

    def _dividir_seccion(
        self, extracted_doc: ExtractedDocument, seccion: Section, config: ChunkingConfig
    ) -> List[Chunk]:
        """Aplica la ventana deslizante semántica al texto de la sección."""
        oraciones = self.segmenter.split_oraciones(seccion.texto)
        ventanas = self.segmenter.ventanas_deslizantes(
            oraciones, config.chunk_size, config.overlap_size
        )
        chunks: List[Chunk] = []
        for posicion_local, (ini, fin) in enumerate(ventanas):
            texto_chunk = " ".join(oraciones[ini : fin + 1]).strip()
            overlap_con = None
            if posicion_local > 0:
                overlap_con = chunks[-1].chunk_id
            self._anadir_chunks(
                extracted_doc, texto_chunk, config, chunks,
                seccion=seccion.titulo, overlap_con=overlap_con,
            )
        return chunks

    # Utilidades ---------------------------------------------------------------

    def _tokens(self, seccion: Section) -> int:
        return self.segmenter.count_tokens(seccion.texto)

    @staticmethod
    def _concatenar(anterior: Section, siguiente: Section) -> Section:
        """Concatena dos secciones conservando título y orden de la primera."""
        texto = f"{anterior.texto.rstrip()}\n{siguiente.texto.strip()}"
        titulo = anterior.titulo or siguiente.titulo
        return Section(titulo=titulo, texto=texto, orden=anterior.orden, splittable=True)
