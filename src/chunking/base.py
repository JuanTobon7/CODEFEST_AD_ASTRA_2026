"""
Clase base abstracta de estrategias de chunking (patrón Strategy).

Todas las estrategias comparten la misma interfaz ``chunk()`` y dependen de
dos colaboradores inyectados (inversión de dependencias):
- :class:`TextSegmenter`: cuenta tokens y segmenta oraciones.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument
from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer, TokenizerFactory

logger = logging.getLogger(__name__)


class TextSegmenter:
    """Colaborador compartido: conteo de tokens y segmentación oracional."""

    def __init__(self, tokenizer: Tokenizer, splitter: SentenceSplitter) -> None:
        self.tokenizer = tokenizer
        self.splitter = splitter

    @classmethod
    def crear(cls, tokenizer_model: str = "cl100k_base", sentence_model: str = "es_core_news_sm") -> "TextSegmenter":
        """Construye un segmentador con factories (tokenizador cacheado)."""
        return cls(
            tokenizer=TokenizerFactory(tokenizer_model).create(),
            splitter=SentenceSplitter(sentence_model),
        )

    def count_tokens(self, texto: str) -> int:
        """Conteo de tokens con el tokenizador del encoder elegido."""
        return self.tokenizer.count_tokens(texto)

    def dividir_por_tokens(self, texto: str, max_tokens: int) -> List[str]:
        """Parte el texto en piezas de a lo sumo ``max_tokens`` (corte duro).

        Se usa cuando una oración/unidad aislada supera el límite del encoder:
        preserva todo el contenido a costa de cortar en mitad de la oración.
        """
        return self.tokenizer.split_tokens(texto, max_tokens)

    def split_oraciones(self, texto: str) -> List[str]:
        """Divide en oraciones completas (nunca corta una oración)."""
        return self.splitter.split(texto)

    def fin_oracion_mas_cercana(self, oraciones: List[str], presupuesto: int) -> int:
        """Índice de la última oración que cabe dentro del presupuesto de tokens.

        Siempre se conserva al menos una oración; si ni siquiera la primera
        cabe (oración más larga que la ventana), se devuelve la oración entera
        (la división por tokens forzada queda registrada como warning por el
        validador/estrategia).
        """
        if not oraciones:
            return 0
        acumulado = 0
        ultimo_valido = 0
        for i, oracion in enumerate(oraciones):
            costo = self.count_tokens(oracion)
            if acumulado + costo > presupuesto:
                break
            acumulado += costo
            ultimo_valido = i
        # Aunque la primera oración no quepa, no podemos cortarla: la tomamos entera.
        return max(ultimo_valido, 0)

    def ventanas_deslizantes(
        self, oraciones: List[str], chunk_size: int, overlap_size: int
    ) -> List[Tuple[int, int]]:
        """Calcula ventanas [ini, fin] (índices de oración) con solapamiento.

        Tanto el inicio como el fin de cada ventana caen en límites de oración:
        nunca se corta una oración. El solapamiento se cuantiza por oraciones
        (aproximación por debajo de ``overlap_size`` tokens).

        Returns:
            Lista de tuplas (inicio_incluido, fin_incluido).
        """
        ventanas: List[Tuple[int, int]] = []
        n = len(oraciones)
        if n == 0:
            return ventanas
        idx = 0
        while idx < n:
            presupuesto = chunk_size
            acumulado = 0
            fin = idx
            for j in range(idx, n):
                costo = self.count_tokens(oraciones[j])
                if acumulado + costo > presupuesto:
                    break
                acumulado += costo
                fin = j
            ventanas.append((idx, fin))
            # Avance con solapamiento: retroceder oraciones hasta llenar el overlap.
            nuevo_idx = fin + 1
            acum_overlap = 0
            k = fin
            while k > idx and acum_overlap < overlap_size:
                costo = self.count_tokens(oraciones[k])
                if acum_overlap + costo > overlap_size:
                    break
                acum_overlap += costo
                nuevo_idx = k
                k -= 1
            if nuevo_idx <= idx:
                nuevo_idx = fin + 1  # garantiza progreso
            idx = nuevo_idx
        return ventanas


class ChunkingStrategy(ABC):
    """Contrato común de todas las estrategias de fragmentación."""

    nombre: str = "base"

    def __init__(self, segmenter: TextSegmenter) -> None:
        self.segmenter = segmenter

    @abstractmethod
    def chunk(self, extracted_doc: ExtractedDocument, config: ChunkingConfig) -> List[Chunk]:
        """Fragmenta el documento extraído en :class:`Chunk`.

        Args:
            extracted_doc: Documento extraído (y limpio) con sus secciones.
            config: Parámetros de chunking (tamaño, solapamiento, mínimos).

        Returns:
            Lista de fragmentos con metadata base ya poblada
            (``doc_id``, ``fuente``, ``formato``, ``fenomeno``, ``posicion``,
            ``num_tokens``, ``texto``, ``chunking_strategy``).
        """

    # Helper compartido --------------------------------------------------------

    def _construir_chunk(
        self,
        extracted_doc: ExtractedDocument,
        texto: str,
        posicion: int,
        config: ChunkingConfig,
        seccion: str | None = None,
        overlap_con: str | None = None,
    ) -> Chunk:
        """Crea un :class:`Chunk` con la metadata base de la Tabla 1."""
        return Chunk(
            doc_id=extracted_doc.doc_id,
            chunk_id=f"{extracted_doc.doc_id}__chunk_{posicion:05d}",
            fuente=extracted_doc.fuente,
            formato=extracted_doc.formato.value if hasattr(extracted_doc.formato, "value") else str(extracted_doc.formato),
            fenomeno=extracted_doc.fenomeno,
            posicion=posicion,
            num_tokens=self.segmenter.count_tokens(texto),
            texto=texto,
            chunking_strategy=self.nombre,
            seccion=seccion,
            overlap_con=overlap_con,
            idioma=extracted_doc.idioma,
            titulo_documento=extracted_doc.titulo_documento,
            fecha_publicacion=extracted_doc.fecha_publicacion,
        )
