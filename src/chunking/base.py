"""
Clase base abstracta de estrategias de chunking (patrón Strategy).

Todas las estrategias comparten la misma interfaz ``chunk()`` y dependen de
dos colaboradores inyectados (inversión de dependencias):
- :class:`TextSegmenter`: cuenta tokens y segmenta oraciones.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple

from src.models.chunk import ADVERTENCIA_CORTE_FORZADO, Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument
from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer, TokenizerFactory

logger = logging.getLogger(__name__)


class PiezaTexto(NamedTuple):
    """Fragmento de texto listo para convertirse en chunk.

    Atributos:
        texto: Contenido del fragmento.
        corte_forzado: ``True`` si hubo que cortar DENTRO de una oración
            (única situación en la que no se puede cumplir el requisito de
            completitud lingüística: una oración aislada más larga que el
            límite del encoder).
    """

    texto: str
    corte_forzado: bool


class TextSegmenter:
    """Colaborador compartido: conteo de tokens y segmentación oracional."""

    def __init__(self, tokenizer: Tokenizer, splitter: SentenceSplitter) -> None:
        self.tokenizer = tokenizer
        self.splitter = splitter

    @classmethod
    def crear(cls, tokenizer_model: str = "google-bert/bert-base-multilingual-cased", sentence_model: str = "es_core_news_sm") -> "TextSegmenter":
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

    def empaquetar_por_oraciones(self, texto: str, max_tokens: int) -> List[PiezaTexto]:
        """Reparte ``texto`` en piezas de ``max_tokens`` cortando entre oraciones.

        Sustituye al corte ciego por tokens cuando hay que fragmentar un texto
        más largo que el límite del encoder: el texto se descompone en párrafos
        y estos en oraciones, y las oraciones se acumulan de forma greedy hasta
        que la siguiente no cabe. Así el corte efectivo retrocede al final de la
        última oración completa que entra en el límite (requisito de completitud
        lingüística: ninguna oración cruza la frontera entre dos fragmentos).

        Los separadores originales se conservan: ``\\n\\n`` entre párrafos y un
        espacio entre oraciones del mismo párrafo.

        Único caso residual: una oración que por sí sola supera ``max_tokens``.
        No hay forma de guardarla sin cortarla, así que se parte por tokens y
        las piezas resultantes se marcan con ``corte_forzado=True``.

        Returns:
            Lista de :class:`PiezaTexto`, cada una con a lo sumo ``max_tokens``.
        """
        if not texto or not texto.strip():
            return []
        # Cabe entero: se devuelve intacto (preserva el texto original tal cual).
        if self.count_tokens(texto) <= max_tokens:
            return [PiezaTexto(texto, False)]

        piezas: List[PiezaTexto] = []
        acumulado: List[str] = []
        tokens_acumulados = 0

        def cerrar_pieza() -> None:
            nonlocal acumulado, tokens_acumulados
            if acumulado:
                piezas.append(PiezaTexto("".join(acumulado).strip(), False))
            acumulado = []
            tokens_acumulados = 0

        for separador, oracion in self._unidades_oracionales(texto):
            costo = self.count_tokens(oracion)
            if acumulado and tokens_acumulados + costo > max_tokens:
                cerrar_pieza()
            if costo > max_tokens:
                # Oración indivisible más larga que el encoder: corte duro.
                cerrar_pieza()
                for trozo in self.dividir_por_tokens(oracion, max_tokens):
                    piezas.append(PiezaTexto(trozo, True))
                continue
            acumulado.append(oracion if not acumulado else separador + oracion)
            tokens_acumulados += costo
        cerrar_pieza()
        return [p for p in piezas if p.texto]

    def _unidades_oracionales(self, texto: str) -> List[Tuple[str, str]]:
        """Oraciones del texto con el separador que las precede.

        El separador es ``\\n\\n`` cuando la oración abre un párrafo nuevo y un
        espacio cuando continúa el párrafo actual; así el reempaquetado no
        destruye la estructura de párrafos del documento original.
        """
        unidades: List[Tuple[str, str]] = []
        for parrafo in self.split_parrafos(texto):
            oraciones = self.split_oraciones(parrafo) or [parrafo]
            for indice, oracion in enumerate(oraciones):
                if not unidades:
                    separador = ""
                elif indice == 0:
                    separador = "\n\n"
                else:
                    separador = " "
                unidades.append((separador, oracion))
        return unidades

    def split_oraciones(self, texto: str) -> List[str]:
        """Divide en oraciones completas (nunca corta una oración)."""
        return self.splitter.split(texto)

    def split_parrafos(self, texto: str) -> List[str]:
        """Divide en párrafos respetando los saltos de párrafo originales.

        Un párrafo es un bloque separado por línea en blanco (``\\n\\n``), que
        es la convención que dejan los extractores (PDF/JSON unen bloques con
        ``\\n\\n``) y el cleaner (colapsa ``\\n{3,}`` a ``\\n\\n``). Los saltos
        de línea simples se conservan DENTRO del párrafo (líneas envueltas).

        Nunca fusiona párrafos ni parte uno por tamaño: es la unidad más
        cercana a la estructura semántica del autor.
        """
        normalizado = texto.replace("\r\n", "\n").replace("\r", "\n")
        parrafos = [p.strip() for p in re.split(r"\n\s*\n", normalizado)]
        return [p for p in parrafos if p]

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
        self, unidades: List[str], chunk_size: int, overlap_size: int
    ) -> List[Tuple[int, int]]:
        """Calcula ventanas [ini, fin] (índices de unidad) con solapamiento.

        Acepta cualquier lista de unidades atómicas contables por tokens
        (oraciones, párrafos, etc.): tanto el inicio como el fin de cada
        ventana caen en límites de unidad (nunca se corta una). El
        solapamiento se cuantiza por unidades (aproximación por debajo de
        ``overlap_size`` tokens).

        Returns:
            Lista de tuplas (inicio_incluido, fin_incluido).
        """
        ventanas: List[Tuple[int, int]] = []
        n = len(unidades)
        if n == 0:
            return ventanas
        idx = 0
        while idx < n:
            presupuesto = chunk_size
            acumulado = 0
            fin = idx
            for j in range(idx, n):
                costo = self.count_tokens(unidades[j])
                if acumulado + costo > presupuesto:
                    break
                acumulado += costo
                fin = j
            ventanas.append((idx, fin))
            if fin == n - 1:
                # La ventana ya cubre el final: retroceder el solapamiento solo
                # generaría ventanas degeneradas (subconjuntos de la última).
                break
            # Avance con solapamiento: retroceder unidades hasta llenar el overlap.
            nuevo_idx = fin + 1
            acum_overlap = 0
            k = fin
            while k > idx and acum_overlap < overlap_size:
                costo = self.count_tokens(unidades[k])
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

    # Helpers compartidos ------------------------------------------------------

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

        Cuando el texto excede el límite del encoder se reparte con
        :meth:`TextSegmenter.empaquetar_por_oraciones`, de modo que los cortes
        caen en fronteras de oración completa. Las piezas que aun así hubo que
        cortar dentro de una oración (oración más larga que ``max_tokens``)
        quedan marcadas con :data:`ADVERTENCIA_CORTE_FORZADO`.
        """
        for pieza in self.segmenter.empaquetar_por_oraciones(texto, config.max_tokens):
            chunk = self._construir_chunk(
                extracted_doc, pieza.texto, len(chunks), config,
                seccion=seccion, overlap_con=overlap_con,
            )
            if pieza.corte_forzado:
                logger.debug(
                    "Corte forzado dentro de una oración | doc=%s pos=%s",
                    extracted_doc.doc_id,
                    chunk.posicion,
                )
                chunk.validation_warnings.append(ADVERTENCIA_CORTE_FORZADO)
            chunks.append(chunk)

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
