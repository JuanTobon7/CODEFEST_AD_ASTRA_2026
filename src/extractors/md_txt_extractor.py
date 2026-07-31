"""
Extractor de Markdown y texto plano.

- Markdown: encabezados ``#``-``######``, párrafos y listas generan secciones.
- TXT plano: secciones por bloques separados por líneas en blanco (párrafos).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section

logger = logging.getLogger(__name__)

_PATRON_ENCABEZADO = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PATRON_LISTA = re.compile(r"^\s*[-*+]\s+")
_PATRON_LISTA_NUM = re.compile(r"^\s*\d+[.)]\s+")


@register_extractor(".md", ".markdown", ".txt")
class MarkdownTxtExtractor(BaseExtractor):
    """Extrae texto de archivos Markdown o texto plano."""

    supported_formats = ["md", "markdown", "txt"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el documento en secciones por encabezado/párrafo."""
        texto = self._decodificar(self._leer_bytes(filepath))
        es_markdown = filepath.suffix.lower() in (".md", ".markdown")
        secciones: List[Section] = self._segmentar(texto, es_markdown)

        if not secciones or not any(s.texto.strip() for s in secciones):
            raise ExtractorError(f"El archivo no contiene texto útil: {filepath.name}")

        titulo = None
        primera = secciones[0]
        if primera.titulo and primera.orden == 0:
            titulo = primera.titulo

        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.MD if es_markdown else Formato.TXT,
            fenomeno=1,
            secciones=secciones,
            titulo_documento=titulo,
        )

    def _segmentar(self, texto: str, es_markdown: bool) -> List[Section]:
        """Convierte el texto plano en secciones estructurales."""
        secciones: List[Section] = []
        bloque: List[str] = []
        titulo_actual: Optional[str] = None
        orden = 0
        en_lista = False

        def cerrar_bloque():
            nonlocal orden, en_lista
            contenido = _colapsar(bloque)
            if contenido:
                secciones.append(
                    Section(titulo=titulo_actual, texto=contenido, orden=orden, splittable=True)
                )
                orden += 1
            bloque.clear()
            en_lista = False

        for linea in texto.splitlines():
            desnuda = linea.strip()
            if not desnuda:
                cerrar_bloque()
                continue
            if es_markdown:
                coincidencia = _PATRON_ENCABEZADO.match(linea)
                if coincidencia:
                    cerrar_bloque()
                    titulo_actual = coincidencia.group(2).strip()
                    continue
                es_item_lista = bool(_PATRON_LISTA.match(linea) or _PATRON_LISTA_NUM.match(linea))
                if es_item_lista:
                    if not en_lista:
                        cerrar_bloque()
                        en_lista = True
                    bloque.append(re.sub(r"^\s*[-*+]\s+|\s*\d+[.)]\s+", "", linea).strip())
                    continue
                if en_lista:
                    cerrar_bloque()
            bloque.append(desnuda)

        cerrar_bloque()
        return secciones


def _colapsar(lineas: List[str]) -> str:
    """Une las líneas del bloque conservando un solo salto entre ellas."""
    return "\n".join(l.strip() for l in lineas if l.strip())
