"""
Extractor de HTML (BeautifulSoup4).

- Descarta markup pero conserva la estructura: encabezados h1-h6, párrafos y
  listas se convierten en secciones con su título estructural.
- Extrae ``<title>``, idioma ``lang`` y fecha de ``<meta>``.
- Si ``beautifulsoup4`` no está instalado, se usa un respaldo con regex.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from pathlib import Path
from typing import List, Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section

logger = logging.getLogger(__name__)

_TAGS_ENCABEZADO = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


@register_extractor(".html", ".htm")
class HTMLExtractor(BaseExtractor):
    """Extrae texto de documentos HTML conservando su estructura jerárquica."""

    supported_formats = ["html", "htm"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el documento HTML en secciones por encabezado/párrafo."""
        texto = self._decodificar(self._leer_bytes(filepath))
        try:
            from bs4 import BeautifulSoup  # type: ignore

            sopa = BeautifulSoup(texto, "html.parser")
        except ImportError:
            return self._extract_sin_bs4(filepath, texto)

        for tag in sopa(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()

        titulo = None
        title_tag = sopa.find("title")
        if title_tag and title_tag.get_text(strip=True):
            titulo = title_tag.get_text(" ", strip=True)

        secciones: List[Section] = []
        bloque_actual: List[str] = []
        encabezado_actual: Optional[str] = None
        orden = 0

        def cerrar_seccion():
            nonlocal orden
            if encabezado_actual and not bloque_actual:
                # Un encabezado sin contenido posterior aún así aporta texto.
                bloque_actual.append(encabezado_actual)
            # Cada <p>/<li> es un párrafo: se separan con línea en blanco
            # (\n\n), la convención que espera el chunking por párrafo.
            texto_seccion = _normalizar_espacios("\n\n".join(bloque_actual))
            if texto_seccion:
                secciones.append(
                    Section(titulo=encabezado_actual, texto=texto_seccion, orden=orden, splittable=True)
                )
                orden += 1
            bloque_actual.clear()

        def abrir_encabezado(nombre_tag: str):
            nonlocal encabezado_actual
            h = sopa.find(nombre_tag)
            if h:
                encabezado_actual = h.get_text(" ", strip=True)
                # El texto del encabezado forma parte del contenido indexable.
                bloque_actual.append(encabezado_actual)

        for elemento in sopa.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "article", "section"]):
            nombre = elemento.name
            if nombre in _TAGS_ENCABEZADO:
                cerrar_seccion()
                abrir_encabezado(nombre)
                continue
            if nombre in ("p", "li"):
                texto_elem = _normalizar_espacios(elemento.get_text(" ", strip=True))
                if texto_elem:
                    bloque_actual.append(texto_elem)
            elif nombre in ("article", "section"):
                # Límite estructural adicional: cierra el bloque en curso.
                cerrar_seccion()
                encabezado_actual = None

        cerrar_seccion()

        if not secciones:
            # Cada nodo de texto es un párrafo (línea en blanco entre ellos).
            cuerpo = sopa.get_text("\n\n", strip=True)
            if cuerpo:
                secciones.append(Section(texto=cuerpo, orden=0, splittable=True))

        idioma = sopa.html.get("lang") if sopa.html else None

        fecha = None
        meta = sopa.find("meta", attrs={"name": re.compile(r"date|published", re.I)})
        if meta and meta.get("content"):
            fecha = meta["content"].strip()

        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.HTML,
            fenomeno=1,
            secciones=secciones,
            titulo_documento=titulo,
            fecha_publicacion=fecha,
            metadata={"idioma_html": idioma} if idioma else {},
        )

    def _extract_sin_bs4(self, filepath: Path, texto: str) -> ExtractedDocument:
        """Respaldo por regex cuando no hay beautifulsoup4."""
        logger.warning("beautifulsoup4 no instalado; usando regex para %s", filepath.name)
        cuerpo = re.sub(r"<script.*?</script>|<style.*?</style>", " ", texto, flags=re.S | re.I)
        cuerpo = re.sub(r"<[^>]+>", " ", cuerpo)
        cuerpo = _normalizar_espacios(unescape(cuerpo))
        if not cuerpo:
            raise ExtractorError(f"HTML sin texto útil: {filepath.name}")
        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.HTML,
            fenomeno=1,
            secciones=[Section(texto=cuerpo, orden=0, splittable=True)],
        )


def _normalizar_espacios(texto: str) -> str:
    """Colapsa espacios y saltos de línea redundantes."""
    return re.sub(r"[ \t]+", " ", texto).replace("\n ", "\n").strip()
