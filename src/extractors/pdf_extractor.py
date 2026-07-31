"""
Extractor de PDFs mediante PyMuPDF.

- Preserva el orden de lectura (bloques en orden de aparición).
- Detecta encabezados por heurística de tamaño de fuente/negrita.
- Cada página genera una sección estructural (``splittable=True``).
- Si ``pymupdf`` no está instalado, se usa un respaldo por bytes mínimos
  (los textos embebidos suelen aparecer como fragmentos de 8 bytes).
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


def _limpiar_linea(linea: str) -> str:
    """Normaliza espacios y elimina basura de ordenamiento de bloques."""
    linea = re.sub(r"\s+", " ", linea).strip()
    # Basura típica de ordenamiento de fuentes (ej. "nnrI ", "nnt ").
    if re.fullmatch(r"[nrIlt ]{2,}", linea):
        return ""
    return linea


@register_extractor(".pdf")
class PDFExtractor(BaseExtractor):
    """Extrae texto de documentos PDF respetando el orden de lectura."""

    supported_formats = ["pdf"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el documento PDF completo.

        Raises:
            ExtractorError: Si el PDF no existe, está cifrado o no tiene texto.
        """
        self._leer_bytes(filepath)  # valida existencia/legibilidad antes de pymupdf
        try:
            import pymupdf  # type: ignore
        except ImportError:
            return self._extract_sin_pymupdf(filepath)

        try:
            doc = pymupdf.open(filepath)
        except Exception as exc:
            raise ExtractorError(f"PDF ilegible ({filepath.name}): {exc}") from exc

        try:
            secciones: List[Section] = []
            titulo: Optional[str] = None
            orden = 0
            for numero_pagina, pagina in enumerate(doc):
                if pagina.rotation:
                    pagina.set_rotation(0)
                bloques = pagina.get_text("dict")["blocks"]
                bloques.sort(key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
                texto_pagina: List[str] = []
                encabezado: Optional[str] = None
                for bloque in bloques:
                    if bloque.get("type") != 0:  # 0 = texto; imágenes se ignoran aquí
                        continue
                    tamano_max, en_negrita = self._estilo_bloque(bloque)
                    # El texto vive en los "spans" de cada línea, no en la línea.
                    lineas: List[str] = []
                    for linea in bloque.get("lines", []):
                        for span in linea.get("spans", []):
                            texto_span = _limpiar_linea(span.get("text", ""))
                            if texto_span:
                                lineas.append(texto_span)
                    texto_bloque = " ".join(lineas)
                    if not texto_bloque:
                        continue
                    if encabezado is None and self._es_encabezado(tamano_max, en_negrita, texto_bloque):
                        encabezado = texto_bloque
                    texto_pagina.append(texto_bloque)
                texto = "\n\n".join(t for t in texto_pagina if t.strip())
                secciones.append(
                    Section(
                        titulo=encabezado or (titulo if numero_pagina == 0 else None),
                        texto=texto,
                        orden=orden,
                        splittable=True,
                    )
                )
                orden += 1
                if numero_pagina == 0 and titulo is None and encabezado:
                    titulo = encabezado
            doc.close()
        except Exception as exc:
            try:
                doc.close()
            except Exception:
                pass
            raise ExtractorError(f"Fallo al extraer texto de {filepath.name}: {exc}") from exc

        if not any(s.texto.strip() for s in secciones):
            raise ExtractorError(f"El PDF no contiene texto extraíble: {filepath.name}")

        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.PDF,
            fenomeno=1,
            secciones=secciones,
            titulo_documento=titulo,
        )

    # Heurísticas de encabezado ----------------------------------------------

    @staticmethod
    def _estilo_bloque(bloque: dict) -> tuple[float, bool]:
        """Tamaño máximo de fuente y si hay negrita dominante en el bloque."""
        tamano_max = 0.0
        negrita_total = 0
        lineas = bloque.get("lines", [])
        for linea in lineas:
            for span in linea.get("spans", []):
                tamano_max = max(tamano_max, float(span.get("size", 0)))
                flags = int(span.get("flags", 0))
                if flags & 16:  # bit de negrita en PyMuPDF
                    negrita_total += 1
        return tamano_max, negrita_total > 0

    @staticmethod
    def _es_encabezado(tamano: float, en_negrita: bool, texto: str) -> bool:
        """Heurística: fuente grande o negrita + texto corto y sin punto final."""
        if len(texto) > 120:
            return False
        sin_punto = texto.rstrip().endswith((".", "?", "!")) is False
        return sin_punto and (tamano >= 14 or (en_negrita and tamano >= 11))

    # Respaldo sin PyMuPDF ---------------------------------------------------

    def _extract_sin_pymupdf(self, filepath: Path) -> ExtractedDocument:
        """Respaldo mínimo: extrae fragmentos de texto puro de 8 bytes."""
        logger.warning("pymupdf no instalado; usando extracción mínima para %s", filepath.name)
        data = self._leer_bytes(filepath)
        coincidencias = re.findall(rb"\(([^()\\]{8,})\)", data)
        texto = " ".join(m.decode("latin-1", errors="replace") for m in coincidencias)
        texto = re.sub(r"\s+", " ", texto).strip()
        if not texto:
            raise ExtractorError(f"Sin texto en el PDF: {filepath.name}")
        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.PDF,
            fenomeno=1,
            secciones=[Section(texto=texto, orden=0, splittable=True)],
        )
