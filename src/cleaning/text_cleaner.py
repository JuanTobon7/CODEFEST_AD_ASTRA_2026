"""
Limpieza y normalización del texto extraído.

El ``TextCleaner`` aplica, en orden:
1. Normalización de codificación a UTF-8.
2. Eliminación de caracteres de control y espacios redundantes.
3. Eliminación de boilerplate repetitivo (headers/footers de página,
   numeración, boilerplate web) mediante heurísticas de repetición entre
   secciones del mismo documento y, opcionalmente, del mismo lote.
4. Detección de idioma predominante (langdetect) como metadata.

NUNCA altera el contenido semántico: no resume ni reformula.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, Optional, Set

from src.models.extracted_document import ExtractedDocument

logger = logging.getLogger(__name__)

# Caracteres de control excepto \t \n \r.
_PATRON_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Líneas que parecen numeración de página, saltos decorativos, etc.
_PATRON_NUMERACION = re.compile(r"^[\s\d\.,\-–—/\\|·•*_=\-~]+$")
# Líneas que parecen "clave: valor" (no se tratan como boilerplate).
_PATRON_CLAVE_VALOR = re.compile(r"^.{1,30}:\s*\S")


class TextCleaner:
    """Limpia documentos extraídos y detecta su idioma.

    Atributos:
        default_language: Idioma por defecto si la detección falla.
        threshold_repetidos: Veces mínimas (en secciones distintas) para
            considerar una línea como boilerplate dentro de un documento.
    """

    def __init__(self, default_language: str = "es", threshold_repetidos: int = 2) -> None:
        self.default_language = default_language
        self.threshold_repetidos = threshold_repetidos
        self._boilerplate_lote: Set[str] = set()

    # API pública ---------------------------------------------------------------

    def clean(self, doc: ExtractedDocument) -> ExtractedDocument:
        """Limpia el documento in-place y devuelve la misma instancia."""
        lineas_repetidas = self._detectar_boilerplate_local(doc)
        doc.idioma = self._detectar_idioma(doc)

        for seccion in doc.secciones:
            texto = seccion.texto
            texto = self._normalizar_codificacion(texto)
            texto = self._eliminar_controles(texto)
            texto = self._colapsar_espacios(texto)
            texto = self._quitar_boilerplate(texto, lineas_repetidas, seccion.splittable)
            seccion.texto = texto.strip()
        return doc

    def set_corpus_boilerplate(self, docs: Iterable[ExtractedDocument], min_docs: int = 3) -> None:
        """Detecta líneas repetidas entre documentos del mismo lote.

        Las líneas cortas repetidas en al menos ``min_docs`` documentos
        distintos se consideran boilerplate de sitio web/plantilla.
        """
        conteo: Dict[str, int] = defaultdict(int)
        for doc in docs:
            vistos: Set[str] = set()
            for seccion in doc.secciones:
                if not seccion.splittable:
                    continue
                for linea in seccion.texto.splitlines():
                    desnuda = linea.strip()
                    if self._es_candidata_boilerplate(desnuda):
                        vistos.add(desnuda)
            for linea in vistos:
                conteo[linea] += 1
        self._boilerplate_lote = {l for l, c in conteo.items() if c >= min_docs}
        logger.info("Boilerplate de lote detectado: %d líneas", len(self._boilerplate_lote))

    # Paso 1: codificación ------------------------------------------------------

    @staticmethod
    def _normalizar_codificacion(texto: str) -> str:
        """Garantiza str UTF-8 (los bytes ya se decodificaron en el extractor)."""
        if isinstance(texto, bytes):
            return texto.decode("utf-8", errors="replace")
        return texto.encode("utf-8", errors="replace").decode("utf-8")

    # Paso 2: caracteres de control y espacios ----------------------------------

    @staticmethod
    def _eliminar_controles(texto: str) -> str:
        return _PATRON_CONTROL.sub("", texto)

    @staticmethod
    def _colapsar_espacios(texto: str) -> str:
        """Colapsa espacios múltiples y elimina espacios al fin de línea."""
        texto = re.sub(r"[ \t]+", " ", texto)
        texto = re.sub(r" *\n", "\n", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto

    # Paso 3: boilerplate --------------------------------------------------------

    def _detectar_boilerplate_local(self, doc: ExtractedDocument) -> Set[str]:
        """Líneas repetidas en secciones distintas del mismo documento."""
        apariciones: Dict[str, Set[int]] = defaultdict(set)
        for seccion in doc.secciones:
            if not seccion.splittable:
                continue
            for linea in seccion.texto.splitlines():
                desnuda = linea.strip()
                if self._es_candidata_boilerplate(desnuda):
                    apariciones[desnuda].add(seccion.orden)
        return {
            linea
            for linea, secciones in apariciones.items()
            if len(secciones) >= self.threshold_repetidos and len(linea) <= 80
        }

    @staticmethod
    def _es_candidata_boilerplate(linea: str) -> bool:
        """Una línea es candidata si es corta, no es numeración pura y no
        parece un par ``clave: valor`` (protege CSV/XLSX/PBF)."""
        if not linea or len(linea) > 80:
            return False
        if _PATRON_NUMERACION.fullmatch(linea):
            return False
        if _PATRON_CLAVE_VALOR.match(linea):
            return False
        return True

    def _quitar_boilerplate(self, texto: str, locales: Set[str], splittable: bool) -> str:
        """Elimina líneas repetidas (locales o de lote) de secciones partibles."""
        if not splittable or (not locales and not self._boilerplate_lote):
            return texto
        prohibidas = locales | self._boilerplate_lote
        lineas = []
        for linea in texto.splitlines():
            desnuda = linea.strip()
            if desnuda in prohibidas and self._es_candidata_boilerplate(desnuda):
                continue
            lineas.append(linea)
        return "\n".join(lineas)

    # Paso 4: idioma --------------------------------------------------------------

    def _detectar_idioma(self, doc: ExtractedDocument) -> str:
        """Detecta el idioma predominante del documento con langdetect."""
        muestra = " ".join(s.texto for s in doc.secciones[:3] if s.texto)
        if len(muestra) < 50:
            return self.default_language
        try:
            from langdetect import detect  # type: ignore

            return detect(muestra)
        except Exception as exc:
            logger.debug("langdetect no disponible o falló: %s", exc)
            return self.default_language
