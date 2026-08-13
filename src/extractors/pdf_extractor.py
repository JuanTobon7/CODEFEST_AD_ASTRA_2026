"""
Extractor de PDFs mediante PyMuPDF.

- Preserva el orden de lectura (bloques en orden de aparición).
- Detecta encabezados por heurística de tamaño de fuente/negrita.
- Cada página genera una sección estructural (``splittable=True``).
- Si ``pymupdf`` no está instalado, se usa un respaldo por bytes mínimos
  (los textos embebidos suelen aparecer como fragmentos de 8 bytes).

Reconstrucción de párrafos
--------------------------
La extracción de PDF pierde con facilidad los saltos de párrafo reales: el
maquetado parte un mismo párrafo en varios bloques (columnas, texto justificado,
cambio de página) y corta palabras con guion al final de línea. Como el chunking
usa ``\\n\\n`` como frontera de párrafo, cada bloque suelto se convertiría en un
"párrafo" que empieza y termina a mitad de oración. Por eso aquí se:

1. unen las líneas de un bloque deshaciendo la partición por guion;
2. descartan los ruidos de página (números de página, filetes decorativos);
3. fusionan los bloques consecutivos que continúan la misma oración;
4. reflota entre páginas la oración que quedó abierta al final de la anterior.

El resultado es que ``\\n\\n`` marca fronteras de párrafo reales, que son
también fronteras de oración.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section
from src.support.ocr import configurar_tesseract, idiomas_ocr

logger = logging.getLogger(__name__)


# Puntuación con la que un bloque cierra una unidad de texto completa.
_PUNTUACION_CIERRE = ".!?…:;»›。！？．｡"
# Comillas y paréntesis que pueden ir después de la puntuación de cierre.
_RE_SIGNOS_CIERRE = re.compile(r"[\"'”’)\]\}\s]+$")
# Un bloque que continúa el anterior empieza en minúscula o en apertura baja.
_RE_INICIO_CONTINUACION = re.compile(r"^[a-záéíóúñüàèìòùâêîôûãõäëïöç]|^[,;)]")
# Marcadores de lista: nunca son continuación del bloque anterior.
_RE_MARCADOR_LISTA = re.compile(r"^\s*(?:[-–—•*·▪◦]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s")
# Numeración de página suelta ("12", "- 12 -") y pies tipo "Página 3 de 40".
_RE_NUMERACION_SUELTA = re.compile(r"^[\s\d.,\-–—/|·]+$")
_RE_PIE_PAGINA = re.compile(
    r"^(?:p[áa]g(?:ina)?\.?|page)\s*\d+(?:\s*(?:de|of|/)\s*\d+)?$", re.IGNORECASE
)
_MAX_CARACTERES_RUIDO = 20


def _limpiar_linea(linea: str) -> str:
    """Normaliza espacios y elimina basura de ordenamiento de bloques."""
    linea = re.sub(r"\s+", " ", linea).strip()
    # Basura típica de ordenamiento de fuentes (ej. "nnrI ", "nnt ").
    if re.fullmatch(r"[nrIlt ]{2,}", linea):
        return ""
    return linea


def _cierra_unidad(texto: str) -> bool:
    """True si el bloque termina una oración o unidad completa."""
    nucleo = _RE_SIGNOS_CIERRE.sub("", texto.rstrip())
    return bool(nucleo) and nucleo[-1] in _PUNTUACION_CIERRE


def _es_ruido_de_pagina(texto: str) -> bool:
    """True si el bloque es numeración de página o un filete decorativo.

    Estos bloques no llevan contenido pero rompen la continuidad del párrafo
    (quedan intercalados entre el final de una página y el inicio de la
    siguiente), así que se descartan antes de reconstruir los párrafos.
    """
    desnudo = texto.strip()
    if not desnudo:
        return True
    if _RE_PIE_PAGINA.fullmatch(desnudo):
        return True
    # Solo se descarta numeración corta: una fila de tabla numérica es contenido.
    return len(desnudo) <= _MAX_CARACTERES_RUIDO and bool(_RE_NUMERACION_SUELTA.fullmatch(desnudo))


def _es_continuacion(anterior: str, siguiente: str) -> bool:
    """True si ``siguiente`` continúa la oración abierta en ``anterior``.

    Se exige evidencia por ambos lados —el bloque previo no cierra unidad y el
    siguiente arranca en minúscula sin marcador de lista— para no fusionar
    párrafos legítimamente distintos.
    """
    if not anterior.strip() or not siguiente.strip():
        return False
    if _cierra_unidad(anterior):
        return False
    if _RE_MARCADOR_LISTA.match(siguiente):
        return False
    return bool(_RE_INICIO_CONTINUACION.match(siguiente.lstrip()))


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
                    texto_bloque = self._texto_bloque(bloque)
                    if not texto_bloque or _es_ruido_de_pagina(texto_bloque):
                        continue
                    if encabezado is None and self._es_encabezado(tamano_max, en_negrita, texto_bloque):
                        encabezado = texto_bloque
                    # Un párrafo partido por el maquetado en varios bloques se
                    # reconstruye; si no, cada trozo sería un "párrafo" cortado.
                    if texto_pagina and _es_continuacion(texto_pagina[-1], texto_bloque):
                        texto_pagina[-1] = f"{texto_pagina[-1]} {texto_bloque}"
                        continue
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
            self._reflotar_entre_paginas(secciones)
            doc.close()
        except Exception as exc:
            try:
                doc.close()
            except Exception:
                pass
            raise ExtractorError(f"Fallo al extraer texto de {filepath.name}: {exc}") from exc

        if not any(s.texto.strip() for s in secciones):
            ocr_doc = self._extract_con_ocr(filepath, doc)
            if ocr_doc is not None:
                return ocr_doc
            raise ExtractorError(f"El PDF no contiene texto extraíble: {filepath.name}")

        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.PDF,
            fenomeno=1,
            secciones=secciones,
            titulo_documento=titulo,
        )

    # Reconstrucción de párrafos ------------------------------------------------

    @staticmethod
    def _texto_bloque(bloque: dict) -> str:
        """Texto de un bloque con sus líneas envueltas ya reunidas.

        Los ``spans`` de una misma línea son trozos contiguos (un cambio de
        fuente parte una palabra en dos spans), así que se concatenan sin
        separador añadido y se normalizan los espacios al final; unirlos con
        un espacio insertaría espacios dentro de las palabras.
        """
        lineas: List[str] = []
        for linea in bloque.get("lines", []):
            texto_linea = _limpiar_linea("".join(span.get("text", "") for span in linea.get("spans", [])))
            if texto_linea:
                lineas.append(texto_linea)
        return PDFExtractor._unir_lineas(lineas)

    @staticmethod
    def _unir_lineas(lineas: List[str]) -> str:
        """Une líneas envueltas deshaciendo la partición de palabra por guion.

        ``informa-`` + ``ción`` → ``información`` (la línea siguiente empieza en
        minúscula: el guion era de partición). Si la siguiente línea empieza en
        mayúscula el guion se conserva y se une sin espacio, porque es parte de
        un compuesto (``Norte-`` + ``Sur`` → ``Norte-Sur``).
        """
        resultado = ""
        for linea in lineas:
            if not resultado:
                resultado = linea
                continue
            if re.search(r"[^\W\d_]-$", resultado, re.UNICODE):
                # Guion al final de línea: partición de palabra o compuesto.
                resultado = (
                    resultado[:-1] + linea
                    if re.match(r"^[a-záéíóúñü]", linea)
                    else resultado + linea
                )
                continue
            resultado = f"{resultado} {linea}"
        return resultado

    @staticmethod
    def _reflotar_entre_paginas(secciones: List[Section]) -> None:
        """Cierra en su página la oración que continúa en la siguiente.

        Cada página es una sección y el chunking nunca cruza secciones, así que
        una oración que salta de página quedaría partida en dos fragmentos. Si
        la página termina con una oración abierta y la siguiente arranca
        continuándola, ese primer párrafo se mueve al final de la página previa
        (se conserva el orden de lectura del documento).
        """
        for actual, siguiente in zip(secciones, secciones[1:]):
            if not actual.texto.strip() or not siguiente.texto.strip():
                continue
            parrafos_actual = re.split(r"\n\s*\n", actual.texto)
            parrafos_siguiente = re.split(r"\n\s*\n", siguiente.texto)
            if not _es_continuacion(parrafos_actual[-1], parrafos_siguiente[0]):
                continue
            parrafos_actual[-1] = f"{parrafos_actual[-1]} {parrafos_siguiente[0].strip()}"
            actual.texto = "\n\n".join(parrafos_actual)
            siguiente.texto = "\n\n".join(parrafos_siguiente[1:])

    # OCR de PDFs escaneados ----------------------------------------------------

    def _extract_con_ocr(self, filepath: Path, doc) -> Optional[ExtractedDocument]:
        """OCR página por página de un PDF sin capa de texto (escaneado).

        Requiere el binario de Tesseract en el sistema (pytesseract es solo el
        envoltorio). Devuelve ``None`` si el OCR no está disponible; en ese
        caso el llamador mantiene el error de "sin texto extraíble".
        """
        if configurar_tesseract() is None:  # pragma: no cover - entorno sin Tesseract
            logger.warning(
                "PDF sin texto y OCR no disponible para %s. Instala el binario de "
                "Tesseract (winget install UB-Mannheim.TesseractOCR) o indica su "
                "ruta con la variable TESSERACT_CMD",
                filepath.name,
            )
            return None
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - entorno sin Pillow
            logger.warning("OCR no disponible para %s: %s", filepath.name, exc)
            return None

        import io

        logger.info("PDF escaneado; aplicando OCR a %s", filepath.name)
        secciones: List[Section] = []
        try:
            idiomas = idiomas_ocr()
            for numero_pagina, pagina in enumerate(doc):
                pix = pagina.get_pixmap(dpi=300)
                imagen = Image.open(io.BytesIO(pix.tobytes("png")))
                texto = pytesseract.image_to_string(imagen, lang=idiomas)
                texto = " ".join(texto.split())
                if not texto.strip():
                    continue
                secciones.append(
                    Section(texto=texto, orden=len(secciones), splittable=True)
                )
        except Exception as exc:
            raise ExtractorError(f"OCR falló para {filepath.name}: {exc}") from exc

        if not secciones:
            raise ExtractorError(f"El PDF no contiene texto extraíble: {filepath.name}")
        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.PDF,
            fenomeno=1,
            secciones=secciones,
            metadata={"ocr": True, "ocr_language": idiomas_ocr()},
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
