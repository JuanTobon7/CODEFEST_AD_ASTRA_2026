"""
Extractor de imágenes con OCR (pytesseract o easyocr).

El texto OCR de cada imagen es una unidad estructural única.
Si el idioma de OCR no se especifica, se detecta automáticamente entre los
idiomas realmente instalados en Tesseract (``spa+eng+por`` por preferencia).

La localización del binario de Tesseract vive en :mod:`src.support.ocr`, que lo
busca fuera del ``PATH`` (el instalador de Windows no lo añade).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section
from src.support.ocr import (
    configurar_tesseract,
    idiomas_easyocr,
    idiomas_ocr,
)

logger = logging.getLogger(__name__)


@register_extractor(".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".avif")
class ImageExtractor(BaseExtractor):
    """Realiza OCR sobre imágenes y devuelve el texto como única sección."""

    supported_formats = ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "avif"]

    def __init__(self, ocr_language: Optional[str] = None) -> None:
        """Inicializa el extractor.

        Args:
            ocr_language: Idiomas para OCR (p. ej. ``spa+eng``). Si es ``None``
                se intenta autodetección.
        """
        self.ocr_language = ocr_language
        # Motor e idiomas realmente empleados; se resuelven durante el OCR y se
        # guardan como metadata para poder auditar la calidad del texto.
        self._motor_usado: str = "ninguno"
        self._idiomas_usados: str = ""

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el texto de la imagen mediante OCR."""
        self._leer_bytes(filepath)  # valida existencia/lectura
        texto = self._ocr(filepath)
        texto = " ".join(texto.split()) if texto else ""
        if not texto:
            raise ExtractorError(f"OCR no devolvió texto para la imagen: {filepath.name}")
        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.IMAGE,
            fenomeno=1,
            secciones=[Section(texto=texto, orden=0, splittable=True)],
            metadata={
                "ocr": True,
                "ocr_engine": self._motor_usado,
                "ocr_language": self._idiomas_usados or self.ocr_language or "auto",
            },
        )

    def _ocr(self, filepath: Path) -> str:
        """Intenta Tesseract y cae a easyocr si no está disponible."""
        # 1) Tesseract: se localiza el binario antes de usar pytesseract, ya que
        # el instalador de Windows no lo deja en el PATH.
        if configurar_tesseract() is not None:
            try:
                import pytesseract  # type: ignore
                from PIL import Image  # type: ignore

                idiomas = self.ocr_language or idiomas_ocr()
                with Image.open(filepath) as imagen:
                    texto = pytesseract.image_to_string(imagen, lang=idiomas)
                self._motor_usado, self._idiomas_usados = "tesseract", idiomas
                return texto
            except ImportError:
                pass  # falta Pillow: probar easyocr
            except Exception as exc:
                logger.warning("Tesseract falló para %s: %s", filepath.name, exc)

        # 2) easyocr: sin binario externo (pero pesado, requiere torch).
        try:
            import easyocr  # type: ignore

            idiomas = idiomas_easyocr(self.ocr_language)
            lector = easyocr.Reader(idiomas, gpu=False, verbose=False)
            resultados = lector.readtext(str(filepath), detail=0, paragraph=True)
            self._motor_usado, self._idiomas_usados = "easyocr", "+".join(idiomas)
            return "\n".join(str(r) for r in resultados)
        except ImportError as exc:
            raise ExtractorError(
                f"OCR no disponible (imagen: {filepath.name}): no se encontró el "
                f"binario de Tesseract ni el paquete easyocr. Instala Tesseract "
                f"(winget install UB-Mannheim.TesseractOCR), indica su ruta con la "
                f"variable TESSERACT_CMD, o instala easyocr."
            ) from exc
        except Exception as exc:
            raise ExtractorError(f"easyocr falló para {filepath.name}: {exc}") from exc
