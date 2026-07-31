"""
Extractor de imágenes con OCR (pytesseract o easyocr).

El texto OCR de cada imagen es una unidad estructural única.
Si el idioma de OCR no se especifica, se detecta automáticamente probando
los idiomas disponibles (``spa+eng`` por defecto).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section

logger = logging.getLogger(__name__)


@register_extractor(".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp")
class ImageExtractor(BaseExtractor):
    """Realiza OCR sobre imágenes y devuelve el texto como única sección."""

    supported_formats = ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"]

    def __init__(self, ocr_language: Optional[str] = None) -> None:
        """Inicializa el extractor.

        Args:
            ocr_language: Idiomas para OCR (p. ej. ``spa+eng``). Si es ``None``
                se intenta autodetección.
        """
        self.ocr_language = ocr_language

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
            metadata={"ocr_language": self.ocr_language or "auto"},
        )

    def _ocr(self, filepath: Path) -> str:
        """Intenta pytesseract y cae a easyocr si no está disponible."""
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            idiomas = self.ocr_language or self._detectar_idiomas_pytesseract()
            texto = pytesseract.image_to_string(Image.open(filepath), lang=idiomas)
            return texto
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pytesseract falló para %s: %s", filepath.name, exc)

        try:
            import easyocr  # type: ignore

            idiomas = self.ocr_language or "es,en,pt"
            lector = easyocr.Reader([l.strip() for l in idiomas.split("+") if l.strip()], gpu=False, verbose=False)
            resultados = lector.readtext(str(filepath), detail=0, paragraph=True)
            return "\n".join(str(r) for r in resultados)
        except ImportError as exc:
            raise ExtractorError(
                f"OCR no disponible: instala pytesseract o easyocr (imagen: {filepath.name})"
            ) from exc
        except Exception as exc:
            raise ExtractorError(f"easyocr falló para {filepath.name}: {exc}") from exc

    @staticmethod
    def _detectar_idiomas_pytesseract() -> str:
        """Idiomas disponibles para pytesseract (prioridad es/en/pt)."""
        try:
            import pytesseract  # type: ignore

            disponibles = set(pytesseract.get_languages(config=""))
        except Exception:
            disponibles = set()
        preferidos = [l for l in ("spa", "eng", "por") if l in disponibles]
        if preferidos:
            return "+".join(preferidos)
        return "eng"
