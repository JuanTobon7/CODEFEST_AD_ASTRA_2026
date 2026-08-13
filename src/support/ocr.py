"""
Localización y configuración del motor de OCR (Tesseract).

``pytesseract`` es solo un envoltorio: invoca el ejecutable ``tesseract`` y lo
busca en el ``PATH``. El instalador de Windows (UB-Mannheim) NO añade Tesseract
al ``PATH`` por defecto, así que el OCR fallaba con "falta el binario" aun
estando instalado. Este módulo lo busca en las rutas habituales de cada sistema
y lo registra en ``pytesseract`` una sola vez por proceso.

La ruta se puede forzar con la variable de entorno ``TESSERACT_CMD``.
"""

from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional, Sequence, Set

logger = logging.getLogger(__name__)

# Idiomas preferidos del corpus (español, inglés, portugués) en código Tesseract.
IDIOMAS_PREFERIDOS = ("spa", "eng", "por")

# Equivalencia entre los códigos de Tesseract y los de easyocr.
_CODIGOS_EASYOCR = {"spa": "es", "eng": "en", "por": "pt"}

# Ubicaciones típicas del ejecutable cuando no está en el PATH.
_RUTAS_WINDOWS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
_RUTAS_UNIX = (
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def _rutas_candidatas() -> Iterator[str]:
    """Rutas donde buscar el ejecutable, en orden de prioridad."""
    forzada = os.getenv("TESSERACT_CMD")
    if forzada:
        yield forzada
    en_path = shutil.which("tesseract")
    if en_path:
        yield en_path
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        yield str(Path(local_appdata) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
    yield from _RUTAS_WINDOWS
    yield from _RUTAS_UNIX


@lru_cache(maxsize=1)
def configurar_tesseract() -> Optional[str]:
    """Registra el binario de Tesseract en ``pytesseract``.

    Returns:
        Ruta del ejecutable encontrado, o ``None`` si no hay Tesseract usable
        (ya sea porque falta ``pytesseract`` o porque no está el binario).
    """
    try:
        import pytesseract  # type: ignore
    except ImportError:
        logger.warning("pytesseract no está instalado: el OCR por Tesseract no está disponible.")
        return None

    for ruta in _rutas_candidatas():
        if not ruta or not Path(ruta).is_file():
            continue
        pytesseract.pytesseract.tesseract_cmd = ruta
        try:
            version = pytesseract.get_tesseract_version()
        except Exception as exc:  # binario presente pero inservible
            logger.debug("Tesseract en %s no es utilizable: %s", ruta, exc)
            continue
        logger.info("Tesseract %s localizado en %s", version, ruta)
        return ruta

    logger.warning(
        "No se encontró el binario de Tesseract. Instálalo "
        "(winget install UB-Mannheim.TesseractOCR) o indica su ruta con "
        "la variable de entorno TESSERACT_CMD."
    )
    return None


def tesseract_disponible() -> bool:
    """True si hay un Tesseract utilizable en el sistema."""
    return configurar_tesseract() is not None


@lru_cache(maxsize=1)
def idiomas_disponibles() -> frozenset:
    """Idiomas (``traineddata``) instalados en Tesseract."""
    if not tesseract_disponible():
        return frozenset()
    try:
        import pytesseract  # type: ignore

        return frozenset(pytesseract.get_languages(config=""))
    except Exception as exc:
        logger.debug("No se pudieron listar los idiomas de Tesseract: %s", exc)
        return frozenset()


def idiomas_ocr(preferidos: Sequence[str] = IDIOMAS_PREFERIDOS) -> str:
    """Cadena ``lang`` para Tesseract con los idiomas realmente instalados.

    Si falta el español (el idioma dominante del corpus) se avisa una vez: el
    OCR seguirá funcionando con inglés, pero con bastante menos precisión sobre
    texto en español.
    """
    disponibles: Set[str] = set(idiomas_disponibles())
    presentes = [idioma for idioma in preferidos if idioma in disponibles]
    if "spa" in preferidos and "spa" not in disponibles and disponibles:
        _avisar_falta_espanol()
    return "+".join(presentes) if presentes else "eng"


@lru_cache(maxsize=1)
def _avisar_falta_espanol() -> None:
    """Aviso único: falta el paquete de idioma español."""
    logger.warning(
        "Tesseract no tiene instalado el idioma español (spa): el OCR usará los "
        "idiomas disponibles y perderá precisión. Añade spa.traineddata al "
        "directorio tessdata de la instalación."
    )


def idiomas_easyocr(idiomas: Optional[str] = None) -> list:
    """Convierte una lista de idiomas a los códigos que espera easyocr.

    Acepta las dos convenciones que circulan por el pipeline: la de Tesseract
    (``spa+eng``) y la separada por comas (``es,en``); easyocr usa códigos
    ISO de dos letras (``["es", "en"]``).
    """
    crudos = idiomas or ",".join(_CODIGOS_EASYOCR[i] for i in IDIOMAS_PREFERIDOS)
    partes = [p.strip() for p in crudos.replace("+", ",").split(",") if p.strip()]
    return [_CODIGOS_EASYOCR.get(p, p) for p in partes] or ["es", "en"]
