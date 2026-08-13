"""
Tests de la capa de soporte de OCR (localización de Tesseract e idiomas).
"""

from __future__ import annotations

import pytest

from src.support import ocr


@pytest.fixture(autouse=True)
def _sin_cache():
    """Los resultados están cacheados por proceso: se limpian en cada test."""
    ocr.configurar_tesseract.cache_clear()
    ocr.idiomas_disponibles.cache_clear()
    yield
    ocr.configurar_tesseract.cache_clear()
    ocr.idiomas_disponibles.cache_clear()


def test_ruta_forzada_por_variable_de_entorno_tiene_prioridad(monkeypatch, tmp_path):
    """TESSERACT_CMD permite apuntar a una instalación fuera del PATH."""
    binario = tmp_path / "tesseract.exe"
    binario.write_text("", encoding="utf-8")
    monkeypatch.setenv("TESSERACT_CMD", str(binario))

    assert next(ocr._rutas_candidatas()) == str(binario)


def test_busca_el_binario_fuera_del_path(monkeypatch):
    """El instalador de Windows no añade Tesseract al PATH: hay que buscarlo."""
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)

    candidatas = list(ocr._rutas_candidatas())
    assert r"C:\Program Files\Tesseract-OCR\tesseract.exe" in candidatas
    assert "/usr/bin/tesseract" in candidatas


def test_sin_binario_no_hay_ocr(monkeypatch):
    """Si no hay ejecutable, se informa en vez de fallar con un error opaco."""
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr, "_rutas_candidatas", lambda: iter(()))

    assert ocr.configurar_tesseract() is None
    assert ocr.tesseract_disponible() is False


def test_idiomas_ocr_usa_solo_los_instalados(monkeypatch):
    monkeypatch.setattr(ocr, "idiomas_disponibles", lambda: frozenset({"spa", "eng", "osd"}))
    assert ocr.idiomas_ocr() == "spa+eng"


def test_idiomas_ocr_degrada_a_ingles_si_falta_el_espanol(monkeypatch):
    """Sin spa.traineddata el OCR sigue, pero con los idiomas que haya."""
    monkeypatch.setattr(ocr, "idiomas_disponibles", lambda: frozenset({"eng", "osd"}))
    assert ocr.idiomas_ocr() == "eng"


def test_idiomas_ocr_sin_tesseract_devuelve_ingles(monkeypatch):
    monkeypatch.setattr(ocr, "idiomas_disponibles", lambda: frozenset())
    assert ocr.idiomas_ocr() == "eng"


@pytest.mark.parametrize(
    "entrada,esperado",
    [
        ("spa+eng", ["es", "en"]),  # convención Tesseract
        ("es,en", ["es", "en"]),  # convención easyocr
        ("spa", ["es"]),
        (None, ["es", "en", "pt"]),  # por defecto: idiomas del corpus
    ],
)
def test_idiomas_easyocr_traduce_los_codigos(entrada, esperado):
    """easyocr usa ISO de dos letras, no los códigos de Tesseract."""
    assert ocr.idiomas_easyocr(entrada) == esperado
