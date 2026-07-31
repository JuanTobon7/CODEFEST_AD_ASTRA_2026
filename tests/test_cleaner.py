"""
Tests del TextCleaner (limpieza, boilerplate e idioma).
"""

from __future__ import annotations

import pytest

from src.cleaning.text_cleaner import TextCleaner
from src.models.extracted_document import ExtractedDocument, Formato, Section


def _doc(secciones: list[Section]) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="d1", fuente="f.txt", formato=Formato.TXT, fenomeno=1, secciones=secciones
    )


def test_elimina_caracteres_de_control_y_espacios():
    secciones = [Section(texto="Hola\x00mundo.\n\n\n\nSegunda  oración.\r\n", orden=0)]
    doc = TextCleaner().clean(_doc(secciones))
    assert "\x00" not in doc.secciones[0].texto
    assert "\r" not in doc.secciones[0].texto
    assert "  " not in doc.secciones[0].texto
    assert "Segunda oración." in doc.secciones[0].texto


def test_elimina_boilerplate_repetido_entre_secciones():
    secciones = [
        Section(texto="Contenido real uno.\nPie de página - Confidencial", orden=0),
        Section(texto="Contenido real dos.\nPie de página - Confidencial", orden=1),
    ]
    doc = TextCleaner(threshold_repetidos=2).clean(_doc(secciones))
    assert "Confidencial" not in doc.secciones[0].texto
    assert "Contenido real uno." in doc.secciones[0].texto


def test_no_toca_lineas_clave_valor_ni_unidades_atomicas():
    secciones = [
        Section(texto="ciudad: Bogotá", orden=0, splittable=False),
        Section(texto="ciudad: Bogotá", orden=1, splittable=False),
    ]
    doc = TextCleaner(threshold_repetidos=2).clean(_doc(secciones))
    assert "ciudad: Bogotá" in doc.secciones[0].texto
    assert "ciudad: Bogotá" in doc.secciones[1].texto


def test_boilerplate_de_lote():
    doc_a = _doc([Section(texto="Artículo sobre clima.\nCopyright 2026 SitioWeb", orden=0)])
    doc_b = _doc([Section(texto="Otro artículo.\nCopyright 2026 SitioWeb", orden=0)])
    doc_c = _doc([Section(texto="Más contenido.\nCopyright 2026 SitioWeb", orden=0)])
    cleaner = TextCleaner()
    cleaner.set_corpus_boilerplate([doc_a, doc_b, doc_c], min_docs=3)
    doc_a = cleaner.clean(doc_a)
    assert "Copyright 2026 SitioWeb" not in doc_a.secciones[0].texto


def test_detecta_idioma():
    secciones = [Section(texto="Esta es una oración en español con suficiente longitud para detectar el idioma.", orden=0)]
    doc = TextCleaner(default_language="es").clean(_doc(secciones))
    assert doc.idioma == "es"


def test_no_altera_contenido_semantico():
    texto_original = "El contenido semántico debe permanecer idéntico."
    secciones = [Section(texto=texto_original, orden=0)]
    doc = TextCleaner().clean(_doc(secciones))
    assert doc.secciones[0].texto == texto_original
