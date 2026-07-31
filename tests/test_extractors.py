"""
Tests del ExtractorFactory y de los extractores por formato.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extractors.base import ExtractorError
from src.extractors.factory import ExtractorFactory, register_extractor
from src.models.extracted_document import Formato


@pytest.fixture
def factory() -> ExtractorFactory:
    return ExtractorFactory()


def test_factory_selecciona_por_extension(factory):
    """El factory devuelve el extractor adecuado según la extensión."""
    assert type(factory.create(Path("doc.pdf"))).__name__ == "PDFExtractor"
    assert type(factory.create(Path("doc.html"))).__name__ == "HTMLExtractor"
    assert type(factory.create(Path("doc.md"))).__name__ == "MarkdownTxtExtractor"
    assert type(factory.create(Path("doc.txt"))).__name__ == "MarkdownTxtExtractor"
    assert type(factory.create(Path("doc.json"))).__name__ == "JSONExtractor"
    assert type(factory.create(Path("doc.csv"))).__name__ == "CSVExtractor"
    assert type(factory.create(Path("doc.xlsx"))).__name__ == "XLSXExtractor"
    assert type(factory.create(Path("img.png"))).__name__ == "ImageExtractor"
    assert type(factory.create(Path("map.pbf"))).__name__ == "PBFExtractor"


def test_factory_extensibilidad_por_decorador(factory):
    """Un nuevo formato se registra sin tocar el código de la fábrica."""

    @register_extractor(".epub")
    class EPUBExtractor:  # solo para verificar el registro
        pass

    assert "epub" in ExtractorFactory.extensiones_registradas()
    assert type(factory.create(Path("libro.epub"))).__name__ == "EPUBExtractor"


def test_factory_formato_desconocido(factory):
    """Una extensión no registrada lanza ExtractorError descriptivo."""
    with pytest.raises(ExtractorError, match="no soportado"):
        factory.create(Path("archivo.xyz"))


# --- Fixtures de archivos de ejemplo ------------------------------------------


@pytest.fixture
def archivo_md(tmp_path: Path) -> Path:
    ruta = tmp_path / "articulo.md"
    ruta.write_text(
        "# Introducción\n\n"
        "Este es el primer párrafo de ejemplo.\n\n"
        "## Detalles\n\n"
        "Segundo párrafo con más contenido.\n\n"
        "- punto uno\n- punto dos\n",
        encoding="utf-8",
    )
    return ruta


@pytest.fixture
def archivo_html(tmp_path: Path) -> Path:
    ruta = tmp_path / "pagina.html"
    ruta.write_text(
        "<html lang='es'><head><title>Mi página</title></head><body>"
        "<h1>Titular principal</h1>"
        "<p>Primer párrafo de la página.</p>"
        "<p>Segundo párrafo.</p>"
        "<h2>Subtítulo</h2><p>Último párrafo.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    return ruta


@pytest.fixture
def archivo_json(tmp_path: Path) -> Path:
    ruta = tmp_path / "noticias.json"
    ruta.write_text(
        json.dumps(
            [
                {
                    "title": "Primera noticia",
                    "body_paragraphs": ["Párrafo del cuerpo uno.", "Párrafo dos."],
                    "author": "Redacción",
                    "date": "2026-07-30",
                },
                {
                    "title": "Segunda noticia",
                    "body_paragraphs": ["Otro cuerpo de texto."],
                    "author": "Equipo",
                },
            ]
        ),
        encoding="utf-8",
    )
    return ruta


@pytest.fixture
def archivo_csv(tmp_path: Path) -> Path:
    ruta = tmp_path / "datos.csv"
    ruta.write_text(
        "ciudad,poblacion,region\n"
        "Bogotá,7743955,Andina\n"
        "Medellín,2529403,Andina\n",
        encoding="utf-8",
    )
    return ruta


# --- Pruebas de extractores ----------------------------------------------------


def test_markdown_extractor_segmenta_por_encabezados(archivo_md, factory):
    doc = factory.create(archivo_md).extract(archivo_md)
    assert doc.formato == Formato.MD
    assert doc.fuente == "articulo.md"
    titulos = [s.titulo for s in doc.secciones if s.titulo]
    assert "Introducción" in titulos
    assert "Detalles" in titulos
    texto_total = doc.texto_completo
    assert "punto uno" in texto_total
    assert "punto dos" in texto_total


def test_html_extractor_descarta_markup(archivo_html, factory):
    doc = factory.create(archivo_html).extract(archivo_html)
    assert "<h1>" not in doc.texto_completo
    assert doc.titulo_documento == "Mi página"
    assert "Titular principal" in doc.texto_completo
    assert doc.metadata.get("idioma_html") == "es"


def test_json_extractor_concatena_titulo_y_cuerpo(archivo_json, factory):
    doc = factory.create(archivo_json).extract(archivo_json)
    assert doc.formato == Formato.JSON
    assert len(doc.secciones) == 2
    assert "Primera noticia" in doc.secciones[0].texto
    assert "Párrafo del cuerpo uno." in doc.secciones[0].texto
    assert doc.fecha_publicacion == "2026-07-30"
    assert doc.metadata.get("author") == "Redacción"


def test_csv_extractor_filas_independientes(archivo_csv, factory):
    doc = factory.create(archivo_csv).extract(archivo_csv)
    assert doc.formato == Formato.CSV
    assert len(doc.secciones) == 2
    assert all(not s.splittable for s in doc.secciones)  # unidades atómicas
    assert "ciudad: Bogotá" in doc.secciones[0].texto
    assert "poblacion: 7743955" in doc.secciones[0].texto


def test_extractor_archivo_inexistente(factory):
    with pytest.raises(ExtractorError, match="no existe"):
        factory.create(Path("doc.pdf")).extract(Path("no_existe.pdf"))
