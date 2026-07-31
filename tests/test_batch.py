"""
Tests del CorpusService y BatchSummary (lógica refactorizada de run_ingestion).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.batch_summary import BatchSummary
from src.models.pipeline_result import IngestionResult
from src.pipeline.corpus_service import CorpusService

MAAPA = {"fenomeno_1": 1, "fenomeno_2": 2, "sismo": 1, "inunda": 2}


def _crear_corpus_anidado(tmp_path: Path) -> Path:
    """Corpus con carpetas dentro de carpetas."""
    corpus = tmp_path / "corpus"
    (corpus / "fenomeno_1" / "informe_tecnico").mkdir(parents=True)
    (corpus / "fenomeno_2" / "noticias").mkdir(parents=True)
    (corpus / "sueltos").mkdir(parents=True)
    (corpus / "fenomeno_1" / "informe_tecnico" / "sismo_andino.md").write_text(
        "# Sismo\n\nTexto de ejemplo.", encoding="utf-8"
    )
    (corpus / "fenomeno_2" / "noticias" / "inundaciones.json").write_text(
        "[]", encoding="utf-8"
    )
    (corpus / "sueltos" / "reporte_inundaciones.pdf").write_bytes(b"%PDF-1.4 fake")
    (corpus / "sueltos" / "README.txt").write_text("Solo una línea.", encoding="utf-8")
    return corpus


def test_scan_recorre_carpetas_anidadas(tmp_path: Path):
    """El escaneo encuentra archivos a cualquier profundidad."""
    corpus = _crear_corpus_anidado(tmp_path)
    servicio = CorpusService(corpus)
    archivos = servicio.scan()
    assert {a.name for a in archivos} == {
        "sismo_andino.md",
        "inundaciones.json",
        "reporte_inundaciones.pdf",
        "README.txt",
    }


def test_scan_filtra_por_extension(tmp_path: Path):
    corpus = _crear_corpus_anidado(tmp_path)
    servicio = CorpusService(corpus)
    archivos = servicio.scan(extensiones=["md", "json"])
    assert {a.name for a in archivos} == {"sismo_andino.md", "inundaciones.json"}


def test_determine_fenomeno_por_carpeta_ancestral(tmp_path: Path):
    """La carpeta ancestral (no la inmediata) asigna el fenómeno."""
    corpus = _crear_corpus_anidado(tmp_path)
    servicio = CorpusService(corpus, MAAPA)
    archivo = corpus / "fenomeno_2" / "noticias" / "inundaciones.json"
    assert servicio.determine_fenomeno(archivo) == 2


def test_determine_fenomeno_por_patron_en_nombre(tmp_path: Path):
    """El patrón en el nombre del archivo también asigna fenómeno."""
    corpus = _crear_corpus_anidado(tmp_path)
    servicio = CorpusService(corpus, MAAPA)
    archivo = corpus / "sueltos" / "reporte_inundaciones.pdf"
    assert servicio.determine_fenomeno(archivo) == 2


def test_determine_fenomeno_por_defecto(tmp_path: Path):
    corpus = _crear_corpus_anidado(tmp_path)
    servicio = CorpusService(corpus, MAAPA)
    archivo = corpus / "sueltos" / "README.txt"
    assert servicio.determine_fenomeno(archivo) == 1


def test_load_fenomenos_map_desde_json(tmp_path: Path):
    ruta = tmp_path / "fenomenos.json"
    ruta.write_text('{"fenomeno_1": 1, "fenomeno_3": 3}', encoding="utf-8")
    assert CorpusService.load_fenomenos_map(ruta) == {"fenomeno_1": 1, "fenomeno_3": 3}
    # Archivo inexistente -> mapa vacío, sin excepción.
    assert CorpusService.load_fenomenos_map(tmp_path / "no_existe.json") == {}


def test_read_plain_text_docs_ignora_binarios(tmp_path: Path):
    corpus = _crear_corpus_anidado(tmp_path)
    servicio = CorpusService(corpus)
    stubs = servicio.read_plain_text_docs()
    fuentes = {s.fuente for s in stubs}
    assert "sismo_andino.md" in fuentes
    assert "inundaciones.json" in fuentes
    assert "reporte_inundaciones.pdf" not in fuentes  # binario: se omite


def test_batch_summary_agrega_resultados():
    """BatchSummary agrega conteos y errores desde IngestionResult."""
    resultados = [
        IngestionResult(doc_id="a", fuente="a.md", status="ok", num_guardados=3),
        IngestionResult(doc_id="b", fuente="b.json", status="ok", num_guardados=1),
        IngestionResult(
            doc_id="c",
            fuente="c.pdf",
            status="error",
            errores=["PDF ilegible"],
        ),
    ]
    resumen = BatchSummary.from_results(resultados)
    assert resumen.total == 3
    assert resumen.ok == 2
    assert resumen.error == 1
    assert resumen.chunks_guardados == 4
    assert any("c.pdf" in e and "PDF ilegible" in e for e in resumen.errores)
