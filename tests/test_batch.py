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
        IngestionResult(doc_id="a", fuente="a.md", status="ok", num_guardados=3, fenomeno=1),
        IngestionResult(doc_id="b", fuente="b.json", status="ok", num_guardados=1, fenomeno=2),
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


def test_batch_summary_desglosa_archivos_y_chunks_por_fenomeno():
    """La segregación F1/F2/F3 queda visible en el resumen del lote."""
    resultados = [
        IngestionResult(doc_id="a", fuente="a.md", status="ok", num_guardados=3, fenomeno=1),
        IngestionResult(doc_id="b", fuente="b.json", status="ok", num_guardados=2, fenomeno=1),
        IngestionResult(doc_id="c", fuente="c.pdf", status="ok", num_guardados=5, fenomeno=2),
        IngestionResult(doc_id="d", fuente="d.xlsx", status="ok", num_guardados=7, fenomeno=3),
        IngestionResult(doc_id="e", fuente="e.csv", status="error", errores=["fallo"]),
    ]
    resumen = BatchSummary.from_results(resultados)
    assert resumen.archivos_por_fenomeno == {1: 2, 2: 1, 3: 1}
    assert resumen.chunks_por_fenomeno == {1: 5, 2: 5, 3: 7}
    # Los resultados sin fenómeno (p. ej. errores) no rompen el desglose.
    assert resumen.archivos_por_fenomeno.get(1) == 2


def test_repartir_por_fenomeno_no_deja_fuera_a_f2_f3_con_limite():
    """Un límite debe repartirse entre los fenómenos, no tomar solo los primeros
    (que por orden alfabético serían casi todos de F1)."""
    from src.pipeline.batch_ingestor import BatchIngestor

    archivos = [Path(f"F{p}_carpeta/archivo_{i}.pdf") for p in (1, 2, 3) for i in range(10)]
    # Resolver por número de fenómeno del prefijo F{n}_ del padre.
    seleccionados = BatchIngestor._repartir_por_fenomeno(
        archivos, 6, lambda p: int(p.parent.name[1])
    )
    assert len(seleccionados) == 6
    fenomenos = [int(p.parent.name[1]) for p in seleccionados]
    assert fenomenos.count(1) == 2
    assert fenomenos.count(2) == 2
    assert fenomenos.count(3) == 2


def test_repartir_por_fenomeno_sin_limite_devuelve_todo(tmp_path: Path):
    from src.pipeline.batch_ingestor import BatchIngestor

    archivos = [Path(f"F{p}_carpeta/archivo_{i}.pdf") for p in (1, 2) for i in range(5)]
    assert BatchIngestor._repartir_por_fenomeno(archivos, 0, lambda p: 1) == archivos
    assert BatchIngestor._repartir_por_fenomeno(archivos, 99, lambda p: 1) == archivos
