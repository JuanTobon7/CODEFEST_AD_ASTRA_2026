"""
Tests de ``QueryLoader`` (src/queries/loader.py): carga desde ``.jsonl``/
``.csv`` con validación estricta (50 consultas, q001..q050) y exportación
desde el PDF oficial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.queries.loader import QueryLoader
from src.queries.models import Query


def _escribir_jsonl(path: Path, n: int = 50) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            f.write(
                json.dumps(
                    {"query_id": f"q{i:03d}", "query_text": f"¿Consulta {i}?"},
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_cargar_desde_jsonl(tmp_path):
    ruta = tmp_path / "consultas.jsonl"
    _escribir_jsonl(ruta)
    consultas = QueryLoader.cargar(str(ruta))
    assert len(consultas) == 50
    assert consultas[0] == Query(query_id="q001", query_text="¿Consulta 1?")
    assert [c.query_id for c in consultas] == [f"q{i:03d}" for i in range(1, 51)]


def test_cargar_desde_csv(tmp_path):
    ruta = tmp_path / "consultas.csv"
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write("query_id,query_text\n")
        for i in range(1, 51):
            f.write(f"q{i:03d},¿Consulta {i}?\n")
    consultas = QueryLoader.cargar(str(ruta))
    assert len(consultas) == 50


def test_cargar_conteo_incorrecto(tmp_path):
    ruta = tmp_path / "consultas.jsonl"
    _escribir_jsonl(ruta, n=49)
    with pytest.raises(ValueError, match="exactamente 50"):
        QueryLoader.cargar(str(ruta))


def test_cargar_id_duplicado(tmp_path):
    ruta = tmp_path / "consultas.jsonl"
    _escribir_jsonl(ruta)
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "q001", "query_text": "¿Duplicada?"}, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="duplicado"):
        QueryLoader.cargar(str(ruta))


def test_cargar_patron_invalido(tmp_path):
    ruta = tmp_path / "consultas.jsonl"
    _escribir_jsonl(ruta)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query_id": "consulta-1", "query_text": "¿X?"}, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="qNNN"):
        QueryLoader.cargar(str(ruta))


def test_cargar_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        QueryLoader.cargar(str(tmp_path / "no_existe.jsonl"))


def test_exportar_desde_pdf(tmp_path, monkeypatch):
    """Exporta desde el PDF y escribe el JSONL en el formato de carga."""
    consultas_fake = [Query(query_id=f"q{i:03d}", query_text=f"¿P{i}?") for i in range(1, 51)]
    monkeypatch.setattr(
        "src.queries.loader.QueryExtractor.extraer_desde_pdf",
        lambda _self, _pdf: consultas_fake,
    )

    salida = tmp_path / "consultas.jsonl"
    devueltas = QueryLoader.exportar_desde_pdf("Extracto_Preguntas_50_v2.pdf", str(salida))

    assert devueltas == consultas_fake
    lineas = [l for l in salida.read_text(encoding="utf-8").split("\n") if l]
    assert len(lineas) == 50
    primero = json.loads(lineas[0])
    assert primero == {"query_id": "q001", "query_text": "¿P1?"}

    # Idempotencia: lo escrito es legible de nuevo por cargar().
    recargadas = QueryLoader.cargar(str(salida))
    assert [c.query_id for c in recargadas] == [f"q{i:03d}" for i in range(1, 51)]
