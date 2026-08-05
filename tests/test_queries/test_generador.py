"""
Tests del orquestador ``generador.py``: generación E2E de ``resultados.jsonl``
con ``retrieve()`` sustituido por un doble sintético (sin FAISS, modelos ni
Mongo) y el auto-export de consultas desde el PDF cuando falta el archivo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import generador
from src.queries.loader import QueryLoader
from src.queries.models import Query
from src.queries.resultado import build_result_object


def _escribir_consultas(path: Path, n: int = 50) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            f.write(
                json.dumps(
                    {"query_id": f"q{i:03d}", "query_text": f"¿Consulta {i}?"},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _salida_valida() -> dict:
    """Salida de ``retrieve()`` que cumple el esquema (3 docs, 10 fragmentos)."""
    return {
        "documents": ["DOC-001", "DOC-002", "DOC-003"],
        "fragments": [
            {
                "chunk_id": f"chunk-{i:03d}",
                "doc_id": f"DOC-{((i - 1) % 3) + 1:03d}",
                "text": f"Texto del fragmento {i}",
            }
            for i in range(1, 11)
        ],
    }


def _fake_retrieve_valido(_texto: str, **_kwargs) -> dict:
    return _salida_valida()


def test_generate_results_escribe_50_lineas_validas(tmp_path, monkeypatch):
    consultas = tmp_path / "consultas.jsonl"
    _escribir_consultas(consultas)
    salida = tmp_path / "resultados.jsonl"

    monkeypatch.setattr(generador, "retrieve", _fake_retrieve_valido)
    generador.generate_results(str(consultas), str(salida))

    lineas = [l for l in salida.read_text(encoding="utf-8").split("\n") if l]
    assert len(lineas) == 50
    for numero, linea in enumerate(lineas, start=1):
        obj = json.loads(linea)
        assert obj["query_id"] == f"q{numero:03d}"
        assert len(obj["documents"]) == 3
        assert len(obj["fragments"]) == 10
        # La línea es exactamente lo que valida build_result_object.
        assert build_result_object(obj["query_id"], _salida_valida()) == obj


def test_generate_results_consulta_fallida_se_omite(tmp_path, monkeypatch):
    """Una consulta que falla se loguea y no impide escribir las demás."""
    consultas = tmp_path / "consultas.jsonl"
    _escribir_consultas(consultas)
    salida = tmp_path / "resultados.jsonl"

    def _fake_con_fallo(texto: str, **_kwargs) -> dict:
        if texto == "¿Consulta 2?":
            raise RuntimeError("boom sintético")
        return _salida_valida()

    monkeypatch.setattr(generador, "retrieve", _fake_con_fallo)
    generador.generate_results(str(consultas), str(salida))

    lineas = [l for l in salida.read_text(encoding="utf-8").split("\n") if l]
    assert len(lineas) == 49
    ids = [json.loads(l)["query_id"] for l in lineas]
    assert "q002" not in ids


def test_main_genera_consultas_desde_pdf_si_falta(tmp_path, monkeypatch):
    """Si --queries no existe, main() exporta desde el PDF y sigue."""
    salida = tmp_path / "resultados.jsonl"

    def _fake_exportar(_pdf: str, _output: str):
        with open(_output, "w", encoding="utf-8") as f:
            for i in range(1, 51):
                f.write(json.dumps({"query_id": f"q{i:03d}", "query_text": f"¿C{i}?"}, ensure_ascii=False) + "\n")

    monkeypatch.setattr(QueryLoader, "exportar_desde_pdf", staticmethod(_fake_exportar))
    monkeypatch.setattr(generador, "retrieve", _fake_retrieve_valido)

    codigo = generador.main(
        [
            "--queries", str(tmp_path / "consultas.jsonl"),
            "--output", str(salida),
        ]
    )

    assert codigo == 0
    assert (tmp_path / "consultas.jsonl").exists()
    lineas = [l for l in salida.read_text(encoding="utf-8").split("\n") if l]
    assert len(lineas) == 50
    assert json.loads(lineas[0])["query_id"] == "q001"


def test_main_error_si_no_hay_consultas_ni_pdf(tmp_path, monkeypatch):
    """Sin archivo de consultas ni PDF, main() aborta con código 1."""
    monkeypatch.setattr(generador, "retrieve", _fake_retrieve_valido)
    codigo = generador.main(
        [
            "--queries", str(tmp_path / "no_existe.jsonl"),
            "--queries-pdf", str(tmp_path / "no_existe.pdf"),
            "--output", str(tmp_path / "resultados.jsonl"),
        ]
    )
    assert codigo == 1
