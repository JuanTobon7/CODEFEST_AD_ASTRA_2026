"""
Tests de ``build_result_object`` (src/queries/resultado.py): construcción y
validación del objeto de resultado según el esquema de la Sección 9.3.1.
"""

from __future__ import annotations

import pytest

from src.queries.resultado import build_result_object


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


def test_resultado_valido():
    """Un output válido produce el esquema exacto (ranks 1..3 y 1..10)."""
    resultado = build_result_object("q001", _salida_valida())
    assert resultado["query_id"] == "q001"
    assert [d["rank"] for d in resultado["documents"]] == [1, 2, 3]
    assert [d["doc_id"] for d in resultado["documents"]] == ["DOC-001", "DOC-002", "DOC-003"]
    assert [f["rank"] for f in resultado["fragments"]] == list(range(1, 11))
    assert "chunk_id" in resultado["fragments"][0] and "text" in resultado["fragments"][0]


def test_resultado_query_id_invalido():
    with pytest.raises(ValueError, match="qNNN"):
        build_result_object("consulta-1", _salida_valida())


def test_resultado_documents_incompletos():
    salida = _salida_valida()
    salida["documents"] = ["DOC-001", "DOC-002"]
    with pytest.raises(ValueError, match="exactamente 3"):
        build_result_object("q001", salida)


def test_resultado_fragments_incompletos():
    salida = _salida_valida()
    salida["fragments"] = salida["fragments"][:9]
    with pytest.raises(ValueError, match="exactamente 10"):
        build_result_object("q001", salida)


def test_resultado_fragmento_muy_largo():
    """Un fragmento de más de 250 palabras debe haberse dividido antes."""
    salida = _salida_valida()
    salida["fragments"][0]["text"] = " ".join(["palabra"] * 251)
    with pytest.raises(ValueError, match="250"):
        build_result_object("q001", salida)


def test_resultado_campos_faltantes():
    salida = _salida_valida()
    del salida["fragments"][0]["text"]
    with pytest.raises(ValueError, match="sin text"):
        build_result_object("q001", salida)

    salida = _salida_valida()
    del salida["fragments"][0]["chunk_id"]
    with pytest.raises(ValueError, match="sin chunk_id"):
        build_result_object("q001", salida)


def test_resultado_doc_id_vacio():
    salida = _salida_valida()
    salida["documents"][1] = "   "
    with pytest.raises(ValueError, match="doc_id vacío"):
        build_result_object("q001", salida)
