"""
Tests del extractor de consultas: parseo puro de los marcadores qNNN y
validación de las 50 consultas (q001..q050 sin huecos ni duplicados).
"""

from __future__ import annotations

import pytest

from src.queries.extractor import QueryExtractor
from src.queries.models import Query


def _texto_50_consultas() -> str:
    """Texto sintético con 50 consultas; algunas ocupan varias líneas y una
    'continúa' en el párrafo siguiente (como q043 en el PDF real)."""
    parrafos: list[str] = []
    for i in range(1, 51):
        qid = f"q{i:03d}"
        if i == 8:  # pregunta multilínea (salto de línea dentro del párrafo)
            parrafos.append(
                f"{qid} ¿Cómo están empleando las fuerzas militares el análisis\n"
                "de inteligencia asistido por IA en operaciones?"
            )
        elif i == 43:  # la pregunta continúa en el párrafo siguiente
            parrafos.append(f"{qid} ¿De qué manera el narcotráfico financia el fortalecimiento")
        elif i == 44:
            parrafos.append("Residuales (GAOR) y GDO en departamentos clave?")
            parrafos.append(f"{qid} ¿Pregunta sintética número {i}?")
        else:
            parrafos.append(f"{qid} ¿Pregunta sintética número {i}?")
    return "\n\n".join(parrafos)


def test_extrae_50_consultas_en_orden():
    """Devuelve exactamente 50 consultas en orden estricto q001..q050."""
    consultas = QueryExtractor.extraer_consultas(_texto_50_consultas())
    assert len(consultas) == 50
    assert [c.query_id for c in consultas] == [f"q{i:03d}" for i in range(1, 51)]


def test_texto_multilinea_se_normaliza():
    """Las preguntas multilínea se unen con un espacio (sin saltos internos)."""
    consultas = QueryExtractor.extraer_consultas(_texto_50_consultas())
    q008 = next(c for c in consultas if c.query_id == "q008")
    assert "análisis de inteligencia asistido por IA" in q008.query_text
    assert "\n" not in q008.query_text


def test_continuacion_tras_salto_de_pagina():
    """q043 captura el texto que continúa después del salto de página."""
    consultas = QueryExtractor.extraer_consultas(_texto_50_consultas())
    q043 = next(c for c in consultas if c.query_id == "q043")
    assert "Residuales (GAOR) y GDO" in q043.query_text


def test_sin_marcadores_lanza_error():
    """Texto sin ningún marcador qNNN es un error claro."""
    with pytest.raises(ValueError, match="ningún marcador"):
        QueryExtractor.extraer_consultas("Un texto sin consultas.")


def test_conteo_incorrecto_lanza_error():
    """Menos de 50 consultas es un error (nunca se aceptan huecos)."""
    texto = "".join(f"q{i:03d} ¿Pregunta {i}?\n" for i in range(1, 49))
    with pytest.raises(ValueError, match="exactamente 50"):
        QueryExtractor.extraer_consultas(texto)


def test_ids_faltantes_lanzan_error():
    """Un hueco en el rango q001..q050 se detecta y reporta.

    Se usan 50 marcadores (uno fuera de rango, q099) para que el conteo sea
    correcto y la validación llegue a detectar el ID faltante (q017).
    """
    ids = [i for i in range(1, 51) if i != 17] + [99]
    texto = "".join(f"q{i:03d} ¿Pregunta {i}?\n" for i in ids)
    with pytest.raises(ValueError, match="q017"):
        QueryExtractor.extraer_consultas(texto)


def test_id_duplicado_lanza_error():
    """Un marcador repetido es un error (los IDs deben ser únicos)."""
    texto = "".join(f"q{i:03d} ¿Pregunta {i}?\n" for i in range(1, 51))
    texto += "q001 ¿Otra pregunta duplicada?\n"
    with pytest.raises(ValueError, match="duplicado"):
        QueryExtractor.extraer_consultas(texto)


def test_query_model_valida_patron_qnnn():
    """El modelo Query rechaza IDs que no sigan el patrón exacto qNNN."""
    with pytest.raises(ValueError, match="qNNN"):
        Query(query_id="q1", query_text="¿Consulta?")
    with pytest.raises(ValueError, match="qNNN"):
        Query(query_id="consulta1", query_text="¿Consulta?")
    assert Query(query_id="q001", query_text="¿Consulta?").query_id == "q001"
