"""
Tests del ChunkValidator (validaciones duras y blandas).
"""

from __future__ import annotations

import pytest

from src.models.chunk import Chunk
from src.validation.chunk_validator import ChunkValidator


def _chunk(posicion: int = 0, **overrides) -> Chunk:
    """Chunk válido por defecto, modificable por ``overrides``."""
    base = dict(
        doc_id="doc_1",
        chunk_id=f"doc_1__chunk_{posicion:05d}",
        fuente="prueba.md",
        formato="md",
        fenomeno=1,
        posicion=posicion,
        num_tokens=10,
        texto="Oración completa de prueba.",
    )
    base.update(overrides)
    return Chunk(**base)


def _chunk_malformado(**overrides) -> Chunk:
    """Chunk construido sin validación de pydantic (para probar al validador)."""
    base = dict(
        doc_id="doc_1",
        chunk_id="doc_1__chunk_00000",
        fuente="prueba.md",
        formato="md",
        fenomeno=1,
        posicion=0,
        num_tokens=10,
        texto="Oración completa de prueba.",
    )
    base.update(overrides)
    return Chunk.model_construct(**base)


def test_chunk_valido_pasa_sin_rechazo():
    resultado = ChunkValidator().validate([_chunk(0), _chunk(1)])
    assert len(resultado.validos) == 2
    assert resultado.rechazados == []
    assert resultado.validos[0].validation_warnings == []


def test_campos_obligatorios_ausentes_se_rechazan():
    sin_fuente = _chunk_malformado(fuente="")
    sin_texto = _chunk_malformado(texto="")
    resultado = ChunkValidator().validate([sin_fuente, sin_texto])
    assert len(resultado.rechazados) == 2
    assert any("fuente" in m for m in resultado.motivos)
    assert any("texto" in m for m in resultado.motivos)


def test_tipo_incorrecto_se_rechaza():
    mal_fenomeno = _chunk_malformado(fenomeno="uno")  # str en lugar de int
    resultado = ChunkValidator().validate([mal_fenomeno])
    assert len(resultado.rechazados) == 1
    assert "tipo incorrecto" in resultado.motivos[0]


def test_fenomeno_fuera_de_rango_se_rechaza():
    resultado = ChunkValidator().validate([_chunk_malformado(fenomeno=9)])
    assert len(resultado.rechazados) == 1


def test_num_tokens_sobre_limite_se_rechaza():
    resultado = ChunkValidator(max_tokens=512).validate([_chunk(0, num_tokens=513)])
    assert len(resultado.rechazados) == 1
    assert "supera el límite" in resultado.motivos[0]


def test_posicion_con_huecos_se_rechaza():
    resultado = ChunkValidator().validate([_chunk(0), _chunk(2)])  # falta la 1
    assert len(resultado.rechazados) == 1
    assert "fuera de secuencia" in resultado.motivos[0]


def test_chunk_id_duplicado_se_rechaza():
    repetido = _chunk(1, chunk_id="doc_1__chunk_00000")
    resultado = ChunkValidator().validate([_chunk(0), repetido])
    assert len(resultado.rechazados) == 1
    assert "duplicado" in resultado.motivos[0]


def test_texto_sin_puntuacion_terminal_es_warning():
    chunk = _chunk(0, texto="Oración cortada a la mitad sin punto")
    resultado = ChunkValidator().validate([chunk])
    assert len(resultado.validos) == 1  # no se rechaza
    assert any("puntuación terminal" in w for w in chunk.validation_warnings)


def test_filas_tabulares_sin_punto_no_generan_warning():
    fila = _chunk(0, formato="csv", texto="ciudad: Bogotá\npoblacion: 7743955")
    resultado = ChunkValidator().validate([fila])
    assert len(resultado.validos) == 1
    assert fila.validation_warnings == []
