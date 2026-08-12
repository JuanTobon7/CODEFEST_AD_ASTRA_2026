"""Pruebas del repositorio JSON de chunks."""

from __future__ import annotations

import json

import pytest

from src.models.chunk import Chunk
from src.persistence.json_repository import JsonChunkRepository, JsonChunkRepositoryError


def _chunk(
    chunk_id: str,
    doc_id: str = "documento",
    posicion: int = 0,
    fenomeno: int = 1,
    texto: str | None = None,
) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        fuente="origen.md",
        formato="md",
        fenomeno=fenomeno,
        posicion=posicion,
        num_tokens=4,
        texto=texto or f"Texto de {chunk_id}.",
        idioma="es",
        hash_texto="no-debe-guardarse-en-el-json",
    )


def test_guarda_una_lista_json_con_solo_metadata_obligatoria_y_orden_estable(tmp_path):
    ruta = tmp_path / "salida" / "chunks.json"
    repositorio = JsonChunkRepository(ruta)

    repositorio.save_many([_chunk("c-2", posicion=1), _chunk("c-1", posicion=0)])

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert isinstance(datos, list)
    assert [registro["chunk_id"] for registro in datos] == ["c-1", "c-2"]
    assert set(datos[0]) == set(Chunk.CAMPOS_METADATA_OBLIGATORIA)
    assert datos[0] == {
        "doc_id": "documento",
        "chunk_id": "c-1",
        "fuente": "origen.md",
        "formato": "md",
        "fenomeno": 1,
        "posicion": 0,
        "num_tokens": 4,
        "texto": "Texto de c-1.",
    }


def test_upsert_por_chunk_id_no_duplica_y_conserva_las_consultas(tmp_path):
    repositorio = JsonChunkRepository(tmp_path / "chunks.json")
    repositorio.save_many([_chunk("c-1", posicion=0), _chunk("c-2", posicion=1)])
    repositorio.save_many([_chunk("c-1", posicion=0, texto="Texto actualizado.")])

    chunks = repositorio.find_by_doc_id("documento")
    assert [chunk.chunk_id for chunk in chunks] == ["c-1", "c-2"]
    assert chunks[0].texto == "Texto actualizado."
    assert repositorio.exists("c-1") is True
    assert repositorio.exists("inexistente") is False


def test_lote_balanceado_reparte_cuotas_por_fenomeno(tmp_path):
    repositorio = JsonChunkRepository(tmp_path / "chunks.json")
    chunks = [
        _chunk(f"f{fenomeno}-{posicion}", f"F{fenomeno}/fuente/doc.md", posicion, fenomeno)
        for fenomeno in (1, 2, 3)
        for posicion in range(3)
    ]
    repositorio.save_many(chunks)

    seleccionados = repositorio.find_all_balanceado(6)
    assert {fenomeno: sum(c.fenomeno == fenomeno for c in seleccionados) for fenomeno in (1, 2, 3)} == {
        1: 2,
        2: 2,
        3: 2,
    }


def test_rechaza_un_archivo_que_no_es_una_lista_json(tmp_path):
    ruta = tmp_path / "chunks.json"
    ruta.write_text('{"chunks": []}', encoding="utf-8")

    with pytest.raises(JsonChunkRepositoryError, match="lista"):
        JsonChunkRepository(ruta).find_all()
