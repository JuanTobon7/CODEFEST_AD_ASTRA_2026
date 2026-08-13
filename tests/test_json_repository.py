"""Pruebas del repositorio JSON de chunks."""

from __future__ import annotations

import gzip
import json
import shutil

import pytest

from src.models.chunk import Chunk
from src.persistence import json_repository
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


# -- metadata.json: metadata completa y escritura masiva ----------------------

def test_write_all_con_metadata_completa_conserva_los_campos_recomendados(tmp_path):
    """``metadata.json`` es la fuente de verdad: no puede perder los deseables."""
    ruta = tmp_path / "metadata.json"
    repositorio = JsonChunkRepository(ruta, solo_obligatorios=False)

    escritos = repositorio.write_all(iter([_chunk("c-1"), _chunk("c-2", posicion=1)]))

    assert escritos == 2
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert datos[0]["idioma"] == "es"
    assert datos[0]["hash_texto"] == "no-debe-guardarse-en-el-json"
    assert "validation_warnings" not in datos[0]
    assert set(Chunk.CAMPOS_METADATA_OBLIGATORIA) <= set(datos[0])


def test_write_all_reemplaza_el_contenido_previo(tmp_path):
    """No hace upsert: vuelca el corpus dado, sin leer lo que hubiera antes."""
    ruta = tmp_path / "metadata.json"
    repositorio = JsonChunkRepository(ruta)
    repositorio.write_all([_chunk("viejo-1"), _chunk("viejo-2", posicion=1)])

    repositorio.write_all([_chunk("nuevo-1")])

    assert [c.chunk_id for c in repositorio.find_all()] == ["nuevo-1"]


def test_lee_ambos_esquemas_de_registro(tmp_path):
    """La lectura acepta registros con y sin los campos recomendados."""
    ruta = tmp_path / "metadata.json"
    JsonChunkRepository(ruta, solo_obligatorios=False).write_all([_chunk("completo")])
    completo = JsonChunkRepository(ruta).find_all()

    JsonChunkRepository(ruta).write_all([_chunk("minimo")])
    minimo = JsonChunkRepository(ruta).find_all()

    assert completo[0].idioma == "es"
    assert minimo[0].idioma is None


# -- Parser incremental --------------------------------------------------------

@pytest.mark.parametrize(
    "contenido, esperado",
    [
        ("[]", []),
        ("[\n]\n", []),
        ("   ", []),
        ('[{"doc_id":"d","chunk_id":"c","fuente":"f","formato":"md","fenomeno":1,'
         '"posicion":0,"num_tokens":1,"texto":"t"}]', ["c"]),
    ],
)
def test_parser_incremental_casos_borde(tmp_path, contenido, esperado):
    ruta = tmp_path / "metadata.json"
    ruta.write_text(contenido, encoding="utf-8")

    assert [c.chunk_id for c in JsonChunkRepository(ruta).iter_chunks()] == esperado


def test_parser_incremental_con_registros_mayores_que_el_bloque(tmp_path, monkeypatch):
    """Un chunk más grande que la ventana de lectura se decodifica igual.

    Se reduce el bloque a 64 bytes para forzar el camino de ampliación de la
    ventana sin escribir un archivo enorme en el test.
    """
    monkeypatch.setattr(json_repository, "_TAMANO_BLOQUE", 64)
    monkeypatch.setattr(json_repository, "_MINIMO_RESIDUO", 16)
    ruta = tmp_path / "metadata.json"
    originales = [
        _chunk(f"c-{i}", posicion=i, texto="palabra " * 200) for i in range(5)
    ]
    JsonChunkRepository(ruta).write_all(originales)

    leidos = list(JsonChunkRepository(ruta).iter_chunks())

    assert [c.chunk_id for c in leidos] == [c.chunk_id for c in originales]
    assert leidos[0].texto == originales[0].texto


def test_lee_la_variante_gz_si_no_existe_el_json_plano(tmp_path):
    """Un clon del repo solo trae ``metadata.json.gz``: debe funcionar igual.

    La configuración apunta a ``metadata.json`` en ambos casos; el repositorio
    resuelve el respaldo comprimido sin que nadie tenga que descomprimir.
    """
    ruta = tmp_path / "metadata.json"
    JsonChunkRepository(ruta, solo_obligatorios=False).write_all(
        [_chunk("c-1"), _chunk("c-2", posicion=1)]
    )
    with ruta.open("rb") as plano, gzip.open(f"{ruta}.gz", "wb") as comprimido:
        shutil.copyfileobj(plano, comprimido)
    ruta.unlink()

    repositorio = JsonChunkRepository(ruta)

    assert [c.chunk_id for c in repositorio.find_all()] == ["c-1", "c-2"]
    assert repositorio.exists("c-2") is True
    assert repositorio.find_all()[0].idioma == "es"


def test_el_json_plano_gana_sobre_el_gz(tmp_path):
    """Si están los dos, manda el descomprimido (es el que se reescribe)."""
    ruta = tmp_path / "metadata.json"
    with gzip.open(f"{ruta}.gz", "wt", encoding="utf-8") as comprimido:
        comprimido.write('[{"doc_id":"d","chunk_id":"del-gz","fuente":"f","formato":"md",'
                         '"fenomeno":1,"posicion":0,"num_tokens":1,"texto":"t"}]')
    JsonChunkRepository(ruta).write_all([_chunk("del-plano")])

    assert [c.chunk_id for c in JsonChunkRepository(ruta).find_all()] == ["del-plano"]


def test_sin_json_ni_gz_no_hay_chunks(tmp_path):
    assert JsonChunkRepository(tmp_path / "metadata.json").find_all() == []


def test_parser_incremental_acepta_json_indentado(tmp_path):
    """El formato importa poco: se acepta cualquier lista JSON válida."""
    ruta = tmp_path / "metadata.json"
    registros = [_chunk("c-1").como_dict_json, _chunk("c-2", posicion=1).como_dict_json]
    ruta.write_text(json.dumps(registros, indent=4), encoding="utf-8")

    assert [c.chunk_id for c in JsonChunkRepository(ruta).iter_chunks()] == ["c-1", "c-2"]
