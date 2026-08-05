"""
Tests del split/merge a nivel de chunk (Sección 9.2.1): división de
fragmentos > max_words respetando oraciones completas, y fusión de
fragmentos cortos con su siguiente chunk del mismo documento.
"""

from __future__ import annotations

import re

from src.retrieval.chunk_ops import contar_palabras, split_or_merge_fragments
from src.retrieval.models import FusedFragment

_REGEX_ORACIONES = re.compile(r"(?<=[.!?])\s+")

MAX_WORDS = 250


def _fragmento(chunk_id, doc_id, texto, posicion=0, rrf=1.0):
    return FusedFragment(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=texto,
        rrf_score=rrf,
        cosine_score=0.9,
        encoders=["A"],
        metadata={"posicion": posicion, "doc_id": doc_id},
    )


def _oraciones_enteras(texto):
    """Todas las oraciones completas que componen ``texto``."""
    return [o for o in _REGEX_ORACIONES.split(texto) if o.strip()]


def test_fragmento_corto_se_concatena_con_siguiente_chunk():
    frag = _fragmento("c1", "d1", "Palabra " * 50, posicion=0)  # 50 palabras

    def siguiente_chunk(doc_id, posicion):
        assert (doc_id, posicion) == ("d1", 0)
        return "Palabra " * 100  # siguiente chunk: 100 palabras -> 150 totales

    resultado = split_or_merge_fragments([frag], max_words=MAX_WORDS, siguiente_chunk=siguiente_chunk)
    assert len(resultado) == 1
    assert resultado[0].chunk_id == "c1"
    assert resultado[0].doc_id == "d1"
    assert contar_palabras(resultado[0].text) == 150
    assert resultado[0].merged_with is not None
    assert resultado[0].score == frag.rrf_score  # hereda el score RRF


def test_fragmento_corto_no_se_fusiona_si_supera_max_words():
    frag = _fragmento("c1", "d1", "Palabra " * 200, posicion=0)  # 200 palabras

    def siguiente_chunk(doc_id, posicion):
        return "Palabra " * 100  # 300 totales > 250: no fusionar

    resultado = split_or_merge_fragments([frag], max_words=MAX_WORDS, siguiente_chunk=siguiente_chunk)
    assert len(resultado) == 1
    assert resultado[0].merged_with is None
    assert contar_palabras(resultado[0].text) == 200


def test_fragmento_corto_sin_siguiente_chunk_queda_igual():
    frag = _fragmento("c1", "d1", "Palabra " * 10, posicion=0)
    resultado = split_or_merge_fragments([frag], max_words=MAX_WORDS, siguiente_chunk=lambda d, p: None)
    assert len(resultado) == 1
    assert resultado[0].text == frag.text
    assert resultado[0].merged_with is None


def test_fragmento_largo_se_dividide_en_oraciones_completas():
    # 20 oraciones de 15 palabras = 300 palabras > 250.
    oraciones = [" ".join(f"palabra{i}_{j}" for j in range(15)) + "." for i in range(20)]
    texto = " ".join(oraciones)
    frag = _fragmento("c1", "d1", texto, posicion=3)

    resultado = split_or_merge_fragments([frag], max_words=MAX_WORDS, siguiente_chunk=lambda d, p: None)

    assert len(resultado) > 1
    # Todos los sub-fragmentos conservan el mismo chunk_id y el score del padre.
    assert all(s.chunk_id == "c1" for s in resultado)
    assert all(s.sub_indice > 0 for s in resultado)
    assert all(s.score == frag.rrf_score for s in resultado)
    # Ningún sub-fragmento excede el tope...
    assert all(contar_palabras(s.text) <= MAX_WORDS for s in resultado)
    # ...y cada sub-fragmento es una concatenación exacta de oraciones enteras:
    # el conjunto de oraciones reconstruido coincide con el original.
    reconstruidas = []
    for sub in resultado:
        reconstruidas.extend(_oraciones_enteras(sub.text))
    assert reconstruidas == oraciones


def test_split_con_splitter_inyectado():
    """El splitter inyectable (p. ej. SentenceSplitter) se usa si se pasa."""
    llamadas = []

    def splitter(texto):
        llamadas.append(texto)
        return _REGEX_ORACIONES.split(texto)

    oraciones = [" ".join(f"p{i}_{j}" for j in range(60)) + "." for i in range(10)]  # 610 palabras
    frag = _fragmento("c1", "d1", " ".join(oraciones), posicion=0)

    resultado = split_or_merge_fragments([frag], max_words=MAX_WORDS, splitter=splitter)
    assert llamadas, "el splitter inyectado debe usarse"
    assert all(contar_palabras(s.text) <= MAX_WORDS for s in resultado)


def test_sin_fragmentos_devuelve_vacio():
    assert split_or_merge_fragments([], max_words=MAX_WORDS) == []
