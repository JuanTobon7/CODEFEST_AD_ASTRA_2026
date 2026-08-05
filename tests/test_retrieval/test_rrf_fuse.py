"""
Tests de la fusión multi-encoder con Reciprocal Rank Fusion (RRF) con
rankings sintéticos de 2-3 índices ficticios (sin FAISS ni modelos).

Verifica que el orden de fusión sea exactamente el esperado por
s_RRF(c) = sum_j 1/(k0 + r_j(c)), que los fragmentos ausentes de un índice
no aporten término, y que los desempates sean deterministas.
"""

from __future__ import annotations

import pytest

from src.retrieval.models import SearchHit
from src.retrieval.rrf import RRFError, rrf_fuse


def _hit(chunk_id, doc_id, rank, encoder, score=0.9):
    """Helper: un :class:`SearchHit` de un índice ficticio."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        encoder_name=encoder,
        rank=rank,
        score=score,
        metadata={"text": f"texto de {chunk_id}", "doc_id": doc_id},
    )


def test_rrf_fuse_orden_esperado_con_tres_indices():
    """Escenario del reto: 3 índices, solapamientos parciales.

    Índice A: c1(1), c2(2), c3(3)
    Índice B: c2(1), c4(2)
    Índice C: c1(1), c4(2)          (c2 no aparece aquí)

    Con k0=60:
      s(c1) = 1/61 + 1/61          = 2/61   ≈ 0.032787
      s(c2) = 1/62 + 1/61          ≈ 0.032522
      s(c4) = 1/62 + 1/62          = 2/62   ≈ 0.032258
      s(c3) = 1/63                  ≈ 0.015873

    Orden esperado: c1 > c2 > c4 > c3
    """
    ranking_a = [
        _hit("c1", "d1", rank=1, encoder="A", score=0.95),
        _hit("c2", "d1", rank=2, encoder="A", score=0.80),
        _hit("c3", "d2", rank=3, encoder="A", score=0.70),
    ]
    ranking_b = [
        _hit("c2", "d1", rank=1, encoder="B", score=0.90),
        _hit("c4", "d2", rank=2, encoder="B", score=0.75),
    ]
    ranking_c = [
        _hit("c1", "d1", rank=1, encoder="C", score=0.92),
        _hit("c4", "d2", rank=2, encoder="C", score=0.72),
    ]

    fusionados = rrf_fuse([ranking_a, ranking_b, ranking_c], k0=60)
    orden = [f.chunk_id for f in fusionados]

    assert orden == ["c1", "c2", "c4", "c3"]

    c1, c2, c4, c3 = fusionados
    assert c1.rrf_score == pytest.approx(2 / 61)
    assert c2.rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert c4.rrf_score == pytest.approx(2 / 62)
    assert c3.rrf_score == pytest.approx(1 / 63)

    # El fragmento c1 conserva la mejor similitud coseno entre A y C.
    assert c1.cosine_score == pytest.approx(0.95)
    assert c1.encoders == ["A", "C"]
    assert c1.doc_id == "d1"
    # El texto sale de la metadata del primer hit que lo vio.
    assert c1.text == "texto de c1"


def test_rrf_fuse_fragmento_solo_en_un_indice():
    """Un fragmento que aparece en un único índice aporta un solo término."""
    ranking_a = [_hit("x1", "d1", rank=1, encoder="A")]
    ranking_b = [_hit("y1", "d2", rank=1, encoder="B")]

    fusionados = rrf_fuse([ranking_a, ranking_b], k0=60)
    por_id = {f.chunk_id: f for f in fusionados}

    # x1 solo recibe el término del índice A: no se le asigna rango
    # artificial en B.
    assert por_id["x1"].rrf_score == pytest.approx(1 / 61)
    assert por_id["x1"].encoders == ["A"]
    assert por_id["y1"].rrf_score == pytest.approx(1 / 61)
    assert por_id["y1"].encoders == ["B"]


def test_rrf_fuse_empate_se_desempata_por_coseno_y_chunk_id():
    """RRF con puntuaciones iguales: orden determinista (coseno, luego id)."""
    ranking = [
        _hit("b", "d2", rank=1, encoder="A", score=0.50),
        _hit("a", "d1", rank=1, encoder="A", score=0.80),
    ]
    fusionados = rrf_fuse([ranking], k0=60)
    # Ambos tienen rrf_score = 1/61; gana el de mayor coseno ("a").
    assert [f.chunk_id for f in fusionados] == ["a", "b"]


def test_rrf_fuse_k0_invalido():
    with pytest.raises(RRFError):
        rrf_fuse([[]], k0=0)
    with pytest.raises(RRFError):
        rrf_fuse([[]], k0=-5)


def test_rrf_fuse_sin_rankings_devuelve_vacio():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
