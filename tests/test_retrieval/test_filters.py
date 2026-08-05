"""
Tests de los post-filtros (Sección 8.7): filtro por metadata (fenomeno,
formato, idioma, rango de fechas) y filtro por vector (umbral de coseno).
"""

from __future__ import annotations

from datetime import date

from src.retrieval.filters import apply_filters
from src.retrieval.models import FusedFragment, RetrievalFilters


def _fragmento(chunk_id, doc_id, cosine, **meta_extra):
    meta = {
        "fenomeno": 1,
        "formato": "md",
        "idioma": "es",
        "fecha_publicacion": "2024-03-15",
    }
    meta.update(meta_extra)
    return FusedFragment(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"texto de {chunk_id}",
        rrf_score=0.5,
        cosine_score=cosine,
        encoders=["A"],
        metadata=meta,
    )


def test_filtro_theta_descarta_baja_similitud():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.85),
        _fragmento("c2", "d1", cosine=0.40),
    ]
    filtrados = apply_filters(fragmentos, RetrievalFilters(theta=0.5))
    assert [f.chunk_id for f in filtrados] == ["c1"]


def test_filtro_fenomeno():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.9, fenomeno=1),
        _fragmento("c2", "d1", cosine=0.9, fenomeno=2),
    ]
    filtrados = apply_filters(fragmentos, RetrievalFilters(fenomeno=2))
    assert [f.chunk_id for f in filtrados] == ["c2"]


def test_filtro_formato():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.9, formato="md"),
        _fragmento("c2", "d1", cosine=0.9, formato="pdf"),
    ]
    filtrados = apply_filters(fragmentos, RetrievalFilters(formato="pdf"))
    assert [f.chunk_id for f in filtrados] == ["c2"]


def test_filtro_idioma():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.9, idioma="es"),
        _fragmento("c2", "d1", cosine=0.9, idioma="en"),
        _fragmento("c3", "d2", cosine=0.9),  # sin idioma: no se descarta
    ]
    filtrados = apply_filters(fragmentos, RetrievalFilters(idioma="es"))
    assert {f.chunk_id for f in filtrados} == {"c1", "c3"}


def test_filtro_rango_de_fechas():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.9, fecha_publicacion="2024-01-01"),
        _fragmento("c2", "d1", cosine=0.9, fecha_publicacion="2024-06-01"),
        _fragmento("c3", "d2", cosine=0.9, fecha_publicacion="2025-01-01"),
        _fragmento("c4", "d2", cosine=0.9),  # sin fecha: no se descarta
    ]
    filtrados = apply_filters(
        fragmentos,
        RetrievalFilters(date_range=(date(2024, 1, 1), date(2024, 12, 31))),
    )
    assert {f.chunk_id for f in filtrados} == {"c1", "c2", "c4"}


def test_filtros_combinados_theta_y_metadata():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.95, fenomeno=1),
        _fragmento("c2", "d1", cosine=0.30, fenomeno=1),  # theta lo descarta
        _fragmento("c3", "d2", cosine=0.95, fenomeno=3),  # fenomeno lo descarta
    ]
    filtrados = apply_filters(fragmentos, RetrievalFilters(fenomeno=1, theta=0.5))
    assert [f.chunk_id for f in filtrados] == ["c1"]


def test_sin_filtros_devuelve_copia_sin_cambios():
    fragmentos = [
        _fragmento("c1", "d1", cosine=0.2),
        _fragmento("c2", "d1", cosine=-0.1),
    ]
    filtrados = apply_filters(fragmentos, None)
    assert len(filtrados) == 2
    assert filtrados[0] is fragmentos[0]  # misma referencia: no muta
