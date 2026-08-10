"""
Tests: ``DeliveryExporter`` (modo de exportación estricta) — valida chunks
faltantes, dimensión uniforme, reproducibilidad exacta (correr el export
dos veces sobre los mismos datos produce ``index.faiss``/``metadata.jsonl``
idénticos) y el modo parcial (``permitir_faltantes``: índice con los
embeddings disponibles).
"""

from __future__ import annotations

import json
from typing import Dict, Iterator, List

import faiss
import numpy as np
import pytest

from src.models.chunk import Chunk
from src.vectorstore.export_delivery import DeliveryExporter, ExportError
from src.vectorstore.flat_ip_strategy import FlatIPIndexStrategy
from src.vectorstore.index_builder_base import IndexBuildConfig
from src.vectorstore.models import EmbeddingRecord
from src.vectorstore.vector_repository import VectorRepository


class _FakeVectorRepository(VectorRepository):
    """Repositorio en memoria, suficiente para ejercitar ``DeliveryExporter``."""

    def __init__(self, registros: List[EmbeddingRecord]) -> None:
        self._registros = {r.chunk_id: r for r in registros}

    def save_many(self, records: List[EmbeddingRecord]) -> None:
        for r in records:
            self._registros[r.chunk_id] = r

    def find_by_encoder(self, encoder_name: str, batch_size: int = 500) -> Iterator[EmbeddingRecord]:
        yield from self._registros.values()

    def find_by_chunk_ids(self, encoder_name: str, chunk_ids: List[str]) -> List[EmbeddingRecord]:
        return [self._registros[c] for c in chunk_ids if c in self._registros]

    def find_missing(self, encoder_name: str, chunk_ids: List[str]) -> List[str]:
        return [c for c in chunk_ids if c not in self._registros]

    def count_by_encoder(self, encoder_name: str) -> int:
        return len(self._registros)

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        self._registros.pop(chunk_id, None)

    def set_faiss_internal_id(self, chunk_id: str, encoder_name: str, faiss_id: int) -> None:
        pass


def _chunk(chunk_id: str, doc_id: str, posicion: int) -> Chunk:
    return Chunk(
        doc_id=doc_id, chunk_id=chunk_id, fuente="test.md", formato="md",
        fenomeno=1, posicion=posicion, num_tokens=5, texto=f"texto de {chunk_id}",
    )


def _registro(chunk_id: str, dim: int = 4, valor: float = 1.0) -> EmbeddingRecord:
    vector = np.full(dim, valor, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)
    return EmbeddingRecord(
        chunk_id=chunk_id, doc_id="doc-1", fenomeno=1, formato="md",
        encoder_name="e5-fake", model_id="fake/e5", embedding_dim=dim, vector=vector,
    )


def test_export_falla_si_faltan_embeddings(tmp_path):
    chunks = [_chunk("c1", "doc-1", 0), _chunk("c2", "doc-1", 1)]
    repo = _FakeVectorRepository([_registro("c1")])  # falta c2
    exportador = DeliveryExporter(repo, tmp_path)

    with pytest.raises(ExportError):
        exportador.export_encoder("e5-fake", chunks, FlatIPIndexStrategy(), 4, IndexBuildConfig())


def test_export_falla_si_dimension_no_es_uniforme(tmp_path):
    chunks = [_chunk("c1", "doc-1", 0)]
    repo = _FakeVectorRepository([_registro("c1", dim=8)])
    exportador = DeliveryExporter(repo, tmp_path)

    with pytest.raises(ExportError):
        exportador.export_encoder("e5-fake", chunks, FlatIPIndexStrategy(), 4, IndexBuildConfig())


def test_export_genera_index_y_metadata_en_orden_doc_id_posicion(tmp_path):
    chunks = [_chunk("c2", "doc-1", 1), _chunk("c1", "doc-1", 0)]  # desordenados a propósito
    repo = _FakeVectorRepository([_registro("c1"), _registro("c2")])
    exportador = DeliveryExporter(repo, tmp_path)

    resultado = exportador.export_encoder("e5-fake", chunks, FlatIPIndexStrategy(), 4, IndexBuildConfig())

    assert resultado.n_vectores == 2
    lineas = (tmp_path / "encoder_e5-fake" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ids_en_orden = [json.loads(l)["chunk_id"] for l in lineas]
    assert ids_en_orden == ["c1", "c2"]  # orden (doc_id, posicion), no orden de inserción


def test_export_es_reproducible_entre_corridas(tmp_path):
    chunks = [_chunk("c1", "doc-1", 0), _chunk("c2", "doc-1", 1)]
    repo = _FakeVectorRepository([_registro("c1"), _registro("c2")])
    exportador = DeliveryExporter(repo, tmp_path)

    r1 = exportador.export_encoder("e5-fake", chunks, FlatIPIndexStrategy(), 4, IndexBuildConfig())
    r2 = exportador.export_encoder("e5-fake", chunks, FlatIPIndexStrategy(), 4, IndexBuildConfig())

    assert r1.checksum_sha256 == r2.checksum_sha256


def test_export_parcial_omite_chunks_sin_embedding(tmp_path):
    """Modo parcial: construye el índice solo con los embeddings disponibles."""
    chunks = [_chunk("c1", "doc-1", 0), _chunk("c2", "doc-1", 1), _chunk("c3", "doc-1", 2)]
    repo = _FakeVectorRepository([_registro("c1"), _registro("c3")])  # falta c2
    exportador = DeliveryExporter(repo, tmp_path)

    resultado = exportador.export_encoder(
        "e5-fake", chunks, FlatIPIndexStrategy(), 4, IndexBuildConfig(), permitir_faltantes=True
    )

    assert resultado.n_vectores == 2
    ruta_carpeta = tmp_path / "encoder_e5-fake"
    lineas = (ruta_carpeta / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(l)["chunk_id"] for l in lineas]
    assert ids == ["c1", "c3"]  # orden (doc_id, posicion), sin el omitido
    # El índice recién escrito cuadra con la metadata (ID interno = ordinal de línea).
    indice = faiss.read_index(str(ruta_carpeta / "index.faiss"))
    assert indice.ntotal == 2
    assert indice.d == 4
    # El build_log deja trazabilidad del modo parcial.
    entradas = [json.loads(l) for l in (ruta_carpeta / "build_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entradas[-1]["parcial"] is True
    assert entradas[-1]["chunks_omitidos"] == 1
