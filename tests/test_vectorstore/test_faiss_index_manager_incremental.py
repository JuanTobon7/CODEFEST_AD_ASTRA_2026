"""
Tests: ``FaissIndexManager`` (modo operativo incremental) — agregar,
reemplazar (texto reprocesado) y persistir a disco sin duplicar vectores.
"""

from __future__ import annotations

from typing import Dict, Iterator, List

import numpy as np
import pytest

from src.vectorstore.faiss_index_manager import FaissIndexManager
from src.vectorstore.flat_ip_strategy import FlatIPIndexStrategy
from src.vectorstore.index_builder_base import IndexBuildConfig
from src.vectorstore.models import EmbeddingRecord
from src.vectorstore.vector_repository import VectorRepository


class _FakeVectorRepository(VectorRepository):
    """Repositorio en memoria: suficiente para ejercitar ``FaissIndexManager``."""

    def __init__(self) -> None:
        self._por_clave: Dict[tuple, EmbeddingRecord] = {}

    def guardar(self, registro: EmbeddingRecord) -> None:
        self._por_clave[(registro.chunk_id, registro.encoder_name)] = registro

    def save_many(self, records: List[EmbeddingRecord]) -> None:
        for r in records:
            self.guardar(r)

    def find_by_encoder(self, encoder_name: str, batch_size: int = 500) -> Iterator[EmbeddingRecord]:
        for (chunk_id, enc), registro in self._por_clave.items():
            if enc == encoder_name:
                yield registro

    def find_by_chunk_ids(self, encoder_name: str, chunk_ids: List[str]) -> List[EmbeddingRecord]:
        return [
            self._por_clave[(cid, encoder_name)]
            for cid in chunk_ids
            if (cid, encoder_name) in self._por_clave
        ]

    def find_missing(self, encoder_name: str, chunk_ids: List[str]) -> List[str]:
        return [cid for cid in chunk_ids if (cid, encoder_name) not in self._por_clave]

    def count_by_encoder(self, encoder_name: str) -> int:
        return sum(1 for (_, enc) in self._por_clave if enc == encoder_name)

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        for clave in [c for c in self._por_clave if c[0] == chunk_id]:
            del self._por_clave[clave]

    def set_faiss_internal_id(self, chunk_id: str, encoder_name: str, faiss_id: int) -> None:
        registro = self._por_clave.get((chunk_id, encoder_name))
        if registro is not None:
            registro.faiss_internal_id = faiss_id


def _registro(chunk_id: str, valor: float, dim: int = 4) -> EmbeddingRecord:
    vector = np.full(dim, valor, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)
    return EmbeddingRecord(
        chunk_id=chunk_id, doc_id="doc-1", fenomeno=1, formato="md",
        encoder_name="e5-fake", model_id="fake/e5", embedding_dim=dim, vector=vector,
    )


def test_faiss_id_de_es_deterministico():
    assert FaissIndexManager.faiss_id_de("c1") == FaissIndexManager.faiss_id_de("c1")
    assert FaissIndexManager.faiss_id_de("c1") != FaissIndexManager.faiss_id_de("c2")


def test_build_or_update_agrega_vectores_nuevos(tmp_path):
    repo = _FakeVectorRepository()
    repo.save_many([_registro("c1", 1.0), _registro("c2", 2.0)])
    manager = FaissIndexManager(repo, tmp_path)
    estrategia = FlatIPIndexStrategy()
    config = IndexBuildConfig()

    resultado = manager.build_or_update("e5-fake", ["c1", "c2"], estrategia, embedding_dim=4, config=config)

    assert resultado.n_total_en_indice == 2
    assert repo._por_clave[("c1", "e5-fake")].faiss_internal_id == FaissIndexManager.faiss_id_de("c1")


def test_build_or_update_reemplaza_sin_duplicar_si_el_chunk_cambio(tmp_path):
    repo = _FakeVectorRepository()
    repo.save_many([_registro("c1", 1.0)])
    manager = FaissIndexManager(repo, tmp_path)
    estrategia = FlatIPIndexStrategy()
    config = IndexBuildConfig()

    manager.build_or_update("e5-fake", ["c1"], estrategia, embedding_dim=4, config=config)

    # El texto de c1 cambió -> nuevo vector con el mismo chunk_id/faiss_id.
    repo.save_many([_registro("c1", 5.0)])
    resultado = manager.build_or_update("e5-fake", ["c1"], estrategia, embedding_dim=4, config=config)

    assert resultado.n_total_en_indice == 1  # no se duplicó


def test_load_lanza_error_si_no_existe_indice(tmp_path):
    repo = _FakeVectorRepository()
    manager = FaissIndexManager(repo, tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load("no-existe")
