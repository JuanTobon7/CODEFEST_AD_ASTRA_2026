"""
Tests de ``MongoChunkRepository`` sin MongoDB real: la distribución
balanceada por fenómeno (Sección 5 del reto) se prueba con una colección
fake que emula ``find()`` + ``sort()`` + ``limit()``.

Cubre:
- ``_cuotas_por_fenomeno``: reparto determinista del límite entre los 3
  fenómenos (residuo a los primeros).
- ``find_all_balanceado``: el lote incluye chunks de los 3 fenómenos con las
  cuotas esperadas, sin sesgo hacia un solo fenómeno.
"""

from __future__ import annotations

import pytest

from src.models.chunk import Chunk
from src.persistence.mongo_repository import MongoChunkRepository


def _doc(chunk_id: str, fenomeno: int, posicion: int = 0) -> dict:
    """Documento Mongo válido para ``Chunk.model_validate``."""
    return {
        "doc_id": f"doc-{chunk_id}",
        "chunk_id": chunk_id,
        "fuente": "prueba.md",
        "formato": "md",
        "fenomeno": fenomeno,
        "posicion": posicion,
        "num_tokens": 10,
        "texto": f"Texto de {chunk_id}",
    }


class _FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *args, **kwargs):
        return self

    def limit(self, n):
        # Igual que el cursor real de Mongo: recorta al iterar.
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class _FakeColeccion:
    def __init__(self, docs):
        self._docs = docs

    def __getitem__(self, nombre):
        return self

    def find(self, filtro):
        fenomeno = filtro.get("fenomeno") if filtro else None
        docs = [d for d in self._docs if d.get("fenomeno") == fenomeno] if fenomeno else list(self._docs)
        docs.sort(key=lambda d: d["chunk_id"])
        return _FakeCursor(docs)


class _FakeCliente:
    def __init__(self, coleccion):
        self._coleccion = coleccion

    def __getitem__(self, nombre):
        return self._coleccion


def _repo(docs) -> MongoChunkRepository:
    repo = MongoChunkRepository(uri="mongodb://fake", db_name="db")
    repo._cliente = _FakeCliente(_FakeColeccion(docs))
    repo._conectado = True
    return repo


def _corpus_mixto():
    """60 chunks: 30 de F1, 20 de F2, 10 de F3 (carga desbalanceada realista)."""
    docs = []
    for i in range(30):
        docs.append(_doc(f"f1-{i:03d}", 1, posicion=i))
    for i in range(20):
        docs.append(_doc(f"f2-{i:03d}", 2, posicion=i))
    for i in range(10):
        docs.append(_doc(f"f3-{i:03d}", 3, posicion=i))
    return docs


def _corpus_holgado():
    """120 chunks (40 por fenómeno): cada fenómeno tiene más que su cuota."""
    docs = []
    for fenomeno in (1, 2, 3):
        for i in range(40):
            docs.append(_doc(f"f{fenomeno}-{i:03d}", fenomeno, posicion=i))
    return docs


# -- _cuotas_por_fenomeno -----------------------------------------------------

@pytest.mark.parametrize(
    ("limite", "esperado"),
    [
        (1000, {1: 334, 2: 333, 3: 333}),
        (100, {1: 34, 2: 33, 3: 33}),
        (3, {1: 1, 2: 1, 3: 1}),
        (1, {1: 1, 2: 0, 3: 0}),
        (0, {}),
        (-5, {}),
    ],
)
def test_cuotas_por_fenomeno(limite, esperado):
    assert MongoChunkRepository._cuotas_por_fenomeno(limite) == esperado


def test_cuotas_suman_el_limite():
    for limite in range(1, 50):
        cuotas = MongoChunkRepository._cuotas_por_fenomeno(limite)
        assert sum(cuotas.values()) == limite
        assert list(cuotas) == [1, 2, 3]


# -- find_all_balanceado ------------------------------------------------------

def test_lote_balanceado_reparte_entre_fenomenos():
    repo = _repo(_corpus_mixto())
    chunks = repo.find_all_balanceado(30)

    assert len(chunks) == 30
    por_fenomeno = {f: sum(1 for c in chunks if c.fenomeno == f) for f in (1, 2, 3)}
    # Cuotas esperadas: 10 por fenómeno (10 del F3 que solo tiene 10).
    assert por_fenomeno == {1: 10, 2: 10, 3: 10}
    assert all(isinstance(c, Chunk) for c in chunks)


def test_lote_balanceado_cuotas_exactas():
    """Con corpus suficiente, cada fenómeno recibe exactamente su cuota."""
    repo = _repo(_corpus_holgado())
    chunks = repo.find_all_balanceado(100)
    por_fenomeno = {f: sum(1 for c in chunks if c.fenomeno == f) for f in (1, 2, 3)}
    # 100 // 3 = 33 + residuo 1 al F1.
    assert por_fenomeno == {1: 34, 2: 33, 3: 33}
    assert len(chunks) == 100


def test_lote_balanceado_respeta_disponibilidad():
    """Si un fenómeno tiene menos que su cuota, devuelve lo disponible."""
    repo = _repo(_corpus_mixto())  # 30/20/10 disponibles
    chunks = repo.find_all_balanceado(100)
    por_fenomeno = {f: sum(1 for c in chunks if c.fenomeno == f) for f in (1, 2, 3)}
    assert por_fenomeno == {1: 30, 2: 20, 3: 10}


def test_lote_balanceado_sin_limite_delega_en_find_all():
    repo = _repo(_corpus_mixto())
    chunks = repo.find_all_balanceado(0)
    assert len(chunks) == 60


def test_lote_balanceado_determinista():
    repo = _repo(_corpus_mixto())
    primero = repo.find_all_balanceado(30)
    segundo = repo.find_all_balanceado(30)
    assert [c.chunk_id for c in primero] == [c.chunk_id for c in segundo]
