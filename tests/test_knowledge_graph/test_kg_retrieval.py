"""Tests de la recuperación híbrida: adapters (FAISS/grafo), fusión y orquestador.

Verifica el contrato común :class:`Retriever` (el grafo es "un índice más"),
el scoring por evidencia del grafo (directo + vecinos de primer orden) y las
estrategias de fusión RRF/CombSUM/CombMNZ (Sección 8.4-8.5). Determinista:
FAISS real con vectores sintéticos y encoder fake (sin modelos ni red).
"""

from __future__ import annotations

import hashlib

import faiss
import numpy as np
import pytest

from src.encoders.base import EncoderStrategy
from src.knowledge_graph.extract.regex_entity_strategy import RegexEntityExtractor
from src.knowledge_graph.graph.repository import (
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.models import (
    Entity,
    EntityType,
    Query,
    RelationType,
    ScoredChunk,
    Tripleta,
)
from src.knowledge_graph.retrieval.adapters import (
    GraphIndexAdapter,
    VectorIndexAdapter,
)
from src.knowledge_graph.retrieval.base import Retriever
from src.knowledge_graph.retrieval.fusion import (
    CombMNZFusionStrategy,
    CombSUMFusionStrategy,
    RRFusionStrategy,
)
from src.knowledge_graph.retrieval.orchestrator import RetrievalOrchestrator


class FakeEncoder(EncoderStrategy):
    """Encoder sintético: vector determinista por hash del texto (sin modelo)."""

    model_id = "fake-encoder"
    embedding_dim = 4
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "mit"
    is_complementary = False

    def _cargar_modelo(self, device: str):  # pragma: no cover - nunca se usa
        return object()

    def encode(self, texts, is_query=False, batch_size=None):
        vectores = []
        for texto in texts:
            semilla = int(hashlib.sha1(texto.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(semilla)
            vectores.append(rng.standard_normal(self.embedding_dim).astype(np.float32))
        matriz = np.vstack(vectores)
        matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
        return matriz


# -- Helpers ----------------------------------------------------------------

def _indice_faiss_con_metadata(metadatas):
    """Índice FlatIP 4-d con vectores aleatorios alineados a ``metadatas``."""
    index = faiss.IndexFlatIP(4)
    rng = np.random.default_rng(7)
    vectores = rng.standard_normal((len(metadatas), 4)).astype(np.float32)
    vectores /= np.linalg.norm(vectores, axis=1, keepdims=True)
    index.add(vectores)
    return index


def _grafo_ejemplo():
    """Grafo: NASA—COOPERA_CON—ESA (c1), NASA—PERTENECE_A—Artemisa (c2), ESA—COOPERA_CON—SpaceX (c3)."""
    builder = KnowledgeGraphBuilder()
    for eid, nombre, tipo in [
        ("nasa", "NASA", EntityType.ORGANIZACION),
        ("esa", "ESA", EntityType.ORGANIZACION),
        ("spacex", "SpaceX", EntityType.ORGANIZACION),
        ("programa artemisa", "Programa Artemisa", EntityType.PROGRAMA),
    ]:
        builder.agregar_entidad(Entity(id=eid, nombre=nombre, tipo=tipo))
    builder.agregar_tripleta(
        Tripleta(sujeto="nasa", relacion=RelationType.COOPERA_CON, objeto="esa",
                 doc_id="d1", chunk_id="c1")
    )
    builder.agregar_tripleta(
        Tripleta(sujeto="nasa", relacion=RelationType.PERTENECE_A, objeto="programa artemisa",
                 doc_id="d1", chunk_id="c2")
    )
    builder.agregar_tripleta(
        Tripleta(sujeto="esa", relacion=RelationType.COOPERA_CON, objeto="spacex",
                 doc_id="d2", chunk_id="c3")
    )
    return builder.construir()


def _scored(chunk_id, doc_id, score, origen="canal"):
    return ScoredChunk(chunk_id=chunk_id, doc_id=doc_id, score=score, origen=origen)


# -- VectorIndexAdapter -----------------------------------------------------

def test_vector_adapter_devuelve_scored_chunks_ordenados():
    encoder = FakeEncoder()
    metadatas = [
        {"chunk_id": "c1", "doc_id": "d1", "texto": "NASA y ESA en la ISS"},
        {"chunk_id": "c2", "doc_id": "d1", "texto": "China y la órbita baja"},
        {"chunk_id": "c3", "doc_id": "d2", "texto": "Artemisa y SpaceX"},
    ]
    adapter: Retriever = VectorIndexAdapter(
        encoder=encoder, index=_indice_faiss_con_metadata(metadatas),
        metadata=metadatas, encoder_name="fake-encoder",
    )
    assert adapter.origen == "fake-encoder"
    resultados = adapter.retrieve(Query(texto="cooperación NASA ESA"), k=2)
    assert len(resultados) == 2
    assert all(isinstance(r, ScoredChunk) for r in resultados)
    assert resultados[0].score >= resultados[1].score
    assert {r.chunk_id for r in resultados} <= {"c1", "c2", "c3"}


def test_vector_adapter_con_indice_vacio():
    adapter = VectorIndexAdapter(
        encoder=FakeEncoder(), index=faiss.IndexFlatIP(4), metadata=[], encoder_name="fake"
    )
    assert adapter.retrieve(Query(texto="hola"), k=5) == []


# -- GraphIndexAdapter ------------------------------------------------------

def test_graph_adapter_evidencia_directa_y_de_vecinos():
    grafo = _grafo_ejemplo()
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(),
        grafo=KnowledgeGraphQuery(grafo),
    )
    resultados = adapter.retrieve(Query(texto="¿Con quién coopera la NASA?"), k=10)
    por_chunk = {r.chunk_id: r.score for r in resultados}
    # Evidencia directa (NASA): c1 (nasa-esa) y c2 (nasa-artemisa) → 1.0 cada
    # uno; los vecinos de NASA (ESA, Artemisa) añaden 0.5 a c1, c2 y c3.
    # c1 y c2 empatan a 1.5 (desempate por chunk_id) y c3 queda con 0.5.
    assert por_chunk["c1"] == pytest.approx(1.5)
    assert por_chunk["c2"] == pytest.approx(1.5)
    assert por_chunk["c3"] == pytest.approx(0.5)
    assert [r.chunk_id for r in resultados][:2] == ["c1", "c2"]
    assert all(r.origen == "grafo" for r in resultados)
    assert resultados[0].score >= resultados[-1].score


def test_graph_adapter_consulta_sin_entidades_devuelve_vacio():
    grafo = _grafo_ejemplo()
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(grafo)
    )
    assert adapter.retrieve(Query(texto="¿cuál es el clima hoy?"), k=5) == []


def test_graph_adapter_respeta_k():
    grafo = _grafo_ejemplo()
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(grafo)
    )
    assert len(adapter.retrieve(Query(texto="NASA ESA SpaceX Artemisa"), k=1)) <= 1


def test_graph_adapter_explica_camino():
    """El camino por el grafo: entidades → nodos → vecinos → tripletas → chunks."""
    grafo = _grafo_ejemplo()
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(grafo)
    )
    camino = adapter.explicar_camino(Query(texto="¿Con quién coopera la NASA?"))
    assert camino["entidades_en_grafo"] == ["nasa"]
    assert camino["vecinos_primer_orden"]["nasa"] == ["esa", "programa artemisa"]
    # Tripleta directa nasa-esa y tripletas de los vecinos (esa-c1/c3, artemisa-c2).
    claves = {(t["sujeto"], t["objeto"], t["chunk_id"]) for t in camino["tripletas"]}
    assert ("nasa", "esa", "c1") in claves
    assert ("esa", "spacex", "c3") in claves
    assert camino["chunks_evidencia"]["c1"] == pytest.approx(1.5)
    assert camino["aportó"] is True


def test_graph_adapter_explica_camino_sin_entidades():
    grafo = _grafo_ejemplo()
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(grafo)
    )
    camino = adapter.explicar_camino(Query(texto="¿cuál es el clima hoy?"))
    assert camino["entidades_en_grafo"] == []
    assert camino["tripletas"] == []
    assert camino["aportó"] is False


def test_graph_adapter_camino_por_top_k():
    """El camino hasta los chunks del resultado fusionado (auditoría)."""
    grafo = _grafo_ejemplo()
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(grafo)
    )
    camino = adapter.explicar_camino(
        Query(texto="¿Con quién coopera la NASA?"), chunk_ids_objetivo=["c1", "c3"]
    )
    por_chunk = {c["chunk_id"]: c for c in camino["camino_por_top_k"]}
    assert set(por_chunk) == {"c1", "c3"}
    # c1 está en la evidencia del grafo (tripleta nasa-esa) → score > 0.
    assert por_chunk["c1"]["score_grafo"] == pytest.approx(1.5)
    assert por_chunk["c1"]["tripletas"]
    # c3 solo lo alcanzan las tripletas de vecinos (ESA) → también aporta.
    assert por_chunk["c3"]["score_grafo"] == pytest.approx(0.5)
    assert por_chunk["c3"]["tripletas"]


# -- Fusión ----------------------------------------------------------------

def test_rrf_fusion_combina_rankings_de_origenes_distintos():
    ranking_vector = [_scored("c1", "d1", 0.95, "vec"), _scored("c2", "d1", 0.80, "vec")]
    ranking_grafo = [_scored("c2", "d1", 3.0, "grafo"), _scored("c3", "d2", 2.0, "grafo")]
    fusion = RRFusionStrategy(k0=60).fusionar([ranking_vector, ranking_grafo])
    por_id = {c.chunk_id: c.score for c in fusion}
    # c2 = 1/61 (r1 en grafo) + 1/62 (r2 en vector) > c1 = 1/61 > c3 = 1/62
    assert por_id["c2"] == pytest.approx(1 / 61 + 1 / 62)
    assert por_id["c1"] == pytest.approx(1 / 61)
    assert por_id["c3"] == pytest.approx(1 / 62)
    assert [c.chunk_id for c in fusion] == ["c2", "c1", "c3"]


def test_combsum_normaliza_escalas_antes_de_sumar():
    # El coseno (~0.9) y la evidencia del grafo (2, 3) no son comparables:
    # CombSUM debe normalizar cada lista a [0, 1] antes de sumar.
    ranking_vector = [_scored("c1", "d1", 0.95, "vec"), _scored("c2", "d1", 0.90, "vec")]
    ranking_grafo = [_scored("c2", "d1", 3.0, "grafo"), _scored("c3", "d2", 1.0, "grafo")]
    fusion = CombSUMFusionStrategy().fusionar([ranking_vector, ranking_grafo])
    por_id = {c.chunk_id: c.score for c in fusion}
    # c1 = 1.0 (único normalizado de su lista); c2 = 0 + 1.0; c3 = 0 + 0.
    assert por_id["c1"] == pytest.approx(1.0)
    assert por_id["c2"] == pytest.approx(1.0)
    assert por_id["c3"] == pytest.approx(0.0)


def test_combmnz_premia_el_consenso_entre_canales():
    ranking_vector = [_scored("c1", "d1", 0.9, "vec"), _scored("c2", "d1", 0.5, "vec")]
    ranking_grafo = [_scored("c1", "d1", 2.0, "grafo")]
    fusion = CombMNZFusionStrategy().fusionar([ranking_vector, ranking_grafo])
    por_id = {c.chunk_id: c.score for c in fusion}
    # Normalizadas: vector → c1=1.0, c2=0.0; grafo → c1=1.0 (lista única).
    # CombMNZ: c1 = 2 canales × (1.0+1.0) = 4.0; c2 = 1 canal × 0.0 = 0.0.
    assert por_id["c1"] == pytest.approx(4.0)
    assert por_id["c2"] == pytest.approx(0.0)
    assert fusion[0].chunk_id == "c1"


def test_fusion_sin_rankings_devuelve_vacio():
    assert RRFusionStrategy().fusionar([]) == []
    assert CombSUMFusionStrategy().fusionar([[]]) == []
    assert CombMNZFusionStrategy().fusionar([[], []]) == []


# -- RetrievalOrchestrator --------------------------------------------------

def test_orquestador_con_un_solo_canal_devuelve_su_ranking():
    canal = [_scored("c1", "d1", 0.5, "vectorial")]
    orquestador = RetrievalOrchestrator(retrievers=[_CanalFijo(canal)], fusion=RRFusionStrategy())
    resultado = orquestador.recuperar(Query(texto="q"), k_final=5)
    assert [c.chunk_id for c in resultado] == ["c1"]
    assert orquestador.canales == ["fijo"]


def test_orquestador_fusiona_dos_canales():
    vectorial = [_scored("c1", "d1", 0.9, "vectorial"), _scored("c2", "d1", 0.8, "vectorial")]
    grafo = [_scored("c2", "d1", 4.0, "grafo"), _scored("c3", "d2", 3.0, "grafo")]
    orquestador = RetrievalOrchestrator(
        retrievers=[_CanalFijo(vectorial), _CanalFijo(grafo)], fusion=RRFusionStrategy(k0=60)
    )
    resultado = orquestador.recuperar(Query(texto="q"), k_final=10)
    assert [c.chunk_id for c in resultado] == ["c2", "c1", "c3"]
    assert all(c.origen == "rrf" for c in resultado)


def test_orquestador_ignora_canal_que_falla():
    bueno = [_scored("c1", "d1", 0.9, "vectorial")]
    malo = _CanalFallo()
    orquestador = RetrievalOrchestrator(
        retrievers=[malo, _CanalFijo(bueno)], fusion=RRFusionStrategy()
    )
    resultado = orquestador.recuperar(Query(texto="q"), k_final=5)
    assert [c.chunk_id for c in resultado] == ["c1"]


def test_orquestador_sin_canales_devuelve_vacio():
    orquestador = RetrievalOrchestrator(retrievers=[], fusion=RRFusionStrategy())
    assert orquestador.recuperar(Query(texto="q")) == []


def test_orquestador_recorta_al_k_final():
    canal = [_scored(f"c{i}", "d1", float(100 - i), "vectorial") for i in range(1, 11)]
    orquestador = RetrievalOrchestrator(retrievers=[_CanalFijo(canal)], fusion=RRFusionStrategy())
    resultado = orquestador.recuperar(Query(texto="q"), k_final=3)
    assert len(resultado) == 3


class _CanalFijo(Retriever):
    """Retriever de prueba con ranking prefijado (independiente de la consulta)."""

    def __init__(self, ranking) -> None:
        self._ranking = ranking

    @property
    def origen(self) -> str:
        return "fijo"

    def retrieve(self, q: Query, k: int):
        return self._ranking[:k]


class _CanalFallo(Retriever):
    """Retriever que falla (el orquestador debe omitirlo, no romper)."""

    @property
    def origen(self) -> str:
        return "roto"

    def retrieve(self, q: Query, k: int):
        raise RuntimeError("canal roto")
