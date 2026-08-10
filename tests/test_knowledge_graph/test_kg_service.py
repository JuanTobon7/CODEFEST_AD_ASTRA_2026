"""Tests de la fachada KnowledgeGraphService, casos de uso e integración híbrida.

Verifica el patrón Facade (punto de entrada único), los use cases
(Controller) y el escenario completo de la Sección 8.5: el grafo y FAISS
como dos canales :class:`Retriever` que el orquestador fusiona tratándolos
de forma uniforme.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import faiss
import numpy as np
import pytest

from src.knowledge_graph.extract.pipeline import ExtractionPipeline
from src.knowledge_graph.extract.regex_entity_strategy import RegexEntityExtractor
from src.knowledge_graph.extract.cooccurrence_relation_strategy import (
    CooccurrenceRelationExtractor,
)
from src.knowledge_graph.graph.repository import (
    GraphRepository,
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.models import Query
from src.knowledge_graph.retrieval.adapters import GraphIndexAdapter, VectorIndexAdapter
from src.knowledge_graph.retrieval.fusion import RRFusionStrategy
from src.knowledge_graph.retrieval.orchestrator import RetrievalOrchestrator
from src.knowledge_graph.service import KnowledgeGraphService
from src.knowledge_graph.use_cases import (
    IndexKnowledgeGraphUseCase,
    RetrieveViaGraphUseCase,
)
from src.encoders.base import EncoderStrategy


@dataclass
class ChunkFake:
    """Chunk mínimo compatible con el protocolo ChunkFuente."""

    doc_id: str
    chunk_id: str
    texto: str


CHUNKS = [
    ChunkFake(
        "d1", "c1",
        "La NASA coopera con la ESA en la Estación Espacial Internacional.",
    ),
    ChunkFake(
        "d1", "c2",
        "La basura espacial afecta a la órbita terrestre baja. China invierte en inteligencia artificial.",
    ),
    ChunkFake(
        "d2", "c3",
        "El programa Artemisa pertenece a la NASA. La ESA colabora con SpaceX.",
    ),
]


class FakeEncoder(EncoderStrategy):
    """Encoder sintético determinista (sin modelo real)."""

    model_id = "fake-encoder"
    embedding_dim = 4
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "mit"

    def _cargar_modelo(self, device: str):  # pragma: no cover
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


# -- Facade -----------------------------------------------------------------

def test_servicio_construye_exporta_y_resume():
    servicio = KnowledgeGraphService()
    grafo = servicio.construir_desde_chunks(CHUNKS)
    assert grafo.num_entidades >= 8
    assert grafo.num_tripletas >= 10

    resumen = servicio.resumen()
    assert resumen["ner"] == "regex-gazetteer"
    assert resumen["entidades"] == grafo.num_entidades
    assert servicio.grafo_actual is grafo


def test_servicio_exporta_graphml_cargable(tmp_path):
    servicio = KnowledgeGraphService()
    servicio.construir_desde_chunks(CHUNKS)
    ruta = servicio.exportar_graphml(tmp_path / "grafo.graphml")
    assert ruta.exists()
    import networkx as nx

    cargado = nx.read_graphml(str(ruta))
    assert cargado.number_of_nodes() >= 8


def test_servicio_exportar_sin_construir_lanza_error():
    servicio = KnowledgeGraphService()
    with pytest.raises(ValueError):
        servicio.exportar_graphml("grafo.graphml")


def test_servicio_retriever_sin_construir_lanza_error():
    servicio = KnowledgeGraphService()
    with pytest.raises(ValueError):
        servicio.crear_retriever_grafo()


def test_servicio_recupera_via_grafo():
    servicio = KnowledgeGraphService()
    servicio.construir_desde_chunks(CHUNKS)
    resultados = servicio.recuperar_via_grafo("¿Qué organización coopera con la NASA?", k=5)
    assert resultados
    assert resultados[0].chunk_id == "c1"
    assert resultados[0].origen == "grafo"


def test_servicio_repositorio_inyectado_se_usa():
    repo = GraphRepositoryStub()
    servicio = KnowledgeGraphService(repositorio=repo)
    servicio.construir_desde_chunks(CHUNKS)
    ruta = servicio.exportar_graphml("grafo.graphml")
    assert repo.guardado == 1
    assert ruta is not None


class GraphRepositoryStub(GraphRepository):
    """Stub que registra la llamada a guardar (sin tocar disco)."""

    def __init__(self) -> None:
        self.guardado = 0

    def guardar(self, grafo, ruta):
        self.guardado += 1
        from pathlib import Path

        return Path(str(ruta))

    def cargar(self, ruta):
        from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph

        return KnowledgeGraph()


# -- Use cases --------------------------------------------------------------

def test_use_case_index_construye_con_trazabilidad():
    pipeline = ExtractionPipeline(
        ner=RegexEntityExtractor(), re=CooccurrenceRelationExtractor()
    )
    builder = KnowledgeGraphBuilder()
    caso = IndexKnowledgeGraphUseCase(pipeline=pipeline, builder=builder)
    grafo = caso.ejecutar(CHUNKS)
    assert grafo.num_tripletas > 0
    for tripleta in grafo.tripletas():
        assert tripleta.doc_id and tripleta.chunk_id


def test_use_case_retrieve_delega_en_el_adapter():
    pipeline = ExtractionPipeline(
        ner=RegexEntityExtractor(), re=CooccurrenceRelationExtractor()
    )
    builder = KnowledgeGraphBuilder()
    IndexKnowledgeGraphUseCase(pipeline=pipeline, builder=builder).ejecutar(CHUNKS)
    adapter = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(builder.construir())
    )
    caso = RetrieveViaGraphUseCase(adapter=adapter)
    resultados = caso.ejecutar("cooperación NASA", k=3)
    assert isinstance(resultados, list)
    assert all(r.chunk_id for r in resultados)


# -- Integración híbrida (Sección 8.5): FAISS + grafo como dos canales -------

def test_integracion_hibrida_grafo_y_faiss_fusionados():
    # 1) Construye el grafo con el servicio (canal simbólico).
    servicio = KnowledgeGraphService()
    servicio.construir_desde_chunks(CHUNKS)

    # 2) Canal vectorial: FAISS real + encoder fake sobre los mismos chunks.
    encoder = FakeEncoder()
    metadatas = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "texto": c.texto} for c in CHUNKS
    ]
    index = faiss.IndexFlatIP(4)
    rng = np.random.default_rng(3)
    vectores = rng.standard_normal((len(metadatas), 4)).astype(np.float32)
    vectores /= np.linalg.norm(vectores, axis=1, keepdims=True)
    index.add(vectores)

    vectorial = VectorIndexAdapter(
        encoder=encoder, index=index, metadata=metadatas, encoder_name="fake-encoder"
    )
    grafo = servicio.crear_retriever_grafo()

    # 3) El orquestador los trata de forma uniforme (Adapter + Strategy).
    orquestador = RetrievalOrchestrator(
        retrievers=[vectorial, grafo], fusion=RRFusionStrategy(k0=60), k_por_canal=50
    )
    resultado = orquestador.recuperar(Query(texto="cooperación entre NASA y ESA"), k_final=5)

    assert resultado, "la fusión debe producir candidatos"
    assert all(isinstance(r.chunk_id, str) for r in resultado)
    # El mismo chunk puede venir de ambos canales; tras RRF el origen es "rrf".
    assert all(r.origen == "rrf" for r in resultado)
    # Los candidatos pertenecen al pool de chunks indexados.
    assert {r.chunk_id for r in resultado} <= {"c1", "c2", "c3"}
