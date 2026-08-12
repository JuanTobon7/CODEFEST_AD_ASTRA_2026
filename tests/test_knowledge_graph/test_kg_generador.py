"""Tests del generador híbrido: pipeline completo FAISS + grafo → Sección 9.

Verifica que ``recuperar_hibrido`` produce el contrato de ``retrieve()``
(``{"documents", "fragments"}``) cuando los canales son fakes (sin modelos),
y que las conversiones de contrato (:class:`SearchHit` / :class:`FusedFragment`)
respetan el texto y la trazabilidad de origen.
"""

from __future__ import annotations

import pytest

from src.knowledge_graph.models import Query, ScoredChunk
from src.knowledge_graph.retrieval.adapters import GraphIndexAdapter
from src.knowledge_graph.retrieval.base import Retriever
from src.knowledge_graph.retrieval.fusion import RRFusionStrategy
from src.knowledge_graph.run_generador_hibrido import (
    _a_fused_fragments,
    recuperar_hibrido,
)
from src.knowledge_graph.extract.regex_entity_strategy import RegexEntityExtractor
from src.knowledge_graph.graph.repository import (
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.models import Entity, EntityType, RelationType, Tripleta
from src.retrieval.index_loader import build_siguiente_chunk_lookup
from src.retrieval.models import FusedFragment

METADATA = [
    {"chunk_id": "c1", "doc_id": "d1", "texto": "La NASA coopera con la ESA.", "posicion": 0},
    {"chunk_id": "c2", "doc_id": "d1", "texto": "La basura espacial afecta a la órbita baja.", "posicion": 1},
    {"chunk_id": "c3", "doc_id": "d2", "texto": "Artemisa pertenece a la NASA.", "posicion": 0},
]


class CanalFijo(Retriever):
    """Canal fake con ranking prefijado (independiente de la consulta)."""

    def __init__(self, ranking, origen: str) -> None:
        self._ranking = ranking
        self._origen = origen

    @property
    def origen(self) -> str:
        return self._origen

    def retrieve(self, q: Query, k: int):
        return self._ranking[:k]


def _grafo_ejemplo():
    builder = KnowledgeGraphBuilder()
    builder.agregar_entidad(Entity(id="nasa", nombre="NASA", tipo=EntityType.ORGANIZACION))
    builder.agregar_entidad(Entity(id="esa", nombre="ESA", tipo=EntityType.ORGANIZACION))
    builder.agregar_tripleta(
        Tripleta(sujeto="nasa", relacion=RelationType.COOPERA_CON, objeto="esa",
                 doc_id="d1", chunk_id="c1")
    )
    return builder.construir()


def _texto_por_chunk() -> dict:
    return {m["chunk_id"]: m["texto"] for m in METADATA}


def test_a_fused_fragments_conserva_texto_y_origen():
    fusionados = [ScoredChunk(chunk_id="c1", doc_id="d1", score=0.5, origen="rrf")]
    fragmentos = _a_fused_fragments(fusionados, _texto_por_chunk())
    assert len(fragmentos) == 1
    f = fragmentos[0]
    assert isinstance(f, FusedFragment)
    assert f.text == "La NASA coopera con la ESA."
    assert f.encoders == ["rrf"]


def test_recuperar_hibrido_produce_contrato_de_retrieve():
    vectorial = CanalFijo(
        [
            ScoredChunk(chunk_id="c1", doc_id="d1", score=0.9, origen="vec"),
            ScoredChunk(chunk_id="c2", doc_id="d1", score=0.8, origen="vec"),
        ],
        origen="vec",
    )
    grafo = GraphIndexAdapter(
        extractor=RegexEntityExtractor(), grafo=KnowledgeGraphQuery(_grafo_ejemplo())
    )
    siguiente_chunk = build_siguiente_chunk_lookup({"enc": METADATA})

    resultado, fusionados = recuperar_hibrido(
        "¿Con quién coopera la NASA?",
        canales_vectoriales=[vectorial],
        canal_grafo=grafo,
        texto_por_chunk=_texto_por_chunk(),
        siguiente_chunk=siguiente_chunk,
        fusion=RRFusionStrategy(k0=60),
        k_search=10,
        k_chunk_out=10,
    )
    assert isinstance(resultado, dict)
    assert set(resultado) == {"documents", "fragments"}
    assert resultado["documents"]
    assert resultado["fragments"]
    for frag in resultado["fragments"]:
        assert frag["text"]  # texto resuelto desde la metadata
        assert frag["chunk_id"] and frag["doc_id"]
    # El grafo aportó c1 (tripleta nasa-esa) a la fusión.
    assert any(f["chunk_id"] == "c1" for f in resultado["fragments"])
    assert fusionados  # ranking fusionado no vacío
