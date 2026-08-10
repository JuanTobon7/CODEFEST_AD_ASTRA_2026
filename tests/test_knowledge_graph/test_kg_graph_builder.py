"""Tests del Builder, del KnowledgeGraph y de la interfaz GraphQuery (ISP).

Verifica el patrón Builder (agregar fragmento a fragmento → grafo), la
resolución de vecinos de primer orden (Sección 8.5) y el fail-fast ante
tripletas con entidades desconocidas.
"""

from __future__ import annotations

import pytest

from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import (
    GraphBuilder,
    GraphQuery,
    KnowledgeGraphBuilder,
    KnowledgeGraphQuery,
)
from src.knowledge_graph.models import Entity, EntityType, RelationType, Tripleta


def _grafo_ejemplo() -> KnowledgeGraph:
    """Grafo: NASA—COOPERA_CON—ESA, NASA—PERTENECE_A—Artemisa, ESA—COOPERA_CON—SpaceX."""
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
                 doc_id="d1", chunk_id="c1", evidencia="...")
    )
    builder.agregar_tripleta(
        Tripleta(sujeto="nasa", relacion=RelationType.PERTENECE_A, objeto="programa artemisa",
                 doc_id="d1", chunk_id="c2", evidencia="...")
    )
    builder.agregar_tripleta(
        Tripleta(sujeto="esa", relacion=RelationType.COOPERA_CON, objeto="spacex",
                 doc_id="d2", chunk_id="c3", evidencia="...")
    )
    return builder.construir()


# -- Builder ----------------------------------------------------------------

def test_builder_agrega_y_construye():
    grafo = _grafo_ejemplo()
    assert grafo.num_entidades == 4
    assert grafo.num_tripletas == 3


def test_builder_es_idempotente_para_entidades():
    builder = KnowledgeGraphBuilder()
    entidad = Entity(id="nasa", nombre="NASA")
    builder.agregar_entidad(entidad)
    builder.agregar_entidad(entidad)
    assert builder.construir().num_entidades == 1


def test_builder_fail_fast_con_entidad_desconocida():
    builder = KnowledgeGraphBuilder()
    builder.agregar_entidad(Entity(id="nasa", nombre="NASA"))
    with pytest.raises(ValueError):
        builder.agregar_tripleta(
            Tripleta(sujeto="nasa", relacion=RelationType.COOPERA_CON,
                     objeto="spacex", doc_id="d1", chunk_id="c1")
        )
    with pytest.raises(ValueError):
        builder.agregar_tripleta(
            Tripleta(sujeto="spacex", relacion=RelationType.COOPERA_CON,
                     objeto="nasa", doc_id="d1", chunk_id="c1")
        )


# -- KnowledgeGraph: Information Expert sobre su estructura ------------------

def test_vecinos_primer_orden():
    grafo = _grafo_ejemplo()
    assert grafo.vecinos("nasa", grado=1) == {"esa", "programa artemisa"}
    assert grafo.vecinos("spacex", grado=1) == {"esa"}


def test_vecinos_segundo_orden():
    grafo = _grafo_ejemplo()
    # A 2 saltos de SpaceX: ESA (1 salto) y NASA (2 saltos vía ESA).
    # Artemisa queda a 3 saltos (SpaceX→ESA→NASA→Artemisa).
    assert grafo.vecinos("spacex", grado=2) == {"esa", "nasa"}


def test_chunk_ids_de_entidad():
    grafo = _grafo_ejemplo()
    assert grafo.chunk_ids_de("nasa") == {"c1", "c2"}
    assert grafo.chunk_ids_de("no-existe") == set()


def test_chunk_ids_de_vecindario():
    grafo = _grafo_ejemplo()
    # SpaceX: directo c3; vecino ESA → c1, c3.
    assert grafo.chunk_ids_de_vecindario("spacex", grado=1) == {"c1", "c3"}


def test_tripletas_de_entidad():
    grafo = _grafo_ejemplo()
    tripletas = grafo.tripletas_de("nasa")
    assert len(tripletas) == 2
    assert all(t.sujeto == "nasa" or t.objeto == "nasa" for t in tripletas)


def test_tripletas_de_chunk():
    grafo = _grafo_ejemplo()
    tripletas = grafo.tripletas_de_chunk("c1")
    assert len(tripletas) == 1
    assert tripletas[0].sujeto == "nasa"
    assert tripletas[0].objeto == "esa"
    assert grafo.tripletas_de_chunk("no-existe") == []


def test_to_networkx_preserva_nodos_y_aristas():
    grafo = _grafo_ejemplo()
    nx_grafo = grafo.to_networkx()
    assert nx_grafo.number_of_nodes() == 4
    assert nx_grafo.number_of_edges() == 3
    assert nx_grafo.nodes["nasa"]["tipo"] == "ORGANIZACION"


# -- GraphQuery (ISP: la recuperación solo ve lectura) ------------------------

def test_knowledge_graph_query_implementa_la_interfaz():
    grafo = _grafo_ejemplo()
    query: GraphQuery = KnowledgeGraphQuery(grafo)
    assert query.tiene("nasa")
    assert not query.tiene("no-existe")
    assert query.vecinos_primer_orden("nasa") == {"esa", "programa artemisa"}
    assert query.chunk_ids_de("esa") == {"c1", "c3"}
    assert len(query.tripletas_de("spacex")) == 1


def test_interfaz_graph_builder_es_abstracta():
    with pytest.raises(TypeError):
        GraphBuilder()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        GraphQuery()  # type: ignore[abstract]
