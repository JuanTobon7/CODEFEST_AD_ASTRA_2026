"""Tests del serializador GraphML y de los repositorios (Sección 7.3).

Verifica que el GraphML generado es válido y cargable con NetworkX SIN
dependencias adicionales (lxml no está instalado en el entorno), que
conserva la trazabilidad (doc_id/chunk_id) y el round-trip del repositorio.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import networkx as nx
import pytest

from src.knowledge_graph.graph.inmemory_repository import (
    GraphMLFileRepository,
    Neo4jRepository,
    NetworkXInMemoryRepository,
)
from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph
from src.knowledge_graph.graph.repository import KnowledgeGraphBuilder
from src.knowledge_graph.graph.serializer import GraphMLSerializer
from src.knowledge_graph.models import Entity, EntityType, RelationType, Tripleta


def _grafo_ejemplo() -> KnowledgeGraph:
    builder = KnowledgeGraphBuilder()
    builder.agregar_entidad(Entity(id="nasa", nombre="NASA", tipo=EntityType.ORGANIZACION))
    builder.agregar_entidad(Entity(id="esa", nombre="ESA", tipo=EntityType.ORGANIZACION))
    builder.agregar_tripleta(
        Tripleta(sujeto="nasa", relacion=RelationType.COOPERA_CON, objeto="esa",
                 doc_id="doc-1", chunk_id="chunk-1", evidencia="La NASA coopera con la ESA.",
                 confianza=0.9)
    )
    return builder.construir()


# -- Serializador -----------------------------------------------------------

def test_serializer_emite_graphml_valido():
    xml = GraphMLSerializer().serializar(_grafo_ejemplo())
    root = ET.fromstring(xml)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    assert root.tag == "{http://graphml.graphdrawing.org/xmlns}graphml"
    nodos = root.findall(".//g:node", ns)
    aristas = root.findall(".//g:edge", ns)
    assert len(nodos) == 2
    assert len(aristas) == 1


def test_serializer_declara_los_atributos():
    xml = GraphMLSerializer().serializar(_grafo_ejemplo())
    root = ET.fromstring(xml)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    claves = {k.get("id") for k in root.findall("g:key", ns)}
    assert "kn0" in claves and "ka0" in claves and "ka2" in claves


def test_graphml_cargable_con_networkx_sin_lxml():
    """Requisito duro: cargable con NetworkX sin dependencias adicionales."""
    xml = GraphMLSerializer().serializar(_grafo_ejemplo())
    import io

    grafo = nx.read_graphml(io.StringIO(xml))
    assert set(grafo.nodes) == {"nasa", "esa"}
    assert grafo.nodes["nasa"]["tipo"] == "ORGANIZACION"
    aristas = list(grafo.edges(data=True))
    assert len(aristas) == 1
    _, _, datos = aristas[0]
    assert datos["relacion"] == "COOPERA_CON"
    assert datos["doc_id"] == "doc-1"
    assert datos["chunk_id"] == "chunk-1"


def test_serializer_escribe_archivo():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ruta = GraphMLSerializer().escribir(_grafo_ejemplo(), Path(tmp) / "grafo.graphml")
        assert ruta.exists()
        assert "graphml" in ruta.read_text(encoding="utf-8")


# -- Repositorios -----------------------------------------------------------

def test_graphml_repository_round_trip(tmp_path):
    repo = GraphMLFileRepository()
    ruta = repo.guardar(_grafo_ejemplo(), tmp_path / "grafo.graphml")
    assert ruta.exists()

    cargado = repo.cargar(ruta)
    assert cargado.num_entidades == 2
    assert cargado.num_tripletas == 1
    tripleta = next(cargado.tripletas())
    assert tripleta.sujeto == "nasa"
    assert tripleta.objeto == "esa"
    assert tripleta.doc_id == "doc-1"
    assert tripleta.chunk_id == "chunk-1"
    assert tripleta.relacion == RelationType.COOPERA_CON


def test_networkx_inmemory_repository_sin_io(tmp_path):
    repo = NetworkXInMemoryRepository()
    grafo = _grafo_ejemplo()
    repo.guardar(grafo, tmp_path / "ignorada.graphml")
    assert repo.grafo is grafo
    # cargar con ruta inexistente devuelve el grafo vivo (semántica de arranque).
    assert repo.cargar(tmp_path / "no-existe.graphml") is grafo


def test_neo4j_repository_es_stub_documentado():
    repo = Neo4jRepository()
    with pytest.raises(NotImplementedError):
        repo.guardar(_grafo_ejemplo(), "grafo.graphml")
    with pytest.raises(NotImplementedError):
        repo.cargar("grafo.graphml")