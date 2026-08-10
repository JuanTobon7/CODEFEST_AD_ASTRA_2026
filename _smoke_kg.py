"""Smoke test del flujo completo del grafo de conocimiento."""
import tempfile
from pathlib import Path

from src.knowledge_graph.extract.pipeline import ExtractionPipeline
from src.knowledge_graph.extract.regex_entity_strategy import RegexEntityExtractor
from src.knowledge_graph.extract.cooccurrence_relation_strategy import CooccurrenceRelationExtractor
from src.knowledge_graph.graph.repository import KnowledgeGraphBuilder, KnowledgeGraphQuery
from src.knowledge_graph.retrieval.adapters import GraphIndexAdapter
from src.knowledge_graph.models import Query
from src.knowledge_graph.service import KnowledgeGraphService

textos = [
    ("doc-1", "chunk-1",
     "La NASA coopera con la ESA en la Estación Espacial Internacional. "
     "México participa en la Agencia Espacial Mexicana."),
    ("doc-1", "chunk-2",
     "La basura espacial afecta a la órbita terrestre baja. "
     "La inteligencia artificial se usa en la vigilancia satelital de China."),
    ("doc-2", "chunk-3",
     "El programa Artemisa pertenece a la NASA. La ESA colabora con SpaceX."),
]

ner = RegexEntityExtractor()
re = CooccurrenceRelationExtractor()
pipe = ExtractionPipeline(ner=ner, re=re)

class C:
    def __init__(self, d, c, t):
        self.doc_id, self.chunk_id, self.texto = d, c, t

chunks = [C(*t) for t in textos]

servicio = KnowledgeGraphService(entity_extractor=ner, relation_extractor=re)
grafo = servicio.construir_desde_chunks(chunks)
print("Entidades:", grafo.num_entidades)
print("Tripletas:", grafo.num_tripletas)
for t in grafo.tripletas():
    print("  ", t.sujeto, "-", t.relacion.value, "->", t.objeto, f"[{t.doc_id}/{t.chunk_id}]")

with tempfile.TemporaryDirectory() as tmp:
    ruta = servicio.exportar_graphml(Path(tmp) / "grafo.graphml")
    import networkx as nx
    g = nx.read_graphml(str(ruta))
    print("GraphML OK:", g.number_of_nodes(), "nodos,", g.number_of_edges(), "aristas")

adapter = servicio.crear_retriever_grafo()
res = adapter.retrieve(Query(texto="¿Qué organización coopera con la NASA?"), k=5)
for r in res:
    print("candidato:", r.chunk_id, r.score, r.origen)
