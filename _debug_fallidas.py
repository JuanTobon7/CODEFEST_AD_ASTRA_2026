"""Debug: por qué fallan q020/q028/q029/q031 en el generador híbrido."""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

from src.knowledge_graph.run_generador_hibrido import (
    _cargar_canal_grafo,
    _cargar_canales_vectoriales,
    _texto_por_chunk,
    recuperar_hibrido,
)
from src.knowledge_graph.retrieval.fusion import RRFusionStrategy
from src.queries import QueryLoader, build_result_object
from src.retrieval.index_loader import build_siguiente_chunk_lookup

consultas = QueryLoader.cargar("consultas.jsonl")
por_id = {c.query_id: c for c in consultas}

canales = _cargar_canales_vectoriales(Path("base_vectorial"))
canal_grafo = _cargar_canal_grafo(Path("base_vectorial"), None, Path("grafo.graphml"))
texto_por_chunk = _texto_por_chunk(Path("base_vectorial"))

config = __import__("src.embeddings.embedding_config", fromlist=["EmbeddingConfig"]).EmbeddingConfig()
lineas = []
for nombre in config.encoder_names:
    ruta = Path("base_vectorial") / f"encoder_{nombre}" / "metadata.jsonl"
    if ruta.exists():
        with ruta.open(encoding="utf-8") as fh:
            lineas = [json.loads(l) for l in fh if l.strip()]
        break
siguiente_chunk = build_siguiente_chunk_lookup({nombre: lineas})

fusion = RRFusionStrategy(k0=60)
for qid in ["q020", "q028", "q029", "q031"]:
    consulta = por_id[qid]
    try:
        resultado, fusionados = recuperar_hibrido(
            consulta.query_text, canales, canal_grafo, texto_por_chunk,
            siguiente_chunk, fusion, k_search=50, doc_agg="max",
        )
        print(f"{qid}: fragments={len(resultado['fragments'])} documents={len(resultado['documents'])}")
        build_result_object(qid, resultado)
        print(f"{qid}: OK")
    except Exception as exc:
        print(f"{qid}: {type(exc).__name__}: {exc}")