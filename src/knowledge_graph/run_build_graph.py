"""CLI: construye el grafo de conocimiento desde los artefactos de entrega.

Lee los chunks de ``base_vectorial/encoder_<n>/metadata.jsonl`` (el MISMO
origen que alimenta el índice FAISS, derivado a su vez de ``metadata.json``),
aplica NER → RE, construye el grafo y exporta ``grafo.graphml`` (Sección 7.3).

Modelos usados, ambos con licencia permisiva y NINGUNO generativo:
- NODOS (NER): ``regex-gazetteer``, simbólico y determinista — es código del
  proyecto, sin checkpoint de terceros ni licencia que arrastrar.
- ARISTAS (RE): ``nli-zero-shot`` con ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli``
  (**MIT**), un encoder con cabecera de clasificación de 3 etiquetas. Elige
  un tipo de relación de un vocabulario cerrado; no genera texto.

Uso:
    python -m src.knowledge_graph.run_build_graph
    python -m src.knowledge_graph.run_build_graph --encoders-dir base_vectorial \
        --output grafo.graphml --limite 5000 --encoder bert-multilingual
    python -m src.knowledge_graph.run_build_graph --re coocurrencia-oracional
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List

from src.embeddings.embedding_config import EmbeddingConfig
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.graph_cost import estimar
from src.knowledge_graph.re_cli_options import agregar_opciones_re, kwargs_re
from src.knowledge_graph.service import KnowledgeGraphService

# Importar la estrategia RE registra su decorador en RelationExtractorFactory
# (nli-zero-shot es el default del CLI; el service por defecto sigue siendo
# coocurrencia-oracional, puramente simbólico).
import src.knowledge_graph.extract.nli_relation_strategy  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_build_graph")


class ChunkMetadata:
    """Adaptador mínimo de una línea de metadata.jsonl a ChunkFuente."""

    __slots__ = ("doc_id", "chunk_id", "texto")

    def __init__(self, linea: dict) -> None:
        self.doc_id = str(linea.get("doc_id", ""))
        self.chunk_id = str(linea.get("chunk_id", ""))
        self.texto = str(linea.get("texto") or linea.get("text", ""))


def _leer_chunks(encoders_dir: Path, encoder: str | None, limite: int | None) -> Iterable[ChunkMetadata]:
    """Itera las líneas de metadata.jsonl del encoder elegido (determinista)."""
    config = EmbeddingConfig()
    nombres = [encoder] if encoder else list(config.encoder_names)
    ruta = None
    for nombre in nombres:
        candidata = encoders_dir / f"encoder_{nombre}" / "metadata.jsonl"
        if candidata.exists():
            ruta = candidata
            logger.info("Usando artefactos de '%s' (%s)", nombre, ruta)
            break
    if ruta is None:
        raise FileNotFoundError(
            f"No hay metadata.jsonl en '{encoders_dir}' para {nombres}. "
            "Ejecute primero 'python -m src.vectorstore.run_export_delivery'."
        )

    contados = 0
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            yield ChunkMetadata(json.loads(linea))
            contados += 1
            if limite is not None and contados >= limite:
                break


def _contar_chunks(encoders_dir: Path, encoder: str | None) -> int:
    """Número de chunks del artefacto (denominador de la extrapolación)."""
    return sum(1 for _ in _leer_chunks(encoders_dir, encoder, None))


def _verificar_graphml(ruta: Path) -> None:
    """Valida que el GraphML se carga con NetworkX sin dependencias extra."""
    import networkx as nx

    grafo = nx.read_graphml(str(ruta))
    logger.info(
        "Verificación NetworkX: %d nodos, %d aristas en '%s'",
        grafo.number_of_nodes(),
        grafo.number_of_edges(),
        ruta,
    )


def _estimar(args) -> int:
    """Mide el coste sobre una muestra y proyecta al corpus (``--estimar N``)."""
    encoders_dir = Path(args.encoders_dir)
    try:
        muestra = list(_leer_chunks(encoders_dir, args.encoder, args.estimar))
        totales = _contar_chunks(encoders_dir, args.encoder)
    except FileNotFoundError as exc:
        logger.exception("%s", exc)
        return 1
    if not muestra:
        logger.error("El artefacto no tiene chunks que muestrear.")
        return 1

    estimacion = estimar(
        muestra,
        totales,
        RelationExtractorFactory.create(args.re, **kwargs_re(args)),
    )
    logger.info("Estrategia RE: %s | %s", args.re, estimacion.como_texto())
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye el grafo de conocimiento (Sección 7)")
    parser.add_argument("--encoders-dir", default="base_vectorial", help="Directorio de artefactos de entrega")
    parser.add_argument("--output", default="grafo.graphml", help="Archivo GraphML de salida")
    parser.add_argument("--encoder", default=None, help="Nombre del encoder cuyo metadata leer (default: el primero activo)")
    parser.add_argument("--limite", type=int, default=None, help="Máximo de chunks a procesar (default: todos)")
    parser.add_argument(
        "--estimar",
        type=int,
        metavar="N",
        default=None,
        help=(
            "No construye el grafo: cronometra N chunks reales con la misma "
            "configuración y proyecta el coste del corpus completo en ESTE "
            "hardware. Úsalo antes de comprometer horas de GPU (p. ej. --estimar 200)"
        ),
    )
    agregar_opciones_re(parser)
    args = parser.parse_args(argv)

    if args.estimar:
        return _estimar(args)

    try:
        chunks = _leer_chunks(Path(args.encoders_dir), args.encoder, args.limite)
        servicio = KnowledgeGraphService(
            relation_extractor=RelationExtractorFactory.create(args.re, **kwargs_re(args))
        )
        grafo = servicio.construir_desde_chunks(chunks)
        ruta = servicio.exportar_graphml(args.output)
    except FileNotFoundError as exc:
        logger.exception("%s", exc)
        return 1

    logger.info(
        "Grafo construido: %d entidades, %d tripletas",
        grafo.num_entidades,
        grafo.num_tripletas,
    )
    logger.info("Resumen del servicio: %s", servicio.resumen())
    _verificar_graphml(ruta)
    logger.info("Exportado: %s", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
