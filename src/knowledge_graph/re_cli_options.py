"""Opciones de línea de comandos compartidas por los CLI del grafo.

Los tres puntos de entrada del grafo (``run_build_graph``,
``run_hybrid_retrieval`` y ``run_generador_hibrido``) eligen la MISMA
estrategia de extracción de relaciones con los mismos parámetros. Centralizar
aquí el parser evita que las tres copias diverjan (DRY) y garantiza que un
grafo construido con un CLI sea reproducible desde cualquier otro.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from src.knowledge_graph.extract.nli_config import (
    MAX_PARES_POR_CHUNK_DEFAULT,
    MAX_PARES_POR_ORACION_DEFAULT,
    VARIANTES_PLANTILLA_DEFAULT,
)

#: Estrategias RE disponibles. Ambas cumplen la restricción del reto (ningún
#: modelo generativo) y tienen licencia permisiva:
#: - ``nli-zero-shot``: encoder de clasificación mDeBERTa-v3 (MIT).
#: - ``coocurrencia-oracional``: simbólica, sin checkpoint de terceros.
ESTRATEGIAS_RE = ("nli-zero-shot", "coocurrencia-oracional")

#: Default de los CLI: el modelo de clasificación de relaciones (Sección 7.2).
ESTRATEGIA_RE_DEFAULT = "nli-zero-shot"


def agregar_opciones_re(parser: argparse.ArgumentParser) -> None:
    """Añade a ``parser`` las opciones de la estrategia de relaciones."""
    parser.add_argument(
        "--re",
        default=ESTRATEGIA_RE_DEFAULT,
        choices=list(ESTRATEGIAS_RE),
        help=(
            "Estrategia de extracción de relaciones "
            f"(default: {ESTRATEGIA_RE_DEFAULT}, clasificador encoder mDeBERTa-v3 MIT)"
        ),
    )
    parser.add_argument(
        "--variantes-plantilla",
        type=int,
        default=VARIANTES_PLANTILLA_DEFAULT,
        help=(
            "Formulaciones evaluadas por tipo de relación (0 = todas). "
            "El coste del canal es lineal en este número. Solo con --re nli-zero-shot"
        ),
    )
    parser.add_argument(
        "--max-pares-oracion",
        type=int,
        default=MAX_PARES_POR_ORACION_DEFAULT,
        help="Tope de pares de entidades clasificados por oración (solo --re nli-zero-shot)",
    )
    parser.add_argument(
        "--max-pares-chunk",
        type=int,
        default=MAX_PARES_POR_CHUNK_DEFAULT,
        help="Tope de pares de entidades clasificados por chunk (solo --re nli-zero-shot)",
    )
    parser.add_argument(
        "--re-batch-size",
        type=int,
        default=64,
        help="Tamaño de lote del forward del clasificador (solo --re nli-zero-shot)",
    )
    parser.add_argument(
        "--sin-fp16",
        action="store_true",
        help="Desactiva la media precisión en GPU (solo --re nli-zero-shot)",
    )


def kwargs_re(args: argparse.Namespace) -> Dict[str, Any]:
    """Traduce los argumentos parseados a ``kwargs`` de la estrategia RE.

    La estrategia simbólica no acepta parámetros, así que devuelve ``{}``
    salvo para ``nli-zero-shot``.
    """
    if args.re != "nli-zero-shot":
        return {}
    return {
        "variantes_plantilla": args.variantes_plantilla,
        "max_pares_por_oracion": args.max_pares_oracion,
        "max_pares_por_chunk": args.max_pares_chunk,
        "batch_size": args.re_batch_size,
        "use_fp16": not args.sin_fp16,
    }
