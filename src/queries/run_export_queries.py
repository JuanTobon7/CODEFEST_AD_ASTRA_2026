"""
Script CLI: extrae las 50 consultas (q001..q050) del PDF oficial de
evaluación reutilizando el extractor del pipeline (``PDFExtractor`` vía
``ExtractorFactory``) y las escribe en ``consultas.jsonl``.

Controlador delgado (GRASP): la lógica vive en ``QueryLoader.exportar_desde_pdf``
(``src/queries/loader.py``), la misma que usa ``generador.py`` cuando el
archivo de consultas no existe. Solo escribe si el archivo no existe
(idempotente); con ``--force`` lo regenera.

Formato de salida (el que consume ``generador.py``)::

    {"query_id": "q001", "query_text": "¿Cómo está transformando ..."}

Uso::

    python -m src.queries.run_export_queries
    python -m src.queries.run_export_queries --pdf consultas.pdf --output consultas.jsonl --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.queries.loader import QueryLoader

logger = logging.getLogger("run_export_queries")


def _configurar_logging(verbose: bool) -> None:
    """Logging estructurado a consola (INFO/WARNING/ERROR)."""
    nivel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae las 50 consultas de evaluación desde PDF a consultas.jsonl"
    )
    parser.add_argument(
        "--pdf",
        default="Extracto_Preguntas_50_v2.pdf",
        help="PDF oficial con las 50 consultas (q001..q050)",
    )
    parser.add_argument(
        "--output",
        default="consultas.jsonl",
        help="Archivo JSONL de salida",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera el archivo aunque ya exista (default: no lo sobrescribe)",
    )
    parser.add_argument("--verbose", action="store_true", help="Logging en nivel DEBUG")
    return parser


def main(argv: Optional[list] = None) -> None:
    """Extrae las consultas del PDF y las escribe en ``consultas.jsonl``.

    Si el archivo ya existe y no se pasó ``--force``, no se sobrescribe.
    """
    args = _parser().parse_args(argv)
    _configurar_logging(args.verbose)

    ruta = Path(args.output)
    if ruta.exists() and not args.force:
        logger.info("'%s' ya existe; se omite (usar --force para regenerar).", ruta)
        return

    consultas = QueryLoader.exportar_desde_pdf(args.pdf, str(ruta))
    logger.info("Escritas %d consultas en '%s'", len(consultas), ruta)


if __name__ == "__main__":
    main()

