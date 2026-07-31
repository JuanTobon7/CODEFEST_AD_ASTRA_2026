"""
Script principal de ingesta del corpus (CODEFEST AD ASTRA 2026 — Etapa 1).

Actúa como Controlador (GRASP): solo parsea argumentos, configura el logging
y delega la ejecución en :class:`BatchIngestor`. Toda la lógica de dominio
(escaneo del corpus, asignación de fenómeno, ensamblaje del pipeline, resumen)
vive en los servicios/orquestadores, no aquí.

Uso::

    python -m src.run_ingestion --corpus data/corpus --fenomenos data/fenomenos.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.models.config import Settings
from src.pipeline.batch_ingestor import BatchIngestor
from src.pipeline.corpus_service import CorpusService

logger = logging.getLogger("ingestion")


def _configurar_logging(verbose: bool) -> None:
    """Logging estructurado a consola (INFO/WARNING/ERROR)."""
    nivel = logging.DEBUG if verbose else logging.INFO
    formato = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=nivel, format=formato, stream=sys.stdout, force=True)


def _log_configuracion(settings: Settings) -> None:
    """Registra la configuración activa del pipeline en el log."""
    logger.info(
        "Configuración | mongo=%s/%s.%s | estrategia=%s | chunk=%d overlap=%d min=%d max=%d",
        settings.mongo_uri,
        settings.mongo_db,
        settings.mongo_collection,
        settings.chunking_strategy,
        settings.chunk_size,
        settings.overlap_size,
        settings.min_chunk_tokens,
        settings.max_tokens,
    )


def _parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Ingesta del corpus CODEFEST AD ASTRA 2026 (Etapa 1)"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus"),
        help="Directorio raíz del corpus a ingerir (recorre carpetas anidadas)",
    )
    parser.add_argument(
        "--fenomenos",
        type=Path,
        default=None,
        help="JSON con mapeo carpeta/patrón -> fenómeno (1, 2 o 3)",
    )
    parser.add_argument(
        "--extensiones",
        nargs="*",
        default=None,
        help="Solo procesar estas extensiones, p. ej. pdf md json (opcional)",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=0,
        help="Procesar solo los primeros N archivos (0 = todos)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Logging en nivel DEBUG"
    )
    parser.add_argument(
        "--debug-log",
        nargs="?",
        const="logs/rechazos_debug.txt",
        default=None,
        help=(
            "Escribe en un .txt el detalle de los chunks rechazados (regla de "
            "negocio, motivo y texto cuando un documento rechaza la mayoría). "
            "Sin valor usa logs/rechazos_debug.txt"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Punto de entrada: delega en el orquestador de lote.

    Returns:
        0 si no hubo errores, 1 si el corpus está vacío, 2 si hubo errores.
    """
    args = _parser().parse_args(argv)
    _configurar_logging(args.verbose)
    settings = Settings()
    _log_configuracion(settings)

    mapeo = CorpusService.load_fenomenos_map(args.fenomenos)
    ingestor = BatchIngestor(settings, args.corpus, mapeo)
    try:
        resumen = ingestor.run(
            extensiones=args.extensiones, limite=args.limite, por_defecto=1
        )
        resumen.log_resumen()
        if args.debug_log:
            resumen.escribir_log_rechazos(Path(args.debug_log))
        if resumen.total == 0:
            return 1
        return 0 if resumen.error == 0 else 2
    finally:
        ingestor.close()


if __name__ == "__main__":
    raise SystemExit(main())
