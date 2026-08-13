"""
Script CLI: exporta los chunks persistidos en MongoDB a ``metadata.json``.

``metadata.json`` es la FUENTE DE VERDAD de los fragmentos: contiene los
mismos chunks que había en la colección ``chunks`` de MongoDB, con toda la
metadata de la Tabla 1 (obligatoria + recomendada), y es lo que consumen
después la etapa de embeddings, la exportación FAISS y el grafo de
conocimiento. Una vez generado, MongoDB deja de ser necesario como almacén
de chunks (sigue usándose para vectores y caché de embeddings).

Actúa como Controlador delgado (GRASP): parsea argumentos y delega el
streaming en ``MongoChunkRepository.iter_all`` y la escritura atómica en
``JsonChunkRepository.write_all``.

Uso::

    python -m src.persistence.run_export_metadata
    python -m src.persistence.run_export_metadata --salida metadata.json --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from src.models.chunk import Chunk
from src.models.config import Settings
from src.persistence.json_repository import JsonChunkRepository
from src.persistence.mongo_repository import MongoChunkRepository

logger = logging.getLogger("run_export_metadata")


def _configurar_logging(verbose: bool) -> None:
    """Logging estructurado a consola (INFO/WARNING/ERROR)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _con_progreso(chunks: Iterator[Chunk], cada: int = 25000) -> Iterator[Chunk]:
    """Reemite ``chunks`` registrando el avance cada ``cada`` fragmentos."""
    for indice, chunk in enumerate(chunks, start=1):
        if indice % cada == 0:
            logger.info("Exportados %d fragmentos...", indice)
        yield chunk


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta los chunks de MongoDB a metadata.json (CODEFEST AD ASTRA 2026)"
    )
    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta del JSON de salida (default: CHUNKS_JSON_PATH del .env)",
    )
    parser.add_argument(
        "--solo-obligatorios",
        action="store_true",
        help=(
            "Guarda unicamente los 8 campos obligatorios de la Tabla 1. "
            "Por defecto se incluyen tambien los recomendados (idioma, "
            "hash_texto, fecha_publicacion...), que el post-filtrado y la "
            "cache de embeddings consumen."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Logging en nivel DEBUG")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada: vuelca ``chunks`` de MongoDB al JSON de salida.

    Returns:
        0 si se exportó al menos un chunk, 1 si la colección estaba vacía.
    """
    args = _parser().parse_args(argv)
    _configurar_logging(args.verbose)
    config = Settings()

    destino = Path(args.salida) if args.salida else config.chunks_json_path
    origen = MongoChunkRepository(
        config.mongo_uri,
        config.mongo_db,
        config.mongo_collection,
        username=config.mongo_user,
        password=config.mongo_password,
        auth_source=config.mongo_auth_source,
    )
    repositorio_json = JsonChunkRepository(
        destino, solo_obligatorios=args.solo_obligatorios
    )

    try:
        logger.info(
            "Exportando '%s.%s' -> '%s' (solo_obligatorios=%s)",
            config.mongo_db, config.mongo_collection, destino, args.solo_obligatorios,
        )
        escritos = repositorio_json.write_all(_con_progreso(origen.iter_all()))
    finally:
        origen.close()

    if not escritos:
        logger.warning(
            "No se exportó ningún fragmento: la colección '%s' está vacía.",
            config.mongo_collection,
        )
        return 1

    logger.info(
        "Listo: %d fragmentos en '%s' (%.1f MB)",
        escritos, destino, destino.stat().st_size / (1024 * 1024),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
