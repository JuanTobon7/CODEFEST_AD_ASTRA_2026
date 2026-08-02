"""
Script CLI: lee chunks pendientes desde MongoDB, corre el/los encoder(s)
activos según ``.env`` y persiste los vectores (Sección 4, continuación del
pipeline de chunking en MongoDB).

Actúa como Controlador delgado (GRASP): solo parsea argumentos y delega en
``EncoderOrchestrator`` / ``EmbeddingCache`` / ``EmbeddingWriter``.

Uso::

    python -m src.embeddings.run_embedding --limite 500
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from typing import List, Optional

# Importar el paquete de estrategias concretas registra cada una vía
# @EncoderFactory.register (efecto secundario necesario antes de crear()).
from src.encoders import (  # noqa: F401
    bge_m3_strategy,
    labse_strategy,
    minilm_light_strategy,
    multilingual_e5_strategy,
)
from src.embeddings.embedding_cache import EmbeddingCache
from src.embeddings.embedding_config import EmbeddingConfig
from src.embeddings.embedding_writer import EmbeddingWriter
from src.encoders.base import EncoderConfig
from src.encoders.factory import EncoderFactory
from src.encoders.orchestrator import EncoderOrchestrator
from src.models.chunk import Chunk
from src.persistence.mongo_repository import MongoChunkRepository

logger = logging.getLogger("run_embedding")


def _configurar_logging(verbose: bool) -> None:
    """Logging estructurado a consola (INFO/WARNING/ERROR)."""
    nivel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _hash_texto(chunk: Chunk) -> str:
    """``hash_texto`` del chunk si ya existe (etapa de chunking); si no, se deriva."""
    return chunk.hash_texto or hashlib.sha256(chunk.texto.encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Codificación semántica de chunks (CODEFEST AD ASTRA 2026)"
    )
    parser.add_argument(
        "--limite", type=int, default=0, help="Procesar solo los primeros N chunks (0 = todos)"
    )
    parser.add_argument("--verbose", action="store_true", help="Logging en nivel DEBUG")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada: lee chunks, corre encoders activos, persiste resultados.

    Returns:
        0 si todo salió bien, 1 si no hay chunks en la colección.
    """
    args = _parser().parse_args(argv)
    _configurar_logging(args.verbose)
    config = EmbeddingConfig()

    logger.info(
        "Encoders activos=%s | batch_size=%d | device=%s | output_dir=%s",
        config.encoder_names, config.embedding_batch_size, config.embedding_device,
        config.embedding_output_dir,
    )

    device_preference = None if config.embedding_device == "auto" else config.embedding_device
    encoder_cfg = EncoderConfig(batch_size=config.embedding_batch_size, device_preference=device_preference)
    estrategias = [EncoderFactory.create(nombre, encoder_cfg) for nombre in config.encoder_names]
    orquestador = EncoderOrchestrator(estrategias, batch_size=config.embedding_batch_size)

    repositorio_chunks = MongoChunkRepository(
        config.mongo_uri, config.mongo_db, config.mongo_collection_chunks,
        username=config.mongo_user, password=config.mongo_password, auth_source=config.mongo_auth_source,
    )
    cache = EmbeddingCache(
        config.mongo_uri, config.mongo_db, config.mongo_collection_embeddings_cache,
        username=config.mongo_user, password=config.mongo_password, auth_source=config.mongo_auth_source,
    )
    escritor = EmbeddingWriter(config.embedding_output_dir)

    try:
        chunks = repositorio_chunks.find_all(args.limite)
        if not chunks:
            logger.warning("No hay chunks en la colección '%s'", config.mongo_collection_chunks)
            return 1
        chunk_id_to_hash = {c.chunk_id: _hash_texto(c) for c in chunks}

        for estrategia in estrategias:
            nombre = estrategia.name
            pendientes_ids = set(cache.pendientes(nombre, chunk_id_to_hash))
            chunks_a_codificar = [c for c in chunks if c.chunk_id in pendientes_ids]
            if not chunks_a_codificar:
                logger.info("Encoder '%s': todos los chunks ya estaban en cache, se omite.", nombre)
                continue

            resultado = orquestador.run(chunks_a_codificar)[nombre]
            if resultado.chunk_ids:
                escritor.write(resultado)
                cache.marcar_procesados(nombre, {cid: chunk_id_to_hash[cid] for cid in resultado.chunk_ids})
                for chunk_id in resultado.chunk_ids:
                    repositorio_chunks.mark_encoder_procesado(chunk_id, nombre)

        return 0
    finally:
        repositorio_chunks.close()
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
