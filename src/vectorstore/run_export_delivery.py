"""
Script CLI: exporta el artefacto de entrega final (``index.faiss`` +
``metadata.jsonl``) de cada encoder activo, a partir de los vectores ya
persistidos en MongoDB (Sección 5.4.2).

Actúa como Controlador delgado: parsea argumentos y delega en
``DeliveryExporter`` / ``IndexBuilderFactory``.

Uso::

    python -m src.vectorstore.run_export_delivery
    python -m src.vectorstore.run_export_delivery --encoder bert-multilingual
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

# Registran cada tipo de índice vía @IndexBuilderFactory.register.
from src.vectorstore import flat_ip_strategy, hnsw_strategy, ivf_flat_strategy  # noqa: F401
from src.embeddings.embedding_config import EmbeddingConfig
from src.persistence.base_repository import ChunkRepository
from src.persistence.json_repository import JsonChunkRepository
from src.persistence.mongo_repository import MongoChunkRepository
from src.vectorstore.export_delivery import DeliveryExporter, ExportError
from src.vectorstore.index_builder_base import IndexBuildConfig
from src.vectorstore.index_builder_factory import IndexBuilderFactory
from src.vectorstore.vector_repository import MongoVectorRepository

logger = logging.getLogger("run_export_delivery")


def _crear_repositorio_chunks(config: EmbeddingConfig) -> ChunkRepository:
    """Selecciona la fuente de chunks configurada para la exportacion."""
    tipo = config.chunk_repository.strip().lower()
    if tipo == "json":
        return JsonChunkRepository(config.chunks_json_path)
    if tipo == "mongo":
        return MongoChunkRepository(
            config.mongo_uri, config.mongo_db, config.mongo_collection_chunks,
            username=config.mongo_user, password=config.mongo_password,
            auth_source=config.mongo_auth_source,
        )
    raise ValueError(f"CHUNK_REPOSITORY debe ser 'json' o 'mongo', no '{config.chunk_repository}'")


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
        description="Exporta index.faiss + metadata.jsonl por encoder (CODEFEST AD ASTRA 2026)"
    )
    parser.add_argument(
        "--encoder", action="append", default=None,
        help="Nombre de encoder a exportar (repetible). Por defecto: ACTIVE_ENCODERS del .env",
    )
    parser.add_argument(
        "--parcial", action="store_true",
        help=(
            "Construye el índice con los embeddings disponibles, omitiendo "
            "(con warning) los chunks que aún no tienen vector. La entrega "
            "oficial sigue exigiendo el 100%%."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Logging en nivel DEBUG")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada: exporta el índice de entrega de cada encoder activo.

    Returns:
        0 si todos los encoders se exportaron correctamente, 1 si alguno falló.
    """
    args = _parser().parse_args(argv)
    _configurar_logging(args.verbose)
    config = EmbeddingConfig()
    nombres_encoders = args.encoder or config.encoder_names

    index_cfg = IndexBuildConfig(
        ivf_nlist=config.faiss_ivf_nlist,
        ivf_nprobe=config.faiss_ivf_nprobe,
        hnsw_m=config.faiss_hnsw_m,
        ivf_auto_threshold=config.faiss_ivf_auto_threshold,
    )

    repositorio_chunks = _crear_repositorio_chunks(config)
    repositorio_vectores = MongoVectorRepository(
        config.mongo_uri, config.mongo_db, config.mongo_collection_embeddings,
        username=config.mongo_user, password=config.mongo_password, auth_source=config.mongo_auth_source,
    )
    exportador = DeliveryExporter(repositorio_vectores, config.delivery_output_dir)

    hubo_error = False
    try:
        chunks_activos = repositorio_chunks.find_all()
        if not chunks_activos:
            logger.warning("No hay chunks en la colección '%s'", config.mongo_collection_chunks)
            return 1

        for nombre in nombres_encoders:
            n_vectores = repositorio_vectores.count_by_encoder(nombre)
            estrategia_indice = IndexBuilderFactory.resolve(config.faiss_index_type, n_vectores, index_cfg)
            # embedding_dim se infiere del primer vector persistido de este encoder.
            primer_registro = next(repositorio_vectores.find_by_encoder(nombre), None)
            if primer_registro is None:
                logger.error("Encoder '%s': no hay vectores persistidos, se omite la exportación", nombre)
                hubo_error = True
                continue
            try:
                exportador.export_encoder(
                    nombre, chunks_activos, estrategia_indice, primer_registro.embedding_dim, index_cfg,
                    permitir_faltantes=args.parcial,
                )
            except ExportError as exc:
                logger.error("Encoder '%s': fallo en la exportación: %s", nombre, exc)
                hubo_error = True

        return 1 if hubo_error else 0
    finally:
        repositorio_chunks.close()
        repositorio_vectores.close()


if __name__ == "__main__":
    raise SystemExit(main())
