"""
``FaissIndexManager`` — modo operativo incremental (Sección 5.4.1): permite
reconstruir/actualizar el índice de trabajo de un encoder sin recalcular
todo el corpus, pensado para desarrollo/pruebas iterativas.

Usa ``faiss.IndexIDMap`` para asignar IDs propios y estables (derivados
determinísticamente del ``chunk_id``, no secuenciales por inserción), de
forma que un chunk reprocesado (texto cambiado) pueda reemplazarse con
``remove_ids`` + ``add_with_ids`` sin reconstruir el índice completo.

El índice de trabajo se persiste en ``WORKING_INDEX_DIR`` (separado de
``DELIVERY_OUTPUT_DIR``, que es el artefacto de entrega final generado por
``export_delivery``).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np

from src.vectorstore.index_builder_base import FaissIndexBuilderStrategy, IndexBuildConfig
from src.vectorstore.vector_repository import VectorRepository

logger = logging.getLogger(__name__)


@dataclass
class IndexBuildResult:
    """Resultado de una actualización incremental del índice de trabajo."""

    encoder_name: str
    index_type: str
    n_vectores_actualizados: int
    n_total_en_indice: int
    working_index_path: str


class FaissIndexManager:
    """Construye/actualiza el índice de trabajo (``IndexIDMap``) de un encoder."""

    def __init__(self, vector_repository: VectorRepository, working_index_dir: Union[Path, str]) -> None:
        self.vector_repository = vector_repository
        self.working_index_dir = Path(working_index_dir)

    @staticmethod
    def faiss_id_de(chunk_id: str) -> int:
        """ID FAISS determinístico y estable derivado de ``chunk_id`` (int64)."""
        return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)

    def _ruta_indice(self, encoder_name: str) -> Path:
        return self.working_index_dir / f"encoder_{encoder_name}" / "index.faiss"

    def load(self, encoder_name: str) -> faiss.Index:
        """Carga el índice de trabajo persistido de ``encoder_name``."""
        ruta = self._ruta_indice(encoder_name)
        if not ruta.exists():
            raise FileNotFoundError(f"No hay índice de trabajo para '{encoder_name}' en {ruta}")
        return faiss.read_index(str(ruta))

    def build_or_update(
        self,
        encoder_name: str,
        chunk_ids: List[str],
        index_strategy: FaissIndexBuilderStrategy,
        embedding_dim: int,
        config: IndexBuildConfig,
    ) -> IndexBuildResult:
        """Agrega/reemplaza en el índice de trabajo solo los ``chunk_ids`` dados.

        1. Lee (o crea) el índice de trabajo envuelto en ``IndexIDMap``.
        2. Recupera los vectores actuales de ``chunk_ids`` desde ``VectorRepository``.
        3. Calcula el ``faiss_id`` determinístico de cada uno.
        4. ``remove_ids`` (si ya existían, p. ej. texto reprocesado) + ``add_with_ids``.
        5. Persiste el índice a disco y registra ``faiss_internal_id`` en Mongo.
        """
        ruta = self._ruta_indice(encoder_name)
        if ruta.exists():
            indice = faiss.read_index(str(ruta))
        else:
            indice = faiss.IndexIDMap(index_strategy.build(embedding_dim, config))

        registros = self.vector_repository.find_by_chunk_ids(encoder_name, chunk_ids)
        if not registros:
            logger.warning("Encoder '%s': ningún vector encontrado para los chunk_ids dados", encoder_name)
            return IndexBuildResult(encoder_name, index_strategy.index_type_name, 0, indice.ntotal, str(ruta))

        vectores = np.vstack([np.asarray(r.vector, dtype=np.float32) for r in registros])
        ids = np.array([self.faiss_id_de(r.chunk_id) for r in registros], dtype=np.int64)

        indice_base = indice.index if isinstance(indice, faiss.IndexIDMap) else indice
        index_strategy.train_if_needed(indice_base, vectores)

        try:
            indice.remove_ids(ids)
        except RuntimeError:
            pass  # el índice base no soporta remove_ids (p. ej. HNSW): se ignora en la 1ª carga

        indice.add_with_ids(vectores, ids)

        ruta.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(indice, str(ruta))

        for registro, faiss_id in zip(registros, ids):
            self.vector_repository.set_faiss_internal_id(registro.chunk_id, encoder_name, int(faiss_id))

        logger.info(
            "Encoder '%s': índice de trabajo actualizado (%d vectores, total=%d) -> %s",
            encoder_name, len(registros), indice.ntotal, ruta,
        )
        return IndexBuildResult(encoder_name, index_strategy.index_type_name, len(registros), indice.ntotal, str(ruta))
