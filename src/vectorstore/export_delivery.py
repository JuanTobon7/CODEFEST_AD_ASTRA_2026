"""
Modo de exportación para entrega (Sección 5.4.2, estricto): genera el
artefacto final ``base_vectorial/encoder_<nombre>/{index.faiss,metadata.jsonl}``
exigido por la especificación del reto.

A diferencia del modo operativo (``FaissIndexManager``, con ``IndexIDMap``
e IDs determinísticos), aquí los vectores se insertan con ``index.add()``
secuencial: el orden de líneas de ``metadata.jsonl`` es exactamente el
orden de inserción (0, 1, 2, ..., n-1), que es el identificador interno que
FAISS asigna en un índice sin ``IndexIDMap``. Esto es lo que exige
literalmente la Sección 5.3 de la especificación.

Es determinístico: correr ``export_encoder`` dos veces sobre los mismos
datos de MongoDB debe producir un ``index.faiss``/``metadata.jsonl``
idénticos (mismo orden — ``ORDER BY doc_id, posicion``, nunca ``_id`` de
Mongo, que no es reproducible entre corridas).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

import faiss
import numpy as np

from src.models.chunk import Chunk
from src.vectorstore.index_builder_base import FaissIndexBuilderStrategy, IndexBuildConfig
from src.vectorstore.models import EmbeddingRecord
from src.vectorstore.vector_repository import VectorRepository

logger = logging.getLogger(__name__)


class ExportError(RuntimeError):
    """El corpus de un encoder no está completo o consistente para exportar."""


@dataclass
class ExportResult:
    """Resultado de exportar el índice de entrega de un encoder."""

    encoder_name: str
    index_type: str
    n_vectores: int
    index_path: str
    metadata_path: str
    checksum_sha256: str


class DeliveryExporter:
    """Construye el artefacto de entrega (índice + metadata) de cada encoder."""

    def __init__(self, vector_repository: VectorRepository, output_dir: Union[Path, str]) -> None:
        self.vector_repository = vector_repository
        self.output_dir = Path(output_dir)

    @staticmethod
    def _ordenar_chunk_ids(chunks: List[Chunk]) -> List[str]:
        """Orden determinístico y estable: ``(doc_id, posicion)``, nunca ``_id`` de Mongo."""
        return [c.chunk_id for c in sorted(chunks, key=lambda c: (c.doc_id, c.posicion))]

    @staticmethod
    def _chunk_ids_faltantes(chunk_ids_totales: List[str], registros_por_id: Dict[str, EmbeddingRecord]) -> List[str]:
        """``chunk_id`` activos que no tienen embedding calculado para este encoder."""
        return [cid for cid in chunk_ids_totales if cid not in registros_por_id]

    @staticmethod
    def _validar_dimension_uniforme(
        registros_por_id: Dict[str, EmbeddingRecord], embedding_dim: int, encoder_name: str
    ) -> None:
        """Aborta si algún vector no coincide con ``embedding_dim`` (mezcla de modelos)."""
        for chunk_id, registro in registros_por_id.items():
            dim_real = int(np.asarray(registro.vector).shape[-1])
            if dim_real != embedding_dim:
                raise ExportError(
                    f"Encoder '{encoder_name}': vector de '{chunk_id}' tiene dim={dim_real}, "
                    f"se esperaba {embedding_dim}. Posible mezcla accidental de modelos."
                )

    @staticmethod
    def _checksum_archivo(ruta: Path) -> str:
        """SHA-256 del archivo, para verificar reproducibilidad entre corridas."""
        return hashlib.sha256(ruta.read_bytes()).hexdigest()

    def export_encoder(
        self,
        encoder_name: str,
        chunks_activos: List[Chunk],
        index_strategy: FaissIndexBuilderStrategy,
        embedding_dim: int,
        config: IndexBuildConfig,
    ) -> ExportResult:
        """Exporta el índice + metadata de entrega de ``encoder_name``.

        Raises:
            ExportError: si faltan embeddings para algún chunk activo, o si
                hay vectores con dimensión inconsistente.
        """
        chunk_ids_totales = self._ordenar_chunk_ids(chunks_activos)
        registros_por_id = {r.chunk_id: r for r in self.vector_repository.find_by_encoder(encoder_name)}

        faltantes = self._chunk_ids_faltantes(chunk_ids_totales, registros_por_id)
        if faltantes:
            raise ExportError(
                f"Encoder '{encoder_name}': faltan embeddings para {len(faltantes)} chunk(s): "
                f"{faltantes[:20]}{'...' if len(faltantes) > 20 else ''}"
            )
        self._validar_dimension_uniforme(registros_por_id, embedding_dim, encoder_name)

        vectores = np.vstack(
            [np.asarray(registros_por_id[cid].vector, dtype=np.float32) for cid in chunk_ids_totales]
        )

        indice = index_strategy.build(embedding_dim, config)
        index_strategy.train_if_needed(indice, vectores)
        indice.add(vectores)

        carpeta = self.output_dir / f"encoder_{encoder_name}"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_index = carpeta / "index.faiss"
        faiss.write_index(indice, str(ruta_index))

        chunks_por_id = {c.chunk_id: c for c in chunks_activos}
        ruta_metadata = carpeta / "metadata.jsonl"
        with open(ruta_metadata, "w", encoding="utf-8") as f:
            for chunk_id in chunk_ids_totales:
                chunk = chunks_por_id[chunk_id]
                f.write(
                    json.dumps(
                        {
                            "doc_id": chunk.doc_id,
                            "chunk_id": chunk.chunk_id,
                            "fuente": chunk.fuente,
                            "formato": chunk.formato,
                            "fenomeno": chunk.fenomeno,
                            "posicion": chunk.posicion,
                            "num_tokens": chunk.num_tokens,
                            "texto": chunk.texto,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self._verificar_integridad(ruta_index, ruta_metadata, embedding_dim, len(chunk_ids_totales))
        checksum = self._checksum_archivo(ruta_index)
        self._registrar_build_log(carpeta, encoder_name, index_strategy.index_type_name, len(chunk_ids_totales), checksum)

        logger.info(
            "Encoder '%s': exportado index.faiss + metadata.jsonl (%d vectores) en %s",
            encoder_name, len(chunk_ids_totales), carpeta,
        )
        return ExportResult(
            encoder_name=encoder_name,
            index_type=index_strategy.index_type_name,
            n_vectores=len(chunk_ids_totales),
            index_path=str(ruta_index),
            metadata_path=str(ruta_metadata),
            checksum_sha256=checksum,
        )

    @staticmethod
    def _verificar_integridad(ruta_index: Path, ruta_metadata: Path, embedding_dim: int, n_esperado: int) -> None:
        """Test de humo: el índice recién escrito debe cargar y cuadrar con la metadata."""
        indice_leido = faiss.read_index(str(ruta_index))
        if indice_leido.d != embedding_dim:
            raise ExportError(f"index.faiss recién escrito tiene d={indice_leido.d}, se esperaba {embedding_dim}")
        if indice_leido.ntotal != n_esperado:
            raise ExportError(f"index.faiss tiene ntotal={indice_leido.ntotal}, se esperaban {n_esperado} vectores")
        with open(ruta_metadata, "r", encoding="utf-8") as f:
            n_lineas = sum(1 for _ in f)
        if n_lineas != n_esperado:
            raise ExportError(f"metadata.jsonl tiene {n_lineas} líneas, se esperaban {n_esperado}")

    @staticmethod
    def _registrar_build_log(carpeta: Path, encoder_name: str, index_type: str, n_vectores: int, checksum: str) -> None:
        """Añade una entrada al log de builds (trazabilidad/reproducibilidad)."""
        entrada = {
            "encoder_name": encoder_name,
            "index_type": index_type,
            "n_vectores": n_vectores,
            "checksum_sha256": checksum,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(carpeta / "build_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
