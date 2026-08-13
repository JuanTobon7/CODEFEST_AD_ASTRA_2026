"""
Orquestador multi-encoder (Sección 4.4): corre 1..N ``EncoderStrategy`` sobre
el mismo lote de chunks. Cada encoder produce su propio espacio vectorial de
forma independiente y trazable por ``chunk_id`` + ``encoder_name``, dejando
trivial que el módulo de recuperación (Sección 8, prompt futuro) fusione
después los rankings (RRF/CombSUM/CombMNZ) sin recomputar nada.

Las reglas de negocio del límite de tokens (conteo con el tokenizador propio
del encoder, truncado por oraciones o exclusión) viven como Information
Expert en :class:`EncoderStrategy.ajustar_a_limite` — este orquestador solo
coordina el lote, no decide cómo se trunca cada texto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.encoders.base import EncoderStrategy
from src.models.chunk import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EncoderRunResult:
    """Resultado de correr un encoder sobre un lote de chunks.

    Es el Information Expert de su propia consistencia interna: se
    autovalida al construirse (nadie más necesita repetir estas reglas).
    """

    encoder_name: str
    model_id: str
    embedding_dim: int
    vectors: np.ndarray
    chunk_ids: List[str]
    metadata_criterios: Dict[str, object] = field(default_factory=dict)
    chunks_excluidos: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"'{self.encoder_name}': se esperaban vectores 2D, forma={self.vectors.shape}")
        if self.vectors.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"'{self.encoder_name}': {self.vectors.shape[0]} vectores != {len(self.chunk_ids)} chunk_ids"
            )
        if self.vectors.size and self.vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"'{self.encoder_name}': dimensión de vectores {self.vectors.shape[1]} "
                f"!= embedding_dim declarado {self.embedding_dim}"
            )


class EncoderOrchestrator:
    """Ejecuta cada ``EncoderStrategy`` de forma independiente sobre los chunks."""

    def __init__(self, strategies: List[EncoderStrategy], batch_size: int = 32) -> None:
        if not strategies:
            raise ValueError("El orquestador requiere al menos un EncoderStrategy")
        self.strategies = strategies
        self.batch_size = batch_size

    def run(self, chunks: List[Chunk]) -> Dict[str, EncoderRunResult]:
        """Corre cada estrategia registrada sobre ``chunks``, en secuencia."""
        resultados: Dict[str, EncoderRunResult] = {}
        for estrategia in self.strategies:
            resultados[estrategia.name] = self._run_una_estrategia(estrategia, chunks)
        return resultados

    def run_encoder(self, encoder_name: str, chunks: List[Chunk]) -> EncoderRunResult:
        """Corre SOLO la estrategia ``encoder_name`` sobre ``chunks``.

        Cada encoder tiene su propia caché, así que el lote pendiente es
        distinto para cada uno: pedir el lote de un encoder a :meth:`run`
        codificaría también con los demás y tiraría el resultado (N veces el
        trabajo con N encoders activos).

        Raises:
            ValueError: si ``encoder_name`` no está entre las estrategias.
        """
        for estrategia in self.strategies:
            if estrategia.name == encoder_name:
                return self._run_una_estrategia(estrategia, chunks)
        raise ValueError(
            f"Encoder '{encoder_name}' no registrado en el orquestador. "
            f"Disponibles: {', '.join(e.name for e in self.strategies)}"
        )

    def _run_una_estrategia(self, estrategia: EncoderStrategy, chunks: List[Chunk]) -> EncoderRunResult:
        textos, chunk_ids, excluidos = self._preparar_lote(estrategia, chunks)
        if not textos:
            logger.warning("Encoder '%s': ningún chunk codificable en el lote", estrategia.name)
            vectores = np.empty((0, estrategia.embedding_dim), dtype=np.float32)
        else:
            # Sin batch_size explícito: usa estrategia.config.batch_size (permite
            # un valor distinto por encoder, p. ej. EMBEDDING_BATCH_SIZE_OVERRIDES).
            vectores = estrategia.encode(textos, is_query=False)

        logger.info(
            "Encoder '%s': %d vectores generados (dim=%d, excluidos=%d)",
            estrategia.name, len(chunk_ids), estrategia.embedding_dim, len(excluidos),
        )
        return EncoderRunResult(
            encoder_name=estrategia.name,
            model_id=estrategia.model_id,
            embedding_dim=estrategia.embedding_dim,
            vectors=vectores,
            chunk_ids=chunk_ids,
            metadata_criterios=estrategia.to_metadata(),
            chunks_excluidos=excluidos,
        )

    def _preparar_lote(
        self, estrategia: EncoderStrategy, chunks: List[Chunk]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Delega en la estrategia (Information Expert) el ajuste al límite de tokens."""
        estrategia.load()
        textos: List[str] = []
        chunk_ids: List[str] = []
        excluidos: List[str] = []
        for chunk in chunks:
            texto_ajustado = estrategia.ajustar_a_limite(chunk.texto)
            if texto_ajustado is None:
                logger.warning(
                    "Chunk '%s' excluido del encoder '%s': excede max_input_tokens=%d "
                    "y no se pudo truncar preservando completitud lingüística.",
                    chunk.chunk_id, estrategia.name, estrategia.max_input_tokens,
                )
                excluidos.append(chunk.chunk_id)
                continue
            if texto_ajustado != chunk.texto:
                logger.warning(
                    "Chunk '%s' truncado para el encoder '%s' (max_input_tokens=%d).",
                    chunk.chunk_id, estrategia.name, estrategia.max_input_tokens,
                )
            textos.append(texto_ajustado)
            chunk_ids.append(chunk.chunk_id)
        return textos, chunk_ids, excluidos

