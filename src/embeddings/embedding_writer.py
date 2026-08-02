"""
Persistencia de vectores de embedding a disco (Sección 4): un directorio
por encoder con ``vectors.npy`` (n×d) y ``chunk_ids.jsonl`` (orden ordinal).
Refleja la estructura ``base_vectorial/encoder_<nombre>/`` que consumirá la
construcción del índice FAISS (Sección 5, prompt siguiente).

La consistencia de ``vectors``/``chunk_ids``/``embedding_dim`` ya la valida
``EncoderRunResult`` (Information Expert de su propia forma) al construirse;
este escritor no repite esas reglas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import numpy as np

from src.encoders.orchestrator import EncoderRunResult

logger = logging.getLogger(__name__)


class EmbeddingWriter:
    """Escribe los resultados de un ``EncoderRunResult`` en disco."""

    def __init__(self, output_dir: Union[Path, str]) -> None:
        self.output_dir = Path(output_dir)

    def write(self, resultado: EncoderRunResult) -> Path:
        """Escribe ``vectors.npy`` + ``chunk_ids.jsonl`` + metadata para un encoder."""
        carpeta = self.output_dir / f"encoder_{resultado.encoder_name}"
        carpeta.mkdir(parents=True, exist_ok=True)

        np.save(carpeta / "vectors.npy", resultado.vectors)
        with open(carpeta / "chunk_ids.jsonl", "w", encoding="utf-8") as f:
            for ordinal, chunk_id in enumerate(resultado.chunk_ids):
                f.write(json.dumps({"ordinal": ordinal, "chunk_id": chunk_id}, ensure_ascii=False) + "\n")

        (carpeta / "metadata_criterios.json").write_text(
            json.dumps(resultado.metadata_criterios, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            "Encoder '%s': %d vectores (dim=%d) escritos en %s",
            resultado.encoder_name, len(resultado.chunk_ids), resultado.embedding_dim, carpeta,
        )
        return carpeta
