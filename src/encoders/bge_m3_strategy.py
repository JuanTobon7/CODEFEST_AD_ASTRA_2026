"""
BGE-M3 (``BAAI/bge-m3``): multi-idioma, multi-granularidad, hasta 8192
tokens. Útil cuando el chunking produce fragmentos largos (chunks > 512
tokens de otros encoders) sin necesidad de truncar.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("bge-m3")
class BGEM3Strategy(EncoderStrategy):
    """MIT, 100+ idiomas, 1024d, máx. 8192 tokens (model card BAAI/bge-m3)."""

    model_id = "BAAI/bge-m3"
    embedding_dim = 1024
    max_input_tokens = 8192
    supported_languages = ["es", "en", "pt"]  # + 100 idiomas adicionales (model card)
    license = "mit"
    mteb_retrieval_score = 48.8  # MTEB-Retrieval avg reportado en el paper/model card BGE-M3
    benchmark_reference = "MTEB Retrieval + MIRACL (model card/paper BAAI/bge-m3)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device=device)
