"""
E5 multilingüe small (``intfloat/multilingual-e5-small``): variante
reducida (384d) del mismo checkpoint afinado con *contrastive learning*
para embeddings de oración, ya empaquetado como modelo
``sentence-transformers``. Cubre el criterio de granularidad dimensional
baja (eficiencia) con benchmark de recuperación densa publicado.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("e5-multilingual-small")
class E5MultilingualSmallStrategy(EncoderStrategy):
    """MIT, ~100 idiomas (incl. es/en/pt), 384d, máx. 512 tokens
    (model card intfloat/multilingual-e5-small; Wang et al. 2024)."""

    model_id = "intfloat/multilingual-e5-small"
    embedding_dim = 384
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "mit"
    requires_prefix = {"query": "query: ", "passage": "passage: "}
    mteb_retrieval_score = 60.8
    benchmark_reference = (
        "MIRACL dev set, promedio de 16 idiomas, nDCG@10 "
        "(Wang et al., 'Multilingual E5 Text Embeddings: A Technical Report', 2024, Tabla 4)"
    )

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device=device)
