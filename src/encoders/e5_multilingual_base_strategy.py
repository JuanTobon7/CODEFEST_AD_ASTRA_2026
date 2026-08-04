"""
E5 multilingüe base (``intfloat/multilingual-e5-base``): checkpoint
XLM-RoBERTa afinado con *contrastive learning* para embeddings de oración,
ya empaquetado como modelo ``sentence-transformers`` (pooling propio, sin
ensamblar ``models.Transformer`` + ``models.Pooling`` manual). Cubre el
criterio de benchmark de recuperación densa publicado (Sección 4.3) que los
checkpoints BERT puros no reportan.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("e5-multilingual-base")
class E5MultilingualBaseStrategy(EncoderStrategy):
    """MIT, ~100 idiomas (incl. es/en/pt), 768d, máx. 512 tokens
    (model card intfloat/multilingual-e5-base; Wang et al. 2024)."""

    model_id = "intfloat/multilingual-e5-base"
    embedding_dim = 768
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "mit"
    requires_prefix = {"query": "query: ", "passage": "passage: "}
    mteb_retrieval_score = 62.3
    benchmark_reference = (
        "MIRACL dev set, promedio de 16 idiomas, nDCG@10 "
        "(Wang et al., 'Multilingual E5 Text Embeddings: A Technical Report', 2024, Tabla 4)"
    )

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device=device)
