"""
BERT puro (``google-bert/bert-base-multilingual-cased``): encoder original
de la familia BERT, sin fine-tuning para *sentence embeddings*. Se ensambla
con *mean pooling* manual (no viene empaquetado como modelo
``sentence-transformers``) y es el encoder multilingüe por defecto de esta
etapa; complementan a este las variantes uncased/large/tiny/monolingües.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("bert-multilingual")
class BertMultilingualStrategy(EncoderStrategy):
    """Apache-2.0, 104 idiomas, 768d, máx. 512 tokens (model card google-bert/bert-base-multilingual-cased)."""

    model_id = "google-bert/bert-base-multilingual-cased"
    embedding_dim = 768
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]  # + 101 idiomas adicionales (model card)
    license = "apache-2.0"
    # BERT puro no reporta score MTEB-Retrieval: no fue afinado para embeddings de oración.
    mteb_retrieval_score = None
    benchmark_reference = "Model card google-bert/bert-base-multilingual-cased (sin fine-tuning de embeddings)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer, models

        transformer = models.Transformer(self.model_id, max_seq_length=self.max_input_tokens)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[transformer, pooling], device=device)
