"""
BERT multilingüe *uncased* (``google-bert/bert-base-multilingual-uncased``):
variante sin distinción de mayúsculas del mismo checkpoint oficial de Google,
102 idiomas. Complementa a ``bert-multilingual`` (variante *cased*) cuando el
corpus tiene ruido de capitalización inconsistente.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("bert-multilingual-uncased")
class BertMultilingualUncasedStrategy(EncoderStrategy):
    """Apache-2.0, 102 idiomas, 768d, máx. 512 tokens (model card google-bert/bert-base-multilingual-uncased)."""

    model_id = "google-bert/bert-base-multilingual-uncased"
    embedding_dim = 768
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]  # + 99 idiomas adicionales (model card)
    license = "apache-2.0"
    # BERT puro no reporta score MTEB-Retrieval: no fue afinado para embeddings de oración.
    mteb_retrieval_score = None
    benchmark_reference = "Model card google-bert/bert-base-multilingual-uncased (sin fine-tuning de embeddings)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer, models

        transformer = models.Transformer(self.model_id, max_seq_length=self.max_input_tokens)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[transformer, pooling], device=device)
