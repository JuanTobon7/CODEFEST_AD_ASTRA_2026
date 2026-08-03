"""
BERT-Large (``google-bert/bert-large-uncased``): checkpoint oficial de mayor
capacidad de la familia BERT (24 capas, 1024d). Complementa a las variantes
*base* cuando se prioriza precisión sobre latencia. Solo inglés — se marca
``is_complementary`` para no exigir cobertura es/en/pt en el registro.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("bert-large")
class BertLargeStrategy(EncoderStrategy):
    """Apache-2.0, inglés, 1024d, máx. 512 tokens (model card google-bert/bert-large-uncased)."""

    model_id = "google-bert/bert-large-uncased"
    embedding_dim = 1024
    max_input_tokens = 512  # límite real de position embeddings de BERT (no 8192 como BGE-M3)
    supported_languages = ["en"]
    license = "apache-2.0"
    is_complementary = True
    # BERT puro no reporta score MTEB-Retrieval: no fue afinado para embeddings de oración.
    mteb_retrieval_score = None
    benchmark_reference = "Model card google-bert/bert-large-uncased (sin fine-tuning de embeddings)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer, models

        transformer = models.Transformer(self.model_id, max_seq_length=self.max_input_tokens)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[transformer, pooling], device=device)
