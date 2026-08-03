"""
BERT-Tiny (``prajjwal1/bert-tiny``): checkpoint oficial más pequeño de la
familia "Well-Read Students Learn Better" (L=2, H=128), portado a Pytorch
desde los checkpoints de Google. Perfila el criterio de **eficiencia
computacional** (baja latencia) frente a las variantes *base*/*large*. Solo
inglés — se marca ``is_complementary`` para no exigir cobertura es/en/pt.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("bert-tiny")
class BertTinyStrategy(EncoderStrategy):
    """MIT, inglés, 128d, máx. 512 tokens (model card prajjwal1/bert-tiny)."""

    model_id = "prajjwal1/bert-tiny"
    embedding_dim = 128
    max_input_tokens = 512
    supported_languages = ["en"]
    license = "mit"
    is_complementary = True
    # No reportado en el leaderboard MTEB oficial; su perfil es de eficiencia, no de precisión.
    mteb_retrieval_score = None
    benchmark_reference = "Well-Read Students Learn Better, Turc et al. 2019 (model card prajjwal1/bert-tiny)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer, models

        transformer = models.Transformer(self.model_id, max_seq_length=self.max_input_tokens)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[transformer, pooling], device=device)
