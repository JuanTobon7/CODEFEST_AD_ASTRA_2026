"""
Encoder liviano de baja latencia (``sentence-transformers/distiluse-base-multilingual-cased-v2``).

Perfila el criterio de **eficiencia computacional** (Sección 4.3) en un
esquema multi-encoder: se combina con un encoder de alta precisión
(p. ej. ``e5-large``) para cubrir "granularidad complementaria".
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("minilm-light")
class MiniLMLightStrategy(EncoderStrategy):
    """Apache-2.0, 512d, máx. 512 tokens (model card sentence-transformers/distiluse-base-multilingual-cased-v2)."""

    model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    embedding_dim = 512
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]  # + idiomas adicionales (model card)
    license = "apache-2.0"
    # No reportado en el leaderboard MTEB oficial; su perfil es de eficiencia, no de precisión.
    mteb_retrieval_score = None
    benchmark_reference = "STS/paraphrase mining (model card sentence-transformers)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device=device)
