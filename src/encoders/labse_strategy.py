"""
LaBSE (``sentence-transformers/LaBSE``): fuerte alineación cross-lingual,
109 idiomas. Útil como encoder complementario para portugués/idiomas menos
representados en otros modelos (Sección 4.4: cobertura por idioma).
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


@EncoderFactory.register("labse")
class LaBSEStrategy(EncoderStrategy):
    """Apache-2.0, 109 idiomas, 768d, máx. 512 tokens (model card sentence-transformers/LaBSE)."""

    model_id = "sentence-transformers/LaBSE"
    embedding_dim = 768
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]  # + 106 idiomas adicionales (model card)
    license = "apache-2.0"
    # LaBSE no reporta score MTEB-Retrieval oficial (se evalúa en alineación cross-lingual, no BEIR).
    mteb_retrieval_score = None
    benchmark_reference = "Tatoeba cross-lingual alignment (model card sentence-transformers/LaBSE)"

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device=device)
