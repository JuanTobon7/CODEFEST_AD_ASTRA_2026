"""
Familia E5 multilingüe (``intfloat/multilingual-e5-*``).

Valores de metadata verificados en las model cards de HuggingFace
(agosto 2026). Requiere prefijos ``"query: "``/``"passage: "`` (ver
model card), resueltos internamente vía ``requires_prefix`` — el
orquestador nunca necesita saber este detalle.
"""

from __future__ import annotations

from typing import Optional

from src.encoders.base import EncoderConfig, EncoderStrategy
from src.encoders.factory import EncoderFactory

# (model_id, embedding_dim) por variante — model cards de intfloat/multilingual-e5-*
_VARIANTES = {
    "e5-small": ("intfloat/multilingual-e5-small", 384),
    "e5-base": ("intfloat/multilingual-e5-base", 768),
    "e5-large": ("intfloat/multilingual-e5-large", 1024),
}


class MultilingualE5Strategy(EncoderStrategy):
    """Familia E5 multilingüe: MIT, 100+ idiomas, máx. 512 tokens."""

    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]  # + 100 idiomas adicionales (model card)
    license = "mit"
    requires_prefix = {"query": "query: ", "passage": "passage: "}

    def __init__(self, variante: str = "e5-base", config: Optional[EncoderConfig] = None) -> None:
        super().__init__(config)
        model_id, dim = _VARIANTES[variante]
        self.model_id = model_id
        self.embedding_dim = dim
        self._variante = variante

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id, device=device)


@EncoderFactory.register("e5-small")
class E5SmallStrategy(MultilingualE5Strategy):
    """Variante liviana (384d) — balance eficiencia/precisión."""

    mteb_retrieval_score = 46.6  # MTEB Retrieval (avg), model card intfloat/multilingual-e5-small
    benchmark_reference = "MTEB Retrieval avg (model card intfloat/multilingual-e5-small)"

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        super().__init__("e5-small", config)


@EncoderFactory.register("e5-base")
class E5BaseStrategy(MultilingualE5Strategy):
    """Variante balanceada (768d) — encoder por defecto recomendado."""

    mteb_retrieval_score = 48.9  # MTEB Retrieval (avg), model card intfloat/multilingual-e5-base
    benchmark_reference = "MTEB Retrieval avg (model card intfloat/multilingual-e5-base)"

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        super().__init__("e5-base", config)


@EncoderFactory.register("e5-large")
class E5LargeStrategy(MultilingualE5Strategy):
    """Variante de alta precisión (1024d) — mayor costo computacional."""

    mteb_retrieval_score = 51.4  # MTEB Retrieval (avg), model card intfloat/multilingual-e5-large
    benchmark_reference = "MTEB Retrieval avg (model card intfloat/multilingual-e5-large)"

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        super().__init__("e5-large", config)
