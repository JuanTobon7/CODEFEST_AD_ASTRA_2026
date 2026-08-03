"""
BERT monolingües por idioma (checkpoints oficiales BERT-Base entrenados
específicamente para es/en/pt), como complemento de granularidad idiomática
a los BERT multilingües (``bert-multilingual``/``bert-multilingual-uncased``):

- ``bert-es``: BETO (``dccuchile/bert-base-spanish-wwm-cased``), Universidad de Chile.
- ``bert-en``: ``google-bert/bert-base-cased``, checkpoint oficial de Google.
- ``bert-pt``: BERTimbau (``neuralmind/bert-base-portuguese-cased``).

Cada uno cubre un único idioma con mayor fidelidad que el multilingüe
compartido, a costa de no servir para los otros dos — se marcan
``is_complementary`` para no exigir cobertura es/en/pt en el registro.
"""

from __future__ import annotations

from src.encoders.base import EncoderStrategy
from src.encoders.factory import EncoderFactory


class BertLanguageStrategy(EncoderStrategy):
    """Base común: BERT-Base monolingüe, 768d, máx. 512 tokens, un solo idioma."""

    embedding_dim = 768
    max_input_tokens = 512
    is_complementary = True

    def _cargar_modelo(self, device: str):
        from sentence_transformers import SentenceTransformer, models

        transformer = models.Transformer(self.model_id, max_seq_length=self.max_input_tokens)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[transformer, pooling], device=device)


@EncoderFactory.register("bert-es")
class BertEsStrategy(BertLanguageStrategy):
    """CC-BY-4.0 (model card BETO). Español, entrenado con Whole Word Masking."""

    model_id = "dccuchile/bert-base-spanish-wwm-cased"
    supported_languages = ["es"]
    license = "cc-by-4.0"
    mteb_retrieval_score = None
    benchmark_reference = "BETO: Spanish Pre-Trained BERT (Cañete et al., PML4DC 2020)"


@EncoderFactory.register("bert-en")
class BertEnStrategy(BertLanguageStrategy):
    """Apache-2.0 (model card google-bert/bert-base-cased). Inglés."""

    model_id = "google-bert/bert-base-cased"
    supported_languages = ["en"]
    license = "apache-2.0"
    mteb_retrieval_score = None
    benchmark_reference = "Model card google-bert/bert-base-cased (sin fine-tuning de embeddings)"


@EncoderFactory.register("bert-pt")
class BertPtStrategy(BertLanguageStrategy):
    """MIT (model card BERTimbau). Portugués brasileño, brWaC corpus."""

    model_id = "neuralmind/bert-base-portuguese-cased"
    supported_languages = ["pt"]
    license = "mit"
    mteb_retrieval_score = None
    benchmark_reference = "BERTimbau (Souza, Nogueira & Lotufo, BRACIS 2020)"
