"""
Paquete de estrategias de codificación semántica (patrón Strategy, Sección 4).

Deliberadamente sin importaciones de las estrategias concretas: así los
tests de :mod:`src.encoders.factory` y :mod:`src.encoders.orchestrator` no
requieren tener instalado ``sentence-transformers``/``torch``. Para
registrar las estrategias reales (checkpoints BERT de HuggingFace)
impórtalas explícitamente, p. ej. desde ``run_embedding.py``::

    from src.encoders import (
        bert_strategy, bert_multilingual_uncased_strategy, bert_large_strategy,
        bert_tiny_strategy, bert_language_strategy,
    )
"""
