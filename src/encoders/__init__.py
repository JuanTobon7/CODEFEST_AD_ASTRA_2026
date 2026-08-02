"""
Paquete de estrategias de codificación semántica (patrón Strategy, Sección 4).

Deliberadamente sin importaciones de las estrategias concretas: así los
tests de :mod:`src.encoders.factory` y :mod:`src.encoders.orchestrator` no
requieren tener instalado ``sentence-transformers``/``torch``. Para
registrar las estrategias reales (E5, BGE-M3, LaBSE, MiniLM) impórtalas
explícitamente, p. ej. desde ``run_embedding.py``::

    from src.encoders import (
        bge_m3_strategy, labse_strategy, minilm_light_strategy, multilingual_e5_strategy,
    )
"""
