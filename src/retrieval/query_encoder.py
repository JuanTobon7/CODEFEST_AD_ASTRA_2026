"""
Etapa 1 del pipeline: codificación de la consulta (Sección 8.1).

La consulta en lenguaje natural se codifica con el MISMO encoder que se usó
para indexar el índice respectivo (cada índice FAISS se corresponde con un
encoder activo). El encoder aplica internamente el prefijo query/passage si
el modelo lo requiere (p. ej. familia E5) y ya normaliza a norma unitaria
(``normalize_embeddings=True`` en ``EncoderStrategy.encode``); aquí se
refuerza la normalización de forma defensiva para que el producto punto del
índice ``IndexFlatIP`` sea exactamente similitud coseno.
"""

from __future__ import annotations

import numpy as np

from src.encoders.base import EncoderStrategy


def encode_query(estrategia: EncoderStrategy, query: str) -> np.ndarray:
    """Codifica ``query`` con ``estrategia`` en un vector fila 1×d normalizado.

    Se usa ``is_query=True`` para que el encoder aplique el prefijo de
    consulta (p. ej. ``"query: "`` en E5) si el modelo lo requiere.

    Args:
        estrategia: estrategia de codificación activa (misma que indexó el
            índice contra el que se buscará).
        query: consulta en lenguaje natural.

    Returns:
        Vector ``np.ndarray`` de forma ``(1, embedding_dim)``, float32 y con
        norma unitaria (listo para ``index.search`` con ``IndexFlatIP``).
    """
    vector = estrategia.encode([query], is_query=True)
    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    norma = float(np.linalg.norm(vector))
    if norma > 0:
        vector = vector / norma
    return vector
