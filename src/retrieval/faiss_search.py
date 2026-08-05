"""
Etapa 2 del pipeline: búsqueda por índice FAISS (Sección 8.2).

Recupera el top-``k_search`` de fragmentos más similares de UN índice,
con su score de similitud coseno y su rango (posición 1..k_search).

Asume el artefacto de ENTREGA (Sección 5.3): el índice fue construido con
``index.add()`` secuencial y la línea ``i`` de ``metadata.jsonl`` corresponde
al vector con ID interno FAISS ``i`` (0-based). Por eso el ID devuelto por
``index.search`` se usa directamente como índice de la lista de metadata.
No es compatible con índices operativos ``IndexIDMap`` (IDs determinísticos
derivados del ``chunk_id``); para esos, alinee la metadata por
``faiss_internal_id`` antes de llamar a esta función.
"""

from __future__ import annotations

from typing import List, Sequence

import faiss
import numpy as np

from src.retrieval.models import SearchHit


def search_faiss(
    index: faiss.Index,
    query_vector: np.ndarray,
    metadata: Sequence[dict],
    encoder_name: str,
    k: int = 50,
) -> List[SearchHit]:
    """Recupera los ``k`` fragmentos más similares del índice ``index``.

    Args:
        index: índice FAISS (tipo ``IndexFlatIP`` sobre vectores normalizados;
            cualquier índice con ``search()`` sirve, p. ej. ``IndexIVFFlat``).
        query_vector: vector de consulta normalizado, forma ``(1, d)``.
        metadata: líneas de ``metadata.jsonl`` del encoder; ``metadata[i]``
            corresponde al ID interno FAISS ``i``.
        encoder_name: nombre del encoder que construyó este índice (para
            trazabilidad en la fusión RRF).
        k: ``k_search`` parametrizable (por defecto 50).

    Returns:
        Lista de hasta ``k`` :class:`SearchHit` ordenada por score
        descendente (el orden lo da FAISS); ``rank`` es 1-based.
    """
    if index.ntotal == 0 or query_vector.shape[1] != index.d:
        if query_vector.shape[1] != index.d:
            raise ValueError(
                f"Encoder '{encoder_name}': dimensión de la consulta "
                f"({query_vector.shape[1]}) no coincide con el índice ({index.d})."
            )
        return []

    k_efectivo = min(max(1, k), int(index.ntotal))
    scores, ids = index.search(query_vector, k_efectivo)

    hits: List[SearchHit] = []
    for posicion, (score, faiss_id) in enumerate(zip(scores[0], ids[0])):
        faiss_id = int(faiss_id)
        if faiss_id == -1:
            continue  # FAISS marca huecos sin vector con -1
        meta = dict(metadata[faiss_id])
        hits.append(
            SearchHit(
                chunk_id=str(meta.get("chunk_id", "")),
                doc_id=str(meta.get("doc_id", "")),
                encoder_name=encoder_name,
                rank=posicion + 1,
                score=float(score),
                metadata=meta,
            )
        )
    return hits
