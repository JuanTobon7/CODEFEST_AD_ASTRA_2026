"""
Construcción y validación del objeto de resultado por consulta (Sección 9.3.1
del reto CODEFEST AD ASTRA 2026).

Transforma la salida cruda de ``retrieve()`` (``{"documents": [doc_id...],
"fragments": [{"chunk_id", "doc_id", "text", ...}]}``) en el esquema exacto
del reglamento y valida las reglas obligatorias ANTES de que la línea se
escriba al archivo:

- ``documents`` con EXACTAMENTE 3 elementos y ranks 1,2,3 sin huecos.
- ``fragments`` con EXACTAMENTE 10 elementos y ranks 1..10 sin huecos.
- Cada fragmento con ``text`` de hasta 250 palabras (``str.split()``); si lo
  supera, debió dividirse antes con ``split_or_merge_fragments`` del módulo
  de recuperación (cada sub-fragmento conserva el ``chunk_id`` del padre).
- Todos los campos obligatorios presentes y no vacíos (query_id, rank,
  doc_id, chunk_id, text).
- ``query_id`` con el patrón exacto ``qNNN``.

Ninguna regla depende de modelos generativos: solo conteo de palabras,
patrones y estructura.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.queries.models import QUERY_ID_PATTERN
from src.retrieval.chunk_ops import contar_palabras

N_DOCS = 3
N_FRAGMENTS = 10
MAX_WORDS_FRAGMENT = 250

# Reutilizado para mensajes de error (patrón exacto qNNN).
_QUERY_ID_MENSAJE = re.compile(r"^q\d{3}$")


def _build_documents(query_id: str, documentos_crudos: List[Any]) -> List[Dict[str, Any]]:
    """Construye y valida la sección ``documents`` del esquema."""
    if len(documentos_crudos) != N_DOCS:
        raise ValueError(
            f"[{query_id}] 'documents' debe tener exactamente {N_DOCS} "
            f"elementos, se obtuvieron {len(documentos_crudos)}"
        )
    documents: List[Dict[str, Any]] = []
    for rank, doc_id in enumerate(documentos_crudos, start=1):
        doc_id = str(doc_id).strip()
        if not doc_id:
            raise ValueError(f"[{query_id}] documents[{rank}] tiene doc_id vacío")
        documents.append({"rank": rank, "doc_id": doc_id})

    ranks_docs = sorted(d["rank"] for d in documents)
    if ranks_docs != list(range(1, N_DOCS + 1)):
        raise ValueError(f"[{query_id}] ranks de 'documents' con huecos/duplicados: {ranks_docs}")
    return documents


def _build_one_fragment(query_id: str, rank: int, fragmento: Dict[str, Any]) -> Dict[str, Any]:
    """Construye y valida un elemento individual de ``fragments``."""
    chunk_id = str(fragmento.get("chunk_id", "")).strip()
    doc_id = str(fragmento.get("doc_id", "")).strip()
    texto = str(fragmento.get("text", "")).strip()

    if not chunk_id:
        raise ValueError(f"[{query_id}] fragments[{rank}] sin chunk_id")
    if not doc_id:
        raise ValueError(f"[{query_id}] fragments[{rank}] sin doc_id")
    if not texto:
        raise ValueError(f"[{query_id}] fragments[{rank}] sin text")

    n_palabras = contar_palabras(texto)
    if n_palabras > MAX_WORDS_FRAGMENT:
        raise ValueError(
            f"[{query_id}] fragments[{rank}] (chunk_id='{chunk_id}') "
            f"tiene {n_palabras} palabras (> {MAX_WORDS_FRAGMENT}); debió "
            "dividirse antes con split_or_merge_fragments"
        )
    return {"rank": rank, "chunk_id": chunk_id, "doc_id": doc_id, "text": texto}


def _build_fragments(query_id: str, fragmentos_crudos: List[Any]) -> List[Dict[str, Any]]:
    """Construye y valida la sección ``fragments`` del esquema."""
    if len(fragmentos_crudos) != N_FRAGMENTS:
        raise ValueError(
            f"[{query_id}] 'fragments' debe tener exactamente {N_FRAGMENTS} "
            f"elementos, se obtuvieron {len(fragmentos_crudos)}"
        )
    fragments = [
        _build_one_fragment(query_id, rank, fragmento)
        for rank, fragmento in enumerate(fragmentos_crudos, start=1)
    ]

    ranks_frag = sorted(f["rank"] for f in fragments)
    if ranks_frag != list(range(1, N_FRAGMENTS + 1)):
        raise ValueError(f"[{query_id}] ranks de 'fragments' con huecos/duplicados: {ranks_frag}")
    return fragments


def build_result_object(query_id: str, retrieval_output: Dict[str, Any]) -> Dict[str, Any]:
    """Construye y valida el objeto JSON de una consulta (Sección 9.3.1).

    Args:
        query_id: identificador ``qNNN`` de la consulta.
        retrieval_output: salida cruda de ``retrieve()``:
            ``{"documents": [doc_id, ...], "fragments": [{"chunk_id",
            "doc_id", "text", "score", ...}, ...]}``.

    Returns:
        Objeto listo para serializar con el esquema exacto del reglamento:
        ``{"query_id", "documents": [{"rank", "doc_id"} x3],
        "fragments": [{"rank", "chunk_id", "doc_id", "text"} x10]}``.

    Raises:
        ValueError: con el ``query_id`` y la regla violada, si alguna de las
            validaciones obligatorias del punto 3 no se cumple. Nunca
            devuelve (ni da pie a escribir) un objeto inválido.
    """
    if not _QUERY_ID_MENSAJE.match(query_id):
        raise ValueError(f"[{query_id}] query_id no sigue el patrón qNNN")

    documentos_crudos = retrieval_output.get("documents") or []
    fragmentos_crudos = retrieval_output.get("fragments") or []

    documents = _build_documents(query_id, documentos_crudos)
    fragments = _build_fragments(query_id, fragmentos_crudos)

    return {"query_id": query_id, "documents": documents, "fragments": fragments}
