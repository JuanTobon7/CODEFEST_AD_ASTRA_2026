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

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from src.queries.loader import N_QUERIES
from src.queries.models import QUERY_ID_PATTERN
from src.retrieval.chunk_ops import contar_palabras

logger = logging.getLogger(__name__)

N_DOCS = 3
N_FRAGMENTS = 10
MAX_WORDS_FRAGMENT = 250

# Reutilizado para mensajes de error (patrón exacto qNNN).
_QUERY_ID_MENSAJE = re.compile(r"^q\d{3}$")


def _build_documents(query_id: str, documentos_crudos: List[Any]) -> List[Dict[str, Any]]:
    """Construye y valida la sección ``documents`` del esquema.

    ``retrieve()`` ya devuelve top-3 (``aggregate_to_documents`` recorta a
    ``n_docs``); por robustez, si vinieran más se recortan a los 3 primeros
    (orden de relevancia) y si vinieran menos es un error (no se inventan
    ``doc_id``).
    """
    if len(documentos_crudos) < N_DOCS:
        raise ValueError(
            f"[{query_id}] 'documents' debe tener exactamente {N_DOCS} "
            f"elementos, se obtuvieron {len(documentos_crudos)}"
        )
    documents: List[Dict[str, Any]] = []
    for rank, doc_id in enumerate(documentos_crudos[:N_DOCS], start=1):
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
    """Construye y valida la sección ``fragments`` del esquema.

    Tras el split/merge (Sección 9.2.1), ``retrieve()`` puede devolver MÁS de
    10 fragmentos: un fragmento largo se divide en sub-fragmentos que ocupan
    sus propios ranks conservando el ``chunk_id`` del padre. El reglamento
    exige EXACTAMENTE 10 en orden de relevancia -> se recortan los primeros 10
    (la lista ya viene ordenada por score RRF descendente). Si vinieran menos
    de 10 es un error (no se inventa texto).
    """
    if len(fragmentos_crudos) < N_FRAGMENTS:
        raise ValueError(
            f"[{query_id}] 'fragments' debe tener exactamente {N_FRAGMENTS} "
            f"elementos, se obtuvieron {len(fragmentos_crudos)}"
        )
    if len(fragmentos_crudos) > N_FRAGMENTS:
        logger.info(
            "[%s] retrieve() devolvió %d fragmentos (split/merge); se conservan los top-%d",
            query_id, len(fragmentos_crudos), N_FRAGMENTS,
        )
    fragments = [
        _build_one_fragment(query_id, rank, fragmento)
        for rank, fragmento in enumerate(fragmentos_crudos[:N_FRAGMENTS], start=1)
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


def verificar_resultados(ruta_salida: Path, esperadas: int, fallidas: List[str]) -> None:
    """Segunda pasada de verificación del archivo ``resultados.jsonl`` escrito.

    Relee el archivo y valida que cada línea sea JSON parseable y que el
    conteo coincida con lo escrito; advierte si quedó por debajo de las 50.
    """
    with open(ruta_salida, "r", encoding="utf-8") as f:
        lineas = [linea for linea in f.read().split("\n") if linea]

    for numero, linea in enumerate(lineas, start=1):
        try:
            json.loads(linea)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Verificación falló: línea {numero} de '{ruta_salida}' no es JSON válido: {exc}"
            ) from exc

    if len(lineas) != esperadas:
        raise ValueError(
            f"Verificación falló: se escribieron {esperadas} líneas pero se "
            f"leyeron {len(lineas)}"
        )

    if len(lineas) < N_QUERIES:
        logger.warning(
            "Solo se escribieron %d/%d líneas (consultas fallidas: %s)",
            len(lineas), N_QUERIES, fallidas,
        )
    else:
        logger.info("Verificación OK: %d líneas, todas JSON válido.", len(lineas))
