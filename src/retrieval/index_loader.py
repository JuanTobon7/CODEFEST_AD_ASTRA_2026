"""
Carga de los artefactos de entrega por encoder (Sección 5.3) para
recuperación: ``base_vectorial/encoder_<nombre>/{index.faiss, metadata.jsonl}``.

El ``index.faiss`` se construyó con ``index.add()`` secuencial, así que el
ID interno FAISS (0-based) de cada vector coincide con el ordinal de su
línea en ``metadata.jsonl`` — exactamente el supuesto de
:func:`src.retrieval.faiss_search.search_faiss`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss

logger = logging.getLogger(__name__)


class ArtifactosFaltantesError(FileNotFoundError):
    """No existen los artefactos de entrega de un encoder."""


def _carpeta_encoder(base_dir: Path, encoder_name: str) -> Path:
    return base_dir / f"encoder_{encoder_name}"


def load_encoder_index(base_dir: Union[Path, str], encoder_name: str) -> faiss.Index:
    """Carga el ``index.faiss`` de entrega de ``encoder_name``."""
    ruta = _carpeta_encoder(Path(base_dir), encoder_name) / "index.faiss"
    if not ruta.exists():
        raise ArtifactosFaltantesError(f"No existe el índice de entrega: {ruta}")
    return faiss.read_index(str(ruta))


def load_encoder_metadata(base_dir: Union[Path, str], encoder_name: str) -> List[dict]:
    """Carga ``metadata.jsonl`` de entrega de ``encoder_name``.

    La línea ``i`` corresponde al vector con ID interno FAISS ``i``.
    """
    ruta = _carpeta_encoder(Path(base_dir), encoder_name) / "metadata.jsonl"
    if not ruta.exists():
        raise ArtifactosFaltantesError(f"No existe la metadata de entrega: {ruta}")
    lineas: List[dict] = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                lineas.append(json.loads(linea))
    return lineas


def load_encoder_artifacts(
    base_dir: Union[Path, str], encoder_name: str
) -> Tuple[faiss.Index, List[dict]]:
    """Carga índice + metadata de entrega de un encoder en una sola llamada."""
    return load_encoder_index(base_dir, encoder_name), load_encoder_metadata(base_dir, encoder_name)


def build_siguiente_chunk_lookup(
    metadata_por_encoder: Dict[str, List[dict]],
) -> "Optional[object]":
    """Construye un lookup ``(doc_id, posicion) -> texto del chunk siguiente``.

    El callback que consume este lookup (la fusión del paso 5) pide
    ``(doc_id, posicion)`` y recibe el texto del chunk en la posición
    ``posicion + 1`` del mismo documento, o ``None`` si no existe.

    Args:
        metadata_por_encoder: metadata de cada encoder (todas comparten los
            mismos ``doc_id``/``posicion``/``texto``; basta una para el lookup).

    Returns:
        Función con firma ``(doc_id: str, posicion: int) -> Optional[str]``.
    """
    por_doc_posicion: Dict[Tuple[str, int], str] = {}
    for metadata in metadata_por_encoder.values():
        for meta in metadata:
            por_doc_posicion.setdefault(
                (str(meta.get("doc_id", "")), int(meta.get("posicion", 0))),
                str(meta.get("texto", "")),
            )
    return _SiguienteChunkLookup(por_doc_posicion)


class _SiguienteChunkLookup:
    """Implementación concreta del callback ``siguiente_chunk``."""

    def __init__(self, por_doc_posicion: Dict[Tuple[str, int], str]) -> None:
        self._por_doc_posicion = por_doc_posicion

    def __call__(self, doc_id: str, posicion: int) -> Optional[str]:
        return self._por_doc_posicion.get((doc_id, posicion + 1))
