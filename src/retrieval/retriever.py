"""
Orquestador del módulo de recuperación (Sección 8) — Controlador delgado
(GRASP): conecta las etapas puras (``encode_query``, ``search_faiss``,
``rrf_fuse``, ``apply_filters``, ``split_or_merge_fragments``,
``aggregate_to_documents``) sin implementar ninguna de sus reglas.

La clase :class:`Retriever` recibe índices, metadata y encoders ya
instanciados (ideal para un servicio que reutiliza los modelos cargados).
La función de módulo :func:`retrieve` es la conveniencia que construye un
``Retriever`` por defecto leyendo los artefactos de entrega de
``base_vectorial/`` y los encoders activos de la configuración.

NINGÚN modelo generativo interviene: todo opera sobre vectores (FAISS),
puntuaciones (RRF) y metadata (filtros/agregación).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import faiss

from src.embeddings.embedding_config import EmbeddingConfig
from src.encoders.base import EncoderStrategy
from src.retrieval.aggregation import aggregate_to_documents
from src.retrieval.chunk_ops import SplitterProtocol, split_or_merge_fragments
from src.retrieval.faiss_search import search_faiss
from src.retrieval.filters import apply_filters
from src.retrieval.index_loader import (
    ArtifactosFaltantesError,
    build_siguiente_chunk_lookup,
    load_encoder_artifacts,
)
from src.retrieval.models import Fragment, RetrievalFilters, RetrievalResult
from src.retrieval.query_encoder import encode_query
from src.retrieval.rrf import rrf_fuse

logger = logging.getLogger(__name__)

# Número de documentos a devolver (Sección 8.6: top-3).
N_DOCS_DEFAULT = 3


class Retriever:
    """Recuperación multi-encoder con RRF + filtros + agregación a doc."""

    def __init__(
        self,
        indices: Dict[str, faiss.Index],
        metadata: Dict[str, List[dict]],
        encoders: Sequence[EncoderStrategy],
        k0: int = 60,
    ) -> None:
        """Inicializa con los recursos ya cargados (sin I/O ni modelos aquí).

        Args:
            indices: ``{encoder_name: index.faiss}``, uno por encoder activo.
            metadata: ``{encoder_name: [líneas de metadata.jsonl]}`` alineadas
                con los IDs internos FAISS de su índice.
            encoders: estrategias activas (la misma que indexó cada índice).
            k0: constante de suavizado de RRF (parametrizable).
        """
        self.indices = indices
        self.metadata = metadata
        self.encoders = {e.name: e for e in encoders}
        self.k0 = k0
        self._siguiente_chunk = build_siguiente_chunk_lookup(metadata)

    # -- Pipeline ---------------------------------------------------------

    def _rankings_por_encoder(
        self, query: str, k_search: int
    ) -> List[List[Any]]:
        """Etapas 1-2: codifica la consulta y busca en cada índice activo.

        Devuelve un ranking (:class:`SearchHit`) por encoder. Un encoder sin
        artefacto o que falle al codificar se omite con warning (tolerancia
        multi-encoder: la fusión RRF no depende de ninguno en particular).
        """
        rankings: List[List[Any]] = []
        for nombre, estrategia in self.encoders.items():
            indice = self.indices.get(nombre)
            metadata = self.metadata.get(nombre)
            if indice is None or metadata is None:
                logger.warning("Encoder '%s': sin índice/metadata, se omite", nombre)
                continue
            try:
                vector_query = encode_query(estrategia, query)
                ranking = search_faiss(
                    indice, vector_query, metadata, encoder_name=nombre, k=k_search
                )
            except Exception as exc:  # noqa: BLE001 - un encoder no debe tumbar el resto
                logger.warning("Encoder '%s' falló (%s); se omite", nombre, exc)
                continue
            if ranking:
                rankings.append(ranking)
        return rankings

    def retrieve(
        self,
        query: str,
        phenomenon_filter: Optional[int] = None,
        format_filter: Optional[str] = None,
        lang_filter: Optional[str] = None,
        date_range: Optional[Sequence[Any]] = None,
        theta: float = 0.0,
        k_search: int = 50,
        k_chunk_out: int = 10,
        doc_agg: str = "max",
        k0: Optional[int] = None,
        n_docs: int = N_DOCS_DEFAULT,
        splitter: Optional[SplitterProtocol] = None,
    ) -> RetrievalResult:
        """Ejecuta el pipeline completo de recuperación para ``query``.

        Args:
            query: consulta en lenguaje natural.
            phenomenon_filter: restringe a un fenómeno (1, 2 o 3).
            format_filter: restringe a un formato (pdf, md, html, ...).
            lang_filter: restringe a un idioma (metadata opcional).
            date_range: rango ``(inicio, fin)`` de ``fecha_publicacion``
                (date/datetime/str ISO, extremos abiertos con ``None``).
            theta: umbral de similitud coseno original (filtro por vector).
            k_search: top-k por índice antes de fusionar.
            k_chunk_out: fragmentos finales tras RRF + filtros.
            doc_agg: estrategia de agregación a documento
                (``max`` | ``sum`` | ``weighted_mean``).
            k0: sobreescribe la constante RRF del constructor si se pasa.
            n_docs: cuántos ``doc_id`` devolver (top-3 por defecto).
            splitter: segmentador de oraciones inyectable (default: regex).

        Returns:
            :class:`RetrievalResult` con ``documents`` (top-``n_docs``) y
            ``fragments`` (hasta ``k_chunk_out``, tras split/merge).
        """
        filtros = RetrievalFilters(
            fenomeno=phenomenon_filter,
            formato=format_filter,
            idioma=lang_filter,
            date_range=date_range,
            theta=theta,
        )

        rankings = self._rankings_por_encoder(query, k_search)
        if not rankings:
            logger.warning("Ningún encoder devolvió resultados para la consulta")
            return RetrievalResult(documents=[], fragments=[])

        # Etapa 3: fusión RRF.
        fusionados = rrf_fuse(rankings, k0=self.k0 if k0 is None else k0)

        # Etapa 4: post-filtros ANTES de recortar al top final.
        fusionados = apply_filters(fusionados, filtros)

        # Etapa 5: selección final a nivel de chunk (split/merge).
        fragmentos_finales: List[Fragment] = split_or_merge_fragments(
            fusionados[:k_chunk_out],
            splitter=splitter,
            siguiente_chunk=self._siguiente_chunk,
        )

        # Etapa 6: agregación a documento (top-n_docs).
        documentos = [
            doc_id
            for doc_id, _ in aggregate_to_documents(fragmentos_finales, strategy=doc_agg)[:n_docs]
        ]

        return RetrievalResult(
            documents=documentos,
            fragments=[f.as_dict() for f in fragmentos_finales],
        )


# -- Conveniencia: función pública con la firma exacta del reto -------------

def _construir_retriever_default(config: Optional[EmbeddingConfig] = None) -> Retriever:
    """Levanta un ``Retriever`` con los encoders activos y artefactos de entrega.

    Importa las estrategias concretas para disparar su registro en el
    ``EncoderFactory`` (efecto secundario necesario, igual que en
    ``run_embedding.py``) y crea cada encoder con el batch/device de config.
    """
    from src.encoders import (  # noqa: F401 - registro por decorador
        bert_language_strategy,
        bert_large_strategy,
        bert_multilingual_uncased_strategy,
        bert_strategy,
        bert_tiny_strategy,
        e5_multilingual_base_strategy,
        e5_multilingual_small_strategy,
    )
    from src.encoders.factory import EncoderFactory
    from src.encoders.base import EncoderConfig

    config = config or EmbeddingConfig()
    device = None if config.embedding_device == "auto" else config.embedding_device

    encoders: List[EncoderStrategy] = []
    indices: Dict[str, faiss.Index] = {}
    metadata: Dict[str, List[dict]] = {}
    base_dir = config.embedding_output_dir

    for nombre in config.encoder_names:
        try:
            indice, meta = load_encoder_artifacts(base_dir, nombre)
        except ArtifactosFaltantesError as exc:
            logger.warning("Se omite encoder '%s': %s", nombre, exc)
            continue
        estrategia = EncoderFactory.create(
            nombre,
            EncoderConfig(batch_size=config.batch_size_para(nombre), device_preference=device),
        )
        indices[nombre] = indice
        metadata[nombre] = meta
        encoders.append(estrategia)

    if not encoders:
        raise ArtifactosFaltantesError(
            f"No hay artefactos de entrega en '{base_dir}' para ningún encoder activo "
            f"({config.encoder_names}). Ejecuta primero 'python -m src.vectorstore.run_export_delivery'."
        )
    return Retriever(indices=indices, metadata=metadata, encoders=encoders)


def retrieve(
    query: str,
    phenomenon_filter=None,
    format_filter=None,
    lang_filter=None,
    date_range=None,
    theta=0.0,
    k_search=50,
    k_chunk_out=10,
    doc_agg="max",
) -> dict:
    """Punto de entrada del reto (Sección 8): recupera fragmentos y documentos.

    Construye un :class:`Retriever` por defecto (encoders activos de la
    config + artefactos de entrega en ``base_vectorial/``) y delega en él.

    Returns:
        ``{"documents": [doc_id_1, doc_id_2, doc_id_3],
            "fragments": [{"chunk_id", "doc_id", "text", "score", ...}, ...]}``
    """
    retriever = _construir_retriever_default()
    resultado = retriever.retrieve(
        query=query,
        phenomenon_filter=phenomenon_filter,
        format_filter=format_filter,
        lang_filter=lang_filter,
        date_range=date_range,
        theta=theta,
        k_search=k_search,
        k_chunk_out=k_chunk_out,
        doc_agg=doc_agg,
    )
    return resultado.as_dict()
