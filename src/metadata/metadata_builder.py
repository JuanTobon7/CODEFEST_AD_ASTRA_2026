"""
MetadataBuilder: completa y valida la metadata obligatoria de la Tabla 1.

Garantiza que cada fragmento tenga presentes y con el tipo correcto los
campos obligatorios (``doc_id``, ``chunk_id``, ``fuente``, ``formato``,
``fenomeno``, ``posicion``, ``num_tokens``, ``texto``), añade los campos
recomendados (``hash_texto``, ``created_at``) y reordena ``chunk_id``
por si el doc_id fue asignado después del chunking.
"""

from __future__ import annotations

import logging
from typing import List

from src.models.chunk import Chunk
from src.models.extracted_document import ExtractedDocument
from src.support.utils import hash_texto

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """Enriquece los fragmentos con la metadata obligatoria y deseable."""

    def __init__(self, default_fuente: str = "documento") -> None:
        self.default_fuente = default_fuente

    def enrich(self, chunks: List[Chunk], doc: ExtractedDocument) -> List[Chunk]:
        """Completa la metadata de cada fragmento y la valida.

        Args:
            chunks: Fragmentos producidos por la estrategia de chunking.
            doc: Documento de origen (para propagar metadata de documento).

        Returns:
            Los mismos fragmentos con metadata completa.
        """
        for chunk in chunks:
            chunk.doc_id = doc.doc_id
            chunk.fuente = chunk.fuente or doc.fuente or self.default_fuente
            chunk.fenomeno = doc.fenomeno
            if chunk.idioma is None:
                chunk.idioma = doc.idioma
            if chunk.titulo_documento is None:
                chunk.titulo_documento = doc.titulo_documento
            if chunk.fecha_publicacion is None:
                chunk.fecha_publicacion = doc.fecha_publicacion
            # Campos recomendados.
            chunk.hash_texto = hash_texto(chunk.texto)
            if not chunk.created_at:
                from datetime import datetime, timezone

                chunk.created_at = datetime.now(timezone.utc).isoformat()
            # Reordenar chunk_id por si el doc_id cambió tras el chunking.
            chunk.chunk_id = f"{chunk.doc_id}__chunk_{chunk.posicion:05d}"
        return chunks
