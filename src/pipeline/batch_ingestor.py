"""
BatchIngestor: orquesta la ingesta de un lote completo de archivos.

Recibe la configuración, construye el pipeline vía factories (inyección de
dependencias) y coordina: escanear corpus -> asignar fenómeno -> pipeline.run
por archivo -> agregar resultados en un :class:`BatchSummary`.

Un archivo corrupto no detiene el lote: su error queda registrado en el
resumen y el proceso continúa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.factory import ExtractorFactory
from src.metadata.metadata_builder import MetadataBuilder
from src.models.batch_summary import BatchSummary
from src.models.config import Settings
from src.models.pipeline_result import IngestionResult
from src.persistence.mongo_repository import MongoChunkRepository
from src.pipeline.corpus_service import CorpusService
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.validation.chunk_validator import ChunkValidator

logger = logging.getLogger(__name__)


class BatchIngestor:
    """Orquesta el procesamiento de un lote de archivos del corpus."""

    def __init__(
        self,
        settings: Settings,
        corpus: Path,
        fenomenos_map: Optional[Dict[str, int]] = None,
    ) -> None:
        """Inicializa el lote construyendo sus dependencias.

        Args:
            settings: Configuración global del pipeline.
            corpus: Directorio raíz del corpus.
            fenomenos_map: Mapeo carpeta/patrón -> fenómeno (1, 2 o 3).
        """
        self._settings = settings
        self._corpus_service = CorpusService(corpus, fenomenos_map or {})
        self._pipeline, self._repository = self._build_pipeline()

    @staticmethod
    def _doc_id_relativo(filepath: Path, corpus: Path) -> str:
        """doc_id único y legible: ruta relativa al corpus.

        Usar solo ``filepath.stem`` colisiona cuando dos archivos de carpetas
        distintas comparten nombre (p. ej. tiles de OSM en ``tiles/*/*/``),
        lo que mezcla documentos y provoca rechazos en cascada por posiciones.
        """
        try:
            rel = Path(filepath).resolve().relative_to(Path(corpus).resolve())
            return str(rel)
        except ValueError:
            return Path(filepath).stem

    def _build_pipeline(self) -> Tuple[IngestionPipeline, MongoChunkRepository]:
        """Ensambla el pipeline con todas sus dependencias (factories)."""
        segmenter = TextSegmenter.crear(
            tokenizer_model=self._settings.tokenizer_model,
            sentence_model=self._settings.sentence_model,
        )
        estrategia = ChunkingStrategyFactory(segmenter).create(
            self._settings.chunking_strategy, self._settings.chunking_config
        )
        cleaner = TextCleaner(default_language=self._settings.default_language)
        # Boilerplate de lote: líneas repetidas entre documentos del corpus.
        cleaner.set_corpus_boilerplate(self._corpus_service.read_plain_text_docs())

        repositorio = MongoChunkRepository(
            uri=self._settings.mongo_uri,
            db_name=self._settings.mongo_db,
            collection_name=self._settings.mongo_collection,
            username=self._settings.mongo_user,
            password=self._settings.mongo_password,
            auth_source=self._settings.mongo_auth_source,
        )

        pipeline = IngestionPipeline(
            extractor_factory=ExtractorFactory(),
            chunking_strategy=estrategia,
            config=self._settings.chunking_config,
            cleaner=cleaner,
            metadata_builder=MetadataBuilder(default_fuente=self._settings.default_fuente),
            validator=ChunkValidator(max_tokens=self._settings.max_tokens),
            repository=repositorio,
            doc_id_generator=lambda fp, fenomeno: self._doc_id_relativo(
                fp, self._corpus_service.corpus
            ),
        )
        return pipeline, repositorio

    # Orquestación del lote ------------------------------------------------------

    def run(
        self,
        extensiones: Optional[List[str]] = None,
        limite: int = 0,
        por_defecto: int = 1,
    ) -> BatchSummary:
        """Procesa todos los archivos del corpus y agrega el resultado.

        Args:
            extensiones: Solo estas extensiones (opcional).
            limite: Máximo de archivos a procesar (0 = todos).
            por_defecto: Fenómeno cuando el mapeo no coincide.

        Returns:
            :class:`BatchSummary` con conteos y errores por archivo.
        """
        archivos = self._corpus_service.scan(extensiones)
        if limite > 0:
            archivos = archivos[:limite]
        if not archivos:
            logger.error("No hay archivos para procesar en %s", self._corpus_service.corpus)
            return BatchSummary(total=0, ok=0, error=0, chunks_guardados=0)

        self._repository.connect()
        resultados: List[IngestionResult] = []
        try:
            for filepath in archivos:
                fenomeno = self._corpus_service.determine_fenomeno(filepath, por_defecto)
                try:
                    resultado = self._pipeline.run(filepath, fenomeno)
                except Exception as exc:  # noqa: BLE001 - el lote continúa
                    logger.exception("Fallo al procesar %s", filepath)
                    resultado = IngestionResult(
                        fuente=filepath.name, status="error", errores=[str(exc)]
                    )
                resultados.append(resultado)
        finally:
            self.close()
        return BatchSummary.from_results(resultados)

    def close(self) -> None:
        """Cierra la conexión con el repositorio si está abierta."""
        self._repository.close()
