"""
IngestionPipeline: orquesta el flujo completo de un archivo.

Flujo (inversión de dependencias: todo se inyecta en el constructor):

    extractor_factory.create(filepath) -> extractor.extract()
        -> cleaner.clean() -> chunking_strategy.chunk()
            -> metadata_builder.enrich() -> validator.validate()
                -> repository.save_many() -> IngestionResult
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.chunking.base import ChunkingStrategy, TextSegmenter
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.base import ExtractorError
from src.extractors.factory import ExtractorFactory
from src.metadata.metadata_builder import MetadataBuilder
from src.models.chunk import Chunk
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument
from src.models.pipeline_result import IngestionResult
from src.persistence.base_repository import ChunkRepository
from src.validation.chunk_validator import ChunkValidator

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orquesta la ingesta de un archivo del corpus."""

    def __init__(
        self,
        extractor_factory: ExtractorFactory,
        chunking_strategy: ChunkingStrategy,
        config: ChunkingConfig,
        cleaner: TextCleaner,
        metadata_builder: MetadataBuilder,
        validator: ChunkValidator,
        repository: ChunkRepository,
        doc_id_generator=None,
    ) -> None:
        """Inicializa el pipeline con todas sus dependencias inyectadas.

        Args:
            extractor_factory: Fábrica de extractores por extensión.
            chunking_strategy: Estrategia de fragmentación seleccionada.
            config: Parámetros de chunking.
            cleaner: Limpiador de texto.
            metadata_builder: Constructor de metadata obligatoria.
            validator: Validador previo a la persistencia.
            repository: Repositorio donde se guardan los fragmentos.
            doc_id_generator: Callable(filepath, fenomeno) -> str. Si es
                ``None`` se usa el nombre del archivo como doc_id.
        """
        self.extractor_factory = extractor_factory
        self.chunking_strategy = chunking_strategy
        self.config = config
        self.cleaner = cleaner
        self.metadata_builder = metadata_builder
        self.validator = validator
        self.repository = repository
        self.doc_id_generator = doc_id_generator or (
            lambda filepath, fenomeno: filepath.stem
        )

    def run(self, filepath: Path, fenomeno: int) -> IngestionResult:
        """Procesa un archivo de principio a fin y persiste los fragmentos.

        Args:
            filepath: Ruta al archivo del corpus.
            fenomeno: Fenómeno asociado (1, 2 o 3).

        Returns:
            :class:`IngestionResult` con conteos, advertencias y errores.
        """
        inicio = time.perf_counter()
        filepath = Path(filepath)
        resultado = IngestionResult(
            fuente=filepath.name,
            fenomeno=fenomeno,
            formato=self._detectar_formato(filepath),
        )
        logger.info("Ingesta iniciada | archivo=%s | fenomeno=%d", filepath.name, fenomeno)

        try:
            extractor = self.extractor_factory.create(filepath)
            extracted: ExtractedDocument = extractor.extract(filepath)
            extracted.fenomeno = fenomeno
            extracted.doc_id = self.doc_id_generator(filepath, fenomeno)
            resultado.doc_id = extracted.doc_id

            extracted = self.cleaner.clean(extracted)
            chunks: list[Chunk] = self.chunking_strategy.chunk(extracted, self.config)
            chunks = self.metadata_builder.enrich(chunks, extracted)

            validacion = self.validator.validate(chunks)
            resultado.num_chunks = len(chunks)
            resultado.num_rechazados = len(validacion.rechazados)
            resultado.warnings = [
                f"{c.chunk_id}: {w}" for c in validacion.validos for w in c.validation_warnings
            ]

            self.repository.save_many(validacion.validos)
            resultado.num_guardados = len(validacion.validos)
            resultado.status = "ok" if validacion.validos else "error"
            if validacion.rechazados:
                resultado.errores.extend(validacion.motivos)
            logger.info(
                "Ingesta completada | archivo=%s | chunks=%d guardados=%d rechazados=%d",
                filepath.name,
                resultado.num_chunks,
                resultado.num_guardados,
                resultado.num_rechazados,
            )
        except ExtractorError as exc:
            resultado.status = "error"
            resultado.errores.append(str(exc))
            logger.error("Error de extracción | archivo=%s | %s", filepath.name, exc)
        except Exception as exc:  # noqa: BLE001 - se reporta y continúa el batch
            resultado.status = "error"
            resultado.errores.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Fallo inesperado | archivo=%s", filepath.name)

        resultado.tiempo_seg = round(time.perf_counter() - inicio, 3)
        return resultado

    @staticmethod
    def _detectar_formato(filepath: Path) -> Optional[str]:
        """Extensión normalizada o ``None``."""
        sufijo = filepath.suffix.lower().lstrip(".")
        if filepath.name.endswith(".osm.pbf"):
            return "pbf"
        return sufijo or None
