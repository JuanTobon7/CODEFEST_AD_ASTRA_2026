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
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.factory import ExtractorFactory
from src.metadata.metadata_builder import MetadataBuilder
from src.models.batch_summary import BatchSummary
from src.models.config import Settings
from src.models.pipeline_result import IngestionResult
from src.persistence.base_repository import ChunkRepository
from src.persistence.json_repository import JsonChunkRepository
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

    def _build_pipeline(self) -> Tuple[IngestionPipeline, ChunkRepository]:
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

        tipo_repositorio = self._settings.chunk_repository.strip().lower()
        if tipo_repositorio == "json":
            repositorio: ChunkRepository = JsonChunkRepository(self._settings.chunks_json_path)
        elif tipo_repositorio == "mongo":
            repositorio = MongoChunkRepository(
                uri=self._settings.mongo_uri,
                db_name=self._settings.mongo_db,
                collection_name=self._settings.mongo_collection,
                username=self._settings.mongo_user,
                password=self._settings.mongo_password,
                auth_source=self._settings.mongo_auth_source,
            )
        else:
            raise ValueError(
                "CHUNK_REPOSITORY debe ser 'json' o 'mongo', "
                f"no '{self._settings.chunk_repository}'"
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

    @staticmethod
    def _repartir_por_fenomeno(
        archivos: List[Path],
        limite: int,
        fenomeno_de: Callable[[Path], int],
    ) -> List[Path]:
        """Reparte ``limite`` archivos equitativamente entre los fenómenos.

        ``archivos[:limite]`` tomaría los primeros N en orden alfabético
        (F1_ < F2_ < F3_), dejando casi solo archivos de F1 cuando hay
        límite. Aquí se agrupa por fenómeno y se toma una cuota de cada uno
        (``limite // n``, residuo uno a cada uno de los primeros), igual que
        ``MongoChunkRepository.find_all_balanceado`` hace para chunks.

        Args:
            archivos: Lista de archivos escaneados (orden estable).
            limite: Máximo de archivos a devolver (0 o > len = sin recorte).
            fenomeno_de: Resolver fenómeno (1, 2 o 3) de un archivo.
        """
        if limite <= 0 or len(archivos) <= limite:
            return archivos
        por_fenomeno: Dict[int, List[Path]] = defaultdict(list)
        for archivo in archivos:
            por_fenomeno[fenomeno_de(archivo)].append(archivo)
        n_fenomenos = len(por_fenomeno)
        cuota, residuo = divmod(limite, n_fenomenos)
        seleccionados: List[Path] = []
        for i, (fenomeno, lista) in enumerate(sorted(por_fenomeno.items())):
            tope = cuota + (1 if i < residuo else 0)
            seleccionados.extend(lista[:tope])
        return seleccionados

    def _log_fenomenos(self, archivos: List[Path], por_defecto: int, etiqueta: str) -> None:
        """Log de trazabilidad: distribución de archivos por fenómeno (por número)."""
        conteo: Dict[int, int] = defaultdict(int)
        for archivo in archivos:
            conteo[self._corpus_service.determine_fenomeno(archivo, por_defecto)] += 1
        detalle = ", ".join(f"F{f}={conteo[f]}" for f in sorted(conteo))
        logger.info("FENÓMENOS | %s: %s (%d archivos)", etiqueta, detalle, len(archivos))

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
        self._log_fenomenos(archivos, por_defecto, "corpus escaneado")
        archivos = self._repartir_por_fenomeno(
            archivos,
            limite,
            lambda fp: self._corpus_service.determine_fenomeno(fp, por_defecto),
        )
        if not archivos:
            logger.error("No hay archivos para procesar en %s", self._corpus_service.corpus)
            return BatchSummary(total=0, ok=0, error=0, chunks_guardados=0)
        self._log_fenomenos(
            archivos,
            por_defecto,
            f"lote a procesar (límite={limite})" if limite > 0 else "lote a procesar (completo)",
        )

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
                        fuente=filepath.name,
                        fenomeno=fenomeno,
                        status="error",
                        errores=[str(exc)],
                    )
                resultados.append(resultado)
        finally:
            self.close()
        return BatchSummary.from_results(resultados)

    def close(self) -> None:
        """Cierra la conexión con el repositorio si está abierta."""
        self._repository.close()
