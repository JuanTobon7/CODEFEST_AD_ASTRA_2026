"""
Script TEMPORAL: reanuda la ingesta interrumpida (corte de luz).

Qué hace, en orden:

1. Escanea el corpus con el mismo orden estable que ``BatchIngestor`` (así el
   índice del archivo N de esta corrida es el mismo que el de la corrida caída).
2. Recorre el JSON de chunks en streaming y recoge los ``doc_id`` persistidos.
3. Identifica el ÚLTIMO documento del scan que tiene chunks: ese es el que
   estaba a medias cuando se fue la luz (o el que acababa de terminar; en
   cualquier caso reprocesarlo es idempotente).
4. Borra sus chunks del JSON (reescritura en streaming + reemplazo atómico).
5. Reanuda la ingesta DESDE ese documento (incluido) hasta el final del corpus.

Nota sobre documentos sin chunks: un archivo que no produjo ningún fragmento
válido (error de extracción, todo rechazado) no deja rastro en el JSON, así que
el punto de reanudación se calcula sobre el último documento CON chunks. Los
archivos sin chunks que hubiera entre medias se reprocesan: es la dirección
conservadora (repetir trabajo barato antes que saltarse un documento).

Uso::

    python _reanudar_ingesta.py --solo-diagnostico     # no toca nada, solo informa
    python _reanudar_ingesta.py                        # borra y reanuda

Se apoya en dos métodos privados de ``JsonChunkRepository``
(``_iter_objetos_json`` y ``_ruta_efectiva``) para no duplicar aquí el parser
incremental del JSON de ~330 MB. Es deliberado: esto es un script de un solo uso.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Set, Tuple

from src.models.batch_summary import BatchSummary
from src.models.chunk import Chunk
from src.models.config import Settings
from src.models.pipeline_result import IngestionResult
from src.persistence.json_repository import JsonChunkRepository
from src.pipeline.batch_ingestor import BatchIngestor
from src.pipeline.corpus_service import CorpusService

logger = logging.getLogger("reanudar")


class _RepositorioBufferizado:
    """Acumula los chunks de varios documentos antes de escribir el JSON.

    ``JsonChunkRepository.save_many`` reescribe el archivo COMPLETO en cada
    llamada, es decir una vez por documento: con ~330 MB ya persistidos eso son
    cientos de MB de I/O por archivo del corpus (coste cuadrático). Aquí se
    agrupan ``cada`` documentos en una sola escritura.

    El precio es que un nuevo corte de luz pierde como mucho ``cada``
    documentos, que es exactamente lo que este script sabe recuperar.
    Con ``--flush-cada 1`` el comportamiento es idéntico al original.
    """

    def __init__(self, base: JsonChunkRepository, cada: int) -> None:
        self._base = base
        self._cada = max(1, cada)
        self._pendientes: List[Chunk] = []
        self._docs_sin_volcar = 0

    def connect(self) -> None:
        self._base.connect()

    def save_many(self, chunks: List[Chunk]) -> None:
        self._pendientes.extend(chunks)
        self._docs_sin_volcar += 1
        if self._docs_sin_volcar >= self._cada:
            self.flush()

    def flush(self) -> None:
        """Vuelca lo pendiente al JSON (upsert por ``chunk_id``)."""
        self._docs_sin_volcar = 0
        if not self._pendientes:
            return
        logger.info("Volcando %d fragmentos pendientes al JSON…", len(self._pendientes))
        self._base.save_many(self._pendientes)
        self._pendientes.clear()

    def close(self) -> None:
        self.flush()
        self._base.close()


def _configurar_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _doc_ids_persistidos(repositorio: JsonChunkRepository) -> Set[str]:
    """``doc_id`` distintos presentes en el JSON, leído en streaming."""
    if repositorio._ruta_efectiva() is None:
        return set()
    doc_ids: Set[str] = set()
    total = 0
    for objeto in repositorio._iter_objetos_json():
        total += 1
        doc_id = objeto.get("doc_id")
        if doc_id:
            doc_ids.add(doc_id)
    logger.info(
        "JSON actual | %d fragmentos | %d documentos distintos", total, len(doc_ids)
    )
    return doc_ids


def _eliminar_chunks_de(repositorio: JsonChunkRepository, doc_id: str) -> Tuple[int, int]:
    """Reescribe el JSON sin los chunks de ``doc_id``.

    Se copia registro a registro a un temporal en el mismo directorio y se
    reemplaza atómicamente, igual que hace el repositorio: si esto se corta a
    mitad, el JSON original queda intacto.

    Returns:
        ``(eliminados, conservados)``.
    """
    eliminados = 0
    conservados = 0
    temporal: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=repositorio.path.parent,
            prefix=f".{repositorio.path.stem}.reanudar.",
            suffix=".tmp",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            archivo.write("[\n")
            for objeto in repositorio._iter_objetos_json():
                if objeto.get("doc_id") == doc_id:
                    eliminados += 1
                    continue
                if conservados:
                    archivo.write(",\n")
                json.dump(objeto, archivo, ensure_ascii=False)
                conservados += 1
            archivo.write("\n]\n")
        os.replace(temporal, repositorio.path)
        temporal = None
    finally:
        if temporal is not None and temporal.exists():
            temporal.unlink()
    return eliminados, conservados


def _indice_reanudacion(
    archivos: List[Path], corpus: Path, persistidos: Set[str]
) -> Tuple[int, Optional[str]]:
    """Índice (en el scan) del último documento con chunks persistidos.

    Returns:
        ``(indice, doc_id)``; ``(0, None)`` si el JSON está vacío.
    """
    ultimo_indice = -1
    ultimo_doc_id: Optional[str] = None
    vistos: Set[str] = set()
    for indice, archivo in enumerate(archivos):
        doc_id = BatchIngestor._doc_id_relativo(archivo, corpus)
        vistos.add(doc_id)
        if doc_id in persistidos:
            ultimo_indice = indice
            ultimo_doc_id = doc_id
    huerfanos = persistidos - vistos
    if huerfanos:
        logger.warning(
            "%d doc_id del JSON no están en el corpus escaneado (p. ej. %s). "
            "Se dejan intactos.",
            len(huerfanos),
            ", ".join(sorted(huerfanos)[:3]),
        )
    if ultimo_indice < 0:
        return 0, None
    return ultimo_indice, ultimo_doc_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reanuda la ingesta interrumpida")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("repo/CORPUS_CODEFEST_AD_ASTRA_2026"),
        help="Directorio raíz del corpus (debe ser el MISMO de la corrida caída)",
    )
    parser.add_argument("--fenomenos", type=Path, default=Path("data/fenomenos.json"))
    parser.add_argument("--extensiones", nargs="*", default=None)
    parser.add_argument(
        "--desde",
        default=None,
        help="Forzar el doc_id de reanudación en vez de detectarlo (ruta relativa al corpus)",
    )
    parser.add_argument(
        "--solo-diagnostico",
        action="store_true",
        help="Informa el punto de reanudación sin borrar nada ni ingerir",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copia el JSON a <nombre>.bak antes de borrar (ocupa lo mismo que el original)",
    )
    parser.add_argument(
        "--flush-cada",
        type=int,
        default=20,
        help="Documentos a agrupar por escritura del JSON (1 = comportamiento original)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _configurar_logging(args.verbose)

    settings = Settings()
    if settings.chunk_repository.strip().lower() != "json":
        logger.error(
            "Este script solo cubre CHUNK_REPOSITORY=json (actual: %s)",
            settings.chunk_repository,
        )
        return 2

    repositorio = JsonChunkRepository(settings.chunks_json_path)
    logger.info("Repositorio de chunks: %s", repositorio.path)

    corpus = args.corpus
    mapeo: Dict[str, int] = CorpusService.load_fenomenos_map(args.fenomenos)
    servicio = CorpusService(corpus, mapeo)
    archivos = servicio.scan(args.extensiones)
    if not archivos:
        logger.error("El corpus %s no tiene archivos que procesar", corpus)
        return 1
    logger.info("Corpus escaneado: %d archivos", len(archivos))

    # 1. Punto de reanudación -------------------------------------------------
    if args.desde:
        doc_ids = [BatchIngestor._doc_id_relativo(a, corpus) for a in archivos]
        if args.desde not in doc_ids:
            logger.error("--desde '%s' no corresponde a ningún archivo del corpus", args.desde)
            return 2
        indice, doc_objetivo = doc_ids.index(args.desde), args.desde
    else:
        persistidos = _doc_ids_persistidos(repositorio)
        indice, doc_objetivo = _indice_reanudacion(archivos, corpus, persistidos)

    if doc_objetivo is None:
        logger.warning("No hay chunks persistidos: se ingiere el corpus desde el principio")
    else:
        logger.info(
            "Último documento analizado: %s (archivo %d de %d)",
            doc_objetivo,
            indice + 1,
            len(archivos),
        )
    pendientes = archivos[indice:]
    logger.info("Quedan %d archivos por procesar (reprocesando el último)", len(pendientes))

    if args.solo_diagnostico:
        logger.info("Modo diagnóstico: no se borra ni se ingiere nada.")
        return 0

    # 2. Borrado de los chunks del documento a medias -------------------------
    if doc_objetivo is not None:
        if args.backup:
            destino = repositorio.path.with_suffix(repositorio.path.suffix + ".bak")
            logger.info("Copia de seguridad -> %s", destino)
            shutil.copy2(repositorio.path, destino)
        logger.info("Borrando los chunks de %s…", doc_objetivo)
        eliminados, conservados = _eliminar_chunks_de(repositorio, doc_objetivo)
        logger.info(
            "Chunks eliminados=%d | conservados en el JSON=%d", eliminados, conservados
        )

    # 3. Reanudación ----------------------------------------------------------
    ingestor = BatchIngestor(settings, corpus, mapeo)
    buffer = _RepositorioBufferizado(ingestor._repository, args.flush_cada)
    ingestor._pipeline.repository = buffer
    buffer.connect()

    resultados: List[IngestionResult] = []
    try:
        for numero, filepath in enumerate(pendientes, start=1):
            fenomeno = servicio.determine_fenomeno(filepath, 1)
            logger.info(
                "[%d/%d] (global %d/%d) %s",
                numero,
                len(pendientes),
                indice + numero,
                len(archivos),
                filepath.name,
            )
            try:
                resultados.append(ingestor._pipeline.run(filepath, fenomeno))
            except Exception as exc:  # noqa: BLE001 - el lote continúa
                logger.exception("Fallo al procesar %s", filepath)
                resultados.append(
                    IngestionResult(
                        fuente=filepath.name,
                        fenomeno=fenomeno,
                        status="error",
                        errores=[str(exc)],
                    )
                )
    except KeyboardInterrupt:
        logger.warning("Interrumpido por el usuario: se vuelca lo pendiente antes de salir")
    finally:
        buffer.close()

    resumen = BatchSummary.from_results(resultados)
    resumen.log_resumen()
    return 0 if resumen.error == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
