"""
Script TEMPORAL de depuración: extrae los chunks de archivos PBF a un JSON.

Usa el pipeline real (extractor -> cleaner -> chunking -> metadata ->
validación) pero con un repositorio "noop": NO escribe en MongoDB, vuelca
los chunks a ``logs/pbf_chunks_debug.json``.

Uso::

    python _debug_pbf_chunks.py                 # todos los .pbf del corpus
    python _debug_pbf_chunks.py <ruta.pbf>      # solo un archivo

Se elimina al terminar la depuración.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.factory import ExtractorFactory
from src.metadata.metadata_builder import MetadataBuilder
from src.models.chunk import Chunk
from src.models.config import Settings
from src.persistence.base_repository import ChunkRepository
from src.pipeline.corpus_service import CorpusService
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.validation.chunk_validator import ChunkValidator

CORPUS = Path("repo/CORPUS_CODEFEST_AD_ASTRA_2026")
FENOMENOS = Path("data/fenomenos.json")
SALIDA = Path("logs/pbf_chunks_debug.json")


class _RepoNoop(ChunkRepository):
    """Repositorio en memoria: retiene los chunks válidos, no persiste nada."""

    def __init__(self) -> None:
        self.guardados: list[Chunk] = []

    def save_many(self, chunks: list[Chunk]) -> None:
        self.guardados = list(chunks)

    def find_by_doc_id(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.guardados if c.doc_id == doc_id]

    def exists(self, chunk_id: str) -> bool:
        return any(c.chunk_id == chunk_id for c in self.guardados)


def _doc_id_relativo(filepath: Path, corpus: Path) -> str:
    """Mismo doc_id que producción: ruta relativa al corpus."""
    try:
        return str(filepath.resolve().relative_to(corpus.resolve()))
    except ValueError:
        return filepath.stem


def _chunk_a_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "posicion": chunk.posicion,
        "num_tokens": chunk.num_tokens,
        "fenomeno": chunk.fenomeno,
        "formato": chunk.formato,
        "chunking_strategy": chunk.chunking_strategy,
        "seccion": chunk.seccion,
        "overlap_con": chunk.overlap_con,
        "texto": chunk.texto,
    }


def _procesar(pipeline: IngestionPipeline, filepath: Path, fenomeno: int) -> dict:
    repo: _RepoNoop = pipeline.repository  # type: ignore[assignment]
    resultado = pipeline.run(filepath, fenomeno)
    entrada = {
        "archivo": filepath.name,
        "ruta": str(filepath),
        "fenomeno": fenomeno,
        "status": resultado.status,
        "num_chunks": resultado.num_chunks,
        "num_guardados": resultado.num_guardados,
        "num_rechazados": resultado.num_rechazados,
        "tiempo_seg": resultado.tiempo_seg,
        "chunks": [_chunk_a_dict(c) for c in repo.guardados],
        "rechazos": list(resultado.rechazos_detalle),
    }
    if resultado.errores:
        entrada["errores"] = list(resultado.errores)
    return entrada


def main(argv: list[str]) -> int:
    settings = Settings()
    mapa = CorpusService.load_fenomenos_map(FENOMENOS)

    # Pipeline completo con repositorio noop (sin MongoDB).
    segmenter = TextSegmenter.crear(
        tokenizer_model=settings.tokenizer_model, sentence_model=settings.sentence_model
    )
    estrategia = ChunkingStrategyFactory(segmenter).create(
        settings.chunking_strategy, settings.chunking_config
    )
    repositorio = _RepoNoop()
    pipeline = IngestionPipeline(
        extractor_factory=ExtractorFactory(),
        chunking_strategy=estrategia,
        config=settings.chunking_config,
        cleaner=TextCleaner(default_language=settings.default_language),
        metadata_builder=MetadataBuilder(default_fuente=settings.default_fuente),
        validator=ChunkValidator(max_tokens=settings.max_tokens),
        repository=repositorio,
        doc_id_generator=lambda fp, fen: _doc_id_relativo(fp, CORPUS),
    )
    servicio = CorpusService(CORPUS, mapa)

    # Blancos: un archivo puntual o todos los .pbf del corpus.
    if len(argv) > 1:
        blancos = [Path(argv[1])]
    else:
        blancos = [p for p in servicio.scan() if p.suffix.lower() == ".pbf"]
    if not blancos:
        print("No se encontraron archivos PBF.")
        return 1

    archivos: list[dict] = []
    for p in blancos:
        try:
            fenomeno = servicio.determine_fenomeno(p)
            archivos.append(_procesar(pipeline, p, fenomeno))
            print(f"  OK  {p.name} | chunks={archivos[-1]['num_chunks']} "
                  f"guardados={archivos[-1]['num_guardados']} rechazados={archivos[-1]['num_rechazados']}")
        except Exception as exc:  # noqa: BLE001 - el script reporta y continúa
            archivos.append({
                "archivo": p.name,
                "ruta": str(p),
                "status": "error",
                "errores": [f"{type(exc).__name__}: {exc}"],
            })
            print(f" ERR  {p.name} | {exc}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(
        json.dumps(
            {
                "generado": datetime.now(timezone.utc).isoformat(),
                "total_archivos": len(archivos),
                "archivos": archivos,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ok = sum(1 for a in archivos if a["status"] == "ok")
    print(f"\nListo: {ok}/{len(archivos)} OK -> {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
