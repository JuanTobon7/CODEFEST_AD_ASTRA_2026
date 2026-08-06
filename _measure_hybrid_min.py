"""Medición temporal: clasifica los chunks < min_chunk_tokens del híbrido real."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.factory import ExtractorFactory
from src.models.config import ChunkingConfig
from src.pipeline.batch_ingestor import BatchIngestor
from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer

CONFIG = ChunkingConfig(chunk_size=400, overlap_size=80, min_chunk_tokens=150, max_tokens=512)
segmenter = TextSegmenter(Tokenizer(), SentenceSplitter())
estrategia = ChunkingStrategyFactory(segmenter).create("hybrid", CONFIG)

CORPUS = Path("repo/CORPUS_CODEFEST_AD_ASTRA_2026")


def procesar(archivo: Path, fenomeno: int) -> Counter:
    doc = ExtractorFactory().create(archivo).extract(archivo)
    doc = TextCleaner().clean(doc)
    doc.doc_id = BatchIngestor._doc_id_relativo(archivo, CORPUS)
    doc.fenomeno = fenomeno
    chunks = estrategia.chunk(doc, CONFIG)
    clasificacion = Counter()
    for c in chunks:
        if c.num_tokens >= 150:
            clasificacion[">=150"] += 1
            continue
        if c.overlap_con is not None:
            clasificacion["<150: cola de ventana"] += 1
        else:
            clasificacion["<150: seccion/unidad unica"] += 1
    return clasificacion, len(chunks)


pdfs = sorted(CORPUS.rglob("*.pdf"))
csvs = sorted(CORPUS.rglob("*.csv"))
total = Counter()
n_docs = 0
for archivo in pdfs[:60]:
    try:
        clas, total_chunks = procesar(archivo, 1)
    except Exception as exc:
        print("ERROR", archivo.name, type(exc).__name__, str(exc)[:80])
        continue
    n_docs += 1
    for k, v in clas.items():
        total[k] += v
print(f"PDFs procesados: {n_docs} de {len(pdfs)}")
print("TOTAL por clase:", dict(total))

total_csv = Counter()
for archivo in csvs[:10]:
    try:
        clas, _ = procesar(archivo, 1)
    except Exception as exc:
        print("ERROR", archivo.name, type(exc).__name__, str(exc)[:80])
        continue
    for k, v in clas.items():
        total_csv[k] += v
print()
print(f"CSVs procesados: {min(10, len(csvs))}")
print("CSV por clase:", dict(total_csv))
