# Medición temporal: chunks excluidos por el encoder en Alertas_Tempranas.
# Se ejecuta ANTES y DESPUÉS del fix del extractor JSON.
import json
import re
import sys
from pathlib import Path

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.factory import ExtractorFactory
from src.models.config import ChunkingConfig
from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer

CORPUS = Path(r"repo/CORPUS_CODEFEST_AD_ASTRA_2026")
DIR_ALERTAS = CORPUS / "F3_Dinamicas_Territoriales" / "Alertas_Tempranas"
MAX_INPUT = 512

# Regex de oraciones del encoder (src/encoders/base.py).
REGEX_ORACIONES = re.compile(r"(?<=[.!?。！？])\s+")

segmenter = TextSegmenter(
    tokenizer=Tokenizer("google-bert/bert-base-multilingual-cased"),
    splitter=SentenceSplitter(),  # regex determinista (sin spacy)
)
estrategia = ChunkingStrategyFactory(segmenter).create(
    "hybrid",
    ChunkingConfig(chunk_size=400, overlap_size=80, min_chunk_tokens=50, max_tokens=512),
)
cleaner = TextCleaner(default_language="es")
factory = ExtractorFactory()

# tokenizador del encoder: cuenta CON [CLS]/[SEP]
tok_encoder = Tokenizer("google-bert/bert-base-multilingual-cased")


def contar_tokens_encoder(texto: str) -> int:
    return len(tok_encoder._enc.encode(texto, add_special_tokens=True))


def ajustar_a_limite_equiv(texto: str):
    """Réplica de EncoderStrategy.ajustar_a_limite -> None = excluido."""
    if contar_tokens_encoder(texto) <= MAX_INPUT:
        return texto
    oraciones = REGEX_ORACIONES.split(texto.strip())
    if not oraciones or not oraciones[0]:
        return None
    acumulado = ""
    for oracion in oraciones:
        candidato = f"{acumulado} {oracion}".strip()
        if contar_tokens_encoder(candidato) > MAX_INPUT:
            break
        acumulado = candidato
    return acumulado or None


archivos = sorted((DIR_ALERTAS / "alertas").glob("*.json"))
excluidos = []
truncados = []
n_chunks = 0
n_docs = 0
fallos = 0

for f in archivos:
    try:
        doc = factory.create(f).extract(f)
        doc.doc_id = str(f.resolve().relative_to(CORPUS.resolve()))
        cleaner.clean(doc)
        chunks = estrategia.chunk(doc, ChunkingConfig(chunk_size=400, overlap_size=80, min_chunk_tokens=50, max_tokens=512))
        n_docs += 1
        n_chunks += len(chunks)
        for c in chunks:
            ajustado = ajustar_a_limite_equiv(c.texto)
            if ajustado is None:
                excluidos.append((c.chunk_id, contar_tokens_encoder(c.texto)))
            elif ajustado != c.texto:
                truncados.append((c.chunk_id, contar_tokens_encoder(c.texto), contar_tokens_encoder(ajustado)))
    except Exception as exc:
        fallos += 1
        print(f"ERROR {f.name}: {exc}", file=sys.stderr)

print(f"Archivos JSON de alertas: {len(archivos)} | procesados OK: {n_docs} | fallos: {fallos}")
print(f"Chunks generados: {n_chunks}")
print(f"Chunks EXCLUIDOS (ajustar_a_limite -> None): {len(excluidos)}")
for cid, ntok in excluidos[:10]:
    print(f"  EXCLUIDO {cid} tokens_encoder={ntok}")
print(f"Chunks TRUNCADOS (parciales): {len(truncados)}")
for cid, antes, despues in truncados[:10]:
    print(f"  TRUNCADO {cid} {antes}->{despues}")
