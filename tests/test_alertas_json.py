"""
Tests del manejo estructural de JSON de alertas tempranas (Sección 3.2/4.3).

Los archivos ``Alertas_Tempranas/alertas/*.json`` concatenan campos sin
puntuación real (listados de códigos ``001-17 001-18 ...`` y enumeraciones
separadas por ``;``). Sin normalización, la sección se lee como una sola
oración de >512 tokens y el encoder excluye el chunk del índice
(``ajustar_a_limite`` -> ``None``).

Estos tests verifican el fix en ``JSONExtractor._normalizar_parrafo``:
  * ningún chunk supera ``max_input_tokens=512`` (tokenizador propio del
    encoder, con tokens especiales ``[CLS]``/``[SEP]``),
  * ninguna oración queda partida entre chunks consecutivos,
  * ninguna oración de la sección original se pierde.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.cleaning.text_cleaner import TextCleaner
from src.extractors.json_extractor import JSONExtractor
from src.models.config import ChunkingConfig
from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer

MODELO_ENCODER = "google-bert/bert-base-multilingual-cased"
MAX_INPUT_TOKENS = 512

# Tokenizador real del encoder (el mismo que usa EncoderStrategy.contar_tokens).
_tokenizer = Tokenizer(MODELO_ENCODER)
_ENCODER_DISPONIBLE = _tokenizer._enc is not None

# Splitter de oraciones DETERMINISTA: modelo spacy inexistente -> fallback regex
# (independiente de si spacy está instalado en la máquina que corre los tests).
_splitter = SentenceSplitter(model="__force_regex__")


def _codigos(n: int = 85) -> str:
    """Genera un listado de códigos NNN-NN como el de las alertas reales."""
    return " ".join(
        f"{n:03d}-{y}"
        for n in range(1, n + 1)
        for y in (18, 19, 20, 21, 22, 23)
    )


def _alerta_representativa() -> dict:
    """Réplica fiel de la estructura de ALERTAS_030-23-91888.json."""
    return {
        "url": "https://alertastempranas.defensoria.gov.co/Alerta/Details/91888",
        "title": "Mapa",
        "fields": {},
        "body_paragraphs": [
            "Narcotráfico; Minería ilegal; Contrabando; Préstamos gota a gota",
            "ELN EPL Facciones disidentes de las FARC-EP Autodefensas Gaitanistas "
            "de Colombia (AGC) Comandos de la Frontera Grupos Armados de Crimen Organizado",
            _codigos(),
            "El escenario de riesgo se centra en las conductas contra los mecanismos "
            "de participación democrática que, en el marco del conflicto armado interno "
            "y violencias conexas, puedan constituir violaciones a los derechos humanos "
            "y al DIH, durante el proceso electoral previsto para el año 2023.",
            "Mujeres; Personas con Orientación Sexual e Identidad de Género Diversas; "
            "Afrodescendientes; Indígenas; Periodistas; Servidores públicos; Personas "
            "defensoras de Derechos Humanos, líderes y lideresas sociales; Candidatos "
            "a cargos de elección popular; Miembros de Juntas de Acción Comunal; "
            "Adolescentes; Comerciantes; Campesinos; Población socialmente estigmatizada",
        ],
    }


def _chunkear_alerta(alerta: dict) -> tuple:
    """Extrae -> limpia -> fragmenta una alerta con el pipeline real (híbrido)."""
    ruta = Path(__file__).parent / "_alerta_tmp.json"
    ruta.write_text(json.dumps(alerta, ensure_ascii=False), encoding="utf-8")
    try:
        doc = JSONExtractor().extract(ruta)
        doc.doc_id = "alerta_fake"
        TextCleaner(default_language="es").clean(doc)
        segmenter = TextSegmenter(tokenizer=_tokenizer, splitter=_splitter)
        estrategia = ChunkingStrategyFactory(segmenter).create(
            "hybrid",
            ChunkingConfig(chunk_size=400, overlap_size=80, min_chunk_tokens=50, max_tokens=512),
        )
        chunks = estrategia.chunk(
            doc, ChunkingConfig(chunk_size=400, overlap_size=80, min_chunk_tokens=50, max_tokens=512)
        )
        return doc, chunks
    finally:
        ruta.unlink(missing_ok=True)


def _tokens_con_speciales(texto: str) -> int:
    """Conteo del encoder: tokenizador propio CON [CLS]/[SEP]."""
    return len(_tokenizer._enc.encode(texto, add_special_tokens=True))


@pytest.mark.skipif(not _ENCODER_DISPONIBLE, reason="tokenizador BERT no disponible")
def test_ningun_chunk_supera_512_tokens_del_encoder():
    """Sección 4.3: cada chunk cabe en max_input_tokens usando el tokenizador del encoder."""
    _, chunks = _chunkear_alerta(_alerta_representativa())
    assert len(chunks) >= 3, "la alerta representativa debe producir varios chunks"
    for chunk in chunks:
        n = _tokens_con_speciales(chunk.texto)
        assert n <= MAX_INPUT_TOKENS, (
            f"{chunk.chunk_id}: {n} tokens > {MAX_INPUT_TOKENS} (sería excluido por el encoder)"
        )


@pytest.mark.skipif(not _ENCODER_DISPONIBLE, reason="tokenizador BERT no disponible")
def test_ninguna_oracion_queda_partida_entre_chunks():
    """Sección 3.3: los cortes caen en fronteras oracionales completas.

    (a) cada chunk es una concatenación exacta de oraciones completas, y
    (b) entre chunks consecutivos del mismo documento no hay contenido perdido
        ni oraciones partidas: la primera oración del siguiente ya apareció
        completa en el anterior (solapamiento) o el anterior termina en
        puntuación de cierre.
    """
    doc, chunks = _chunkear_alerta(_alerta_representativa())

    # (a) frontera limpia dentro de cada chunk.
    for chunk in chunks:
        oraciones = _splitter.split(chunk.texto)
        assert oraciones, f"{chunk.chunk_id}: chunk vacío de oraciones"
        reconstruido = re.sub(r"\s+", " ", " ".join(oraciones)).strip()
        assert re.sub(r"\s+", " ", chunk.texto).strip() == reconstruido, (
            f"{chunk.chunk_id}: el texto no es una concatenación de oraciones completas"
        )

    # (b) continuidad entre chunks consecutivos del mismo documento.
    for previo, siguiente in zip(chunks, chunks[1:]):
        if previo.doc_id != siguiente.doc_id:
            continue
        primeras_siguiente = _splitter.split(siguiente.texto)
        oraciones_previo = _splitter.split(previo.texto)
        assert (
            primeras_siguiente[0] in oraciones_previo
            or previo.texto.rstrip().endswith((".", "!", "?"))
        ), (
            f"frontera partida entre {previo.chunk_id} y {siguiente.chunk_id}: "
            f"la primera oración del siguiente no apareció completa en el anterior"
        )


@pytest.mark.skipif(not _ENCODER_DISPONIBLE, reason="tokenizador BERT no disponible")
def test_toda_oracion_de_la_seccion_esta_cubierta_y_cabe_en_el_limite():
    """Nada se pierde: cada oración del documento original aparece en algún
    chunk, y ninguna oración aislada supera 512 tokens (=> sin exclusiones)."""
    doc, chunks = _chunkear_alerta(_alerta_representativa())
    oraciones_seccion = _splitter.split(doc.secciones[0].texto)
    oraciones_chunks = [s for c in chunks for s in _splitter.split(c.texto)]
    for oracion in oraciones_seccion:
        assert oracion in oraciones_chunks, f"oración perdida del corpus: {oracion[:60]!r}"
        assert _tokens_con_speciales(oracion) <= MAX_INPUT_TOKENS, (
            f"oración de {_tokens_con_speciales(oracion)} tokens > {MAX_INPUT_TOKENS}"
        )


# --- Pruebas unitarias de la normalización estructural ------------------------


def test_normalizar_lista_de_codigos_a_oraciones():
    assert (
        JSONExtractor._normalizar_parrafo("001-17 001-18 001-19")
        == "001-17. 001-18. 001-19."
    )


def test_normalizar_enumeracion_con_punto_y_coma_a_oraciones():
    assert (
        JSONExtractor._normalizar_parrafo("Narcotráfico; Minería ilegal; Contrabando")
        == "Narcotráfico. Minería ilegal. Contrabando."
    )


def test_normalizar_agrega_punto_final_a_parrafo_sin_puntuacion():
    assert (
        JSONExtractor._normalizar_parrafo("ELN EPL Facciones disidentes")
        == "ELN EPL Facciones disidentes."
    )


def test_normalizar_no_toca_prosa_ya_puntuada():
    texto = "El escenario de riesgo se centra en 2023."
    assert JSONExtractor._normalizar_parrafo(texto) == texto
