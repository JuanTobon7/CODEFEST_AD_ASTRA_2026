"""
Tests de las estrategias de chunking (híbrida, estructural y semántica).
"""

from __future__ import annotations

import pytest

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.chunking.hybrid_strategy import HybridChunkingStrategy
from src.chunking.semantic_overlap_strategy import SemanticOverlapChunkingStrategy
from src.chunking.structural_strategy import StructuralChunkingStrategy
from src.models.config import ChunkingConfig
from src.models.extracted_document import ExtractedDocument, Formato, Section


def _doc(secciones: list[Section], fenomeno: int = 1) -> ExtractedDocument:
    """Documento de prueba con doc_id fijo."""
    return ExtractedDocument(
        doc_id="doc_test",
        fuente="prueba.md",
        formato=Formato.MD,
        fenomeno=fenomeno,
        secciones=secciones,
    )


def _sec(texto: str, orden: int = 0, titulo=None, splittable: bool = True) -> Section:
    return Section(titulo=titulo, texto=texto, orden=orden, splittable=splittable)


def _oraciones_repetidas(frase: str, n: int) -> str:
    """Genera n oraciones repetidas terminadas en punto."""
    return " ".join(f"{frase} {i}." for i in range(n))


# --- Casos borde de la estrategia híbrida -------------------------------------


def test_hybrid_seccion_pequena_se_fusiona_con_la_anterior(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Secciones menores a min_chunk_tokens se concatenan con la adyacente."""
    secciones = [
        _sec("Sección larga con bastante contenido para superar el umbral mínimo de tokens requerido.", orden=0),
        _sec("Fragmento corto.", orden=1),
    ]
    estrategia = HybridChunkingStrategy(segmenter)
    chunks = estrategia.chunk(_doc(secciones), chunking_config)

    assert len(chunks) == 1, "La sección pequeña debe fusionarse con la anterior"
    assert "Fragmento corto." in chunks[0].texto
    assert "Sección larga" in chunks[0].texto


def test_hybrid_seccion_mayor_al_limite_se_dividide(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Una sección que excede la ventana se divide en fragmentos con overlap."""
    texto_largo = _oraciones_repetidas("Esta es una oración de ejemplo con contenido", 30)
    secciones = [_sec(texto_largo, orden=0, titulo="Introducción")]
    estrategia = HybridChunkingStrategy(segmenter)
    chunks = estrategia.chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) >= 2, "La sección grande debe producir varios fragmentos"
    # Ningún fragmento supera el tamaño de ventana (oración entera de más).
    assert all(c.num_tokens <= chunking_config_pequeno.chunk_size + 20 for c in chunks)
    # Los fragmentos consecutivos se solapan y quedan referenciados.
    for c in chunks[1:]:
        assert c.overlap_con is not None
    # El texto original se reconstruye uniendo fragmentos (con solapamiento).
    assert texto_largo.split()[0] in chunks[0].texto
    assert texto_largo.split()[-1] in chunks[-1].texto


def test_hybrid_nunca_corta_una_oracion(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """El corte de la ventana retrocede al final de la última oración completa."""
    texto = " ".join(
        f"Oración número {i} con un poco de contenido para llenar la ventana." for i in range(20)
    )
    secciones = [_sec(texto, orden=0)]
    chunks = HybridChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    for chunk in chunks:
        assert chunk.texto.rstrip().endswith((".", "?", "!")), (
            f"El chunk no termina en límite de oración: ...{chunk.texto[-40:]!r}"
        )


def test_hybrid_oracion_que_cruza_la_ventana(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Si una oración cruza la ventana, se toma completa aunque exceda el tamaño."""
    oracion_gigante = " ".join(f"palabra_{i}" for i in range(150)) + "."
    texto = f"{oracion_gigante} Segunda oración corta."
    secciones = [_sec(texto, orden=0)]
    chunks = HybridChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) == 2
    # La oración gigante no se corta: se conserva completa en el primer chunk.
    assert chunks[0].texto == oracion_gigante
    assert "Segunda oración corta." in chunks[1].texto


def test_hybrid_unidades_atomicas_no_se_parten_ni_fusionan(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Filas CSV/XLSX y elementos PBF son unidades atómicas (splittable=False)."""
    fila_larga = "col1: " + " ".join(f"v{i}" for i in range(200)) + " col2: fin"
    secciones = [
        _sec("fila1: a b c", orden=0, splittable=False),
        _sec(fila_larga, orden=1, splittable=False),
    ]
    chunks = HybridChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) == 2, "Las unidades atómicas nunca se fusionan ni se dividen"
    assert "col1:" in chunks[1].texto and "col2: fin" in chunks[1].texto


def test_hybrid_metadata_obligatoria(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Cada fragmento lleva la metadata base de la Tabla 1."""
    filler = (
        "Este es un texto de relleno suficientemente extenso para superar con holgura "
        "el umbral mínimo de tokens definido en la configuración de la estrategia de "
        "fragmentación híbrida del pipeline de ingesta. "
    ) * 3
    secciones = [
        _sec(filler + "Primera oración de prueba.", orden=0, titulo="Título uno"),
        _sec(filler + "Segunda oración de prueba.", orden=1, titulo="Título dos"),
    ]
    chunks = HybridChunkingStrategy(segmenter).chunk(_doc(secciones, fenomeno=2), chunking_config)

    assert len(chunks) == 2
    for i, chunk in enumerate(chunks):
        assert chunk.doc_id == "doc_test"
        assert chunk.chunk_id == f"doc_test__chunk_{i:05d}"
        assert chunk.fuente == "prueba.md"
        assert chunk.formato == "md"
        assert chunk.fenomeno == 2
        assert chunk.posicion == i
        assert chunk.num_tokens > 0
        assert chunk.texto
        assert chunk.chunking_strategy == "hybrid"
        assert chunk.seccion in ("Título uno", "Título dos")


# --- Comparación entre estrategias ---------------------------------------------


def test_factory_estrategias_por_nombre(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """El factory instancia cada estrategia por nombre (configurable en runtime)."""
    factory = ChunkingStrategyFactory(segmenter)
    assert isinstance(factory.create("structural", chunking_config), StructuralChunkingStrategy)
    assert isinstance(factory.create("semantic", chunking_config), SemanticOverlapChunkingStrategy)
    assert isinstance(factory.create("hybrid", chunking_config), HybridChunkingStrategy)
    with pytest.raises(ValueError, match="desconocida"):
        factory.create("otra_estrategia", chunking_config)


def test_structural_un_chunk_por_seccion(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """La estrategia estructural pura no divide por tamaño."""
    secciones = [_sec(_oraciones_repetidas("Frase de ejemplo", 30), orden=0)]
    chunks = StructuralChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)
    assert len(chunks) == 1
    assert chunks[0].chunking_strategy == "structural"


def test_semantic_ignora_estructura_y_solapa(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """La semántica pura junta todo el documento y aplica la ventana con overlap."""
    secciones = [
        _sec(_oraciones_repetidas("Primera sección con texto", 12), orden=0),
        _sec(_oraciones_repetidas("Segunda sección con texto", 12), orden=1),
    ]
    chunks = ChunkingStrategyFactory(segmenter).create("semantic", chunking_config_pequeno).chunk(
        _doc(secciones), chunking_config_pequeno
    )
    assert len(chunks) >= 2
    assert chunks[0].seccion is None  # ignora la estructura
    for c in chunks[1:]:
        assert c.overlap_con is not None
