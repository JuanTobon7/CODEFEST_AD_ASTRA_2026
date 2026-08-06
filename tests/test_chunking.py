"""
Tests de las estrategias de chunking (híbrida, estructural, semántica y por párrafo).
"""

from __future__ import annotations

import pytest

from src.chunking.base import TextSegmenter
from src.chunking.factory import ChunkingStrategyFactory
from src.chunking.hybrid_strategy import HybridChunkingStrategy
from src.chunking.paragraph_overlap_strategy import ParagraphOverlapChunkingStrategy
from src.chunking.paragraph_strategy import ParagraphChunkingStrategy
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


def test_hybrid_cola_corta_se_absorbe_en_la_ventana_anterior(
    segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig
):
    """La última ventana por debajo de min_chunk_tokens se absorbe en la anterior.

    El texto (20 oraciones de 6 tokens) deja un sobrante final de 4 oraciones
    (24 tokens < min_chunk_tokens=25): sin la absorción serían 3 chunks, con
    ella quedan 2 y ningún fragmento baja del mínimo.
    """
    oraciones = [f"Oración {i} de ejemplo corta." for i in range(20)]
    secciones = [_sec(" ".join(oraciones), orden=0)]
    chunks = HybridChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) == 2, "La cola corta debe incorporarse a la ventana anterior"
    assert all(c.num_tokens >= chunking_config_pequeno.min_chunk_tokens for c in chunks), (
        "Ningún chunk debe quedar por debajo de min_chunk_tokens"
    )
    assert "Oración 0 de ejemplo corta." in chunks[0].texto
    assert "Oración 19 de ejemplo corta." in chunks[-1].texto  # cobertura completa
    assert chunks[-1].overlap_con == chunks[0].chunk_id


def test_hybrid_cola_corta_absorbe_si_cabe(segmenter: TextSegmenter):
    """_absorber_cola_corta: fusiona la cola corta en la ventana anterior."""
    config = ChunkingConfig(chunk_size=500, overlap_size=80, min_chunk_tokens=100, max_tokens=512)
    estrategia = HybridChunkingStrategy(segmenter)
    oraciones = [" ".join(f"p{i}" for i in range(200)) + ".", "Cola corta."]
    ventanas = [(0, 0), (1, 1)]  # 201 tokens + 2 = 203 <= 512
    assert estrategia._absorber_cola_corta(ventanas, oraciones, config) == [(0, 1)]


def test_hybrid_cola_corta_no_se_absorbe_si_supera_max_tokens(segmenter: TextSegmenter):
    """_absorber_cola_corta: si absorberla supera max_tokens, se conserva la cola."""
    config = ChunkingConfig(chunk_size=500, overlap_size=80, min_chunk_tokens=100, max_tokens=512)
    estrategia = HybridChunkingStrategy(segmenter)
    oraciones = [" ".join(f"p{i}" for i in range(520)) + ".", "Cola corta."]
    ventanas = [(0, 0), (1, 1)]  # 521 tokens + 2 = 523 > 512
    assert estrategia._absorber_cola_corta(ventanas, oraciones, config) == [(0, 0), (1, 1)]


def test_hybrid_cola_corta_solo_absorbe_la_ultima_ventana(segmenter: TextSegmenter):
    """_absorber_cola_corta: solo toca la última ventana y solo si llega al final."""
    config = ChunkingConfig(chunk_size=500, overlap_size=80, min_chunk_tokens=100, max_tokens=512)
    estrategia = HybridChunkingStrategy(segmenter)
    oraciones = [
        " ".join(f"p{i}" for i in range(200)) + ".",
        "Cola corta.",
        "Ultima oración de relleno.",
        "Ultima oración de relleno dos.",
    ]
    ventanas = [(0, 0), (1, 1), (2, 2)]
    # La última ventana no termina en la última oración -> intacta.
    assert estrategia._absorber_cola_corta(ventanas, oraciones, config) == ventanas
    # Con una sola ventana -> intacta.
    assert estrategia._absorber_cola_corta([(0, 1)], oraciones, config) == [(0, 1)]


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
    assert isinstance(factory.create("paragraph", chunking_config), ParagraphChunkingStrategy)
    assert isinstance(factory.create("paragraph_overlap", chunking_config), ParagraphOverlapChunkingStrategy)
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


# --- Estrategia por párrafo pura -------------------------------------------------


def test_paragraph_respeta_saltos_de_parrafo(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Cada párrafo original (separado por línea en blanco) es un chunk exacto."""
    texto = (
        "Primer párrafo del documento con contenido sustancial.\n\n"
        "Segundo párrafo del documento con contenido sustancial.\n\n"
        "Tercer párrafo del documento con contenido sustancial."
    )
    secciones = [_sec(texto, orden=0, titulo="Introducción")]
    chunks = ParagraphChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)

    assert len(chunks) == 3
    assert [c.texto for c in chunks] == [
        "Primer párrafo del documento con contenido sustancial.",
        "Segundo párrafo del documento con contenido sustancial.",
        "Tercer párrafo del documento con contenido sustancial.",
    ]
    assert [c.posicion for c in chunks] == [0, 1, 2]
    assert all(c.chunking_strategy == "paragraph" for c in chunks)
    assert all(c.seccion == "Introducción" for c in chunks)


def test_paragraph_conserva_lineas_envueltas_del_parrafo(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Los saltos de línea simples son líneas envueltas: NO separan párrafos."""
    texto = "Primera línea del párrafo\nsegunda línea envuelta\ntercera línea envuelta"
    secciones = [_sec(texto, orden=0)]
    chunks = ParagraphChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)

    assert len(chunks) == 1
    assert "segunda línea envuelta" in chunks[0].texto


def test_paragraph_no_fusiona_parrafos_pequenos(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Aunque estén por debajo de min_chunk_tokens, los párrafos no se fusionan."""
    texto = "Corto uno.\n\nCorto dos.\n\nCorto tres."
    secciones = [_sec(texto, orden=0)]
    chunks = ParagraphChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)

    assert len(chunks) == 3, "La estrategia por párrafo respeta la estructura del autor"


def test_paragraph_parrafo_gigante_se_parte_por_tokens(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Un párrafo que excede max_tokens se parte por tokens sin perder contenido."""
    parrafo_gigante = " ".join(f"palabra_{i}" for i in range(600)) + "."
    secciones = [_sec(parrafo_gigante, orden=0)]
    chunks = ParagraphChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)

    assert len(chunks) >= 2
    assert all(c.num_tokens <= chunking_config.max_tokens for c in chunks)
    reconstruido = " ".join(c.texto for c in chunks)
    assert "palabra_0" in reconstruido and "palabra_599" in reconstruido


def test_paragraph_seccion_atomica_un_solo_chunk(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Filas CSV/XLSX y elementos PBF (splittable=False) no se segmentan en párrafos."""
    fila = "fila1: a b c\nfila2: d e f\nfila3: g h i"
    secciones = [_sec(fila, orden=0, splittable=False)]
    chunks = ParagraphChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)

    assert len(chunks) == 1
    assert "fila3: g h i" in chunks[0].texto


# --- Híbrida por párrafo + superposición ------------------------------------------


def _muchos_parrafos(n: int, frase: str = "Párrafo con contenido temático para la prueba") -> str:
    """Genera n párrafos separados por línea en blanco."""
    return "\n\n".join(f"{frase} {i}." for i in range(n))


def test_paragraph_overlap_seccion_corta_un_chunk(segmenter: TextSegmenter, chunking_config: ChunkingConfig):
    """Sección por debajo de chunk_size: un solo chunk que preserva los párrafos."""
    secciones = [_sec("Un párrafo breve.\n\nOtro párrafo breve.", orden=0)]
    chunks = ParagraphOverlapChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config)

    assert len(chunks) == 1
    assert "\n\n" in chunks[0].texto  # se preserva la estructura de párrafos
    assert chunks[0].overlap_con is None


def test_paragraph_overlap_genera_ventanas_solapadas(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Muchos párrafos se agrupan en ventanas consecutivas con overlap."""
    secciones = [_sec(_muchos_parrafos(12), orden=0)]
    chunks = ParagraphOverlapChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) >= 2
    for c in chunks[1:]:
        assert c.overlap_con is not None, "Las ventanas consecutivas deben solaparse"
    # Cobertura completa: el primer y el último párrafo aparecen.
    assert "Párrafo con contenido temático para la prueba 0." in chunks[0].texto
    assert "Párrafo con contenido temático para la prueba 11." in chunks[-1].texto
    # Cada chunk agrupa párrafos completos (nunca los parte).
    for c in chunks:
        partes = c.texto.split("\n\n")
        assert partes, "El chunk no debe estar vacío"
        assert all("Párrafo con contenido temático" in p for p in partes)


def test_paragraph_overlap_nunca_parte_un_parrafo(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Los límites de ventana caen siempre en límites de párrafo original."""
    parrafos = [f"Párrafo {i} con contenido suficiente para la prueba de ventanas." for i in range(15)]
    secciones = [_sec("\n\n".join(parrafos), orden=0)]
    chunks = ParagraphOverlapChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    for chunk in chunks:
        for parte in chunk.texto.split("\n\n"):
            assert parte in parrafos, f"Fragmento ajeno a los párrafos originales: {parte!r}"


def test_paragraph_overlap_seccion_atomica_un_chunk(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Las unidades atómicas nunca se ventanean ni se parten."""
    fila_larga = "col1: " + " ".join(f"v{i}" for i in range(200)) + " col2: fin"
    secciones = [_sec(fila_larga, orden=0, splittable=False)]
    chunks = ParagraphOverlapChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) == 1
    assert "col1:" in chunks[0].texto and "col2: fin" in chunks[0].texto


def test_paragraph_overlap_parrafo_que_excede_limite(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Un párrafo gigante se parte por tokens pero el resto sigue en ventanas."""
    gigante = " ".join(f"palabra_{i}" for i in range(700)) + "."
    texto = f"{gigante}\n\nPárrafo final corto."
    secciones = [_sec(texto, orden=0)]
    chunks = ParagraphOverlapChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) >= 2
    assert all(c.num_tokens <= chunking_config_pequeno.max_tokens for c in chunks)
    reconstruido = " ".join(c.texto for c in chunks)
    assert "palabra_0" in reconstruido and "palabra_699" in reconstruido
    assert "Párrafo final corto." in reconstruido


def test_paragraph_overlap_metadata(segmenter: TextSegmenter, chunking_config_pequeno: ChunkingConfig):
    """Cada fragmento lleva la metadata base de la Tabla 1 con la estrategia correcta."""
    secciones = [_sec(_muchos_parrafos(10), orden=0, titulo="Sección A")]
    chunks = ParagraphOverlapChunkingStrategy(segmenter).chunk(_doc(secciones), chunking_config_pequeno)

    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"doc_test__chunk_{i:05d}"
        assert c.posicion == i
        assert c.chunking_strategy == "paragraph_overlap"
        assert c.seccion == "Sección A"
        assert c.fuente == "prueba.md"
        assert c.formato == "md"
