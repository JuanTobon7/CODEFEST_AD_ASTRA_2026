"""Tests de NER (gazetteer regex), RE (co-ocurrencia) y pipeline de extracción.

Deterministas y sin red/modelos: verifican la normalización de entidades,
el matching por spans (gana la entrada más larga), la tipificación por
patrones verbales y la trazabilidad de las tripletas (doc_id/chunk_id).
"""

from __future__ import annotations

import pytest

from src.knowledge_graph.extract.base import normalizar_id_entidad
from src.knowledge_graph.extract.cooccurrence_relation_strategy import (
    CooccurrenceRelationExtractor,
)
from src.knowledge_graph.extract.factory import (
    EntityExtractorFactory,
    RelationExtractorFactory,
)
from src.knowledge_graph.extract.pipeline import ExtractionPipeline
from src.knowledge_graph.extract.regex_entity_strategy import (
    RegexEntityExtractor,
    RegexEntityExtractorConfig,
)
from src.knowledge_graph.models import Entity, EntityType, RelationType

# Importar las estrategias registra sus decoradores en las factories.
import src.knowledge_graph.extract.regex_entity_strategy  # noqa: F401
import src.knowledge_graph.extract.cooccurrence_relation_strategy  # noqa: F401


# -- Normalización ----------------------------------------------------------

def test_normalizar_id_entidad_quita_acentos_y_minusculas():
    assert normalizar_id_entidad("Perú") == "peru"
    assert normalizar_id_entidad("  ÓRBITA  Terrestre  ") == "orbita terrestre"
    assert normalizar_id_entidad("Estados Unidos") == "estados unidos"


# -- NER --------------------------------------------------------------------

def test_ner_detecta_entidades_del_gazetteer():
    ner = RegexEntityExtractor()
    entidades = ner.extraer_entidades(
        "La NASA coopera con la ESA en la Estación Espacial Internacional."
    )
    ids = {e.id: e.tipo for e in entidades}
    assert ids["nasa"] == EntityType.ORGANIZACION
    assert ids["esa"] == EntityType.ORGANIZACION
    assert ids["estacion espacial internacional"] == EntityType.PROGRAMA


def test_ner_normaliza_acentos_para_matching():
    ner = RegexEntityExtractor()
    entidades = ner.extraer_entidades("La órbita terrestre baja es crítica para Perú.")
    ids = {e.id for e in entidades}
    assert "orbita terrestre baja" in ids
    assert "peru" in ids


def test_ner_gana_la_entrada_mas_larga_no_la_subcadena():
    # Se añade "inteligencia" suelta al gazetteer: el span de la entrada más
    # larga ("inteligencia artificial") debe cubrirla y dejarla fuera.
    config = RegexEntityExtractorConfig(
        gazetteer_adicional={"inteligencia": EntityType.TECNOLOGIA}
    )
    ner = RegexEntityExtractor(config=config)
    entidades = ner.extraer_entidades(
        "La inteligencia artificial transforma la defensa del sector."
    )
    ids = {e.id for e in entidades}
    assert "inteligencia artificial" in ids
    assert "inteligencia" not in ids


def test_ner_nombres_propios_desactivables():
    # "Fuerza Aérea Brasileña" no está en el gazetteer: solo la detecta el
    # patrón genérico de nombres propios (que se puede desactivar).
    texto = "La Fuerza Aérea Brasileña adoptó drones para vigilancia."
    con_propios = RegexEntityExtractor()
    sin_propios = RegexEntityExtractor(
        config=RegexEntityExtractorConfig(detectar_nombres_propios=False)
    )
    ids_con = {e.id for e in con_propios.extraer_entidades(texto)}
    ids_sin = {e.id for e in sin_propios.extraer_entidades(texto)}
    assert "fuerza aerea brasilena" in ids_con
    assert "fuerza aerea brasilena" not in ids_sin
    assert ids_sin < ids_con


def test_ner_gazetteer_adicional_por_config():
    config = RegexEntityExtractorConfig(
        gazetteer_adicional={"ministerio de defensa": EntityType.ORGANIZACION}
    )
    ner = RegexEntityExtractor(config=config)
    entidades = ner.extraer_entidades("El Ministerio de Defensa aprobó el plan.")
    assert any(e.id == "ministerio de defensa" for e in entidades)


def test_ner_dedup_por_id_canonico():
    ner = RegexEntityExtractor()
    entidades = ner.extraer_entidades("Perú y PERÚ cooperan con MÉXICO y con México.")
    ids = [e.id for e in entidades]
    assert ids.count("peru") == 1
    assert ids.count("mexico") == 1


def test_ner_alias_bilingue_es_en_mismo_nodo():
    """El corpus es EN pero las consultas son ES: los alias fusionan el nodo."""
    ner = RegexEntityExtractor()
    es = ner.extraer_entidades("La basura espacial afecta a la órbita terrestre baja.")
    en = ner.extraer_entidades("Space debris threatens low earth orbit.")
    ids_es = {e.id for e in es}
    ids_en = {e.id for e in en}
    assert "basura espacial" in ids_es
    assert "basura espacial" in ids_en  # "space debris" → alias → mismo nodo
    assert "orbita terrestre baja" in ids_es
    assert "orbita terrestre baja" in ids_en  # "low earth orbit" → alias


def test_ner_alias_estacion_espacial():
    ner = RegexEntityExtractor()
    en = ner.extraer_entidades("Countries cooperate on the International Space Station.")
    assert any(e.id == "estacion espacial internacional" for e in en)


# -- RE ---------------------------------------------------------------------

def test_re_coocurrencia_sin_patron():
    re = CooccurrenceRelationExtractor()
    entidades = [
        Entity(id="nasa", nombre="NASA"),
        Entity(id="esa", nombre="ESA"),
    ]
    relaciones = re.extraer_relaciones("La NASA y la ESA trabajan este año.", entidades)
    assert len(relaciones) == 2  # ambos sentidos
    assert all(r.tipo == RelationType.COOCURRENCIA for r in relaciones)
    assert {r.confianza for r in relaciones} == {0.6}


def test_re_patron_verbal_tipifica_relacion():
    re = CooccurrenceRelationExtractor()
    entidades = [
        Entity(id="basura espacial", nombre="basura espacial"),
        Entity(id="orbita terrestre baja", nombre="órbita terrestre baja"),
    ]
    relaciones = re.extraer_relaciones(
        "La basura espacial afecta a la órbita terrestre baja.", entidades
    )
    por_par = {(r.sujeto, r.objeto): r.tipo for r in relaciones}
    assert por_par[("basura espacial", "orbita terrestre baja")] == RelationType.AFECTA
    # El sentido inverso también se reporta (el grafo es no dirigido).
    assert por_par[("orbita terrestre baja", "basura espacial")] == RelationType.AFECTA


def test_re_solo_relaciones_entre_entidades_presentes():
    re = CooccurrenceRelationExtractor()
    entidades = [
        Entity(id="nasa", nombre="NASA"),
        Entity(id="esa", nombre="ESA"),
        Entity(id="spacex", nombre="SpaceX"),
    ]
    # "SpaceX" no aparece en el texto → no debe generar relaciones.
    relaciones = re.extraer_relaciones("La NASA coopera con la ESA.", entidades)
    assert all(r.sujeto != "spacex" and r.objeto != "spacex" for r in relaciones)


def test_re_dedup_par_tipo():
    re = CooccurrenceRelationExtractor()
    entidades = [
        Entity(id="nasa", nombre="NASA"),
        Entity(id="esa", nombre="ESA"),
    ]
    texto = "La NASA coopera con la ESA. También la NASA coopera con la ESA."
    relaciones = re.extraer_relaciones(texto, entidades)
    assert len(relaciones) == 2  # un par por sentido, sin duplicados


# -- Pipeline ---------------------------------------------------------------

def test_pipeline_procesa_chunk_con_trazabilidad():
    ner = RegexEntityExtractor()
    re = CooccurrenceRelationExtractor()
    pipeline = ExtractionPipeline(ner=ner, re=re)
    resultado = pipeline.procesar_chunk(
        "doc-1",
        "chunk-7",
        "La NASA coopera con la ESA en la Estación Espacial Internacional.",
    )
    assert resultado.doc_id == "doc-1"
    assert resultado.chunk_id == "chunk-7"
    assert len(resultado.entidades) >= 3
    tripletas = resultado.tripletas()
    assert tripletas
    for t in tripletas:
        assert t.doc_id == "doc-1"
        assert t.chunk_id == "chunk-7"
        assert t.evidencia  # fragmento textual de sustento


def test_pipeline_pasos_independientes():
    ner = RegexEntityExtractor()
    re = CooccurrenceRelationExtractor()
    pipeline = ExtractionPipeline(ner=ner, re=re)
    assert [paso.nombre for paso in pipeline.pasos] == [
        "ner:regex-gazetteer",
        "re:coocurrencia-oracional",
    ]


# -- Factories (Factory Method, registro por decorador) ---------------------

def test_factories_registran_estrategias():
    assert "regex-gazetteer" in EntityExtractorFactory.list_available()
    assert "coocurrencia-oracional" in RelationExtractorFactory.list_available()


def test_factory_crea_y_devuelve_la_interfaz():
    extractor = EntityExtractorFactory.create("regex-gazetteer")
    assert isinstance(extractor, RegexEntityExtractor)
    relacionador = RelationExtractorFactory.create("coocurrencia-oracional")
    assert isinstance(relacionador, CooccurrenceRelationExtractor)


def test_factory_nombre_desconocido():
    with pytest.raises(ValueError):
        EntityExtractorFactory.create("no-existe")
    with pytest.raises(ValueError):
        RelationExtractorFactory.create("no-existe")
