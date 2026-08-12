"""Tests de la RE neuronal mREBEL (``mrebel``) y su parser de tripletas.

Sin red ni modelos reales: tokenizer y modelo inyectados como fakes. El
tokenizer fake codifica el ÍNDICE de la oración en ``input_ids[0, 0]`` y el
modelo fake devuelve ese índice como token generado; ``batch_decode`` mapea
el índice al texto de tripletas que el test quiera simular.

Prefijo ``kg_``: evita colisión de basename con tests existentes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.knowledge_graph.extract.base import RelationExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.extract.mrebel_relation_strategy import (
    MrebelRelationExtractor,
    extract_triplets,
)
from src.knowledge_graph.models import Entity, EntityType, RelationType

# Importar la estrategia registra su decorador en la factory (efecto secundario).
import src.knowledge_graph.extract.mrebel_relation_strategy  # noqa: F401


# -- Fakes -------------------------------------------------------------------

class _FakeTokenizadorMrebel:
    """Tokenizador mBART fake: codifica el índice de oración y decodifica."""

    def __init__(self, salidas):
        self._salidas = salidas  # dict: índice de llamada -> texto de tripletas
        self._contador = 0

    def __call__(self, texto, **kwargs):
        idx = self._contador
        self._contador += 1
        return {
            "input_ids": torch.tensor([[idx]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    def convert_tokens_to_ids(self, token):
        return 250058  # id de "tp_XX" (irrelevante para el fake)

    def batch_decode(self, ids, skip_special_tokens=False):
        return [self._salidas.get(int(fila[0]), "") for fila in ids]


class _FakeModeloMrebel:
    """Seq2seq fake: genera el índice de la oración (lo decodifica el tokenizer)."""

    def __init__(self):
        self.generate_llamadas = 0
        self.half_llamadas = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def half(self):
        self.half_llamadas += 1
        return self

    def generate(self, ids, **kwargs):
        self.generate_llamadas += 1
        return torch.tensor([[int(ids[0, 0])]], dtype=torch.long)


def _extractor(salidas=None, **kwargs):
    """Estrategia con fakes inyectados; devuelve (extractor, modelo_fake)."""
    modelo = _FakeModeloMrebel()
    extractor = MrebelRelationExtractor(
        tokenizer=_FakeTokenizadorMrebel(salidas or {}), modelo=modelo, **kwargs
    )
    return extractor, modelo


def _ent(nombre: str, tipo=EntityType.ORGANIZACION) -> Entity:
    return Entity(id=normalizar_id_entidad(nombre), nombre=nombre, tipo=tipo)


# -- Registro e interfaz ------------------------------------------------------

def test_registrada_en_factory():
    assert "mrebel" in RelationExtractorFactory.list_available()


def test_implementa_relation_extractor():
    extractor, _ = _extractor()
    assert isinstance(extractor, RelationExtractor)
    assert extractor.name == "mrebel"


# -- Parser de tripletas ------------------------------------------------------

def test_parser_formato_rebel():
    texto = "<triplet> Punta Cana <subj> Higüey <obj> located in </triplet>"
    assert extract_triplets(texto) == [
        {"head": "Punta Cana", "type": "located in", "tail": "Higüey"}
    ]


def test_parser_formato_redfm_con_tipos():
    texto = "<triplet> ESA <ORG> NASA <ORG> cooperates with </triplet>"
    assert extract_triplets(texto) == [
        {"head": "ESA", "type": "cooperates with", "tail": "NASA"}
    ]


def test_parser_multiples_tripletas():
    texto = (
        "<triplet> ESA <subj> NASA <obj> cooperates with </triplet> "
        "<triplet> ESA <subj> Madrid <obj> located in </triplet>"
    )
    assert extract_triplets(texto) == [
        {"head": "ESA", "type": "cooperates with", "tail": "NASA"},
        {"head": "ESA", "type": "located in", "tail": "Madrid"},
    ]


def test_parser_sin_tripletas_devuelve_vacio():
    assert extract_triplets("sin marcadores de tripleta") == []


# -- Clasificación ------------------------------------------------------------

def test_extrae_tripleta_mapeada():
    """Salida fake 'cooperates with' → COOPERA_CON con confianza 0.9."""
    salidas = {0: "<triplet> ESA <subj> NASA <obj> cooperates with </triplet>"}
    extractor, _ = _extractor(salidas)
    entidades = [_ent("ESA"), _ent("NASA")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    assert [(r.sujeto, r.tipo, r.objeto) for r in relaciones] == [
        ("esa", RelationType.COOPERA_CON, "nasa")
    ]
    assert relaciones[0].confianza == 0.9


def test_extrae_relacion_no_mapeada_a_relacionado_con():
    salidas = {0: "<triplet> ESA <subj> NASA <obj> sister city </triplet>"}
    extractor, _ = _extractor(salidas)
    entidades = [_ent("ESA"), _ent("NASA")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    assert [(r.sujeto, r.tipo, r.objeto) for r in relaciones] == [
        ("esa", RelationType.RELACIONADO_CON, "nasa")
    ]
    assert relaciones[0].confianza == 0.7


def test_filtra_tripletas_con_entidades_fuera_del_ner():
    """China/Rusia no están en las entidades del NER → la tripleta se descarta."""
    salidas = {
        0: (
            "<triplet> ESA <subj> NASA <obj> cooperates with </triplet> "
            "<triplet> China <subj> Rusia <obj> cooperates with </triplet>"
        )
    }
    extractor, _ = _extractor(salidas)
    entidades = [_ent("ESA"), _ent("NASA")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    assert [(r.sujeto, r.tipo, r.objeto) for r in relaciones] == [
        ("esa", RelationType.COOPERA_CON, "nasa")
    ]


def test_entidades_ausentes_sin_relaciones():
    extractor, _ = _extractor()
    entidades = [_ent("China"), _ent("Rusia")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    assert relaciones == []


def test_menos_de_dos_entidades_sin_modelo():
    modelo = _FakeModeloMrebel()
    extractor = MrebelRelationExtractor(
        tokenizer=_FakeTokenizadorMrebel({}), modelo=modelo
    )
    assert extractor.extraer_relaciones("Texto suelto.", [_ent("ESA")]) == []
    assert modelo.generate_llamadas == 0


# -- Caché --------------------------------------------------------------------

def test_cache_evita_regenerar_con_el_modelo():
    salidas = {0: "<triplet> ESA <subj> NASA <obj> cooperates with </triplet>"}
    extractor, modelo = _extractor(salidas)
    entidades = [_ent("ESA"), _ent("NASA")]
    texto = "La ESA coopera con la NASA en la órbita terrestre."
    extractor.extraer_relaciones(texto, entidades)
    primera = modelo.generate_llamadas
    extractor.extraer_relaciones(texto, entidades)  # mismo texto → caché
    assert modelo.generate_llamadas == primera


def test_determinista():
    salidas = {0: "<triplet> ESA <subj> NASA <obj> cooperates with </triplet>"}
    entidades = [_ent("ESA"), _ent("NASA")]
    texto = "La ESA coopera con la NASA en la órbita terrestre."
    a, _ = _extractor(salidas)
    b, _ = _extractor(salidas)
    assert a.extraer_relaciones(texto, entidades) == b.extraer_relaciones(texto, entidades)


# -- FP16 ---------------------------------------------------------------------

def test_use_fp16_se_propaga_al_motor():
    extractor, _ = _extractor(use_fp16=True)
    assert extractor._motor._use_fp16 is True


def test_use_fp16_en_cpu_no_convierte_a_half():
    """Sin CUDA, FP16 se ignora (con warning): el modelo no se toca."""
    modelo = _FakeModeloMrebel()
    extractor = MrebelRelationExtractor(
        tokenizer=_FakeTokenizadorMrebel({}), modelo=modelo, use_fp16=True
    )
    assert extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", [_ent("ESA"), _ent("NASA")]
    ) == []
    assert modelo.half_llamadas == 0
