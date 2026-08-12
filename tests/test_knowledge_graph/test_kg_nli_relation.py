"""Tests de la RE neuronal NLI zero-shot (``nli-zero-shot``).

Sin red ni modelos reales: el tokenizer y el modelo se inyectan como fakes.
El tokenizer fake codifica el ÍNDICE de la hipótesis en ``input_ids[i, 0]`` y
el modelo fake devuelve logits de una tabla por índice → el test controla
qué RelationType "entraña" el texto sin tocar HuggingFace.

Prefijo ``kg_``: evita colisión de basename con tests existentes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.knowledge_graph.extract.base import RelationExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.extract.nli_backend import NLIInferenceEngine
from src.knowledge_graph.extract.nli_config import PLANTILLAS_HIPOTESIS
from src.knowledge_graph.extract.nli_relation_strategy import NLIRelationExtractor
from src.knowledge_graph.models import Entity, EntityType, RelationType

# Importar la estrategia registra su decorador en la factory (efecto secundario).
import src.knowledge_graph.extract.nli_relation_strategy  # noqa: F401


# -- Fakes -------------------------------------------------------------------

class _FakeTokenizador:
    """Devuelve tensores con el índice de la hipótesis en ``input_ids[i, 0]``."""

    def __call__(self, premisas, hipotesis, **kwargs):
        b, t = len(premisas), 8
        input_ids = torch.zeros((b, t), dtype=torch.long)
        for i in range(b):
            input_ids[i, 0] = i
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((b, t), dtype=torch.long),
        }


class _FakeModelo:
    """Modelo de clasificación fake: logits por índice de hipótesis."""

    def __init__(self, logits_por_indice, id2label=None):
        self._logits = logits_por_indice
        self.config = SimpleNamespace(
            id2label=id2label
            or {0: "contradiction", 1: "neutral", 2: "entailment"}
        )
        self.llamadas = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **entradas):
        self.llamadas += 1
        filas = [
            self._logits.get(int(entradas["input_ids"][i, 0]), [1.0, 1.0, 0.05])
            for i in range(entradas["input_ids"].shape[0])
        ]
        return SimpleNamespace(logits=torch.tensor(filas, dtype=torch.float))


def _indices_de(tipo_buscado: RelationType) -> list:
    """Índices de las hipótesis de un RelationType (mismo orden que la clase)."""
    candidatas = [
        (tipo, _p)
        for tipo, plantillas in PLANTILLAS_HIPOTESIS.items()
        for _p in plantillas
    ]
    return [i for i, (tipo, _p) in enumerate(candidatas) if tipo == tipo_buscado]


def _extractor(logits_por_indice=None, **kwargs):
    """Estrategia con fakes inyectados; devuelve (extractor, modelo_fake).

    ``batch_size=64`` por defecto: las ~35 hipótesis de un par entran en un
    solo forward y el índice global codificado por el tokenizer fake coincide
    con la tabla de logits del modelo fake.
    """
    kwargs.setdefault("batch_size", 64)
    modelo = _FakeModelo(logits_por_indice or {})
    extractor = NLIRelationExtractor(
        tokenizer=_FakeTokenizador(), modelo=modelo, **kwargs
    )
    return extractor, modelo


def _ent(nombre: str, tipo=EntityType.ORGANIZACION) -> Entity:
    return Entity(id=normalizar_id_entidad(nombre), nombre=nombre, tipo=tipo)


# -- Registro e interfaz ------------------------------------------------------

def test_registrada_en_factory():
    assert "nli-zero-shot" in RelationExtractorFactory.list_available()


def test_implementa_relacion_extractor():
    extractor, _ = _extractor()
    assert isinstance(extractor, RelationExtractor)
    assert extractor.name == "nli-zero-shot"


def test_default_sigue_siendo_simbólico():
    """El service no cambia su default: coocurrencia-oracional sigue registrado."""
    assert "coocurrencia-oracional" in RelationExtractorFactory.list_available()


# -- Clasificación ------------------------------------------------------------

def test_clasifica_coopera_con_entailment_alto():
    """Entailment alto solo en las hipótesis de COOPERA_CON → relación tipada."""
    logits = {i: [0.05, 0.05, 0.9] for i in _indices_de(RelationType.COOPERA_CON)}
    extractor, _ = _extractor(logits)
    entidades = [_ent("ESA"), _ent("NASA")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    # Ambas direcciones (sujeto→objeto y objeto→sujeto), como la RE simbólica.
    assert {(r.sujeto, r.tipo, r.objeto) for r in relaciones} == {
        ("esa", RelationType.COOPERA_CON, "nasa"),
        ("nasa", RelationType.COOPERA_CON, "esa"),
    }
    esperado = round(float(torch.softmax(torch.tensor([0.05, 0.05, 0.9]), dim=-1)[2]), 4)
    assert all(r.confianza == esperado for r in relaciones)


def test_entidades_ausentes_sin_relaciones():
    extractor, _ = _extractor()
    entidades = [_ent("China"), _ent("Rusia")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    assert relaciones == []


def test_umbral_no_alcanzado_emite_coocurrencia():
    """Entailment < umbral tipado pero ≥ umbral_coocurrencia → COOCURRENCIA."""
    logits = {i: [0.1, 0.7, 0.2] for i in _indices_de(RelationType.CAUSA)}
    extractor, _ = _extractor(logits)
    entidades = [_ent("ESA"), _ent("NASA")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    esperado = round(float(torch.softmax(torch.tensor([0.1, 0.7, 0.2]), dim=-1)[2]), 4)
    assert relaciones
    assert all(r.tipo == RelationType.COOCURRENCIA for r in relaciones)
    assert all(r.confianza == esperado for r in relaciones)


def test_por_debajo_del_umbral_de_coocurrencia_no_emite():
    extractor, _ = _extractor(
        {i: [0.1, 0.7, 0.2] for i in _indices_de(RelationType.CAUSA)},
        umbral_coocurrencia=0.5,
    )
    entidades = [_ent("ESA"), _ent("NASA")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA en la órbita terrestre.", entidades
    )
    assert relaciones == []


def test_menos_de_dos_entidades_sin_modelo():
    modelo = _FakeModelo({})
    extractor = NLIRelationExtractor(tokenizer=_FakeTokenizador(), modelo=modelo)
    assert extractor.extraer_relaciones("Texto suelto.", [_ent("ESA")]) == []
    assert modelo.llamadas == 0


# -- Caché, determinismo y robustez -------------------------------------------

def test_cache_evita_reevaluar_el_modelo():
    logits = {i: [0.05, 0.05, 0.9] for i in _indices_de(RelationType.COOPERA_CON)}
    extractor, modelo = _extractor(logits)
    entidades = [_ent("ESA"), _ent("NASA")]
    texto = "La ESA coopera con la NASA en la órbita terrestre."
    extractor.extraer_relaciones(texto, entidades)
    primera = modelo.llamadas
    extractor.extraer_relaciones(texto, entidades)  # mismo texto → caché
    assert modelo.llamadas == primera


def test_determinista():
    logits = {i: [0.05, 0.05, 0.9] for i in _indices_de(RelationType.COOPERA_CON)}
    entidades = [_ent("ESA"), _ent("NASA")]
    texto = "La ESA coopera con la NASA en la órbita terrestre."
    a, _ = _extractor(logits)
    b, _ = _extractor(logits)
    assert a.extraer_relaciones(texto, entidades) == b.extraer_relaciones(texto, entidades)


def test_indice_entailment_desde_config():
    modelo = _FakeModelo({}, id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    assert NLIInferenceEngine.indice_entailment(modelo) == 0


def test_fallback_indice_entailment():
    modelo = SimpleNamespace()  # sin config
    assert NLIInferenceEngine.indice_entailment(modelo) == 2


def test_puntuar_particiona_por_batch():
    """El motor parte las hipótesis en bloques de ``batch_size`` y conserva el orden."""
    modelo = _FakeModelo({})
    motor = NLIInferenceEngine(
        model_id="fake", batch_size=4, tokenizer=_FakeTokenizador(), modelo=modelo
    )
    motor.asegurar_modelo()
    puntajes = motor.puntuar(["premisa"] * 10, ["hipotesis"] * 10)
    assert len(puntajes) == 10
    assert modelo.llamadas == 3  # 4 + 4 + 2
