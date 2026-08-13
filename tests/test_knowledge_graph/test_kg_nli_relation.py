"""Tests de la RE por clasificación NLI (``nli-zero-shot``).

Sin red ni modelos reales: el tokenizer y el modelo se inyectan como fakes.
El tokenizer fake codifica el ÍNDICE de la secuencia en ``input_ids[i, 0]`` y
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
from src.knowledge_graph.extract.nli_config import (
    FICHA_MODELO_NLI,
    LICENCIA_MODELO_NLI,
    PLANTILLAS_HIPOTESIS,
    VARIANTES_PLANTILLA_DEFAULT,
    plantillas_activas,
)
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


#: Plantillas efectivamente evaluadas por la estrategia, en su mismo orden.
_PLANTILLAS_ACTIVAS = [
    (tipo, plantilla)
    for tipo, plantillas in plantillas_activas(VARIANTES_PLANTILLA_DEFAULT).items()
    for plantilla in plantillas
]
_ANCHO = len(_PLANTILLAS_ACTIVAS)


def _indices_de(tipo_buscado: RelationType, max_pares: int = 8) -> list:
    """Índices GLOBALES de las hipótesis de un tipo dentro del lote.

    La estrategia clasifica todos los pares de una oración en un solo lote:
    el par ``p`` ocupa las posiciones ``p * ancho .. (p+1) * ancho``. Se
    devuelven las posiciones del tipo buscado en los primeros ``max_pares``.
    """
    locales = [i for i, (tipo, _p) in enumerate(_PLANTILLAS_ACTIVAS) if tipo == tipo_buscado]
    return [par * _ANCHO + i for par in range(max_pares) for i in locales]


def _extractor(logits_por_indice=None, **kwargs):
    """Estrategia con fakes inyectados; devuelve (extractor, modelo_fake).

    ``batch_size=512`` por defecto: todos los pares de la oración entran en un
    solo forward, así que el índice global que codifica el tokenizer fake
    coincide con la tabla de logits del modelo fake.
    """
    kwargs.setdefault("batch_size", 512)
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


def test_solo_hay_estrategias_re_con_licencia_permisiva():
    """Regresión: mREBEL (CC BY-NC-SA y generativo) quedó fuera del proyecto."""
    assert set(RelationExtractorFactory.list_available()) == {
        "coocurrencia-oracional",
        "nli-zero-shot",
    }


def test_modelo_declarado_es_mit_y_no_generativo():
    """La ficha del checkpoint documenta el cumplimiento exigido por el reto."""
    assert LICENCIA_MODELO_NLI == "MIT"
    assert FICHA_MODELO_NLI["licencia"] == "MIT"
    assert FICHA_MODELO_NLI["generativo"] == "no"
    assert FICHA_MODELO_NLI["model_id"] == "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


# -- Presupuesto de inferencia ------------------------------------------------

def test_plantillas_activas_recorta_variantes_por_tipo():
    recortadas = plantillas_activas(1)
    assert set(recortadas) == set(PLANTILLAS_HIPOTESIS)
    assert all(len(v) == 1 for v in recortadas.values())
    assert plantillas_activas(0) == PLANTILLAS_HIPOTESIS


def test_max_pares_por_oracion_acota_el_numero_de_pares():
    """Con 3 entidades hay 6 pares dirigidos; el tope deja pasar solo 2."""
    logits = {i: [0.05, 0.05, 0.9] for i in _indices_de(RelationType.COOPERA_CON)}
    extractor, _ = _extractor(logits, max_pares_por_oracion=2)
    entidades = [_ent("ESA"), _ent("NASA"), _ent("SpaceX")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA y con SpaceX.", entidades
    )
    assert len(relaciones) == 2


def test_max_pares_por_chunk_acota_el_gasto_entre_oraciones():
    """El presupuesto del chunk se agota en la primera oración."""
    logits = {i: [0.05, 0.05, 0.9] for i in _indices_de(RelationType.COOPERA_CON)}
    extractor, _ = _extractor(logits, max_pares_por_oracion=2, max_pares_por_chunk=2)
    entidades = [_ent("ESA"), _ent("NASA"), _ent("SpaceX"), _ent("Boeing")]
    relaciones = extractor.extraer_relaciones(
        "La ESA coopera con la NASA. SpaceX coopera con Boeing.", entidades
    )
    assert len(relaciones) == 2
    assert {r.sujeto for r in relaciones} | {r.objeto for r in relaciones} == {"esa", "nasa"}


def test_num_hipotesis_por_par_refleja_las_variantes():
    extractor, _ = _extractor(variantes_plantilla=1)
    assert extractor.num_hipotesis_por_par == len(PLANTILLAS_HIPOTESIS)


def test_un_solo_forward_para_todos_los_pares_de_la_oracion():
    """Los pares de una oración se clasifican en un lote, no uno por uno."""
    extractor, modelo = _extractor()
    entidades = [_ent("ESA"), _ent("NASA"), _ent("SpaceX")]
    extractor.extraer_relaciones("La ESA coopera con la NASA y con SpaceX.", entidades)
    assert modelo.llamadas == 1


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
