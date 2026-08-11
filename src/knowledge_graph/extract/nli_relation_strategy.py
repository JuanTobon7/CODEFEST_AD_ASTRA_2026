"""Estrategia RE neuronal: NLI zero-shot multilingüe (encoder BERT, sin LLM).

Clasifica la relación entre cada PAR de entidades que co-ocurren en la misma
oración usando un modelo de inferencia textual (NLI): la oración es la
premisa y la hipótesis es la tripleta candidata ``"<sujeto> <verbo> <objeto>"``
para cada :class:`RelationType`. Entailment alto → relación tipada.

Encaja con la Sección 7.2 ("modelos de clasificación de relaciones") y NO es
un LLM generativo (restricción dura del reto): es un encoder multilingüe con
una cabecera de clasificación de 3 etiquetas (contradicción/neutral/entailment).

Módulos que la componen (separados por responsabilidad):
- :mod:`nli_config`: constantes (checkpoint, plantillas de hipótesis, umbrales).
- :mod:`nli_backend`: motor de inferencia (carga lazy, device, softmax).
- :mod:`entity_matchers`: matching de entidades en la oración (compartido
  con la RE simbólica).

Registrada como ``nli-zero-shot`` (opcional: el flujo por defecto sigue
siendo ``coocurrencia-oracional``, simbólico y sin red).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.knowledge_graph.extract.base import RelationExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.entity_matchers import formas_buscables, mencionada
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.extract.nli_backend import NLIInferenceEngine
from src.knowledge_graph.extract.nli_config import (
    MODELO_NLI_DEFAULT,
    PLANTILLAS_HIPOTESIS,
    UMBRAL_COOCURRENCIA,
    UMBRAL_TIPADO,
)
from src.knowledge_graph.models import Entity, Relation, RelationType

_REGEX_ORACIONES = re.compile(r"(?<=[.!?。！？])\s+")
_ESPACIO = re.compile(r"\s+")


@RelationExtractorFactory.register("nli-zero-shot")
class NLIRelationExtractor(RelationExtractor):
    """RE por clasificación NLI zero-shot entre pares de entidades.

    Args:
        model_id: checkpoint de HuggingFace (clasificación NLI multilingüe).
        device_preference: ``None``/``"auto"`` (detecta CUDA), ``"cpu"`` o ``"cuda"``.
        umbral: entailment mínimo para relación tipada (0..1).
        umbral_coocurrencia: entailment mínimo para emitir ``COOCURRENCIA``
            (0..1); por defecto ``0.0`` (siempre emite, igual que la
            estrategia simbólica cuando las entidades co-ocurren).
        max_length: truncado de la secuencia premisa+hipótesis.
        batch_size: tamaño de lote del forward del motor NLI.
        tokenizer/modelo: inyectables (tests u otro checkpoint). Si faltan,
            se cargan de HuggingFace en el primer uso.
    """

    name = "nli-zero-shot"

    def __init__(
        self,
        model_id: str = MODELO_NLI_DEFAULT,
        device_preference: Optional[str] = None,
        umbral: float = UMBRAL_TIPADO,
        umbral_coocurrencia: float = UMBRAL_COOCURRENCIA,
        max_length: int = 256,
        batch_size: int = 16,
        tokenizer=None,
        modelo=None,
    ) -> None:
        self._umbral = umbral
        self._umbral_coocurrencia = umbral_coocurrencia
        self._motor = NLIInferenceEngine(
            model_id=model_id,
            device_preference=device_preference,
            max_length=max_length,
            batch_size=batch_size,
            tokenizer=tokenizer,
            modelo=modelo,
        )
        #: Caché por (oración normalizada, sujeto, objeto) -> Relation | None:
        #: el grafo indexa miles de chunks con frases repetidas (boilerplate,
        #: alertas) y no re-evaluar el modelo ahorra horas en CPU.
        self._cache: Dict[Tuple[str, str, str], Optional[Relation]] = {}

    def extraer_relaciones(
        self, texto: str, entidades: Sequence[Entity]
    ) -> List[Relation]:
        """Relaciones entre ``entidades`` que co-ocurren en ``texto`` (NLI).

        Para cada oración: localiza las entidades presentes (por forma
        normalizada) y clasifica cada PAR dirigido (sujeto, objeto) con el
        modelo NLI. Un mismo par con el mismo tipo se reporta una sola vez
        con la máxima confianza observada.
        """
        if not texto or len(entidades) < 2:
            return []
        self._motor.asegurar_modelo()
        formas = formas_buscables(entidades)
        nombres_por_id = {e.id: e.nombre for e in entidades}
        oraciones = _REGEX_ORACIONES.split(_ESPACIO.sub(" ", texto.strip()))
        mejores: Dict[Tuple[str, RelationType, str], float] = {}
        for oracion in oraciones:
            self._acumular_de_oracion(oracion, formas, nombres_por_id, mejores)

        return [
            Relation(tipo=tipo, sujeto=sujeto, objeto=objeto, confianza=confianza)
            for (sujeto, tipo, objeto), confianza in sorted(mejores.items())
        ]

    def _acumular_de_oracion(
        self,
        oracion: str,
        formas: Dict[str, List[re.Pattern[str]]],
        nombres_por_id: Dict[str, str],
        mejores: Dict[Tuple[str, RelationType, str], float],
    ) -> None:
        """Clasifica los pares dirigidos de una oración y los acumula en ``mejores``."""
        noracion = _ESPACIO.sub(" ", normalizar_id_entidad(oracion))
        presentes = [
            eid for eid, patrones in formas.items() if mencionada(noracion, patrones)
        ]
        if len(presentes) < 2:
            return
        for i, sujeto in enumerate(presentes):
            for objeto in presentes[i + 1 :]:
                for par in ((sujeto, objeto), (objeto, sujeto)):
                    relacion = self._clasificar_par(
                        oracion, noracion, par[0], par[1], nombres_por_id
                    )
                    if relacion is None:
                        continue
                    clave = (relacion.sujeto, relacion.tipo, relacion.objeto)
                    mejores[clave] = max(mejores.get(clave, 0.0), relacion.confianza)

    def _clasificar_par(
        self,
        oracion: str,
        noracion: str,
        sujeto: str,
        objeto: str,
        nombres_por_id: Dict[str, str],
    ) -> Optional[Relation]:
        """Clasifica un par dirigido con NLI (con caché por oración+par)."""
        clave_cache = (noracion, sujeto, objeto)
        if clave_cache in self._cache:
            return self._cache[clave_cache]

        candidatas = [
            (tipo, plantilla.format(sujeto=nombres_por_id[sujeto], objeto=nombres_por_id[objeto]))
            for tipo, plantillas in PLANTILLAS_HIPOTESIS.items()
            for plantilla in plantillas
        ]
        scores = self._motor.puntuar([oracion] * len(candidatas), [h for _, h in candidatas])
        mejor_tipo, mejor_score = self._mejor_tipo(candidatas, scores)

        relacion: Optional[Relation] = None
        if mejor_score >= self._umbral:
            relacion = Relation(
                tipo=mejor_tipo, sujeto=sujeto, objeto=objeto,
                confianza=round(float(mejor_score), 4),
            )
        elif mejor_score >= self._umbral_coocurrencia:
            relacion = Relation(
                tipo=RelationType.COOCURRENCIA, sujeto=sujeto, objeto=objeto,
                confianza=round(float(mejor_score), 4),
            )
        self._cache[clave_cache] = relacion
        return relacion

    @staticmethod
    def _mejor_tipo(
        candidatas: List[Tuple[RelationType, str]], scores: List[float]
    ) -> Tuple[RelationType, float]:
        """Tipo con mayor entailment: máximo entre las variantes del mismo tipo."""
        por_tipo: Dict[RelationType, float] = {}
        for (tipo, _plantilla), score in zip(candidatas, scores):
            por_tipo[tipo] = max(por_tipo.get(tipo, 0.0), score)
        tipo = max(por_tipo, key=por_tipo.get)
        return tipo, por_tipo[tipo]
