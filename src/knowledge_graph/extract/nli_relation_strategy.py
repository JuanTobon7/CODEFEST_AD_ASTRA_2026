"""Estrategia RE del grafo: clasificación de relaciones con un encoder NLI.

Es el modelo que construye las ARISTAS del grafo (Sección 7.2, "modelos de
clasificación de relaciones"). Para cada PAR de entidades que co-ocurren en
la misma oración, la oración es la premisa y la tripleta candidata
``"<sujeto> <verbo> <objeto>"`` es la hipótesis, una por
:class:`RelationType`; el tipo con mayor probabilidad de *entailment* gana.

Cumplimiento de las restricciones del reto:

- **Licencia**: ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`` es **MIT**
  (ver :data:`~nli_config.FICHA_MODELO_NLI`).
- **No generativo**: es un encoder bidireccional con cabecera de
  clasificación de 3 etiquetas. No tiene decoder ni cabecera de lenguaje;
  su salida son tres logits, nunca texto. Los sujetos y objetos salen
  siempre del NER sobre el texto real — el modelo solo ELIGE una etiqueta
  de un vocabulario cerrado, nunca inventa entidades ni relaciones nuevas.

Coste y presupuesto: el canal hace ``pares × tipos × variantes × 2``
forwards. Por eso el candidateo es simbólico y barato (co-ocurrencia
oracional) y el modelo solo TIPA los candidatos, con topes explícitos
(``max_pares_por_oracion`` / ``max_pares_por_chunk``) y caché por
(oración, par) — el corpus repite mucho boilerplate.

Módulos que la componen (separados por responsabilidad):
- :mod:`nli_config`: constantes (checkpoint, licencia, plantillas, umbrales).
- :mod:`nli_backend`: motor de inferencia (carga lazy, device, softmax).
- :mod:`entity_matchers`: matching de entidades en la oración (compartido
  con la RE simbólica).

Registrada como ``nli-zero-shot``; es la estrategia RE por defecto de los
CLI del grafo.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.knowledge_graph.extract.base import RelationExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.entity_matchers import formas_buscables, mencionada
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.extract.nli_backend import NLIInferenceEngine
from src.knowledge_graph.extract.nli_config import (
    MAX_PARES_POR_CHUNK_DEFAULT,
    MAX_PARES_POR_ORACION_DEFAULT,
    MODELO_NLI_DEFAULT,
    UMBRAL_COOCURRENCIA,
    UMBRAL_TIPADO,
    VARIANTES_PLANTILLA_DEFAULT,
    plantillas_activas,
)
from src.knowledge_graph.models import Entity, Relation, RelationType

_REGEX_ORACIONES = re.compile(r"(?<=[.!?。！？])\s+")
_ESPACIO = re.compile(r"\s+")

#: Clave de caché: (oración normalizada, id sujeto, id objeto).
_ClavePar = Tuple[str, str, str]


@RelationExtractorFactory.register("nli-zero-shot")
class NLIRelationExtractor(RelationExtractor):
    """RE por clasificación NLI entre pares de entidades co-ocurrentes.

    Args:
        model_id: checkpoint de HuggingFace (clasificación NLI multilingüe).
        device_preference: ``None``/``"auto"`` (detecta CUDA), ``"cpu"`` o ``"cuda"``.
        umbral: entailment mínimo para relación tipada (0..1).
        umbral_coocurrencia: entailment mínimo para emitir ``COOCURRENCIA``
            (0..1); por defecto ``0.0`` (siempre emite, igual que la
            estrategia simbólica cuando las entidades co-ocurren).
        max_length: truncado de la secuencia premisa+hipótesis.
        batch_size: tamaño de lote del forward del motor NLI.
        variantes_plantilla: formulaciones evaluadas por tipo de relación
            (``<= 0`` = todas). Menos variantes = canal más rápido.
        max_pares_por_oracion: tope de pares clasificados por oración
            (las oraciones con muchas entidades explotan combinatoriamente).
        max_pares_por_chunk: tope de pares clasificados por chunk.
        use_fp16: media precisión en CUDA (~2x más rápido, la mitad de VRAM).
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
        batch_size: int = 64,
        variantes_plantilla: int = VARIANTES_PLANTILLA_DEFAULT,
        max_pares_por_oracion: int = MAX_PARES_POR_ORACION_DEFAULT,
        max_pares_por_chunk: int = MAX_PARES_POR_CHUNK_DEFAULT,
        use_fp16: bool = True,
        tokenizer=None,
        modelo=None,
    ) -> None:
        self._umbral = umbral
        self._umbral_coocurrencia = umbral_coocurrencia
        self._max_pares_por_oracion = max_pares_por_oracion
        self._max_pares_por_chunk = max_pares_por_chunk
        #: Plantillas aplanadas una sola vez: (tipo, plantilla). El orden es
        #: fijo, así que los scores de un par llegan siempre alineados.
        self._plantillas: List[Tuple[RelationType, str]] = [
            (tipo, plantilla)
            for tipo, plantillas in plantillas_activas(variantes_plantilla).items()
            for plantilla in plantillas
        ]
        self._motor = NLIInferenceEngine(
            model_id=model_id,
            device_preference=device_preference,
            max_length=max_length,
            batch_size=batch_size,
            use_fp16=use_fp16,
            tokenizer=tokenizer,
            modelo=modelo,
        )
        #: Caché por (oración normalizada, sujeto, objeto) -> Relation | None:
        #: el grafo indexa cientos de miles de chunks con frases repetidas
        #: (boilerplate, alertas) y no re-evaluar el modelo ahorra horas.
        self._cache: Dict[_ClavePar, Optional[Relation]] = {}

    # -- API pública ---------------------------------------------------------

    def extraer_relaciones(
        self, texto: str, entidades: Sequence[Entity]
    ) -> List[Relation]:
        """Relaciones entre ``entidades`` que co-ocurren en ``texto`` (NLI).

        Para cada oración: localiza las entidades presentes (por forma
        normalizada) y clasifica cada PAR dirigido (sujeto, objeto) en UN
        solo forward por oración. Un mismo par con el mismo tipo se reporta
        una sola vez con la máxima confianza observada.
        """
        if not texto or len(entidades) < 2:
            return []
        self._motor.asegurar_modelo()
        formas = formas_buscables(entidades)
        nombres_por_id = {e.id: e.nombre for e in entidades}
        oraciones = _REGEX_ORACIONES.split(_ESPACIO.sub(" ", texto.strip()))

        mejores: Dict[Tuple[str, RelationType, str], float] = {}
        presupuesto = self._max_pares_por_chunk
        for oracion in oraciones:
            if presupuesto <= 0:
                break
            usados = self._acumular_de_oracion(
                oracion, formas, nombres_por_id, mejores, presupuesto
            )
            presupuesto -= usados

        return [
            Relation(tipo=tipo, sujeto=sujeto, objeto=objeto, confianza=confianza)
            for (sujeto, tipo, objeto), confianza in sorted(mejores.items())
        ]

    @property
    def num_hipotesis_por_par(self) -> int:
        """Hipótesis evaluadas por par dirigido (para estimar el coste)."""
        return len(self._plantillas)

    # -- Clasificación por oración -------------------------------------------

    def _acumular_de_oracion(
        self,
        oracion: str,
        formas: Dict[str, List[re.Pattern[str]]],
        nombres_por_id: Dict[str, str],
        mejores: Dict[Tuple[str, RelationType, str], float],
        presupuesto: int,
    ) -> int:
        """Clasifica los pares dirigidos de una oración y los acumula.

        Returns:
            Número de pares dirigidos que consumieron presupuesto (los
            resueltos por caché no cuentan: no invocaron al modelo).
        """
        noracion = _ESPACIO.sub(" ", normalizar_id_entidad(oracion))
        presentes = [
            eid for eid, patrones in formas.items() if mencionada(noracion, patrones)
        ]
        if len(presentes) < 2:
            return 0

        tope = min(self._max_pares_por_oracion, presupuesto)
        pares = self._pares_dirigidos(presentes, tope)
        pendientes = [p for p in pares if (noracion, *p) not in self._cache]
        if pendientes:
            self._clasificar_lote(oracion, noracion, pendientes, nombres_por_id)

        for sujeto, objeto in pares:
            relacion = self._cache.get((noracion, sujeto, objeto))
            if relacion is None:
                continue
            clave = (relacion.sujeto, relacion.tipo, relacion.objeto)
            mejores[clave] = max(mejores.get(clave, 0.0), relacion.confianza)
        return len(pendientes)

    @staticmethod
    def _pares_dirigidos(presentes: List[str], tope: int) -> List[Tuple[str, str]]:
        """Pares dirigidos (sujeto, objeto) de la oración, hasta ``tope``.

        El recorte es determinista (orden de detección de las entidades):
        dos corridas sobre el mismo corpus producen el mismo grafo.
        """
        if tope <= 0:
            return []
        pares: List[Tuple[str, str]] = []
        for i, primero in enumerate(presentes):
            for segundo in presentes[i + 1 :]:
                pares.append((primero, segundo))
                pares.append((segundo, primero))
                if len(pares) >= tope:
                    return pares[:tope]
        return pares

    def _clasificar_lote(
        self,
        oracion: str,
        noracion: str,
        pares: List[Tuple[str, str]],
        nombres_por_id: Dict[str, str],
    ) -> None:
        """Clasifica todos los ``pares`` de la oración en una sola pasada.

        Agrupar los pares en un único lote (en vez de un forward por par)
        es lo que hace viable el canal sobre un corpus grande: el modelo se
        invoca con ``len(pares) × hipótesis`` secuencias de golpe y la GPU
        va saturada.
        """
        premisas: List[str] = []
        hipotesis: List[str] = []
        for sujeto, objeto in pares:
            nombre_sujeto = nombres_por_id[sujeto]
            nombre_objeto = nombres_por_id[objeto]
            for _tipo, plantilla in self._plantillas:
                premisas.append(oracion)
                hipotesis.append(
                    plantilla.format(sujeto=nombre_sujeto, objeto=nombre_objeto)
                )

        scores = self._motor.puntuar(premisas, hipotesis)

        ancho = len(self._plantillas)
        for indice, (sujeto, objeto) in enumerate(pares):
            bloque = scores[indice * ancho : (indice + 1) * ancho]
            self._cache[(noracion, sujeto, objeto)] = self._relacion_de(
                sujeto, objeto, bloque
            )

    def _relacion_de(
        self, sujeto: str, objeto: str, scores: Sequence[float]
    ) -> Optional[Relation]:
        """Convierte los scores de un par en una :class:`Relation` (o ``None``)."""
        tipo, score = self._mejor_tipo(self._plantillas, scores)
        if score >= self._umbral:
            return Relation(
                tipo=tipo, sujeto=sujeto, objeto=objeto,
                confianza=round(float(score), 4),
            )
        if score >= self._umbral_coocurrencia:
            return Relation(
                tipo=RelationType.COOCURRENCIA, sujeto=sujeto, objeto=objeto,
                confianza=round(float(score), 4),
            )
        return None

    @staticmethod
    def _mejor_tipo(
        candidatas: Sequence[Tuple[RelationType, str]], scores: Sequence[float]
    ) -> Tuple[RelationType, float]:
        """Tipo con mayor entailment: máximo entre las variantes del mismo tipo."""
        por_tipo: Dict[RelationType, float] = {}
        for (tipo, _plantilla), score in zip(candidatas, scores):
            por_tipo[tipo] = max(por_tipo.get(tipo, 0.0), score)
        tipo = max(por_tipo, key=por_tipo.get)
        return tipo, por_tipo[tipo]
