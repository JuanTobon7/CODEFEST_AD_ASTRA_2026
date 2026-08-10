"""Estrategia RE simbólica: co-ocurrencia oracional + patrones verbales.

Sin LLMs (restricción dura): la relación entre dos entidades se infiere de
forma determinista cuando ambas aparecen en la MISMA oración. Si la oración
contiene un patrón verbal tipado (p. ej. "coopera con", "ubicado en"), se
etiqueta la relación con ese tipo; si no, queda como ``COOCURRENCIA``.

Implementa :class:`RelationExtractor` (Strategy/OCP: una RE neuronal futura
solo necesita implementar la misma interfaz).
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from src.knowledge_graph.extract.base import RelationExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.models import Entity, Relation, RelationType

# -- Patrones verbales tipados (es/en) --------------------------------------
# Cada patrón es una lista de fragmentos léxicos; se buscan normalizados
# (sin acentos). Se evalúan en el orden de declaración (más específicos
# primero: "es parte de" antes de "parte de").
PATRONES_VERBALES: List[Tuple[RelationType, Tuple[str, ...]]] = [
    (RelationType.UBICADO_EN, ("ubicado en", "ubicada en", "ubicados en", "ubicadas en",
                               "situado en", "situada en", "se encuentra en", "se encuentran en",
                               "localizado en", "localizada en", "located in", "based in")),
    (RelationType.ES_PARTE_DE, ("es parte de", "forma parte de", "es miembro de",
                                "part of", "member of")),
    (RelationType.PARTE_DE, ("parte de", "miembro de")),
    (RelationType.PERTENECE_A, ("pertenece a", "pertenecen a", "es propiedad de",
                                "son propiedad de", "belongs to", "owned by")),
    (RelationType.AFECTA, ("afecta a", "afectan a", "impacta a", "impactan a",
                           "incide en", "inciden en", "influye en", "influyen en",
                           "amenaza a", "amenazan a", "threatens", "impacts")),
    (RelationType.CAUSA, ("causa", "causan", "provoca", "provocan", "genera",
                          "generan", "origina", "origina", "causes", "triggers")),
    (RelationType.DEPENDE_DE, ("depende de", "dependen de", "depende de la",
                               "depende del", "depends on")),
    (RelationType.COOPERA_CON, ("coopera con", "cooperan con", "colabora con", "colaboran con",
                                "trabaja junto a", "trabaja junto con", "es aliado de",
                                "es aliada de", "cooperates with", "collaborates with",
                                "partnership with")),
    (RelationType.OPONE_A, ("se opone a", "se enfrenta a", "en conflicto con",
                            "rival de", "opposes", "conflict with")),
    (RelationType.INVOLUCRA, ("involucra", "involucran", "incluye", "incluyen",
                              "comprende", "comprenden", "involves", "includes")),
    (RelationType.RELACIONADO_CON, ("relacionado con", "relacionada con", "relacionados con",
                                    "relacionadas con", "vinculado a", "vinculada a",
                                    "asociado a", "asociada a", "related to", "linked to",
                                    "associated with")),
]

#: Confianza de cada tipo de relación (heurística documentada del informe).
_CONFIANZA_TIPADA = 0.9
_CONFIANZA_COOCURRENCIA = 0.6

_REGEX_ORACIONES = re.compile(r"(?<=[.!?。！？])\s+")
_ESPACIO = re.compile(r"\s+")


@RelationExtractorFactory.register("coocurrencia-oracional")
class CooccurrenceRelationExtractor(RelationExtractor):
    """RE por co-ocurrencia oracional + patrones verbales (sin LLM)."""

    name = "coocurrencia-oracional"

    def extraer_relaciones(
        self, texto: str, entidades: Sequence[Entity]
    ) -> List[Relation]:
        """Relaciones entre ``entidades`` que co-ocurren en ``texto``.

        Para cada oración: localiza las entidades presentes (por forma
        normalizada), y para cada PAR ordenado (sujeto, objeto) decide el
        tipo por el patrón verbal que aparezca en la oración (o
        ``COOCURRENCIA``). Un mismo par con el mismo tipo se reporta una
        sola vez con la máxima confianza observada.
        """
        if not texto or len(entidades) < 2:
            return []

        formas = self._formas_buscables(entidades)
        oraciones = _REGEX_ORACIONES.split(_ESPACIO.sub(" ", texto.strip()))
        mejores: Dict[Tuple[str, RelationType, str], float] = {}

        for oracion in oraciones:
            noracion = _ESPACIO.sub(" ", normalizar_id_entidad(oracion))
            presentes = [
                eid for eid, patrones in formas.items() if self._mencionada(noracion, patrones)
            ]
            if len(presentes) < 2:
                continue
            self._acumular_pares(presentes, noracion, mejores)

        return [
            Relation(tipo=tipo, sujeto=sujeto, objeto=objeto, confianza=confianza)
            for (sujeto, tipo, objeto), confianza in sorted(mejores.items())
        ]

    @classmethod
    def _acumular_pares(
        cls,
        presentes: List[str],
        oracion_normalizada: str,
        mejores: Dict[Tuple[str, RelationType, str], float],
    ) -> None:
        """Registra los pares (sujeto, objeto) de la oración con su confianza.

        Extraído a método propio para mantener la complejidad cognitiva del
        flujo principal baja (SRP: una responsabilidad por método).
        """
        tipo = cls._tipo_de_oracion(oracion_normalizada)
        confianza = _CONFIANZA_TIPADA if tipo != RelationType.COOCURRENCIA else _CONFIANZA_COOCURRENCIA
        for i, sujeto in enumerate(presentes):
            for objeto in presentes[i + 1 :]:
                for par in ((sujeto, objeto), (objeto, sujeto)):
                    clave = (par[0], tipo, par[1])
                    mejores[clave] = max(mejores.get(clave, 0.0), confianza)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _formas_buscables(
        entidades: Sequence[Entity],
    ) -> Dict[str, List[re.Pattern[str]]]:
        """Patrones por id de entidad: sus formas léxicas normalizadas."""
        formas: Dict[str, List[re.Pattern[str]]] = {}
        for e in entidades:
            variantes = {e.nombre, *e.variantes, e.id}
            patrones = [
                re.compile(rf"(?<!\w){re.escape(_ESPACIO.sub(' ', normalizar_id_entidad(v)))}(?!\w)")
                for v in variantes
                if v.strip()
            ]
            formas[e.id] = patrones
        return formas

    @staticmethod
    def _mencionada(oracion_normalizada: str, patrones: List[re.Pattern[str]]) -> bool:
        """True si la oración normalizada menciona la entidad."""
        return any(p.search(oracion_normalizada) is not None for p in patrones)

    @classmethod
    def _tipo_de_oracion(cls, oracion_normalizada: str) -> RelationType:
        """Tipo de relación por el primer patrón verbal que aparece (o COOCURRENCIA)."""
        for tipo, patrones in PATRONES_VERBALES:
            if any(_ESPACIO.sub(" ", pat) in oracion_normalizada for pat in patrones):
                return tipo
        return RelationType.COOCURRENCIA
