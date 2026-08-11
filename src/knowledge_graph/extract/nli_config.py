"""Configuración de la RE NLI: constantes puras, sin dependencias de modelo.

Separada de la estrategia (:mod:`nli_relation_strategy`) y del motor de
inferencia (:mod:`nli_backend`) para que las plantillas de hipótesis sean
testeables e intercambiables sin instanciar ningún modelo.
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.knowledge_graph.models import RelationType

#: Checkpoint NLI multilingüe por defecto (licencia MIT).
MODELO_NLI_DEFAULT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

#: Hipótesis por RelationType (es/en). ``{sujeto}``/``{objeto}`` se sustituyen
#: con la forma léxica preferida de cada entidad. Varias variantes porque el
#: modelo NLI es más sensible a unas formulaciones que a otras; se toma la
#: máxima probabilidad de entailment entre las variantes del mismo tipo.
PLANTILLAS_HIPOTESIS: Dict[RelationType, Tuple[str, ...]] = {
    RelationType.UBICADO_EN: (
        "{sujeto} está ubicado en {objeto}",
        "{sujeto} se encuentra en {objeto}",
        "{sujeto} is located in {objeto}",
    ),
    RelationType.ES_PARTE_DE: (
        "{sujeto} es parte de {objeto}",
        "{sujeto} es miembro de {objeto}",
        "{sujeto} is part of {objeto}",
        "{sujeto} is a member of {objeto}",
    ),
    RelationType.PARTE_DE: (
        "{sujeto} parte de {objeto}",
        "{sujeto} part of {objeto}",
    ),
    RelationType.PERTENECE_A: (
        "{sujeto} pertenece a {objeto}",
        "{sujeto} es propiedad de {objeto}",
        "{sujeto} belongs to {objeto}",
    ),
    RelationType.AFECTA: (
        "{sujeto} afecta a {objeto}",
        "{sujeto} amenaza a {objeto}",
        "{sujeto} affects {objeto}",
        "{sujeto} threatens {objeto}",
    ),
    RelationType.CAUSA: (
        "{sujeto} causa {objeto}",
        "{sujeto} provoca {objeto}",
        "{sujeto} causes {objeto}",
    ),
    RelationType.DEPENDE_DE: (
        "{sujeto} depende de {objeto}",
        "{sujeto} depends on {objeto}",
    ),
    RelationType.COOPERA_CON: (
        "{sujeto} coopera con {objeto}",
        "{sujeto} colabora con {objeto}",
        "{sujeto} cooperates with {objeto}",
        "{sujeto} collaborates with {objeto}",
    ),
    RelationType.OPONE_A: (
        "{sujeto} se opone a {objeto}",
        "{sujeto} está en conflicto con {objeto}",
        "{sujeto} opposes {objeto}",
    ),
    RelationType.INVOLUCRA: (
        "{sujeto} involucra a {objeto}",
        "{sujeto} incluye a {objeto}",
        "{sujeto} involves {objeto}",
        "{sujeto} includes {objeto}",
    ),
    RelationType.RELACIONADO_CON: (
        "{sujeto} está relacionado con {objeto}",
        "{sujeto} está vinculado a {objeto}",
        "{sujeto} is related to {objeto}",
    ),
}

#: Probabilidad de entailment mínima para emitir una relación TIPADA.
UMBRAL_TIPADO = 0.5
#: Probabilidad mínima para emitir ``COOCURRENCIA`` (por debajo: nada).
UMBRAL_COOCURRENCIA = 0.0
