"""Configuración de la RE con mREBEL: constantes puras, sin dependencias.

Separada de la estrategia (:mod:`mrebel_relation_strategy`) y del motor de
inferencia (:mod:`mrebel_backend`) para que el mapeo de relaciones y las
confianzas sean testeables e intercambiables sin instanciar el modelo.
"""

from __future__ import annotations

from typing import Dict

from src.knowledge_graph.models import RelationType

#: Checkpoint mREBEL (REBEL multilingüe, seq2seq T5-large 560M).
#: OJO licencia: CC BY-NC-SA 4.0 (no comercial) — decisión del usuario.
MODELO_MREBEL_DEFAULT = "Babelscape/mrebel-large"

#: Idioma de origen del texto (tokenizer mBART). El corpus es mayormente EN;
#: mREBEL es multilingüe y el ``src_lang`` solo afecta la tokenización.
SRC_LANG_DEFAULT = "en_XX"

#: Idioma de destino fijo de mREBEL (genera tripletas).
TGT_LANG = "tp_XX"

#: Mapeo relación-generada (normalizada: minúsculas, sin acentos) -> RelationType.
#: mREBEL emite la relación en texto libre (es/en); las que no están aquí
#: caen a RELACIONADO_CON (relación detectada pero no tipada).
MAPEO_RELACIONES: Dict[str, RelationType] = {
    # Ubicación
    "located in": RelationType.UBICADO_EN,
    "located at": RelationType.UBICADO_EN,
    "based in": RelationType.UBICADO_EN,
    "headquartered in": RelationType.UBICADO_EN,
    "headquarters location": RelationType.UBICADO_EN,
    "country": RelationType.UBICADO_EN,
    "capital": RelationType.UBICADO_EN,
    "work location": RelationType.UBICADO_EN,
    "ubicado en": RelationType.UBICADO_EN,
    "se encuentra en": RelationType.UBICADO_EN,
    # Pertenencia / composición
    "part of": RelationType.ES_PARTE_DE,
    "is part of": RelationType.ES_PARTE_DE,
    "member of": RelationType.ES_PARTE_DE,
    "subsidiary of": RelationType.PARTE_DE,
    "parent organization": RelationType.PARTE_DE,
    "es parte de": RelationType.ES_PARTE_DE,
    "es miembro de": RelationType.ES_PARTE_DE,
    "pertenece a": RelationType.PERTENECE_A,
    "belongs to": RelationType.PERTENECE_A,
    "owned by": RelationType.PERTENECE_A,
    "employer": RelationType.PERTENECE_A,
    "owner of": RelationType.PERTENECE_A,
    "controls": RelationType.PERTENECE_A,
    "head of": RelationType.PERTENECE_A,
    "leader": RelationType.PERTENECE_A,
    "president": RelationType.PERTENECE_A,
    # Impacto / causalidad
    "affects": RelationType.AFECTA,
    "threatens": RelationType.AFECTA,
    "impacts": RelationType.AFECTA,
    "influences": RelationType.AFECTA,
    "afecta": RelationType.AFECTA,
    "amenaza": RelationType.AFECTA,
    "causes": RelationType.CAUSA,
    "triggers": RelationType.CAUSA,
    "causa": RelationType.CAUSA,
    "provoca": RelationType.CAUSA,
    # Dependencia
    "depends on": RelationType.DEPENDE_DE,
    "depende de": RelationType.DEPENDE_DE,
    # Cooperación
    "cooperates with": RelationType.COOPERA_CON,
    "collaborates with": RelationType.COOPERA_CON,
    "partnership with": RelationType.COOPERA_CON,
    "alliance with": RelationType.COOPERA_CON,
    "coopera con": RelationType.COOPERA_CON,
    "colabora con": RelationType.COOPERA_CON,
    # Oposición
    "opposes": RelationType.OPONE_A,
    "conflict with": RelationType.OPONE_A,
    "military conflict": RelationType.OPONE_A,
    "armed conflict": RelationType.OPONE_A,
    "se opone a": RelationType.OPONE_A,
    "en conflicto con": RelationType.OPONE_A,
    # Involucramiento
    "involves": RelationType.INVOLUCRA,
    "includes": RelationType.INVOLUCRA,
    "participant in": RelationType.INVOLUCRA,
    "uses": RelationType.INVOLUCRA,
    "develops": RelationType.INVOLUCRA,
    "produces": RelationType.INVOLUCRA,
    "manufactures": RelationType.INVOLUCRA,
    "operates": RelationType.INVOLUCRA,
    "involucra": RelationType.INVOLUCRA,
    "incluye": RelationType.INVOLUCRA,
    # Relación genérica
    "related to": RelationType.RELACIONADO_CON,
    "associated with": RelationType.RELACIONADO_CON,
    "linked to": RelationType.RELACIONADO_CON,
    "relacionado con": RelationType.RELACIONADO_CON,
    "vinculado a": RelationType.RELACIONADO_CON,
}

#: Confianza de las tripletas mREBEL (no emite probabilidades por tripleta;
#: beam search determinista). Tipada = mapeada al vocabulario del grafo.
CONFIANZA_MAPEADA = 0.9
CONFIANZA_NO_MAPEADA = 0.7