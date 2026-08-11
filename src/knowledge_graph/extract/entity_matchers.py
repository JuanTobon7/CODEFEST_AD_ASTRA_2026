"""Matchers de entidades en texto, compartidos por las RE simbólica y NLI.

La técnica es la misma en ambas estrategias: normalizar el texto (id
canónico: minúsculas, sin acentos) y buscar las formas léxicas de cada
entidad (nombre, variantes, id) como PALABRA COMPLETA (``\\b``).

Están aquí (y no en :mod:`base`) porque solo las estrategias de RE las
usan; el NER resuelve las menciones con spans, no con estos patrones.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from src.knowledge_graph.extract.base import normalizar_id_entidad
from src.knowledge_graph.models import Entity

_ESPACIO = re.compile(r"\s+")


def formas_buscables(
    entidades: Sequence[Entity],
) -> Dict[str, List[re.Pattern[str]]]:
    """Patrones por id de entidad: sus formas léxicas normalizadas."""
    formas: Dict[str, List[re.Pattern[str]]] = {}
    for e in entidades:
        variantes = {e.nombre, *e.variantes, e.id}
        patrones = [
            re.compile(
                rf"(?<!\w){re.escape(_ESPACIO.sub(' ', normalizar_id_entidad(v)))}(?!\w)"
            )
            for v in variantes
            if v.strip()
        ]
        formas[e.id] = patrones
    return formas


def mencionada(oracion_normalizada: str, patrones: List[re.Pattern[str]]) -> bool:
    """True si la oración normalizada menciona la entidad."""
    return any(p.search(oracion_normalizada) is not None for p in patrones)
