"""
Etapa 5 del pipeline: selección final de fragmentos a nivel de chunk
(Sección 9.2.1).

Reglas aplicadas sobre los top-``k_chunk_out`` fragmentos tras RRF + filtros:

1. Todo fragmento con más de ``max_words`` (250 por defecto) palabras se
   divide en sub-fragmentos que respetan límites oracionales COMPLETOS
   (nunca se corta una oración a la mitad); todos los sub-fragmentos
   conservan el ``chunk_id`` del padre. Un sub-fragmento conserva el score
   RRF del padre.

2. Un fragmento con menos de ``max_words`` palabras se concatena con el
   ``chunk_id`` de posición inmediatamente siguiente del mismo ``doc_id``
   (que no necesariamente está en el top recuperado), SOLO si el resultado
   no supera ``max_words`` palabras. El fragmento resultante conserva el
   ``chunk_id`` del primero y registra ``merged_with``.

Toda la lógica es pura: no toca ni MongoDB ni ningún modelo.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from src.retrieval.models import Fragment, FusedFragment

# División tras puntuación terminal (ES/EN/PT + CJK), misma idea que
# ``src.support.sentence_splitter`` pero sin dependencias externas.
_REGEX_ORACIONES = re.compile(r"(?<=[.!?»›。！？．｡])\s*")

# Firmas de dependencias inyectables (para tests con stubs).
SplitterProtocol = Callable[[str], List[str]]
SiguienteChunk = Callable[[str, int], Optional[str]]


def contar_palabras(texto: str) -> int:
    """Número de palabras (tokens separados por espacios) de ``texto``."""
    return len(texto.split()) if texto else 0


def _dividir_por_oraciones(
    texto: str, splitter: Optional[SplitterProtocol], max_words: int
) -> List[str]:
    """Empaqueta oraciones completas en sub-fragmentos de <= ``max_words``.

    Greedy: acumula oraciones mientras el sub-fragmento no supere el tope;
    una oración aislada más larga que ``max_words`` se conserva entera
    (respetar límites oracionales prevalece sobre el tope de palabras).
    """
    if splitter is not None:
        oraciones = [o for o in splitter(texto) if o.strip()]
    else:
        oraciones = [o.strip() for o in _REGEX_ORACIONES.split(texto) if o.strip()]

    sub_fragmentos: List[str] = []
    actual: List[str] = []
    palabras_actuales = 0

    for oracion in oraciones:
        n = contar_palabras(oracion)
        if actual and palabras_actuales + n > max_words:
            sub_fragmentos.append(" ".join(actual))
            actual, palabras_actuales = [], 0
        actual.append(oracion)
        palabras_actuales += n

    if actual:
        sub_fragmentos.append(" ".join(actual))
    return sub_fragmentos or [texto]


def _dividir_fragmento(
    fragmento: FusedFragment,
    splitter: Optional[SplitterProtocol],
    max_words: int,
) -> List[Fragment]:
    """Divide un fragmento largo en sub-fragmentos (mismo ``chunk_id``)."""
    sub_textos = _dividir_por_oraciones(fragmento.text, splitter, max_words)
    posicion = int(fragmento.metadata.get("posicion", 0))
    return [
        Fragment(
            chunk_id=fragmento.chunk_id,
            doc_id=fragmento.doc_id,
            text=sub,
            score=fragmento.rrf_score,
            posicion=posicion,
            sub_indice=i + 1,
            metadata=dict(fragmento.metadata),
        )
        for i, sub in enumerate(sub_textos)
    ]


def _fusionar_con_siguiente(
    fragmento: FusedFragment,
    siguiente_chunk: Optional[SiguienteChunk],
    max_words: int,
) -> Fragment:
    """Concatena un fragmento corto con el siguiente chunk del mismo doc
    si el resultado no excede ``max_words`` palabras."""
    posicion = int(fragmento.metadata.get("posicion", 0))
    if siguiente_chunk is None:
        return Fragment(
            chunk_id=fragmento.chunk_id,
            doc_id=fragmento.doc_id,
            text=fragmento.text,
            score=fragmento.rrf_score,
            posicion=posicion,
            metadata=dict(fragmento.metadata),
        )

    siguiente_texto = siguiente_chunk(fragmento.doc_id, posicion)
    if not siguiente_texto:
        return Fragment(
            chunk_id=fragmento.chunk_id,
            doc_id=fragmento.doc_id,
            text=fragmento.text,
            score=fragmento.rrf_score,
            posicion=posicion,
            metadata=dict(fragmento.metadata),
        )

    combinado = f"{fragmento.text.strip()} {siguiente_texto.strip()}"
    if contar_palabras(combinado) > max_words:
        return Fragment(
            chunk_id=fragmento.chunk_id,
            doc_id=fragmento.doc_id,
            text=fragmento.text,
            score=fragmento.rrf_score,
            posicion=posicion,
            metadata=dict(fragmento.metadata),
        )

    return Fragment(
        chunk_id=fragmento.chunk_id,
        doc_id=fragmento.doc_id,
        text=combinado,
        score=fragmento.rrf_score,
        posicion=posicion,
        merged_with=f"{fragmento.doc_id}::{posicion + 1}",
        metadata=dict(fragmento.metadata),
    )


def split_or_merge_fragments(
    fragmentos: Sequence[FusedFragment],
    max_words: int = 250,
    splitter: Optional[SplitterProtocol] = None,
    siguiente_chunk: Optional[SiguienteChunk] = None,
) -> List[Fragment]:
    """Aplica la Sección 9.2.1 (split de largos + fusión de cortos).

    Args:
        fragmentos: top-``k_chunk_out`` fragmentos tras RRF + filtros,
            ordenados por relevancia.
        max_words: tope de palabras (250 por defecto).
        splitter: ``Callable[[str], List[str]]`` que segmenta en oraciones
            completas. ``None`` usa un segmentador regex sin dependencias.
        siguiente_chunk: ``Callable[[doc_id, posicion], Optional[str]]`` que
            devuelve el texto del chunk en la posición ``posicion + 1`` del
            mismo documento (``None`` si no existe). ``None`` desactiva la
            fusión de fragmentos cortos.

    Returns:
        Lista de :class:`Fragment` en el mismo orden de entrada (un
        fragmento largo ocupa el lugar de sus sub-fragmentos).
    """
    resultado: List[Fragment] = []
    for fragmento in fragmentos:
        if contar_palabras(fragmento.text) > max_words:
            resultado.extend(_dividir_fragmento(fragmento, splitter, max_words))
        else:
            resultado.append(_fusionar_con_siguiente(fragmento, siguiente_chunk, max_words))
    return resultado
