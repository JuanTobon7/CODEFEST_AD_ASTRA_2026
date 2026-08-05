"""
Etapa 4 del pipeline: post-filtros de la Sección 8.7, aplicados ANTES de
recortar al top final.

Dos familias de filtros, ambas sobre vectores/metadata puros (nunca sobre
texto generado):

1. **Filtro por metadata**: ``fenomeno``, ``formato``, ``idioma`` y rango de
   fechas (``fecha_publicacion``). Los campos obligatorios de la Tabla 1
   (``fenomeno``/``formato``) siempre existen en la metadata; ``idioma`` y
   ``fecha_publicacion`` son campos recomendados: si la metadata del
   fragmento no los trae, el filtro correspondiente se omite para ese
   fragmento (comportamiento conservador: no se descarta lo que no se
   puede verificar).

2. **Filtro por vector**: se descarta todo fragmento cuya similitud coseno
   ORIGINAL (antes de RRF, conservada en ``FusedFragment.cosine_score``)
   sea menor al umbral ``theta`` (``theta=0.0`` no descarta nada con
   vectores normalizados, salvo anti-correlación exacta).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.retrieval.models import FusedFragment, RetrievalFilters, _parse_fecha


def cumple_filtro_theta(fragmento: FusedFragment, theta: float) -> bool:
    """True si la mejor similitud coseno del fragmento es >= ``theta``."""
    return fragmento.cosine_score >= theta


def _cumple_filtro_fechas(
    fragmento: FusedFragment, rango: Tuple[Optional[object], Optional[object]]
) -> bool:
    """True si ``fecha_publicacion`` del fragmento cae dentro de ``rango``.

    Sin ``fecha_publicacion`` (o si no se puede parsear) se conserva el
    fragmento: no se descarta lo que no se puede verificar.
    """
    inicio, fin = rango
    fecha = fragmento.metadata.get("fecha_publicacion")
    if fecha is None:
        return True
    fecha_parseada = _parse_fecha(fecha)
    if fecha_parseada is None:
        return True
    if inicio is not None and fecha_parseada < inicio:
        return False
    if fin is not None and fecha_parseada > fin:
        return False
    return True


def cumple_filtro_metadata(fragmento: FusedFragment, filtros: RetrievalFilters) -> bool:
    """True si el fragmento pasa los filtros de metadata declarados.

    Regla conservadora: los filtros ``idioma``/``date_range`` solo excluyen
    cuando la metadata existe y NO coincide; la ausencia de la clave no
    excluye.
    """
    meta = fragmento.metadata

    if filtros.fenomeno is not None and meta.get("fenomeno") != filtros.fenomeno:
        return False

    if filtros.formato is not None and meta.get("formato") != filtros.formato:
        return False

    if filtros.idioma is not None:
        idioma = meta.get("idioma")
        if idioma is not None and str(idioma).lower() != filtros.idioma.lower():
            return False

    if filtros.date_range is not None and not _cumple_filtro_fechas(fragmento, filtros.date_range):
        return False

    return True


def apply_filters(
    fragmentos: List[FusedFragment],
    filtros: Optional[RetrievalFilters] = None,
) -> List[FusedFragment]:
    """Aplica post-filtros (Sección 8.7) a la lista fusionada con RRF.

    El orden importa poco (ambos filtros son independientes), pero se aplica
    primero el filtro por vector (más barato) y luego el de metadata.

    Args:
        fragmentos: salida de :func:`src.retrieval.rrf.rrf_fuse`.
        filtros: filtros a aplicar; ``None`` (o todos los campos en ``None``)
            equivale a devolver la lista sin cambios.

    Returns:
        Nueva lista con los fragmentos que superan todos los filtros.
    """
    if filtros is None:
        return list(fragmentos)

    resultado = [f for f in fragmentos if cumple_filtro_theta(f, filtros.theta)]
    resultado = [f for f in resultado if cumple_filtro_metadata(f, filtros)]
    return resultado
