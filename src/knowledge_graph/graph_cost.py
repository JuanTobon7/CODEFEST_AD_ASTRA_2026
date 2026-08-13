"""Estimación del coste de construir el grafo sobre el corpus completo.

El canal de relaciones invoca un modelo por cada par de entidades candidato,
así que el tiempo total depende del hardware y del presupuesto configurado.
Antes de comprometer horas de GPU conviene MEDIRLO en la máquina donde se va
a correr, en vez de estimarlo a ojo: este módulo cronometra la construcción
sobre una muestra real de chunks y extrapola al corpus.

Se usa vía ``run_build_graph --estimar N`` (ver README §11.3).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Sequence

from src.knowledge_graph.service import KnowledgeGraphService

logger = logging.getLogger(__name__)


@dataclass
class EstimacionCoste:
    """Proyección del coste de construir el grafo sobre todo el corpus."""

    chunks_muestra: int
    chunks_totales: int
    segundos_muestra: float
    entidades_muestra: int
    tripletas_muestra: int

    @property
    def segundos_por_chunk(self) -> float:
        """Tiempo medio por chunk medido en la muestra."""
        return self.segundos_muestra / max(self.chunks_muestra, 1)

    @property
    def horas_estimadas(self) -> float:
        """Proyección al corpus completo, en horas.

        Es una cota SUPERIOR razonable: la caché por (oración, par) acierta
        más cuanto más grande es el corpus (boilerplate repetido), así que
        el coste real por chunk tiende a bajar al escalar.
        """
        return self.segundos_por_chunk * self.chunks_totales / 3600.0

    def como_texto(self) -> str:
        """Resumen legible para el log del CLI."""
        return (
            f"Muestra: {self.chunks_muestra} chunks en {self.segundos_muestra:.1f}s "
            f"({self.segundos_por_chunk * 1000:.0f} ms/chunk) -> "
            f"{self.entidades_muestra} entidades, {self.tripletas_muestra} tripletas. "
            f"Proyección a {self.chunks_totales} chunks: "
            f"~{self.horas_estimadas:.1f} h en este hardware."
        )


def estimar(
    chunks_muestra: Sequence,
    chunks_totales: int,
    relation_extractor=None,
) -> EstimacionCoste:
    """Cronometra la construcción sobre ``chunks_muestra`` y extrapola.

    Args:
        chunks_muestra: chunks reales del corpus (no sintéticos: el coste
            depende de cuántas entidades trae cada texto).
        chunks_totales: tamaño del corpus al que se extrapola.
        relation_extractor: la MISMA estrategia RE con la que se construirá
            el grafo definitivo; si es ``None`` se usa el default del
            servicio (simbólico).

    Returns:
        La :class:`EstimacionCoste` medida.
    """
    muestra: List = list(chunks_muestra)
    servicio = KnowledgeGraphService(relation_extractor=relation_extractor)

    # Primera pasada corta para pagar la carga perezosa del modelo fuera del
    # cronómetro (descarga/allocación en GPU no es parte del coste por chunk).
    if muestra:
        servicio.construir_desde_chunks(muestra[:1])

    inicio = time.perf_counter()
    grafo = servicio.construir_desde_chunks(muestra)
    transcurrido = time.perf_counter() - inicio

    return EstimacionCoste(
        chunks_muestra=len(muestra),
        chunks_totales=chunks_totales,
        segundos_muestra=transcurrido,
        entidades_muestra=grafo.num_entidades,
        tripletas_muestra=grafo.num_tripletas,
    )
