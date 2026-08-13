"""Configuración de la RE NLI: constantes puras, sin dependencias de modelo.

Separada de la estrategia (:mod:`nli_relation_strategy`) y del motor de
inferencia (:mod:`nli_backend`) para que las plantillas de hipótesis sean
testeables e intercambiables sin instanciar ningún modelo.
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.knowledge_graph.models import RelationType

#: Checkpoint de clasificación de relaciones del grafo (Sección 7.2).
#:
#: ``mDeBERTa-v3-base-mnli-xnli`` es un ENCODER bidireccional (DeBERTa-v3,
#: ~280M) con una cabecera de clasificación de 3 etiquetas
#: (contradicción/neutral/entailment). NO es generativo: no tiene decoder ni
#: cabecera de lenguaje, no produce texto — solo tres logits por par
#: (premisa, hipótesis). Multilingüe (entrenado en XNLI, ~100 idiomas del
#: preentrenamiento de mDeBERTa), lo que cubre es/en/pt del corpus.
MODELO_NLI_DEFAULT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

#: Licencia del checkpoint (verificada en su model card de HuggingFace).
#: El reto exige licencias permisivas tipo Apache-2.0/MIT: este cumple.
LICENCIA_MODELO_NLI = "MIT"

#: Ficha del modelo para el informe técnico y la auditoría de licencias.
FICHA_MODELO_NLI: Dict[str, str] = {
    "model_id": MODELO_NLI_DEFAULT,
    "arquitectura": "DeBERTa-v3 (encoder bidireccional) + cabecera de clasificación",
    "generativo": "no",
    "tarea": "clasificación de pares (NLI) reutilizada como clasificación de relaciones",
    "licencia": LICENCIA_MODELO_NLI,
    "idiomas": "multilingüe (XNLI: 15 idiomas; mDeBERTa: ~100)",
    "parametros": "~280M",
}

#: Hipótesis por RelationType. ``{sujeto}``/``{objeto}`` se sustituyen con la
#: forma léxica preferida de cada entidad. El ORDEN importa: la variante 0 es
#: la canónica en español y la 1 la canónica en inglés (el corpus es
#: mayoritariamente EN y las consultas del reto son ES; mDeBERTa clasifica
#: pares premisa/hipótesis en idiomas distintos, que es justo el escenario de
#: XNLI). ``plantillas_activas`` recorta cuántas variantes se evalúan: el
#: coste del canal es lineal en el número de hipótesis por par.
#:
#: ``PARTE_DE`` no aparece: es un sinónimo de ``ES_PARTE_DE`` en el
#: vocabulario del grafo y evaluar ambos gastaría presupuesto en dos
#: hipótesis que compiten entre sí. La RE simbólica sí lo sigue emitiendo.
PLANTILLAS_HIPOTESIS: Dict[RelationType, Tuple[str, ...]] = {
    RelationType.UBICADO_EN: (
        "{sujeto} está ubicado en {objeto}",
        "{sujeto} is located in {objeto}",
        "{sujeto} se encuentra en {objeto}",
    ),
    RelationType.ES_PARTE_DE: (
        "{sujeto} es parte de {objeto}",
        "{sujeto} is part of {objeto}",
        "{sujeto} es miembro de {objeto}",
        "{sujeto} is a member of {objeto}",
    ),
    RelationType.PERTENECE_A: (
        "{sujeto} pertenece a {objeto}",
        "{sujeto} belongs to {objeto}",
        "{sujeto} es propiedad de {objeto}",
    ),
    RelationType.AFECTA: (
        "{sujeto} afecta a {objeto}",
        "{sujeto} affects {objeto}",
        "{sujeto} amenaza a {objeto}",
        "{sujeto} threatens {objeto}",
    ),
    RelationType.CAUSA: (
        "{sujeto} causa {objeto}",
        "{sujeto} causes {objeto}",
        "{sujeto} provoca {objeto}",
    ),
    RelationType.DEPENDE_DE: (
        "{sujeto} depende de {objeto}",
        "{sujeto} depends on {objeto}",
    ),
    RelationType.COOPERA_CON: (
        "{sujeto} coopera con {objeto}",
        "{sujeto} cooperates with {objeto}",
        "{sujeto} colabora con {objeto}",
        "{sujeto} collaborates with {objeto}",
    ),
    RelationType.OPONE_A: (
        "{sujeto} se opone a {objeto}",
        "{sujeto} opposes {objeto}",
        "{sujeto} está en conflicto con {objeto}",
    ),
    RelationType.INVOLUCRA: (
        "{sujeto} involucra a {objeto}",
        "{sujeto} involves {objeto}",
        "{sujeto} incluye a {objeto}",
        "{sujeto} includes {objeto}",
    ),
    RelationType.RELACIONADO_CON: (
        "{sujeto} está relacionado con {objeto}",
        "{sujeto} is related to {objeto}",
        "{sujeto} está vinculado a {objeto}",
    ),
}

#: Variantes de hipótesis evaluadas por tipo si no se indica otra cosa.
#:
#: Una sola (la canónica en español) es el presupuesto elegido para la
#: entrega: mDeBERTa clasifica pares premisa/hipótesis en idiomas distintos
#: —es justo el escenario de XNLI—, así que una hipótesis en español sobre un
#: corpus mayoritariamente inglés funciona, y evaluar la segunda variante
#: duplicaría el coste del canal. Súbelo con ``--variantes-plantilla`` si
#: sobra cómputo y se quiere un tipado más rico.
VARIANTES_PLANTILLA_DEFAULT = 1

#: Probabilidad de entailment mínima para emitir una relación TIPADA.
UMBRAL_TIPADO = 0.5
#: Probabilidad mínima para emitir ``COOCURRENCIA`` (por debajo: nada).
UMBRAL_COOCURRENCIA = 0.0

#: Topes del presupuesto de inferencia (el coste del canal es
#: ``pares_dirigidos × tipos × variantes`` forwards del modelo).
#:
#: Calibrados para que el grafo cubra el corpus COMPLETO en un tiempo
#: razonable: para recuperación pesa más la cobertura (que todo chunk tenga
#: nodos y aristas) que la riqueza del tipado — una arista ``COOCURRENCIA``
#: recupera el chunk igual que una tipada.
MAX_PARES_POR_ORACION_DEFAULT = 4
MAX_PARES_POR_CHUNK_DEFAULT = 4


def plantillas_activas(
    variantes: int = VARIANTES_PLANTILLA_DEFAULT,
) -> Dict[RelationType, Tuple[str, ...]]:
    """Recorta :data:`PLANTILLAS_HIPOTESIS` a ``variantes`` por tipo.

    Args:
        variantes: número de formulaciones por :class:`RelationType`
            (``<= 0`` significa todas las declaradas).

    Returns:
        Mapa tipo -> plantillas efectivamente evaluadas.
    """
    if variantes <= 0:
        return dict(PLANTILLAS_HIPOTESIS)
    return {
        tipo: plantillas[:variantes]
        for tipo, plantillas in PLANTILLAS_HIPOTESIS.items()
    }
