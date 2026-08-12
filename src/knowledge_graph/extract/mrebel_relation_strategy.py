"""Estrategia RE neuronal: mREBEL (``Babelscape/mrebel-large``) — RE end-to-end.

mREBEL es la versión multilingüe de REBEL (ACL 2023, REDFM): un
seq2seq (T5-large, 560M) que genera tripletas ``(sujeto, relación, objeto)``
directamente desde el texto, en es/en/pt. NO es un LLM generativo libre:
es un modelo de extracción de relaciones entrenado (Sección 7.2
"modelos de clasificación de relaciones").

Flujo: por cada oración con ≥2 entidades del NER → el motor genera las
secuencias de tripletas → se parsean con :func:`extract_triplets` → se
filtran las tripletas cuyos head/tail corresponden a entidades del NER
(nunca se inventan sujetos/objetos) → se mapea la relación a
:class:`RelationType` (las no mapeadas caen a RELACIONADO_CON).

OJO LICENCIA: ``Babelscape/mrebel-large`` es CC BY-NC-SA 4.0 (no comercial)
— el usuario decidió usarla explícitamente. Registrada como ``mrebel``.

Módulos: :mod:`mrebel_config` (constantes/mapeo) y :mod:`mrebel_backend`
(motor de inferencia). ``tokenizer``/``modelo`` inyectables para tests.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.knowledge_graph.extract.base import RelationExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.entity_matchers import formas_buscables, mencionada
from src.knowledge_graph.extract.factory import RelationExtractorFactory
from src.knowledge_graph.extract.mrebel_backend import MrebelInferenceEngine
from src.knowledge_graph.extract.mrebel_config import (
    CONFIANZA_MAPEADA,
    CONFIANZA_NO_MAPEADA,
    MAPEO_RELACIONES,
    MODELO_MREBEL_DEFAULT,
    SRC_LANG_DEFAULT,
)
from src.knowledge_graph.models import Entity, Relation, RelationType

_REGEX_ORACIONES = re.compile(r"(?<=[.!?。！？])\s+")
_ESPACIO = re.compile(r"\s+")


def extract_triplets(texto: str) -> List[dict]:
    """Parsea una secuencia generada por mREBEL a lista de tripletas.

    Basado en el autómata oficial de REBEL (``extract_triplets``), con dos
    extensiones para cubrir los formatos que emite mREBEL:
    - ``</triplet>`` y los tokens de cierre ya no contaminan la relación.
    - Los tokens de tipo (``<ORG>``, ``<PER>``...) transicionan de estado
      igual que en ``extract_triplets_typed`` (sin usar el tipo).

    Formatos soportados::

        <triplet> SUJETO <subj> OBJETO <obj> RELACIÓN </triplet>   # REBEL
        <triplet> SUJETO <ORG> OBJETO <ORG> RELACIÓN </triplet>    # REDFM/mREBEL

    Returns:
        Lista de ``{"head", "type", "tail"}`` en el orden de la secuencia.
    """
    tripletas: List[dict] = []
    relacion, sujeto, objeto = "", "", ""
    current = "x"
    tokens = (
        texto.replace("<s>", " ").replace("</s>", " ").replace("<pad>", " ").split()
    )
    for token in tokens:
        if token == "<triplet>":
            current = "t"
            if relacion != "":
                tripletas.append({"head": sujeto.strip(), "type": relacion.strip(), "tail": objeto.strip()})
            relacion, sujeto = "", ""
        elif token == "<subj>":
            current = "s"
            if relacion != "":
                tripletas.append({"head": sujeto.strip(), "type": relacion.strip(), "tail": objeto.strip()})
            objeto = ""
        elif token == "<obj>":
            current = "o"
            relacion = ""
        elif token.startswith("<") and token.endswith(">"):
            # Token de tipo (<ORG>, <PER>...) o cierre (</triplet>): transición.
            if current == "t" or current == "o":
                current = "s"
                if relacion != "":
                    tripletas.append({"head": sujeto.strip(), "type": relacion.strip(), "tail": objeto.strip()})
                relacion = ""
                objeto = ""
            elif current == "s":
                current = "o"
                relacion = ""
        else:
            if current == "t":
                sujeto += " " + token
            elif current == "s":
                objeto += " " + token
            elif current == "o":
                relacion += " " + token
    if sujeto != "" and relacion != "" and objeto != "":
        tripletas.append({"head": sujeto.strip(), "type": relacion.strip(), "tail": objeto.strip()})
    return tripletas


@RelationExtractorFactory.register("mrebel")
class MrebelRelationExtractor(RelationExtractor):
    """RE end-to-end con mREBEL entre pares de entidades del NER.

    Args:
        model_id: checkpoint de HuggingFace (mREBEL, seq2seq multilingüe).
        src_lang: idioma de origen del tokenizer mBART (``en_XX`` por defecto).
        device_preference: ``None``/``"auto"`` (detecta CUDA), ``"cpu"`` o ``"cuda"``.
        max_length: longitud máxima de la secuencia generada.
        num_beams: beam search (3 = el de la model card).
        use_fp16: si ``True`` y hay CUDA, el modelo se convierte a float16
            (VRAM ~2.9 GB → ~1.5 GB, ~2x más rápido); en CPU se ignora.
        tokenizer/modelo: inyectables (tests u otro checkpoint).
    """

    name = "mrebel"

    def __init__(
        self,
        model_id: str = MODELO_MREBEL_DEFAULT,
        src_lang: str = SRC_LANG_DEFAULT,
        device_preference: Optional[str] = None,
        max_length: int = 256,
        num_beams: int = 3,
        use_fp16: bool = False,
        tokenizer=None,
        modelo=None,
    ) -> None:
        self._motor = MrebelInferenceEngine(
            model_id=model_id,
            src_lang=src_lang,
            device_preference=device_preference,
            max_length=max_length,
            num_beams=num_beams,
            use_fp16=use_fp16,
            tokenizer=tokenizer,
            modelo=modelo,
        )
        #: Caché por oración normalizada -> tripletas parseadas (frases
        #: repetidas del corpus no re-generan con el modelo).
        self._cache: Dict[str, List[dict]] = {}

    def extraer_relaciones(
        self, texto: str, entidades: Sequence[Entity]
    ) -> List[Relation]:
        """Relaciones entre ``entidades`` que co-ocurren en ``texto`` (mREBEL).

        Solo se emiten tripletas cuyo head y tail corresponden a entidades
        del NER presentes en la oración (nunca se inventan sujetos/objetos).
        """
        if not texto or len(entidades) < 2:
            return []
        self._motor.asegurar_modelo()
        formas = formas_buscables(entidades)
        oraciones = _REGEX_ORACIONES.split(_ESPACIO.sub(" ", texto.strip()))
        mejores: Dict[Tuple[str, RelationType, str], float] = {}
        for oracion in oraciones:
            self._acumular_de_oracion(oracion, formas, mejores)

        return [
            Relation(tipo=tipo, sujeto=sujeto, objeto=objeto, confianza=confianza)
            for (sujeto, tipo, objeto), confianza in sorted(mejores.items())
        ]

    def _acumular_de_oracion(
        self,
        oracion: str,
        formas: Dict[str, List[re.Pattern[str]]],
        mejores: Dict[Tuple[str, RelationType, str], float],
    ) -> None:
        """Genera las tripletas de una oración y las acumula en ``mejores``."""
        noracion = _ESPACIO.sub(" ", normalizar_id_entidad(oracion))
        presentes = [
            eid for eid, patrones in formas.items() if mencionada(noracion, patrones)
        ]
        if len(presentes) < 2:
            return
        for tripleta in self._tripletas_de_oracion(oracion, noracion):
            sujeto = self._entidad_de(tripleta.get("head", ""), presentes, formas)
            objeto = self._entidad_de(tripleta.get("tail", ""), presentes, formas)
            if sujeto is None or objeto is None or sujeto == objeto:
                continue
            tipo, confianza = self._tipo_y_confianza(tripleta.get("type", ""))
            clave = (sujeto, tipo, objeto)
            mejores[clave] = max(mejores.get(clave, 0.0), confianza)

    def _tripletas_de_oracion(self, oracion: str, noracion: str) -> List[dict]:
        """Tripletas generadas por el modelo para la oración (con caché)."""
        if noracion in self._cache:
            return self._cache[noracion]
        tripletas: List[dict] = []
        for secuencia in self._motor.generar(oracion):
            tripletas.extend(extract_triplets(secuencia))
        # Dedup: el beam search puede repetir la misma tripleta.
        unicas = {
            (
                t.get("head", "").strip().lower(),
                t.get("type", "").strip().lower(),
                t.get("tail", "").strip().lower(),
            ): t
            for t in tripletas
        }
        resultado = list(unicas.values())
        self._cache[noracion] = resultado
        return resultado

    @staticmethod
    def _entidad_de(
        forma: str,
        presentes: List[str],
        formas: Dict[str, List[re.Pattern[str]]],
    ) -> Optional[str]:
        """Id de la entidad presente que menciona ``forma`` (o ``None``)."""
        nforma = _ESPACIO.sub(" ", normalizar_id_entidad(forma))
        if not nforma:
            return None
        for eid in presentes:
            if mencionada(nforma, formas[eid]):
                return eid
        return None

    @staticmethod
    def _tipo_y_confianza(relacion_texto: str) -> Tuple[RelationType, float]:
        """Mapea la relación generada al vocabulario del grafo."""
        clave = _ESPACIO.sub(" ", normalizar_id_entidad(relacion_texto))
        tipo = MAPEO_RELACIONES.get(clave)
        if tipo is None:
            return RelationType.RELACIONADO_CON, CONFIANZA_NO_MAPEADA
        return tipo, CONFIANZA_MAPEADA
