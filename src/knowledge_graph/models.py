"""Modelos de dominio del módulo de Grafo de Conocimiento (Sección 7).

Entidades puras (pydantic/dataclass) que representan el grafo
G = (E, R, T) con T ⊆ E × R × E, donde cada tripleta conserva la
trazabilidad de evidencia textual exigida por el reto (``doc_id`` y
``chunk_id`` de origen).

Ninguno de estos modelos depende de NetworkX, Neo4j, MongoDB ni de ningún
modelo generativo: son la representación neutra sobre la que operan el
:class:`GraphBuilder` (escritura) y los serializadores (GraphML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Tipos de entidad reconocidos por el NER (vocabulario controlado)."""

    PERSONA = "PERSONA"
    ORGANIZACION = "ORGANIZACION"
    ORGANIZACION_INTERNACIONAL = "ORGANIZACION_INTERNACIONAL"
    PAIS = "PAIS"
    LUGAR = "LUGAR"
    PROGRAMA = "PROGRAMA"
    TECNOLOGIA = "TECNOLOGIA"
    SECTOR = "SECTOR"
    EVENTO = "EVENTO"
    CONFLICTO = "CONFLICTO"
    OTRO = "OTRO"


class RelationType(str, Enum):
    """Tipos de relación extraídos por la RE (vocabulario controlado).

    ``COOCURRENCIA`` es el tipo por defecto de la estrategia basada en
    co-ocurrencia oracional (sin LLM); el resto son patrones verbales
    explícitos que la estrategia reconoce sobre el texto.
    """

    COOCURRENCIA = "COOCURRENCIA"
    UBICADO_EN = "UBICADO_EN"
    PARTE_DE = "PARTE_DE"
    PERTENECE_A = "PERTENECE_A"
    AFECTA = "AFECTA"
    CAUSA = "CAUSA"
    DEPENDE_DE = "DEPENDE_DE"
    COOPERA_CON = "COOPERA_CON"
    OPONE_A = "OPONE_A"
    ES_PARTE_DE = "ES_PARTE_DE"
    INVOLUCRA = "INVOLUCRA"
    RELACIONADO_CON = "RELACIONADO_CON"


class Entity(BaseModel):
    """Nodo del grafo: una entidad nombrada con su tipo y variantes.

    ``id`` es el identificador canónico normalizado (usado como clave de
    nodo en el grafo y en el GraphML); ``nombre`` es la forma léxica que se
    muestra al consumidor.
    """

    id: str = Field(description="Identificador canónico normalizado (clave del nodo)")
    nombre: str = Field(description="Forma léxica preferida de la entidad")
    tipo: EntityType = Field(default=EntityType.OTRO)
    variantes: List[str] = Field(
        default_factory=list, description="Formas alternativas observadas en el corpus"
    )


class Relation(BaseModel):
    """Arista candidata: relación tipada entre dos entidades.

    ``sujeto``/``objeto`` referencian ``Entity.id``. ``confianza`` es una
    puntuación 0..1 que la estrategia de RE asigna a la extracción (la
    estrategia sin LLM usa evidencia léxica).
    """

    tipo: RelationType = Field(default=RelationType.COOCURRENCIA)
    sujeto: str = Field(description="Entity.id del sujeto")
    objeto: str = Field(description="Entity.id del objeto")
    confianza: float = Field(default=1.0, ge=0.0, le=1.0)


class Tripleta(BaseModel):
    """Tripleta persistida: T ⊆ E × R × E con trazabilidad de evidencia.

    La Sección 7 exige que cada tripleta conserve ``doc_id`` y ``chunk_id``
    de origen; ``evidencia`` guarda el fragmento textual que la sustenta
    (auditoría del informe técnico, sin generar texto nuevo).
    """

    sujeto: str = Field(description="Entity.id del sujeto")
    relacion: RelationType = Field(description="Tipo de relación (R)")
    objeto: str = Field(description="Entity.id del objeto")
    doc_id: str = Field(description="Documento de origen (trazabilidad)")
    chunk_id: str = Field(description="Fragmento de origen (trazabilidad)")
    evidencia: str = Field(default="", description="Fragmento textual de sustento")
    confianza: float = Field(default=1.0, ge=0.0, le=1.0)


class ScoredChunk(BaseModel):
    """Contrato común de salida de cualquier ``Retriever`` (patrón Adapter).

    Tanto el ``VectorIndexAdapter`` (FAISS) como el ``GraphIndexAdapter``
    (grafo de conocimiento) devuelven esta misma estructura para que el
    motor de fusión (Sección 8.5) trate al grafo "como un índice más",
    sin conocer su origen. ``origen`` es una etiqueta de trazabilidad
    (nombre del encoder, o ``"grafo"``).
    """

    chunk_id: str
    doc_id: str
    score: float
    origen: str = Field(default="desconocido")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    """Consulta de recuperación: el texto en lenguaje natural.

    Se ampliará con filtros (fenómeno, formato, idioma, fechas) en la
    etapa de post-filtrado; aquí solo transporta el texto que codifican
    los adapters (vectorial: encoder; grafo: NER).
    """

    texto: str = Field(min_length=1)


@dataclass
class ExtractionResult:
    """Resultado de la extracción sobre UN chunk (NER + RE).

    Es el producto intermedio del pipeline de extracción (etapa texto →
    NER → RE): las entidades detectadas y las relaciones entre ellas,
    junto con la procedencia exacta (``doc_id``/``chunk_id``/``evidencia``)
    que después materializará el :class:`TripleFactory` en tripletas.
    """

    doc_id: str
    chunk_id: str
    entidades: List[Entity] = field(default_factory=list)
    relaciones: List[Relation] = field(default_factory=list)
    texto_original: str = ""

    def tripletas(self) -> List[Tripleta]:
        """Materializa las tripletas (sujeto-relación-objeto) con trazabilidad.

        Creator (GRASP): este resultado es quien AGREGÓ las entidades y las
        relaciones que componen cada tripleta, así que es quien sabe
        materializarlas; usa una entidad solo si aparece en el texto.
        """
        ids = {e.id for e in self.entidades}
        tripletas: List[Tripleta] = []
        for r in self.relaciones:
            if r.sujeto in ids and r.objeto in ids:
                tripletas.append(
                    Tripleta(
                        sujeto=r.sujeto,
                        relacion=r.tipo,
                        objeto=r.objeto,
                        doc_id=self.doc_id,
                        chunk_id=self.chunk_id,
                        evidencia=self.texto_original,
                        confianza=r.confianza,
                    )
                )
        return tripletas

    def __bool__(self) -> bool:
        """Un resultado vacío (sin entidades ni relaciones) es falsy."""
        return bool(self.entidades or self.relaciones)
