"""Pipeline de extracción texto → NER → RE → tripletas (Chain of Responsibility).

Cada etapa es una clase independiente (SRP) que implementa el mismo contrato
``procesar(entrada) -> salida``; el :class:`ExtractionPipeline` las encadena
en orden fijo sin conocer sus detalles internos (bajo acoplamiento). Las
etapas son intercambiables y testeables por separado (OCP): añadir una etapa
nueva (p. ej. normalización previa) no modifica las existentes.

Flujo para un chunk: texto → NER (entidades) → RE (relaciones) →
:class:`ExtractionResult` (producto intermedio con tripletas trazables).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from src.knowledge_graph.extract.base import EntityExtractor, RelationExtractor
from src.knowledge_graph.models import ExtractionResult

E = TypeVar("E")
S = TypeVar("S")


class PipelineStep(ABC, Generic[E, S]):
    """Una etapa del pipeline con contrato ``procesar(E) -> S``."""

    nombre: str = "paso"

    @abstractmethod
    def procesar(self, entrada: E) -> S:
        """Procesa ``entrada`` y devuelve su transformación."""


class NERStep(PipelineStep[str, List]):
    """Etapa 1: texto → entidades nombradas (delega en un EntityExtractor)."""

    def __init__(self, extractor: EntityExtractor) -> None:
        self._extractor = extractor

    @property
    def nombre(self) -> str:
        return f"ner:{self._extractor.name}"

    def procesar(self, entrada: str) -> List:
        """Devuelve la lista de :class:`Entity` de ``entrada``."""
        return self._extractor.extraer_entidades(entrada)


class REStep(PipelineStep[tuple, List]):
    """Etapa 2: (texto, entidades) → relaciones (delega en un RelationExtractor)."""

    def __init__(self, extractor: RelationExtractor) -> None:
        self._extractor = extractor

    @property
    def nombre(self) -> str:
        return f"re:{self._extractor.name}"

    def procesar(self, entrada: tuple) -> List:
        """Devuelve la lista de :class:`Relation` entre las entidades dadas."""
        texto, entidades = entrada
        return self._extractor.extraer_relaciones(texto, entidades)


class TripleFactory:
    """Materializa tripletas desde entidades + relaciones + procedencia.

    Creator (GRASP): agrega las tripletas a partir de las entidades y
    relaciones que ya posee (recibe un :class:`ExtractionResult`), sin
    conocer NER ni RE.
    """

    def crear(self, resultado: ExtractionResult) -> List:
        """Tripletas trazables de ``resultado`` (ver ``ExtractionResult.tripletas``)."""
        return resultado.tripletas()


class ExtractionPipeline:
    """Cadena fija: NER → RE → TripleFactory (Chain of Responsibility).

    El orquestador solo conoce el contrato ``procesar`` de cada paso
    (DIP): si mañana el NER cambia de regex a un modelo neuronal, este
    pipeline no cambia.
    """

    def __init__(
        self,
        ner: EntityExtractor,
        re: RelationExtractor,
        triple_factory: TripleFactory | None = None,
    ) -> None:
        self._pasos: List[PipelineStep] = [
            NERStep(ner),
            REStep(re),
        ]
        self._triple_factory = triple_factory or TripleFactory()

    def procesar_chunk(self, doc_id: str, chunk_id: str, texto: str) -> ExtractionResult:
        """Ejecuta NER → RE sobre ``texto`` y devuelve el resultado intermedio.

        Args:
            doc_id: documento de origen (trazabilidad de la Sección 7.3).
            chunk_id: fragmento de origen (trazabilidad de la Sección 7.3).
            texto: texto del chunk (tal como lo indexó la base vectorial).

        Returns:
            :class:`ExtractionResult` con entidades, relaciones y tripletas
            materializadas (cada tripleta conserva ``doc_id`` y ``chunk_id``).
        """
        entidades = self._pasos[0].procesar(texto)
        relaciones = self._pasos[1].procesar((texto, entidades))
        resultado = ExtractionResult(
            doc_id=doc_id,
            chunk_id=chunk_id,
            entidades=entidades,
            relaciones=relaciones,
            texto_original=texto,
        )
        return resultado

    @property
    def pasos(self) -> List[PipelineStep]:
        """Etapas encadenadas (solo lectura, para inspección/auditoría)."""
        return list(self._pasos)
