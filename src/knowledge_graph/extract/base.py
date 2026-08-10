"""Interfaces de extracción de entidades (NER) y relaciones (RE).

Separa estrictamente ambas responsabilidades (SRP) detrás de interfaces
intercambiables (Strategy / OCP): cualquier modelo NER/RE nuevo se agrega
implementando el ABC, sin tocar el pipeline ni los consumidores (LSP/DIP).

Restricción dura del reto: NINGUNA implementación puede invocar un
LLM/decoder (GPT, LLaMA, Claude, etc.). Las estrategias por defecto de este
módulo son puramente simbólicas (regex + gazetteer + patrones léxicos),
deterministas y sin red.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Sequence

from src.knowledge_graph.models import Entity, EntityType, Relation


def normalizar_id_entidad(nombre: str) -> str:
    """Identificador canónico de una entidad (clave de nodo del grafo).

    Normalización determinista compartida por NER, RE y grafo: minúsculas,
    sin acentos, espacios colapsados y sin puntuación. Así "Perú" y "PERU"
    resuelven al mismo nodo y el GraphML queda estable entre corridas.
    """
    texto = unicodedata.normalize("NFKD", nombre)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return re.sub(r"\s+", " ", texto).strip().lower()


class EntityExtractor(ABC):
    """Reconocimiento de entidades nombradas sobre texto (NER, Sección 7.1).

    Consumido por el pipeline de extracción y por el ``GraphIndexAdapter``
    (mismo extractor al construir el grafo y al consultar, para que las
    entidades de la consulta resuelvan contra el vocabulario indexado).
    """

    #: Nombre corto registrado (traza en metadatos de construcción).
    name: str = "entity-extractor"

    @abstractmethod
    def extraer_entidades(self, texto: str) -> List[Entity]:
        """Devuelve las entidades nombradas detectadas en ``texto``.

        Las entidades se normalizan (id canónico), se tipan y se deduplican
        (una misma forma léxica aparece una sola vez).
        """

    def extraer_entidades_multiples(self, textos: Sequence[str]) -> List[Entity]:
        """Conveniencia: agrega entidades de varios textos, deduplicadas."""
        vistos: dict[str, Entity] = {}
        for texto in textos:
            for entidad in self.extraer_entidades(texto):
                vistos.setdefault(entidad.id, entidad)
        return list(vistos.values())


class RelationExtractor(ABC):
    """Extracción de relaciones tipadas entre entidades (RE, Sección 7.2).

    Recibe las entidades detectadas por el :class:`EntityExtractor` y el
    texto original; devuelve las relaciones que las conectan (la estrategia
    sin LLM las infiere por co-ocurrencia oracional y patrones verbales).
    """

    name: str = "relation-extractor"

    @abstractmethod
    def extraer_relaciones(
        self, texto: str, entidades: Sequence[Entity]
    ) -> List[Relation]:
        """Devuelve las relaciones entre ``entidades`` presentes en ``texto``.

        Solo se reportan relaciones entre entidades que realmente aparecen
        en el texto (nunca se inventan sujetos/objetos).
        """
