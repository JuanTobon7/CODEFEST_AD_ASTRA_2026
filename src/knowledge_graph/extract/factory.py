"""Factories de extractores NER/RE (Factory Method con registro por decorador).

Sigue el patrón del proyecto (:class:`EncoderFactory`): agregar un modelo
NER/RE nuevo NO requiere tocar esta clase ni el pipeline — basta crear la
estrategia, decorarla con ``@EntityExtractorFactory.register("nombre")`` y
(a lo sumo) importarla desde el punto de entrada. Cumple OCP y DIP: los
consumidores piden una estrategia por nombre, nunca construyen concretas.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from src.knowledge_graph.extract.base import EntityExtractor, RelationExtractor

logger = logging.getLogger(__name__)


class EntityExtractorFactory:
    """Crea instancias de :class:`EntityExtractor` por nombre registrado."""

    _registry: Dict[str, Type[EntityExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorador: ``@EntityExtractorFactory.register("regex-gazetteer")``."""

        def _decorador(clase: Type[EntityExtractor]) -> Type[EntityExtractor]:
            clave = name.strip().lower()
            if clave in cls._registry:
                logger.warning("Sobrescribiendo extractor de entidades ya registrado: '%s'", clave)
            cls._registry[clave] = clase
            return clase

        return _decorador

    @classmethod
    def create(cls, name: str, **kwargs) -> EntityExtractor:
        """Instancia la estrategia ``name`` (``kwargs`` van a su constructor).

        Raises:
            ValueError: si ``name`` no está registrado.
        """
        clave = name.strip().lower()
        clase = cls._registry.get(clave)
        if clase is None:
            raise ValueError(
                f"Extractor de entidades desconocido: '{name}'. "
                f"Disponibles: {', '.join(sorted(cls._registry)) or '(ninguno)'}"
            )
        return clase(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Nombres registrados (ordenados), sin instanciarlos."""
        return sorted(cls._registry)


class RelationExtractorFactory:
    """Crea instancias de :class:`RelationExtractor` por nombre registrado."""

    _registry: Dict[str, Type[RelationExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorador: ``@RelationExtractorFactory.register("coocurrencia-oracional")``."""

        def _decorador(clase: Type[RelationExtractor]) -> Type[RelationExtractor]:
            clave = name.strip().lower()
            if clave in cls._registry:
                logger.warning("Sobrescribiendo extractor de relaciones ya registrado: '%s'", clave)
            cls._registry[clave] = clase
            return clase

        return _decorador

    @classmethod
    def create(cls, name: str, **kwargs) -> RelationExtractor:
        """Instancia la estrategia ``name`` (``kwargs`` van a su constructor)."""
        clave = name.strip().lower()
        clase = cls._registry.get(clave)
        if clase is None:
            raise ValueError(
                f"Extractor de relaciones desconocido: '{name}'. "
                f"Disponibles: {', '.join(sorted(cls._registry)) or '(ninguno)'}"
            )
        return clase(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Nombres registrados (ordenados), sin instanciarlos."""
        return sorted(cls._registry)
