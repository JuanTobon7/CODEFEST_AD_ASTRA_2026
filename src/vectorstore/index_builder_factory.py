"""
Factory de estrategias de índice FAISS con registro por decorador (mismo
patrón que ``EncoderFactory``): abierto/cerrado, agregar un tipo de índice
nuevo no requiere tocar esta clase.

Resuelve además el valor especial ``FAISS_INDEX_TYPE=auto``: por debajo de
``ivf_auto_threshold`` vectores usa ``flat_ip`` (exacto), por encima usa
``ivf_flat`` (aproximado, escala mejor) — decisión que se deja trazada en
el log para justificar en el informe técnico.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from src.vectorstore.index_builder_base import FaissIndexBuilderStrategy, IndexBuildConfig

logger = logging.getLogger(__name__)


class IndexBuilderFactory:
    """Crea instancias de :class:`FaissIndexBuilderStrategy` por nombre registrado."""

    _registry: Dict[str, Type[FaissIndexBuilderStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorador: ``@IndexBuilderFactory.register("flat_ip")`` sobre la clase concreta."""

        def _decorador(clase: Type[FaissIndexBuilderStrategy]) -> Type[FaissIndexBuilderStrategy]:
            clave = name.strip().lower()
            if clave in cls._registry:
                logger.warning("Sobrescribiendo tipo de índice ya registrado: '%s'", clave)
            cls._registry[clave] = clase
            return clase

        return _decorador

    @classmethod
    def create(cls, name: str) -> FaissIndexBuilderStrategy:
        """Instancia la estrategia ``name`` (debe estar registrada)."""
        clave = name.strip().lower()
        clase = cls._registry.get(clave)
        if clase is None:
            raise ValueError(
                f"Tipo de índice FAISS desconocido: '{name}'. Disponibles: {', '.join(sorted(cls._registry))}"
            )
        return clase()

    @classmethod
    def resolve(cls, index_type: str, n_vectors: int, config: IndexBuildConfig) -> FaissIndexBuilderStrategy:
        """Resuelve ``index_type`` a una estrategia concreta, incluyendo ``auto``.

        ``auto`` decide entre ``flat_ip`` (exacto, por defecto) e ``ivf_flat``
        (aproximado) comparando ``n_vectors`` contra ``config.ivf_auto_threshold``.
        """
        clave = index_type.strip().lower()
        if clave != "auto":
            return cls.create(clave)
        elegido = "ivf_flat" if n_vectors > config.ivf_auto_threshold else "flat_ip"
        logger.info(
            "FAISS_INDEX_TYPE=auto: %d vectores %s umbral=%d -> se elige '%s'",
            n_vectors, ">" if elegido == "ivf_flat" else "<=", config.ivf_auto_threshold, elegido,
        )
        return cls.create(elegido)

    @classmethod
    def list_available(cls) -> List[str]:
        """Nombres de tipos de índice registrados."""
        return sorted(cls._registry)
