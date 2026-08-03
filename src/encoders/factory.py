"""
Factory de estrategias de codificación semántica con registro por decorador.

Cumple abierto/cerrado: agregar un encoder nuevo solo requiere crear el
archivo de la estrategia y decorar la clase con ``@EncoderFactory.register``,
sin tocar esta clase.

Las reglas de negocio (Sección 6: idiomas mínimos en registro, licencia en
instanciación) están centralizadas como Information Expert en
:class:`EncoderStrategy` (``cubre_idiomas_minimos()`` / ``licencia_permitida()``):
cada estrategia es quien conoce su propia metadata y decide si la cumple.
Este Factory solo orquesta la decisión, no la evalúa.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from src.encoders.base import (
    IDIOMAS_MINIMOS,
    LICENSE_WHITELIST,
    EncoderConfig,
    EncoderStrategy,
    LicenseNotAllowedError,
)

logger = logging.getLogger(__name__)


class EncoderFactory:
    """Crea instancias de :class:`EncoderStrategy` por nombre corto registrado."""

    _registry: Dict[str, Type[EncoderStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorador: ``@EncoderFactory.register("bert-multilingual")`` sobre la clase concreta.

        Raises:
            ValueError: si ``supported_languages`` no cubre es/en/pt y la
                clase no está marcada como ``is_complementary=True``.
        """

        def _decorador(clase: Type[EncoderStrategy]) -> Type[EncoderStrategy]:
            clave = name.strip().lower()
            if not clase.cubre_idiomas_minimos():
                raise ValueError(
                    f"No se puede registrar '{clave}': supported_languages="
                    f"{sorted(clase.supported_languages)} no cubre {sorted(IDIOMAS_MINIMOS)}. "
                    "Márquela con is_complementary=True si es un encoder de dominio/idioma específico."
                )
            if clave in cls._registry:
                logger.warning("Sobrescribiendo encoder ya registrado: '%s'", clave)
            cls._registry[clave] = clase
            clase._registry_name = clave  # type: ignore[attr-defined]
            return clase

        return _decorador

    @classmethod
    def create(cls, name: str, config: Optional[EncoderConfig] = None) -> EncoderStrategy:
        """Instancia la estrategia ``name``, validando la licencia primero.

        Raises:
            ValueError: si ``name`` no está registrado.
            LicenseNotAllowedError: si la licencia no está en la lista blanca
                y ``config.allow_unlisted_license`` es ``False``.
        """
        clave = name.strip().lower()
        clase = cls._registry.get(clave)
        if clase is None:
            raise ValueError(
                f"Encoder desconocido: '{name}'. Disponibles: {', '.join(sorted(cls._registry))}"
            )
        config = config or EncoderConfig()
        if not clase.licencia_permitida(allow_unlisted=config.allow_unlisted_license):
            raise LicenseNotAllowedError(
                f"Encoder '{name}' tiene licencia '{clase.license}' fuera de la lista blanca "
                f"({', '.join(sorted(LICENSE_WHITELIST))}). Instáncielo con "
                "allow_unlisted_license=True si es una decisión consciente."
            )
        if config.allow_unlisted_license and clase.license.lower() not in LICENSE_WHITELIST:
            logger.warning(
                "Licencia '%s' de '%s' fuera de la lista blanca; permitida por "
                "allow_unlisted_license=True (decisión consciente registrada).",
                clase.license, name,
            )
        return clase(config)

    @classmethod
    def list_available(cls) -> List[dict]:
        """Metadata de criterios de cada encoder registrado, sin instanciarlos."""
        return [
            {
                "name": nombre,
                "model_id": getattr(clase, "model_id", None),
                "embedding_dim": getattr(clase, "embedding_dim", None),
                "max_input_tokens": getattr(clase, "max_input_tokens", None),
                "supported_languages": getattr(clase, "supported_languages", []),
                "license": getattr(clase, "license", None),
                "mteb_retrieval_score": getattr(clase, "mteb_retrieval_score", None),
                "benchmark_reference": getattr(clase, "benchmark_reference", None),
                "is_complementary": getattr(clase, "is_complementary", False),
            }
            for nombre, clase in sorted(cls._registry.items())
        ]
