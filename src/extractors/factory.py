"""
Factory de extractores con registro extensible por decorador.

Nuevos formatos se agregan decorando la clase: ``@register_extractor(".epub")``.
No hay que tocar el código de la fábrica (principio abierto/cerrado).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Type

from src.extractors.base import BaseExtractor, ExtractorError
from src.support.utils import extension_de

logger = logging.getLogger(__name__)

# Registro global extensión -> clase extractora.
_REGISTRY: Dict[str, Type[BaseExtractor]] = {}


def register_extractor(*extensiones: str):
    """Decorador que registra una clase extractora para las extensiones dadas.

    Ejemplo::

        @register_extractor(".pdf")
        class PDFExtractor(BaseExtractor): ...
    """

    def decorador(cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
        for ext in extensiones:
            ext_norm = ext.lower().lstrip(".")
            if ext_norm in _REGISTRY:
                logger.warning("Extensión '%s' ya registrada por %s; se reemplaza por %s",
                               ext_norm, _REGISTRY[ext_norm].__name__, cls.__name__)
            _REGISTRY[ext_norm] = cls
        # Refleja las extensiones en el atributo de clase (para documentación).
        soportadas = {*getattr(cls, "supported_formats", []), *[e.lower().lstrip(".") for e in extensiones]}
        cls.supported_formats = sorted(soportadas)
        return cls

    return decorador


class ExtractorFactory:
    """Crea el extractor adecuado según la extensión del archivo."""

    def create(self, filepath: Path) -> BaseExtractor:
        """Devuelve la instancia de extractor para ``filepath``.

        Lanza :class:`ExtractorError` si la extensión no está registrada.
        """
        ext = extension_de(filepath)
        clase = _REGISTRY.get(ext)
        if clase is None:
            soportadas = ", ".join(sorted(_REGISTRY)) or "ninguna"
            raise ExtractorError(
                f"Formato '.{ext}' no soportado para {filepath.name}. "
                f"Extensiones registradas: {soportadas}"
            )
        logger.debug("Extractor seleccionado para '.%s': %s", ext, clase.__name__)
        return clase()

    @staticmethod
    def extensiones_registradas() -> List[str]:
        """Extensiones soportadas actualmente (para logging/CLI)."""
        return sorted(_REGISTRY)
