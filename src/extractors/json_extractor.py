"""
Extractor de JSON de artículos/registros.

Cada registro/artículo del array raíz es su propia unidad estructural:
``title`` + ``body_paragraphs`` se concatenan respetando el orden, y los
campos descriptivos (author, date, tags...) se guardan como metadata del
documento o de la sección.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section

logger = logging.getLogger(__name__)

# Campos que suelen contener el cuerpo principal del artículo.
_CAMPOS_TITULO = ("title", "titulo", "headline", "name", "nombre")
_CAMPOS_CUERPO = ("body", "cuerpo", "body_paragraphs", "content", "text", "texto", "description", "descripcion")
# Campos descriptivos que no forman parte del cuerpo indexable.
_CAMPOS_METADATA = ("author", "autor", "date", "fecha", "published", "publicado",
                    "tags", "categoria", "category", "url", "link", "id", "source", "fuente")


@register_extractor(".json")
class JSONExtractor(BaseExtractor):
    """Extrae texto de archivos JSON (lista de artículos/registros)."""

    supported_formats = ["json"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el documento JSON en secciones (una por registro)."""
        texto = self._decodificar(self._leer_bytes(filepath))
        try:
            datos = json.loads(texto)
        except json.JSONDecodeError as exc:
            raise ExtractorError(f"JSON inválido ({filepath.name}): {exc}") from exc

        registros = self._normalizar_lista(datos)
        if not registros:
            raise ExtractorError(f"JSON sin registros útiles: {filepath.name}")

        secciones: List[Section] = []
        metadata_doc: Dict[str, Any] = {}
        titulo_doc: Optional[str] = None
        fecha_doc: Optional[str] = None

        for indice, registro in enumerate(registros):
            if not isinstance(registro, dict):
                continue
            titulo = self._primer_campo(registro, _CAMPOS_TITULO)
            cuerpo = self._cuerpo_registro(registro)
            if not cuerpo:
                continue
            texto_seccion = f"{titulo}.\n{cuerpo}" if titulo else cuerpo
            secciones.append(
                Section(titulo=titulo, texto=texto_seccion, orden=len(secciones), splittable=True)
            )
            if titulo_doc is None and titulo:
                titulo_doc = titulo
            for clave, valor in registro.items():
                if clave in _CAMPOS_METADATA and isinstance(valor, (str, int, float)):
                    metadata_doc.setdefault(clave, valor)
            if fecha_doc is None:
                fecha = self._primer_campo(registro, ("date", "fecha", "published", "publicado"))
                if fecha:
                    fecha_doc = str(fecha)

        if not secciones:
            raise ExtractorError(f"JSON sin contenido textual: {filepath.name}")

        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.JSON,
            fenomeno=1,
            secciones=secciones,
            titulo_documento=titulo_doc,
            fecha_publicacion=fecha_doc,
            metadata=metadata_doc,
        )

    # Utilidades --------------------------------------------------------------

    @staticmethod
    def _normalizar_lista(datos: Any) -> List[Any]:
        """Acepta un array, un objeto con lista, o un único registro."""
        if isinstance(datos, list):
            return datos
        if isinstance(datos, dict):
            for valor in datos.values():
                if isinstance(valor, list):
                    return valor
            return [datos]
        return []

    @staticmethod
    def _primer_campo(registro: Dict[str, Any], candidatos: tuple) -> Optional[str]:
        """Devuelve el primer campo presente y no vacío entre los candidatos."""
        for clave in candidatos:
            valor = registro.get(clave)
            if valor is None:
                continue
            if isinstance(valor, str) and valor.strip():
                return valor.strip()
            if isinstance(valor, (int, float)):
                return str(valor)
        return None

    @classmethod
    def _cuerpo_registro(cls, registro: Dict[str, Any]) -> str:
        """Concatena title + body_paragraphs respetando el orden."""
        partes: List[str] = []
        for clave in _CAMPOS_CUERPO:
            valor = registro.get(clave)
            if valor is None:
                continue
            if isinstance(valor, list):
                for item in valor:
                    if isinstance(item, str) and item.strip():
                        partes.append(item.strip())
            elif isinstance(valor, str) and valor.strip():
                partes.append(valor.strip())
        if partes:
            return re.sub(r"\n{3,}", "\n\n", "\n".join(partes))
        # Si no hay campos de cuerpo conocidos, indexar pares clave: valor
        # de campos no descriptivos.
        for clave, valor in registro.items():
            if clave in _CAMPOS_METADATA or clave in _CAMPOS_TITULO:
                continue
            if isinstance(valor, str) and valor.strip():
                partes.append(f"{clave}: {valor.strip()}")
        return "\n".join(partes)
