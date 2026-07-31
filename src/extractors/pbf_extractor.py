"""
Extractor de archivos PBF (OpenStreetMap) con osmium.

Cada elemento de capa (par ``atributo: valor``) es su propia unidad
estructural. Las repeticiones entre niveles de zoom se deduplican por
(tipo, id) de elemento: los atributos de un elemento ya visto no se
vuelcan dos veces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section

logger = logging.getLogger(__name__)

# Atributos que no aportan al texto indexable.
_ATRIBUTOS_IGNORADOS = {
    "created_by", "source", "attribution", "comment", "changeset",
    "timestamp", "version", "uid", "user", "odbl", "note",
}


@register_extractor(".pbf", ".osm.pbf")
class PBFExtractor(BaseExtractor):
    """Vuelca los atributos de capas OSM como ``atributo: valor`` por elemento."""

    supported_formats = ["pbf"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae los elementos del PBF deduplicando por (tipo, id)."""
        try:
            import osmium  # type: ignore
        except ImportError as exc:
            raise ExtractorError(
                f"osmium requerido para PBF ({filepath.name}): {exc}"
            ) from exc

        vistos: Set[Tuple[str, int]] = set()
        secciones: List[Section] = []

        def _volar_elemento(tipo: str, elemento) -> None:
            """Vuelca un elemento (node/way/relation) como sección atómica."""
            clave = (tipo, int(elemento.id))
            if clave in vistos:  # deduplicación entre niveles de zoom
                return
            vistos.add(clave)
            pares = self._pares_atributos(elemento)
            if not pares:
                return
            texto = "\n".join(pares)
            secciones.append(Section(texto=texto, orden=len(secciones), splittable=False))

        try:
            # osmium 3.x: osmium.SimpleHandler
            class _Handler(osmium.SimpleHandler):
                def node(self, n):
                    _volar_elemento("node", n)

                def way(self, w):
                    _volar_elemento("way", w)

                def relation(self, r):
                    _volar_elemento("relation", r)

            handler = _Handler()
            handler.apply_file(str(filepath))
        except AttributeError:
            # osmium 4.x: handler por lote (pasar el archivo al constructor)
            class _HandlerV4(osmium.SimpleHandler):
                def __init__(self, archivo):
                    super().__init__(archivo)
                    self.apply()

                def node(self, n):
                    _volar_elemento("node", n)

                def way(self, w):
                    _volar_elemento("way", w)

                def relation(self, r):
                    _volar_elemento("relation", r)

            try:
                _HandlerV4(str(filepath))
            except Exception as exc:
                raise ExtractorError(f"osmium no pudo leer el PBF ({filepath.name}): {exc}") from exc
        except Exception as exc:
            raise ExtractorError(f"osmium falló ({filepath.name}): {exc}") from exc

        if not secciones:
            raise ExtractorError(f"El PBF no contiene elementos con atributos: {filepath.name}")

        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.PBF,
            fenomeno=1,
            secciones=secciones,
            metadata={"num_elementos": len(secciones)},
        )

    @staticmethod
    def _pares_atributos(elemento) -> List[str]:
        """Convierte los tags del elemento en pares ``atributo: valor``."""
        pares: List[str] = []
        try:
            tags = elemento.tags
        except Exception:
            return pares
        for etiqueta in tags:
            clave = str(etiqueta.k)
            valor = str(etiqueta.v)
            if clave in _ATRIBUTOS_IGNORADOS or not valor:
                continue
            pares.append(f"{clave}: {valor}")
        return pares
