"""
Extractor de archivos ``.pbf`` con detección automática de formato.

Los archivos ``.pbf`` pueden ser de dos formatos completamente distintos
(ambos Protobuf, pero incompatibles):

1. **OSM PBF** (OpenStreetMap): capas de elementos ``node/way/relation`` con
   ``tags``. Backends: Pyrosm (primario) y Pyosmium (``osmium.SimpleHandler``,
   respaldo).

2. **Mapbox Vector Tile (MVT)**: tiles ``tiles/{z}/{x}/{y}.pbf`` con capas de
   features geográficas; cada feature tiene ``properties`` (campos tipo
   ``mvt_id``, ``b_ADM2_PCODE``, ...). Backend: ``mapbox-vector-tile``.

La detección lee la cabecera del archivo: OSM PBF contiene la firma
``OSMHeader``/``OSMData`` en su BlobHeader; si no, se intenta decodificar
como MVT.

Cada elemento/feature es su propia unidad estructural (``splittable=False``:
no se parte ni se fusiona). Las repeticiones entre niveles de zoom se
deduplican por (tipo, id).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

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
    """Vuelca los atributos de OSM PBF o los properties de MVT por elemento."""

    supported_formats = ["pbf"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Detecta el formato y extrae (OSM PBF o Mapbox Vector Tile)."""
        if self._es_osm_pbf(filepath):
            doc = self._extract_con_pyrosm(filepath)
            if doc is not None:
                return doc
            return self._extract_con_pyosmium(filepath)

        doc = self._extract_con_mvt(filepath)
        if doc is not None:
            return doc
        raise ExtractorError(
            f"PBF no reconocido como OSM PBF ni Mapbox Vector Tile: {filepath.name}"
        )

    # Detección de formato ------------------------------------------------------

    @staticmethod
    def _es_osm_pbf(filepath: Path) -> bool:
        """True si la cabecera del archivo corresponde a OSM PBF.

        El BlobHeader de OSM PBF contiene el tipo ``OSMHeader``/``OSMData``
        en sus primeros bytes; los MVT no lo tienen.
        """
        try:
            with open(filepath, "rb") as fh:
                cabeza = fh.read(4096)
        except OSError:
            return False
        return b"OSMHeader" in cabeza or b"OSMData" in cabeza

    # Backend MVT (Mapbox Vector Tiles) -----------------------------------------

    def _extract_con_mvt(self, filepath: Path) -> Optional[ExtractedDocument]:
        """Intenta extraer un Mapbox Vector Tile; ``None`` si no es MVT."""
        try:
            import mapbox_vector_tile  # type: ignore
        except ImportError:
            return None

        try:
            data = filepath.read_bytes()
            tile = mapbox_vector_tile.decode(data)
        except Exception as exc:
            logger.warning("mapbox-vector-tile falló para %s: %s", filepath.name, exc)
            return None

        vistos: Set[Tuple[str, int]] = set()
        secciones: List[Section] = []
        capas_nombres: List[str] = []
        for nombre_capa, features in self._iterar_capas_mvt(tile):
            capas_nombres.append(str(nombre_capa))
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                props = feature.get("properties") or {}
                pares = self._pares_tags_dict(props)
                if not pares:
                    continue
                # Deduplicación entre niveles de zoom por (capa, fid).
                fid = props.get("fid", props.get("id"))
                clave = (str(nombre_capa), int(fid) if fid is not None else -1)
                if clave in vistos:
                    continue
                vistos.add(clave)
                secciones.append(
                    Section(texto="\n".join(pares), orden=len(secciones), splittable=False)
                )

        if not secciones:
            return None
        return self._documento(
            filepath,
            secciones,
            metadata={
                "num_elementos": len(secciones),
                "formato_mvt": True,
                "capas": capas_nombres,
            },
        )

    @staticmethod
    def _iterar_capas_mvt(tile) -> List[Tuple[str, List[dict]]]:
        """Normaliza el resultado de ``mapbox_vector_tile.decode``.

        v1.x devuelve ``{"layers": [{"name": ..., "features": [...]}]}``;
        v2.x devuelve ``{nombre_capa: {"extent": ..., "features": [...]}}``.
        """
        capas: List[Tuple[str, List[dict]]] = []
        if isinstance(tile, dict) and "layers" in tile:
            for capa in tile["layers"]:
                capas.append((capa.get("name", ""), capa.get("features", []) or []))
            return capas
        if isinstance(tile, dict):
            for nombre, capa in tile.items():
                if isinstance(capa, dict) and "features" in capa:
                    capas.append((nombre, capa.get("features", []) or []))
        return capas

    # Backend 1: Pyrosm ---------------------------------------------------------

    def _extract_con_pyrosm(self, filepath: Path) -> Optional[ExtractedDocument]:
        """Intenta extraer con Pyrosm; ``None`` si no está disponible o falla."""
        try:
            from pyrosm import Pyrosm  # type: ignore
        except ImportError:
            return None

        try:
            osm = Pyrosm(str(filepath))
            datos = osm.get_data(filter_used_nodes=False, keep_geometry=False)
        except Exception as exc:
            logger.warning("pyrosm falló para %s: %s", filepath.name, exc)
            return None
        if datos is None:
            return None

        vistos: Set[Tuple[str, int]] = set()
        secciones: List[Section] = []
        for tipo, df in (("node", datos[0]), ("way", datos[1]), ("relation", datos[2])):
            if df is None or df.empty:
                continue
            for _, fila in df.iterrows():
                try:
                    eid = int(fila["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                clave = (tipo, eid)
                if clave in vistos:  # deduplicación entre niveles de zoom
                    continue
                vistos.add(clave)
                pares = self._pares_tags_dict(fila.get("tags"))
                if not pares:
                    continue
                secciones.append(
                    Section(texto="\n".join(pares), orden=len(secciones), splittable=False)
                )

        if not secciones:
            return None
        return self._documento(filepath, secciones, metadata={"num_elementos": len(secciones)})

    # Backend 2: Pyosmium -------------------------------------------------------

    def _extract_con_pyosmium(self, filepath: Path) -> ExtractedDocument:
        """Extrae con Pyosmium (osmium.SimpleHandler); lanza si no hay backends."""
        try:
            import osmium  # type: ignore
        except ImportError as exc:
            raise ExtractorError(
                f"PBF ({filepath.name}): instala pyrosm o osmium (pyosmium) "
                f"— pip install pyrosm"
            ) from exc

        vistos: Set[Tuple[str, int]] = set()
        secciones: List[Section] = []

        def _volar_elemento(tipo: str, elemento) -> None:
            """Vuelca un elemento (node/way/relation) como sección atómica."""
            clave = (tipo, int(elemento.id))
            if clave in vistos:  # deduplicación entre niveles de zoom
                return
            vistos.add(clave)
            pares = self._pares_tags(elemento)
            if not pares:
                return
            secciones.append(
                Section(texto="\n".join(pares), orden=len(secciones), splittable=False)
            )

        try:
            # osmium 3.x: SimpleHandler() + apply_file
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
                raise ExtractorError(
                    f"PBF ilegible ({filepath.name}): pyrosm y pyosmium fallaron. "
                    f"Pyosmium: {exc}"
                ) from exc
        except Exception as exc:
            raise ExtractorError(
                f"PBF ilegible ({filepath.name}): pyrosm y pyosmium fallaron. "
                f"Pyosmium: {exc}"
            ) from exc

        if not secciones:
            raise ExtractorError(f"El PBF no contiene elementos con atributos: {filepath.name}")
        return self._documento(filepath, secciones, metadata={"num_elementos": len(secciones)})

    # Utilidades ---------------------------------------------------------------

    @staticmethod
    def _documento(filepath: Path, secciones: List[Section], metadata: dict) -> ExtractedDocument:
        """Construye el documento PBF con su metadata."""
        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=Formato.PBF,
            fenomeno=1,
            secciones=secciones,
            metadata=metadata,
        )

    @staticmethod
    def _pares_tags_dict(tags) -> List[str]:
        """Convierte un dict de tags (Pyrosm) en pares ``atributo: valor``."""
        if not isinstance(tags, dict):
            return []
        pares: List[str] = []
        for clave, valor in tags.items():
            clave = str(clave)
            valor = str(valor)
            if clave in _ATRIBUTOS_IGNORADOS or not valor:
                continue
            pares.append(f"{clave}: {valor}")
        return pares

    @staticmethod
    def _pares_tags(elemento) -> List[str]:
        """Convierte los tags de un elemento osmium en pares ``atributo: valor``."""
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
