"""
Servicio de acceso al corpus: escaneo de archivos y asignación de fenómeno.

Clase "Pure Fabrication" (GRASP): aísla la lógica de recorrido del sistema de
archivos y del mapeo carpeta/patrón -> fenómeno, para que el controlador
(``run_ingestion``) y el orquestador (``BatchIngestor``) solo coordinen.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.models.extracted_document import ExtractedDocument

logger = logging.getLogger(__name__)

# Extensiones leídas como texto plano para el pre-análisis de boilerplate.
_EXTENSIONES_TEXTO = {"md", "markdown", "txt", "html", "htm", "json", "csv"}


class CorpusService:
    """Descubre archivos del corpus y asigna el fenómeno (1, 2 o 3)."""

    def __init__(
        self, corpus: Path, fenomenos_map: Optional[Dict[str, int]] = None
    ) -> None:
        self.corpus = Path(corpus)
        self.fenomenos_map = fenomenos_map or {}

    # Mapeo de fenómenos -----------------------------------------------------

    @staticmethod
    def load_fenomenos_map(ruta: Optional[Path]) -> Dict[str, int]:
        """Carga el mapeo carpeta/patrón -> fenómeno desde un JSON.

        Args:
            ruta: Ruta al JSON (``{"fenomeno_1": 1, "sismo": 1, ...}``).

        Returns:
            Diccionario con claves en minúsculas; vacío si no hay archivo.
        """
        if ruta is None:
            return {}
        try:
            datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        except Exception:
            logger.exception("No se pudo leer el mapeo de fenómenos %s", ruta)
            return {}
        mapeo: Dict[str, int] = {}
        for clave, valor in datos.items():
            try:
                mapeo[str(clave).lower()] = int(valor)
            except (TypeError, ValueError):
                logger.warning("Fenómeno inválido para '%s': %r (se ignora)", clave, valor)
        return mapeo

    def determine_fenomeno(self, filepath: Path, por_defecto: int = 1) -> int:
        """Fenómeno por carpeta (incluidas las ancestrales), por patrón en el
        nombre del archivo, o el valor por defecto.

        Como el corpus puede tener carpetas anidadas dentro de carpetas, se
        recorren los directorios antecesores del archivo de la hoja hacia la
        raíz; gana la carpeta más específica (la más profunda) que aparezca
        en el mapeo.
        """
        for carpeta in Path(filepath).parents:  # parents[0] = carpeta inmediata
            if carpeta.name.lower() in self.fenomenos_map:
                return self.fenomenos_map[carpeta.name.lower()]
        nombre = Path(filepath).stem.lower()
        for patron, fenomeno in self.fenomenos_map.items():
            if patron in nombre:
                return fenomeno
        return por_defecto

    # Escaneo de archivos -------------------------------------------------------

    def scan(self, extensiones: Optional[List[str]] = None) -> List[Path]:
        """Archivos del corpus (recursivo, orden estable) con filtro opcional.

        Args:
            extensiones: Solo estas extensiones (sin punto), p. ej. ``["pdf"]``.

        Returns:
            Lista ordenada de rutas; vacía si el corpus no existe.
        """
        if not self.corpus.exists() or not self.corpus.is_dir():
            logger.error("El directorio del corpus no existe: %s", self.corpus)
            return []
        archivos = [p for p in self.corpus.rglob("*") if p.is_file()]
        if extensiones:
            archivos = [
                p for p in archivos if p.suffix.lower().lstrip(".") in extensiones
            ]
        return sorted(archivos)

    def read_plain_text_docs(self, limite: int = 500) -> List[ExtractedDocument]:
        """Stubs de documentos de texto plano para el análisis de boilerplate.

        Solo lee archivos de texto (md, txt, html, json, csv) por rapidez; los
        binarios (pdf, xlsx, imágenes, pbf) se omiten en esta pasada.
        """
        from src.models.extracted_document import Section

        stubs: List[ExtractedDocument] = []
        for p in sorted(self.corpus.rglob("*"))[:limite]:
            if not p.is_file() or p.suffix.lower().lstrip(".") not in _EXTENSIONES_TEXTO:
                continue
            try:
                texto = p.read_text(encoding="utf-8", errors="replace")
                stubs.append(
                    ExtractedDocument(
                        doc_id=p.stem,
                        fuente=p.name,
                        formato="txt",
                        fenomeno=1,
                        secciones=[Section(texto=texto, orden=0, splittable=True)],
                    )
                )
            except Exception:
                continue
        return stubs
