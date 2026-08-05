"""
Extracción de las 50 consultas de evaluación (q001..q050) desde el PDF oficial.

Reutiliza el extractor del pipeline (``ExtractorFactory`` -> ``PDFExtractor``,
Sección 3 del reto) para leer el documento, y una función pura que detecta
los marcadores ``qNNN`` y captura el texto de cada pregunta hasta el
siguiente marcador. El enfoque por posición es robusto a saltos de línea y a
preguntas que continúan en otra página (p. ej. q043 en el PDF real).

Ningún modelo generativo interviene: el texto de cada consulta es el texto
original del PDF, solo normalizado en espacio en blanco.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from src.extractors.factory import ExtractorFactory
from src.queries.models import Query

logger = logging.getLogger(__name__)

N_QUERIES = 50

# Marcador de inicio de pregunta: "q" + 3 dígitos + espacio.
_MARKER = re.compile(r"q(\d{3})\s+")


class QueryExtractor:
    """Extrae las 50 consultas q001..q050 desde el PDF de evaluación."""

    def __init__(self, extractor_factory: Optional[ExtractorFactory] = None) -> None:
        self._factory = extractor_factory or ExtractorFactory()

    def extraer_desde_pdf(self, pdf_path: str) -> List[Query]:
        """Extrae las consultas del PDF oficial reutilizando el pipeline.

        Args:
            pdf_path: ruta al PDF de consultas (p. ej. ``Extracto_Preguntas_50_v2.pdf``).

        Raises:
            FileNotFoundError: si el PDF no existe.
            ValueError: si el número de consultas o los IDs no cumplen la regla.
        """
        ruta = Path(pdf_path)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el PDF de consultas: {ruta}")
        logger.info("Extrayendo consultas desde '%s' con %s", ruta, type(self._factory.create(ruta)).__name__)
        documento = self._factory.create(ruta).extract(ruta)
        return self.extraer_consultas(documento.texto_completo)

    @staticmethod
    def extraer_consultas(texto: str) -> List[Query]:
        """Parsea el texto del PDF en las 50 consultas (q001..q050).

        Cada pregunta empieza con el marcador ``qNNN`` seguido de un espacio;
        su texto es todo lo que hay hasta el siguiente marcador. Se colapsa el
        espacio en blanco (las preguntas pueden continuar en otra página).

        Raises:
            ValueError: si no hay marcadores, si hay IDs duplicados, si no son
                exactamente 50 o si faltan IDs (huecos en q001..q050).
        """
        coincidencias = list(_MARKER.finditer(texto))
        if not coincidencias:
            raise ValueError("No se encontró ningún marcador qNNN en el texto")

        por_id: dict = {}
        for i, match in enumerate(coincidencias):
            query_id = f"q{match.group(1)}"
            inicio = match.end()
            fin = coincidencias[i + 1].start() if i + 1 < len(coincidencias) else len(texto)
            texto_consulta = re.sub(r"\s+", " ", texto[inicio:fin]).strip()
            if query_id in por_id:
                raise ValueError(f"query_id duplicado en el PDF: '{query_id}'")
            por_id[query_id] = texto_consulta

        esperados = [f"q{i:03d}" for i in range(1, N_QUERIES + 1)]
        if len(por_id) != N_QUERIES:
            raise ValueError(
                f"Se esperaban exactamente {N_QUERIES} consultas en el PDF, "
                f"se encontraron {len(por_id)}"
            )
        faltantes = [qid for qid in esperados if qid not in por_id]
        if faltantes:
            raise ValueError(f"Faltan IDs de consulta en el PDF (sin huecos): {faltantes}")

        return [Query(query_id=qid, query_text=por_id[qid]) for qid in esperados]
