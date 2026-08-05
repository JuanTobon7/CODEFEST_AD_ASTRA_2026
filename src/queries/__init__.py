"""
Módulo de consultas de evaluación (Sección 9 del reto CODEFEST AD ASTRA 2026).

Agrupa la lógica de dominio de la etapa de evaluación:

- ``Query``: modelo de una de las 50 preguntas (q001..q050).
- ``QueryExtractor``: extrae las consultas del PDF oficial reutilizando el
  extractor del pipeline (``PDFExtractor`` vía ``ExtractorFactory``).
- ``QueryLoader``: carga las 50 consultas desde ``consultas.jsonl``/``.csv``
  (validando cantidad e IDs) y exporta desde el PDF a ``consultas.jsonl``.
- ``build_result_object``: construye y valida el objeto de resultado por
  consulta según el esquema de la Sección 9.3.1.

``generador.py`` y ``run_export_queries.py`` son controladores delgados que
delegan en este módulo (GRASP: Información Experta en el módulo de dominio).
"""

from src.queries.extractor import QueryExtractor
from src.queries.loader import QueryLoader
from src.queries.models import QUERY_ID_PATTERN, Query
from src.queries.resultado import build_result_object

__all__ = [
    "QUERY_ID_PATTERN",
    "Query",
    "QueryExtractor",
    "QueryLoader",
    "build_result_object",
]
