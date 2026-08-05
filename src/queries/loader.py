"""
Carga de las 50 consultas de evaluación (Sección 9 del reto CODEFEST AD
ASTRA 2026).

``QueryLoader`` centraliza en un solo lugar:
- La lectura del archivo de consultas (``.jsonl`` o ``.csv`` con columnas
  ``query_id``/``query_text``) con validación estricta: EXACTAMENTE 50
  consultas, IDs ``q001``..``q050`` sin huecos ni duplicados.
- La exportación desde el PDF oficial a ``consultas.jsonl`` reutilizando el
  extractor del pipeline (``PDFExtractor`` vía ``QueryExtractor``).

Así, ni ``generador.py`` (orquestador) ni ``run_export_queries.py`` (CLI)
repiten la lógica de lectura/escritura de consultas.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.queries.extractor import QueryExtractor
from src.queries.models import QUERY_ID_PATTERN, Query

logger = logging.getLogger(__name__)

N_QUERIES = 50


class QueryLoader:
    """Carga las consultas de evaluación desde archivo o desde el PDF oficial."""

    # -- Validación de IDs ----------------------------------------------------

    @staticmethod
    def _query_ids_esperados(n: int = N_QUERIES) -> List[str]:
        """``["q001", "q002", ..., "qNNN"]`` en orden estricto."""
        return [f"q{i:03d}" for i in range(1, n + 1)]

    @classmethod
    def _validar_ids(cls, por_id: Dict[str, str], origen: str) -> List[Query]:
        """Valida 50 consultas con IDs q001..q050 y devuelve ``List[Query]``.

        Raises:
            ValueError: si el conteo o los IDs no cumplen la regla.
        """
        esperados = cls._query_ids_esperados()
        if len(por_id) != N_QUERIES:
            raise ValueError(
                f"Se esperaban exactamente {N_QUERIES} consultas, se encontraron "
                f"{len(por_id)} en '{origen}'"
            )
        faltantes = [qid for qid in esperados if qid not in por_id]
        if faltantes:
            raise ValueError(
                f"Faltan IDs de consulta (sin huecos permitidos) en '{origen}': {faltantes}"
            )
        return [Query(query_id=qid, query_text=por_id[qid]) for qid in esperados]

    # -- Lectura del archivo ---------------------------------------------------

    @staticmethod
    def _leer_jsonl(path: Path) -> List[Dict[str, Any]]:
        registros: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for numero_linea, linea in enumerate(f, start=1):
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"'{path}' línea {numero_linea}: JSON inválido ({exc})"
                    ) from exc
        return registros

    @staticmethod
    def _leer_csv(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @classmethod
    def cargar(cls, path: str) -> List[Query]:
        """Lee el archivo de 50 consultas (``.jsonl`` o ``.csv``).

        Acepta claves ``query_id``/``id`` y ``query_text``/``text``/``query``.
        El orden de salida es siempre q001..q050, independientemente del
        orden del archivo.

        Raises:
            FileNotFoundError: si ``path`` no existe.
            ValueError: si el conteo de consultas o los IDs no cumplen la regla.
        """
        ruta = Path(path)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de consultas: {ruta}")

        crudos = cls._leer_csv(ruta) if ruta.suffix.lower() == ".csv" else cls._leer_jsonl(ruta)

        por_id: Dict[str, str] = {}
        for i, registro in enumerate(crudos, start=1):
            query_id = str(registro.get("query_id") or registro.get("id") or "").strip()
            texto = str(
                registro.get("query_text") or registro.get("text") or registro.get("query") or ""
            ).strip()
            if not query_id or not texto:
                raise ValueError(
                    f"Registro #{i} de '{ruta}' incompleto: "
                    f"query_id='{query_id}', query_text='{texto}'"
                )
            if not QUERY_ID_PATTERN.match(query_id):
                raise ValueError(f"query_id inválido '{query_id}' (debe seguir el patrón qNNN)")
            if query_id in por_id:
                raise ValueError(f"query_id duplicado: '{query_id}'")
            por_id[query_id] = texto

        return cls._validar_ids(por_id, str(ruta))

    # -- Exportación desde el PDF oficial ---------------------------------------

    @staticmethod
    def exportar_desde_pdf(pdf_path: str, output_path: str) -> List[Query]:
        """Extrae las consultas del PDF oficial y las escribe en ``output_path``.

        Reutiliza ``PDFExtractor`` vía ``QueryExtractor`` (mismo extractor del
        pipeline de ingesta) y escribe el ``.jsonl`` con el formato que
        consume :meth:`cargar`. Devuelve las consultas extraídas.
        """
        consultas = QueryExtractor().extraer_desde_pdf(pdf_path)
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for consulta in consultas:
                f.write(
                    json.dumps(
                        {"query_id": consulta.query_id, "query_text": consulta.query_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("Generadas %d consultas desde '%s' -> '%s'", len(consultas), pdf_path, output_path)
        return consultas
