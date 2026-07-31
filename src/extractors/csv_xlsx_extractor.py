"""
Extractores CSV y XLSX basados en pandas.

Cada fila es una unidad estructural independiente: se arma como pares
``columna: valor`` y se marca ``splittable=False`` (no se subdivide ni se
fusiona en la fase 2 del chunking).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import register_extractor
from src.models.extracted_document import ExtractedDocument, Formato, Section

logger = logging.getLogger(__name__)


def _fila_a_texto(columnas: List[str], valores: List[Any]) -> str:
    """Arma pares ``columna: valor`` para una fila, omitiendo nulos."""
    partes = []
    for col, valor in zip(columnas, valores):
        if valor is None or (isinstance(valor, float) and valor != valor):  # NaN
            continue
        texto_valor = str(valor).strip()
        if not texto_valor:
            continue
        partes.append(f"{col}: {texto_valor}")
    return "\n".join(partes)


def _normalizar_columnas(columnas: List[Any]) -> List[str]:
    """Convierte encabezados a strings no vacíos."""
    normalizadas = []
    for i, col in enumerate(columnas):
        nombre = str(col).strip() if col is not None and str(col).strip() else f"col_{i}"
        normalizadas.append(nombre)
    return normalizadas


class _BaseTabularExtractor(BaseExtractor):
    """Lógica común de extracción tabular (CSV/XLSX)."""

    def _construir_documento(
        self, filepath: Path, columnas: List[str], filas: List[List[Any]], formato: Formato
    ) -> ExtractedDocument:
        """Construye el documento con una sección atómica por fila."""
        secciones: List[Section] = []
        for fila in filas:
            texto = _fila_a_texto(columnas, fila)
            if not texto:
                continue
            secciones.append(Section(texto=texto, orden=len(secciones), splittable=False))
        if not secciones:
            raise ExtractorError(f"El archivo tabular no tiene filas con datos: {filepath.name}")
        return ExtractedDocument(
            doc_id="",
            fuente=filepath.name,
            formato=formato,
            fenomeno=1,
            secciones=secciones,
            metadata={"num_filas": len(secciones)},
        )


@register_extractor(".csv")
class CSVExtractor(_BaseTabularExtractor):
    """Extrae cada fila de un CSV como unidad estructural independiente."""

    supported_formats = ["csv"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae el CSV usando pandas (o csv estándar como respaldo)."""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            return self._extract_csv_std(filepath)

        df = self._leer_csv(pd, filepath)
        columnas = _normalizar_columnas(df.columns.tolist())
        filas = df.values.tolist()
        return self._construir_documento(filepath, columnas, filas, Formato.CSV)

    @staticmethod
    def _leer_csv(pd, filepath: Path):
        """Lee el CSV tolerando filas malformadas y codificaciones raras."""
        try:
            return pd.read_csv(
                filepath,
                encoding="utf-8",
                encoding_errors="replace",
                keep_default_na=False,
                on_bad_lines="skip",
            )
        except Exception:
            # Reintento con el motor de Python: tolerante a filas irregulares.
            try:
                return pd.read_csv(
                    filepath,
                    engine="python",
                    encoding="utf-8",
                    encoding_errors="replace",
                    keep_default_na=False,
                    on_bad_lines="skip",
                    sep=None,
                )
            except Exception as exc:
                raise ExtractorError(f"CSV ilegible ({filepath.name}): {exc}") from exc

    def _extract_csv_std(self, filepath: Path) -> ExtractedDocument:
        """Respaldo con el módulo csv cuando no hay pandas."""
        import csv as csv_mod

        logger.warning("pandas no instalado; usando módulo csv para %s", filepath.name)
        with open(filepath, newline="", encoding="utf-8", errors="replace") as fh:
            lector = csv.reader(fh)
            try:
                columnas = _normalizar_columnas(next(lector))
            except StopIteration:
                raise ExtractorError(f"CSV vacío: {filepath.name}") from None
            filas = [list(fila) for fila in lector]
        return self._construir_documento(filepath, columnas, filas, Formato.CSV)


@register_extractor(".xlsx", ".xls")
class XLSXExtractor(_BaseTabularExtractor):
    """Extrae cada fila de un XLSX como unidad estructural independiente."""

    supported_formats = ["xlsx", "xls"]

    def extract(self, filepath: Path) -> ExtractedDocument:
        """Extrae la primera hoja del libro (o la de mayor contenido)."""
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ExtractorError(f"pandas requerido para XLSX ({filepath.name}): {exc}") from exc

        try:
            hojas = pd.read_excel(filepath, sheet_name=None, keep_default_na=False)
        except Exception as exc:
            raise ExtractorError(f"XLSX ilegible ({filepath.name}): {exc}") from exc

        hoja = max(hojas, key=lambda nombre: len(hojas[nombre])) if hojas else None
        if hoja is None:
            raise ExtractorError(f"XLSX sin hojas: {filepath.name}")
        df = hojas[hoja]
        columnas = _normalizar_columnas(df.columns.tolist())
        filas = df.values.tolist()
        return self._construir_documento(filepath, columnas, filas, Formato.XLSX)
