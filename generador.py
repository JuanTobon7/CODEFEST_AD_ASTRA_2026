"""
Script final de entrega (Sección 9) — CODEFEST AD ASTRA 2026.

Orquestador delgado (GRASP: Controller): parsea argumentos, asegura
``consultas.jsonl`` (generándolo desde el PDF oficial si falta) y delega en
``src.queries`` (QueryLoader, build_result_object, verificar_resultados) y
``src.retrieval.retrieve`` (RRF multi-encoder + filtros + agregación).

Produce ``resultados.jsonl`` con el esquema exacto de la Sección 9.3.1:
50 líneas, una por consulta, en orden q001..q050. Sin modelos generativos:
los fragmentos son el texto original de la metadata, solo recortado/
concatenado por límite de palabras en split_or_merge_fragments. Determinista.

Uso:
    python generador.py
    python generador.py --queries consultas.jsonl --queries-pdf otro.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.queries import QueryLoader, build_result_object, verificar_resultados
from src.retrieval.retriever import retrieve

logger = logging.getLogger(__name__)

QUERIES_DEFAULT = "consultas.jsonl"
QUERIES_PDF_DEFAULT = "Extracto_Preguntas_50_v2.pdf"


def generate_results(
    queries_path: str,
    output_path: str = "resultados.jsonl",
    **retrieve_kwargs: Any,
) -> None:
    """Ejecuta consultas -> retrieve() -> resultados.jsonl (solo líneas válidas).

    Una consulta que falle (retrieve o validación) se loguea y no detiene el
    resto; al final se reporta si el total escrito es < 50.
    """
    queries = QueryLoader.cargar(queries_path)
    logger.info("Cargadas %d consultas desde '%s'", len(queries), queries_path)

    lineas: List[str] = []
    fallidas: List[str] = []
    for i, consulta in enumerate(queries, start=1):
        query_id, texto = consulta.query_id, consulta.query_text
        logger.info("[%d/%d] Procesando %s: %r", i, len(queries), query_id, texto[:80])
        try:
            resultado = build_result_object(query_id, retrieve(texto, **retrieve_kwargs))
        except Exception:  # noqa: BLE001 - una consulta no debe tumbar el resto
            logger.exception("%s FALLÓ", query_id)
            fallidas.append(query_id)
            continue

        lineas.append(json.dumps(resultado, ensure_ascii=False))
        logger.info("%s OK", query_id)

    ruta_salida = Path(output_path)
    with open(ruta_salida, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lineas))
        if lineas:
            f.write("\n")

    logger.info("Escritas %d líneas en '%s'", len(lineas), ruta_salida)
    verificar_resultados(ruta_salida, len(lineas), fallidas)


def _parse_date_range(valor: str) -> Optional[List[Optional[str]]]:
    """``"2020-01-01,2023-12-31"`` -> ``["2020-01-01", "2023-12-31"]``."""
    partes = [p.strip() or None for p in valor.split(",")]
    if len(partes) != 2:
        raise argparse.ArgumentTypeError(
            "--date-range debe tener el formato 'inicio,fin' (extremos opcionales)"
        )
    return partes


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera resultados.jsonl para las 50 consultas de evaluación (Sección 9)."
    )
    parser.add_argument("--queries", default=QUERIES_DEFAULT, help="Archivo de consultas (.jsonl o .csv); si no existe, se genera desde el PDF")
    parser.add_argument("--queries-pdf", default=QUERIES_PDF_DEFAULT, help="PDF oficial usado solo si --queries no existe")
    parser.add_argument("--output", default="resultados.jsonl", help="Ruta de salida")
    parser.add_argument("--encoders-dir", default=None, help="Override de EMBEDDING_OUTPUT_DIR (base_vectorial/)")
    parser.add_argument("--k-search", type=int, default=50, help="Top-k por índice antes de fusionar")
    parser.add_argument("--k0-rrf", type=int, default=None, help="Constante k0 de RRF (informativo; no soportado por la firma pública de retrieve())")
    parser.add_argument("--doc-agg", choices=["max", "sum", "weighted_mean"], default="max", help="Estrategia de agregación a documento")
    parser.add_argument("--theta", type=float, default=0.0, help="Umbral de similitud coseno original")
    parser.add_argument("--phenomenon-filter", type=int, default=None, choices=[1, 2, 3], help="Filtra por fenómeno (1, 2 o 3)")
    parser.add_argument("--lang-filter", default=None, help="Filtra por idioma")
    parser.add_argument("--date-range", type=_parse_date_range, default=None, help="Rango 'inicio,fin' (ISO), extremos opcionales")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (DEBUG, INFO, WARNING, ...)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.encoders_dir:
        os.environ["EMBEDDING_OUTPUT_DIR"] = args.encoders_dir
    if args.k0_rrf is not None:
        logger.warning("--k0-rrf=%s ignorado: la firma pública de retrieve() no acepta 'k0'.", args.k0_rrf)

    if not Path(args.queries).exists():
        if not Path(args.queries_pdf).exists():
            logger.error("No existe ni '%s' ni el PDF '%s'.", args.queries, args.queries_pdf)
            return 1
        try:
            QueryLoader.exportar_desde_pdf(args.queries_pdf, args.queries)
        except Exception:  # noqa: BLE001 - reportar el fallo raíz y salir con error
            logger.exception("No se pudo generar el archivo de consultas desde el PDF")
            return 1

    try:
        generate_results(
            args.queries,
            args.output,
            k_search=args.k_search,
            doc_agg=args.doc_agg,
            theta=args.theta,
            phenomenon_filter=args.phenomenon_filter,
            format_filter=None,
            lang_filter=args.lang_filter,
            date_range=args.date_range,
        )
    except Exception:  # noqa: BLE001 - reportar el fallo raíz y salir con error
        logger.exception("Generación abortada")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
