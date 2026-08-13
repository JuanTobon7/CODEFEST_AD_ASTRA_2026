"""
Script TEMPORAL: composición del corpus tras la extracción (Tabla I).

Recorre el JSON de chunks en streaming y agrega por formato: documentos
distintos, bloques (fragmentos), caracteres de texto y violaciones del contrato
de datos de la Tabla 1. Además cruza con el corpus en disco para listar los
archivos que NO produjeron ningún fragmento.

Violación del contrato = un registro que incumple alguna de estas reglas:

  * falta uno de los ocho campos obligatorios, o viene nulo;
  * ``doc_id`` / ``chunk_id`` / ``fuente`` vacíos;
  * ``fenomeno`` fuera de {1, 2, 3};
  * ``posicion`` o ``num_tokens`` negativos;
  * ``texto`` vacío;
  * ``chunk_id`` repetido en todo el archivo.

El exceso sobre el límite del encoder (``num_tokens > MAX_TOKENS``) se cuenta
aparte: no incumple el modelo, pero sí truncaría al codificar.

Uso::

    python _analisis_corpus.py
    python _analisis_corpus.py --latex     # además, el cuerpo de la tabla en LaTeX
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.models.chunk import Chunk
from src.models.config import Settings
from src.persistence.json_repository import JsonChunkRepository
from src.pipeline.batch_ingestor import BatchIngestor
from src.pipeline.corpus_service import CorpusService

OBLIGATORIOS = Chunk.CAMPOS_METADATA_OBLIGATORIA


class _Agregado:
    """Acumulador por formato."""

    def __init__(self) -> None:
        self.docs: Set[str] = set()
        self.bloques = 0
        self.caracteres = 0
        self.tokens = 0
        self.violaciones = 0
        self.exceden_limite = 0


def _violaciones(registro: dict, chunk_ids: Set[str]) -> List[str]:
    """Reglas del contrato que incumple ``registro`` (lista vacía = correcto)."""
    fallos: List[str] = []
    for campo in OBLIGATORIOS:
        if registro.get(campo) is None:
            fallos.append(f"falta {campo}")
    for campo in ("doc_id", "chunk_id", "fuente"):
        valor = registro.get(campo)
        if isinstance(valor, str) and not valor.strip():
            fallos.append(f"{campo} vacío")
    if registro.get("fenomeno") not in (1, 2, 3):
        fallos.append("fenomeno fuera de {1,2,3}")
    for campo in ("posicion", "num_tokens"):
        valor = registro.get(campo)
        if isinstance(valor, int) and valor < 0:
            fallos.append(f"{campo} negativo")
    texto = registro.get("texto")
    if isinstance(texto, str) and not texto.strip():
        fallos.append("texto vacío")
    chunk_id = registro.get("chunk_id")
    if chunk_id in chunk_ids:
        fallos.append("chunk_id duplicado")
    return fallos


def _formato_legible(formato: Optional[str]) -> str:
    """Normaliza la etiqueta de formato para la tabla."""
    if not formato:
        return "(sin formato)"
    return {"txt": "texto", "md": "texto", "image": "imagen", "png": "imagen"}.get(
        formato, formato
    )


def _millones(n: int) -> str:
    """Caracteres en la escala de la tabla (K / M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M".replace(".", ",")
    if n >= 1_000:
        return f"{n / 1_000:.0f} K"
    return str(n)


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Composición del corpus tras la extracción")
    parser.add_argument("--corpus", type=Path, default=Path("repo/CORPUS_CODEFEST_AD_ASTRA_2026"))
    parser.add_argument("--fenomenos", type=Path, default=Path("data/fenomenos.json"))
    parser.add_argument("--latex", action="store_true", help="Imprime el cuerpo de la tabla en LaTeX")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Límite del encoder (por defecto, el de .env)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

    settings = Settings()
    max_tokens = args.max_tokens or settings.max_tokens
    repositorio = JsonChunkRepository(settings.chunks_json_path)

    por_formato: Dict[str, _Agregado] = defaultdict(_Agregado)
    por_fenomeno: Dict[int, _Agregado] = defaultdict(_Agregado)
    chunk_ids: Set[str] = set()
    detalle_violaciones: Dict[str, int] = defaultdict(int)
    docs_totales: Set[str] = set()

    for registro in repositorio._iter_objetos_json():
        formato = _formato_legible(registro.get("formato"))
        doc_id = registro.get("doc_id") or "(sin doc_id)"
        texto = registro.get("texto") or ""
        tokens = registro.get("num_tokens") or 0

        fallos = _violaciones(registro, chunk_ids)
        for fallo in fallos:
            detalle_violaciones[fallo] += 1
        chunk_ids.add(registro.get("chunk_id"))
        docs_totales.add(doc_id)

        for agregado in (por_formato[formato], por_fenomeno[registro.get("fenomeno") or 0]):
            agregado.docs.add(doc_id)
            agregado.bloques += 1
            agregado.caracteres += len(texto)
            agregado.tokens += tokens
            agregado.violaciones += 1 if fallos else 0
            agregado.exceden_limite += 1 if tokens > max_tokens else 0

    # Tabla por formato --------------------------------------------------------
    filas = sorted(por_formato.items(), key=lambda kv: -kv[1].bloques)
    ancho = 76
    print()
    print("COMPOSICIÓN DEL CORPUS TRAS LA EXTRACCIÓN".center(ancho))
    print(f"(fuente: {repositorio.path})".center(ancho))
    print("=" * ancho)
    print(f"{'Formato':<12}{'Docs.':>8}{'Bloques':>12}{'Caracteres':>14}{'Tokens':>14}{'Viol.':>8}")
    print("-" * ancho)
    for formato, agregado in filas:
        print(
            f"{formato:<12}{_miles(len(agregado.docs)):>8}{_miles(agregado.bloques):>12}"
            f"{_millones(agregado.caracteres):>14}{_millones(agregado.tokens):>14}"
            f"{agregado.violaciones:>8}"
        )
    print("-" * ancho)
    total_bloques = sum(a.bloques for a in por_formato.values())
    total_caracteres = sum(a.caracteres for a in por_formato.values())
    total_tokens = sum(a.tokens for a in por_formato.values())
    total_viol = sum(a.violaciones for a in por_formato.values())
    print(
        f"{'Total':<12}{_miles(len(docs_totales)):>8}{_miles(total_bloques):>12}"
        f"{_millones(total_caracteres):>14}{_millones(total_tokens):>14}{total_viol:>8}"
    )
    print("=" * ancho)

    # Tabla por fenómeno -------------------------------------------------------
    print()
    print("DISTRIBUCIÓN POR FENÓMENO".center(ancho))
    print("-" * ancho)
    print(f"{'Fenómeno':<12}{'Docs.':>8}{'Bloques':>12}{'Caracteres':>14}{'Tokens':>14}{'Viol.':>8}")
    for fenomeno, agregado in sorted(por_fenomeno.items()):
        etiqueta = f"F{fenomeno}" if fenomeno else "(sin fenómeno)"
        print(
            f"{etiqueta:<12}{_miles(len(agregado.docs)):>8}{_miles(agregado.bloques):>12}"
            f"{_millones(agregado.caracteres):>14}{_millones(agregado.tokens):>14}"
            f"{agregado.violaciones:>8}"
        )

    # Calidad ------------------------------------------------------------------
    excedidos = sum(a.exceden_limite for a in por_formato.values())
    print()
    print("CONTRATO DE DATOS".center(ancho))
    print("-" * ancho)
    if detalle_violaciones:
        for regla, veces in sorted(detalle_violaciones.items(), key=lambda kv: -kv[1]):
            print(f"  {regla}: {_miles(veces)}")
    else:
        print("  Sin violaciones: los 8 campos obligatorios están completos y son válidos.")
    print(
        f"  Fragmentos que exceden el límite del encoder ({max_tokens} tokens): "
        f"{_miles(excedidos)} ({excedidos / max(total_bloques, 1):.2%})"
    )
    print(
        f"  Media de {total_bloques / max(len(docs_totales), 1):.0f} bloques por documento; "
        f"{total_tokens / max(total_bloques, 1):.0f} tokens por bloque."
    )

    # Cobertura del corpus en disco --------------------------------------------
    mapeo = CorpusService.load_fenomenos_map(args.fenomenos)
    archivos = CorpusService(args.corpus, mapeo).scan()
    if archivos:
        sin_chunks: Dict[str, List[str]] = defaultdict(list)
        for archivo in archivos:
            doc_id = BatchIngestor._doc_id_relativo(archivo, args.corpus)
            if doc_id not in docs_totales:
                sin_chunks[archivo.suffix.lower().lstrip(".") or "(sin ext)"].append(doc_id)
        total_sin = sum(len(v) for v in sin_chunks.values())
        print()
        print("COBERTURA DEL CORPUS".center(ancho))
        print("-" * ancho)
        print(
            f"  Archivos escaneados: {_miles(len(archivos))} | con fragmentos: "
            f"{_miles(len(archivos) - total_sin)} ({(len(archivos) - total_sin) / len(archivos):.1%}) | "
            f"sin fragmentos: {_miles(total_sin)}"
        )
        for extension, docs in sorted(sin_chunks.items(), key=lambda kv: -len(kv[1])):
            ejemplos = ", ".join(Path(d).name for d in docs[:3])
            print(f"    .{extension}: {len(docs)} (p. ej. {ejemplos})")

    if args.latex:
        print()
        print("% Cuerpo de la Tabla I")
        for formato, agregado in filas:
            print(
                f"\\texttt{{{formato}}} & {_miles(len(agregado.docs))} & "
                f"{_miles(agregado.bloques)} & {_millones(agregado.caracteres)} & "
                f"{agregado.violaciones} \\\\"
            )
        print("\\midrule")
        print(
            f"\\textbf{{Total}} & \\textbf{{{_miles(len(docs_totales))}}} & "
            f"\\textbf{{{_miles(total_bloques)}}} & \\textbf{{{_millones(total_caracteres)}}} & "
            f"\\textbf{{{total_viol}}} \\\\"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
