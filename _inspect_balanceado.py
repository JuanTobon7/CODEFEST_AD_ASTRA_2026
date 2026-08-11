"""
Verificación Fase 3: cobertura por subcarpeta del muestreo balanceado.

ANTES (sort global por chunk_id): la cuota de F1 se llenaba solo con
AI_Index_Stanford. DESPUÉS (find_all_balanceado con reparto por subcarpeta):
debe cubrir las 8 subcarpetas de F1 y SWF/INPE/ESA/UNOOSA en F2.

Uso:
    python _inspect_balanceado.py [--limite 5000]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.embeddings.embedding_config import EmbeddingConfig
from src.persistence.mongo_repository import MongoChunkRepository


def _subcarpeta(doc_id: str) -> str:
    return MongoChunkRepository._subcarpeta_de(doc_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=5000)
    args = parser.parse_args()

    config = EmbeddingConfig()
    repo = MongoChunkRepository(
        config.mongo_uri, config.mongo_db, config.mongo_collection_chunks,
        username=config.mongo_user, password=config.mongo_password,
        auth_source=config.mongo_auth_source,
    )
    repo.connect()
    coleccion = repo._cliente[config.mongo_db][config.mongo_collection_chunks]

    total = coleccion.count_documents({})
    print(f"Chunks totales en Mongo: {total}")

    # --- Estado por fenómeno y subcarpeta (población) ----------------------
    por_fenomeno: Counter = Counter()
    por_sub: defaultdict = defaultdict(Counter)
    for d in coleccion.find({}, {"doc_id": 1, "fenomeno": 1}):
        f = d["fenomeno"]
        por_fenomeno[f] += 1
        por_sub[f][_subcarpeta(d["doc_id"])] += 1

    print("\n=== POBLACIÓN por fenómeno ===")
    for f in sorted(por_fenomeno):
        print(f"  F{f}: {por_fenomeno[f]} chunks")

    print("\n=== SUBCARPETAS por fenómeno (población) ===")
    for f in sorted(por_sub):
        print(f"  F{f}:")
        for sub, n in sorted(por_sub[f].items()):
            print(f"    {n:>8,}  {sub}")

    # --- Estado ANTES: sort global por chunk_id ----------------------------
    print("\n=== ANTES (sort global por chunk_id, límite=F1) ===")
    cuotas = MongoChunkRepository._cuotas_por_fenomeno(args.limite)
    for f, tope in sorted(cuotas.items()):
        if tope <= 0:
            continue
        docs = (
            coleccion.find({"fenomeno": f}, {"doc_id": 1, "chunk_id": 1})
            .sort("chunk_id", 1)
            .limit(tope)
        )
        subs = Counter(_subcarpeta(d["doc_id"]) for d in docs)
        print(f"  F{f} (tope={tope}):")
        for sub, n in sorted(subs.items()):
            print(f"    {n:>8,}  {sub}")

    # --- Estado DESPUÉS: find_all_balanceado -------------------------------
    print(f"\n=== DESPUÉS (find_all_balanceado({args.limite})) ===")
    chunks = repo.find_all_balanceado(args.limite)
    print(f"  Total devuelto: {len(chunks)}")
    despues_por_sub: defaultdict = defaultdict(Counter)
    for c in chunks:
        despues_por_sub[c.fenomeno][_subcarpeta(c.doc_id)] += 1
    for f in sorted(despues_por_sub):
        print(f"  F{f}: {sum(despues_por_sub[f].values())} chunks")
        for sub, n in sorted(despues_por_sub[f].items()):
            print(f"    {n:>8,}  {sub}")

    # --- Veredicto ----------------------------------------------------------
    # Los nombres reales de subcarpeta tienen sufijos (AI_Index_Stanford,
    # SWF_Counterspace, ESA_Space_Debris, Defensa21_LatAm): match por sufijo.
    print("\n=== VEREDICTO ===")
    f1_esperadas = {
        "AI_Index_Stanford", "Atlantic_Council", "CENIA", "CSET_Georgetown",
        "DAIO", "Defensa21_LatAm", "ILIA_Latam", "RutaN_GEIAL",
    }
    f1_cubiertas = {e for e in f1_esperadas if any(s.endswith(e) for s in despues_por_sub[1])}
    cubre_f1 = f1_esperadas <= f1_cubiertas
    print(f"  F1 cubre las 8 subcarpetas esperadas: {cubre_f1}")
    if not cubre_f1:
        print(f"    Faltan: {sorted(f1_esperadas - f1_cubiertas)}")
    f2_esperadas = {"SWF_Counterspace", "INPE", "ESA_Space_Debris", "UNOOSA"}
    f2_cubiertas = {e for e in f2_esperadas if any(s.endswith(e) for s in despues_por_sub[2])}
    cubre_f2 = f2_esperadas <= f2_cubiertas
    print(f"  F2 cubre SWF/INPE/ESA/UNOOSA: {cubre_f2}")
    if not cubre_f2:
        print(f"    Faltan: {sorted(f2_esperadas - f2_cubiertas)}")

    # F1 ya no es 100% AI_Index_Stanford
    f1_total = sum(despues_por_sub[1].values())
    ai_chunks = sum(
        n for s, n in despues_por_sub[1].items() if s.endswith("AI_Index_Stanford")
    )
    ai_share = ai_chunks / max(f1_total, 1)
    print(f"  F1: share AI_Index_Stanford = {ai_share:.1%} (debe ser << 100%)")
    print(
        f"  Total {len(chunks)} < {args.limite}: subcarpetas con menos chunks que su "
        "cuota (Defensa21_LatAm=1, CEEEP=104, FASE=11) — respeta disponibilidad."
    )

    repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
