"""Validación de resultados híbridos + grafo (auditoría de calidad).

Uso:
    .venv\\Scripts\\python.exe _validate_hibrido.py

Resumen por consulta: fenómeno esperado vs top-doc recuperado, % de
fragmentos del fenómeno correcto, solape léxico pregunta-fragmento top-1
y veredicto (RESPONDE / parcial / NO RESPONDE / SIN RESULTADO).
"""
import json
import re
from collections import Counter

STOP = set(
    "cómo que de la el los las en y a al con por para qué cuáles cuál cuales su sus se un una del es son han esta este estas estos está están ha como más desde sobre entre cual o e i u no lo le".split()
)


def terminos(texto):
    return set(re.findall(r"[a-záéíóúñü]{4,}", texto.lower())) - STOP


def fenomeno_doc(doc_id):
    if doc_id.startswith("F1_"):
        return 1
    if doc_id.startswith("F2_"):
        return 2
    if doc_id.startswith("F3_"):
        return 3
    return 0


def fenomeno_esperado(qid):
    n = int(qid[1:])
    return 1 if n <= 16 else 2 if n <= 32 else 3


def main() -> None:
    qs = {}
    with open("consultas.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                q = json.loads(l)
                qs[q["query_id"]] = q["query_text"]

    res = {}
    with open("resultados_hibridos.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                res[r["query_id"]] = r

    print(f"{'qid':6} {'exp':3} {'fragFen%':8} {'topDoc':7} {'overlap':7} veredicto")
    print("-" * 80)
    resumen = Counter()
    for qid in sorted(qs):
        if qid not in res:
            print(f"{qid:6} {'-':3} {'FALLÓ':8} {'-':7} {'-':7} SIN RESULTADO")
            resumen["falló"] += 1
            continue
        r = res[qid]
        fe = fenomeno_esperado(qid)
        qt = terminos(qs[qid])
        frags = r["fragments"]
        frags_fen = sum(1 for fr in frags if fenomeno_doc(fr["doc_id"]) == fe)
        pct = round(100 * frags_fen / len(frags)) if frags else 0
        top_doc_fen = fenomeno_doc(r["documents"][0]["doc_id"])
        top_ok = "OK" if top_doc_fen == fe else f"F{top_doc_fen}"
        ov = len(qt & terminos(frags[0]["text"])) if frags else 0
        if top_doc_fen != fe:
            v = "NO RESPONDE"
            resumen["no"] += 1
        elif pct >= 80 and ov >= 2:
            v = "RESPONDE"
            resumen["ok"] += 1
        elif pct >= 80:
            v = "parcial"
            resumen["parcial"] += 1
        else:
            v = "NO RESPONDE"
            resumen["no"] += 1
        print(f"{qid:6} F{fe:<3} {pct:>5}%  {top_ok:7} {ov:>5}  {v}")

    print("-" * 80)
    print("RESUMEN:", dict(resumen))


if __name__ == "__main__":
    main()
