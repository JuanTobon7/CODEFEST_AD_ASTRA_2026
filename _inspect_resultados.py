"""Inspección de resultados híbridos y camino de q010."""
import json

with open("resultados_hibridos.jsonl", encoding="utf-8") as f:
    lineas = [json.loads(l) for l in f if l.strip()]

print("total líneas:", len(lineas))
q010 = next(l for l in lineas if l["query_id"] == "q010")
print("\n== q010 ==")
print("documents:", [(d["rank"], d["doc_id"][:50]) for d in q010["documents"]])
print("fragments top-3:")
for fr in q010["fragments"][:3]:
    print("   ", fr["rank"], fr["chunk_id"][:60], "| palabras:", len(fr["text"].split()))

with open("logs/caminos_grafo.json", encoding="utf-8") as f:
    caminos = json.load(f)
c = next(x for x in caminos if x["query_id"] == "q010")
camino = c["camino_grafo"]
print("\n== camino q010 ==")
print("entidades_consulta:", [(e["id"], e["tipo"]) for e in camino["entidades_consulta"]])
print("entidades_en_grafo:", camino["entidades_en_grafo"])
print("tripletas (n):", len(camino["tripletas"]))
print("chunks_evidencia top-3:", list(camino["chunks_evidencia"].items())[:3])
print("camino_por_top_k (top1):")
t1 = camino["camino_por_top_k"][0]
print("   chunk:", t1["chunk_id"][:60], "| score_grafo:", t1["score_grafo"])
for t in t1["tripletas"][:4]:
    print("      ", t["sujeto"], "-", t["relacion"], "->", t["objeto"])