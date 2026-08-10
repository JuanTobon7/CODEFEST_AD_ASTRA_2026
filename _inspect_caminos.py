"""Inspección del JSON de caminos generado."""
import json

with open("logs/caminos_grafo.json", encoding="utf-8") as f:
    datos = json.load(f)

print("consultas:", [d["query_id"] for d in datos])
d = datos[2]  # q010
print("\n== q010 ==")
print("top1:", d["top_k"][0]["chunk_id"], "score:", d["top_k"][0]["score"])
camino = d["camino_grafo"]
print("entidades_consulta:", [(e["id"], e["tipo"]) for e in camino["entidades_consulta"]])
print("entidades_en_grafo:", camino["entidades_en_grafo"])
print("vecinos de basura espacial:", camino["vecinos_primer_orden"].get("basura espacial", [])[:5], "...")
print("tripletas (n):", len(camino["tripletas"]))
for t in camino["tripletas"][:5]:
    print("   ", t["sujeto"], "-", t["relacion"], "->", t["objeto"], f"[{t['chunk_id']}]")
print("chunks_evidencia (top 3):", list(camino["chunks_evidencia"].items())[:3])
print("aportó:", camino["aportó"])
