"""Inspección temporal (v2): estado real de la colección chunks en Mongo."""
from collections import Counter

from pymongo import MongoClient

c = MongoClient("mongodb://localhost:27017", username="admin", password="admin", authSource="admin")
col = c["rag_corpus"]["chunks"]

print("TOTAL:", col.count_documents({}))
print()
print("=== Campos de un documento de muestra ===")
muestra = col.find_one({})
if muestra:
    print(sorted(muestra.keys()))
    print("chunking_strategy =", muestra.get("chunking_strategy"))
    print("formato =", muestra.get("formato"))
    print("num_tokens =", muestra.get("num_tokens"))
    print("created_at =", muestra.get("created_at"))
print()

por_estrategia = Counter()
por_formato = Counter()
pequenos_por_formato = Counter()
n = 0
for d in col.find({}, {"chunking_strategy": 1, "formato": 1, "num_tokens": 1}):
    n += 1
    por_estrategia[d.get("chunking_strategy")] += 1
    por_formato[d.get("formato")] += 1
    if (d.get("num_tokens") or 0) < 150:
        pequenos_por_formato[d.get("formato")] += 1

print("escaneados:", n)
print()
print("=== Por chunking_strategy ===")
for k, v in por_estrategia.most_common():
    print(f"{k!r:28} {v}")
print()
print("=== Por formato ===")
for k, v in por_formato.most_common():
    print(f"{k!r:12} {v}")
print()
print("=== Chunks < 150 tokens, por formato ===")
for k, v in pequenos_por_formato.most_common():
    print(f"{k!r:12} {v}")
