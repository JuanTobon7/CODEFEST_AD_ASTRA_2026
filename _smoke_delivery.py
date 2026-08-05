"""
Script de humo (debug): construye artefactos de entrega TEMPORALES en
``base_vectorial_smoke/`` con los embeddings YA persistidos en disco
(parciales: 500 vectores por encoder de la corrida ``--limite 500``).

Útil para validar el flujo completo de entrega (generador -> retrieve ->
resultados.jsonl) sin esperar el embedding completo del corpus:

- ``vectors.npy``  (n×d, alineado por ordinal con chunk_ids.jsonl)
- ``chunk_ids.jsonl`` ({"ordinal", "chunk_id"})
- metadata de cada chunk desde MongoDB (los 8 campos obligatorios de la
  Tabla 1 + idioma/titulo/fecha si existen), en el ORDEN de chunk_ids.

El ``index.faiss`` se construye con ``IndexFlatIP`` + ``index.add()``
secuencial (ID interno = ordinal de línea en metadata.jsonl), exactamente el
supuesto de ``src.retrieval.faiss_search`` para los artefactos de entrega.

NO reemplaza la entrega oficial: ``run_export_delivery`` sigue exigiendo
embeddings del 100% del corpus.

Uso:
    python _smoke_delivery.py
    python generador.py --encoders-dir base_vectorial_smoke --output resultados_smoke.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np

from src.embeddings.embedding_config import EmbeddingConfig
from src.persistence.mongo_repository import MongoChunkRepository

# Los 8 campos obligatorios de la Tabla 1 del reto (el artefacto de entrega
# usa la clave ``texto`` para el texto del chunk).
CAMPOS_OBLIGATORIOS = (
    "doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto",
)
CAMPOS_OPCIONALES = ("idioma", "titulo_documento", "fecha_publicacion")

DIR_SALIDA = Path("base_vectorial_smoke")


def _leer_chunk_ids(ruta: Path) -> list[str]:
    """Lee ``chunk_ids.jsonl`` (líneas ``{"ordinal", "chunk_id"}``)."""
    chunk_ids: list[str] = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                chunk_ids.append(json.loads(linea)["chunk_id"])
    return chunk_ids


def main() -> int:
    config = EmbeddingConfig()
    origen = Path(config.embedding_output_dir)

    repositorio = MongoChunkRepository(
        config.mongo_uri, config.mongo_db, config.mongo_collection_chunks,
        username=config.mongo_user, password=config.mongo_password,
        auth_source=config.mongo_auth_source,
    )
    try:
        repositorio.connect()
        chunks_por_id = {c.chunk_id: c for c in repositorio.find_all()}
    finally:
        repositorio.close()

    for nombre in config.encoder_names:
        carpeta = origen / f"encoder_{nombre}"
        chunk_ids = _leer_chunk_ids(carpeta / "chunk_ids.jsonl")
        vectores = np.load(carpeta / "vectors.npy").astype("float32")
        assert len(chunk_ids) == len(vectores), (
            f"encoder '{nombre}': {len(chunk_ids)} chunk_ids != {len(vectores)} vectores"
        )

        faltantes = [cid for cid in chunk_ids if cid not in chunks_por_id]
        if faltantes:
            print(f"encoder '{nombre}': {len(faltantes)} chunk_ids sin metadata en Mongo, se omiten", file=sys.stderr)
            indices_ok = [i for i, cid in enumerate(chunk_ids) if cid in chunks_por_id]
            chunk_ids = [chunk_ids[i] for i in indices_ok]
            vectores = vectores[indices_ok]

        lineas_meta: list[dict] = []
        for cid in chunk_ids:
            chunk = chunks_por_id[cid]
            fila = {campo: getattr(chunk, campo) for campo in CAMPOS_OBLIGATORIOS}
            for campo in CAMPOS_OPCIONALES:
                valor = getattr(chunk, campo)
                if valor:
                    fila[campo] = valor
            lineas_meta.append(fila)

        dim = int(vectores.shape[1])
        indice = faiss.IndexFlatIP(dim)
        indice.add(vectores)

        salida = DIR_SALIDA / f"encoder_{nombre}"
        salida.mkdir(parents=True, exist_ok=True)
        faiss.write_index(indice, str(salida / "index.faiss"))
        with open(salida / "metadata.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for fila in lineas_meta:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")

        print(f"encoder '{nombre}': {len(chunk_ids)} vectores (dim={dim}) -> {salida}")

    print(f"Listo. Ejecuta: python generador.py --encoders-dir {DIR_SALIDA} --output resultados_smoke.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
