# Prompt: Persistencia de Vectores en MongoDB + Construcción de Índices FAISS por Encoder
### CODEFEST AD ASTRA 2026 — Etapa 1 — Complemento del prompt de encoders (Sección 4 → Sección 5)

> Este prompt **complementa** `prompt_encoders_embeddings.md`, no lo reemplaza. Todo lo ya definido ahí (`EncoderStrategy`, `EncoderFactory`, `EncoderOrchestrator`) se mantiene igual. Lo único que cambia es el **destino de persistencia**: en vez de (o además de) escribir `vectors.npy` a disco como caché plana, los vectores se guardan de forma persistente en **MongoDB**, y desde ahí se construyen y exportan los índices **FAISS**, uno por encoder, tal como exige la especificación:
>
> *"Si se emplean múltiples encoders, cada uno genera su propio índice FAISS independiente. [...] este paso se repite para cada uno, generando un conjunto de vectores por fragmento."*

---

## Rol

Actúa como ingeniero de datos/ML especializado en sistemas de recuperación vectorial. Debes implementar en Python 3.11+ un módulo que:

1. Persista en **MongoDB** el vector generado por cada `EncoderStrategy` para cada chunk, de forma trazable (`chunk_id` + `encoder_name` como clave compuesta).
2. Construya, **un índice FAISS independiente por cada encoder activo** (patrón ya establecido por `EncoderOrchestrator`: N encoders → N espacios vectoriales → N índices), sin mezclar vectores de distinta dimensionalidad o de distinto modelo en un mismo índice.
3. Permita seleccionar el **tipo de índice FAISS** (`IndexFlatIP`, `IndexIVFFlat`, `IndexHNSW`) también mediante **Strategy**, ya que la especificación (Sección 5.2) reconoce distintos tipos según el volumen de datos y el objetivo (exactitud vs. velocidad).
4. Exporte los artefactos físicos exigidos por el reto (`index.faiss` + `metadata.jsonl` por `encoder_<nombre>/`) respetando **al pie de la letra** el requisito de la Sección 5.3: *"El orden de las líneas [de `metadata.jsonl`] debe coincidir con los identificadores internos asignados por FAISS al momento de la indexación."*

---

## 1. Corrección/aclaración importante respecto al prompt anterior

En `prompt_encoders_embeddings.md` se propuso `EmbeddingWriter` escribiendo `vectors.npy` + `chunk_ids.jsonl` como caché intermedia en disco. Eso **se mantiene como caché local opcional** (útil para no perder cómputo si se interrumpe un proceso largo), pero **ya no es la fuente de verdad**. La fuente de verdad persistente pasa a ser MongoDB. Ajusta `EmbeddingWriter`/`EmbeddingCache` para que, además de (o en vez de) escribir a disco, invoquen a `VectorRepository.save_many(...)` (ver Sección 2) como paso obligatorio de persistencia.

---

## 2. Persistencia de vectores en MongoDB

### 2.1 Modelo de datos

Nueva colección `embeddings`, **separada** de `chunks` (para no inflar los documentos de `chunks` con arrays grandes y para poder tener 1:N — un chunk puede tener un vector por cada encoder activo):

```json
{
  "_id": "ObjectId(...)",
  "chunk_id": "DOC-042-chunk-007",
  "doc_id": "DOC-042",
  "fenomeno": 1,
  "formato": "pdf",
  "encoder_name": "e5-base",
  "model_id": "intfloat/multilingual-e5-base",
  "embedding_dim": 768,
  "vector": "<BinData: float32[768] empaquetado con numpy.tobytes()>",
  "vector_dtype": "float32",
  "normalized": true,
  "hash_texto": "sha256(...)",
  "faiss_internal_id": null,
  "created_at": "ISODate(...)",
  "updated_at": "ISODate(...)"
}
```

**Reglas de implementación:**

- El vector se guarda como `bson.Binary` (bytes de `numpy.ndarray.astype(np.float32).tobytes()`), **no** como `list[float]` — un array de floats en JSON/BSON es varias veces más pesado que el binario empaquetado, y con miles de chunks × varios encoders esto importa.
- Índice único compuesto sobre `(chunk_id, encoder_name)` — evita duplicados si se reprocesa el mismo chunk con el mismo encoder (upsert idempotente).
- Índice adicional sobre `encoder_name` (para poder extraer eficientemente "todos los vectores de este encoder" al construir el índice FAISS correspondiente).
- `hash_texto` permite detectar si el texto del chunk cambió desde que se calculó el embedding (reutilizando el mismo hash ya definido en la etapa de chunking/validación); si cambió, se recomputa y se hace upsert.
- `faiss_internal_id` queda `null` en el modo de persistencia pura; se rellena solo si se usa el **modo operativo incremental** (Sección 4.2).

### 2.2 Repository

```
VectorRepository (ABC)
 ├── save_many(records: list[EmbeddingRecord]) -> None      # upsert por (chunk_id, encoder_name)
 ├── find_by_encoder(encoder_name: str) -> Iterator[EmbeddingRecord]   # streaming, no cargar todo en memoria
 ├── find_missing(encoder_name: str, chunk_ids: list[str]) -> list[str] # chunks sin embedding aún para ese encoder
 ├── count_by_encoder(encoder_name: str) -> int
 └── delete_by_chunk_id(chunk_id: str) -> None                # para reprocesos/borrados

MongoVectorRepository(VectorRepository)
```

- `find_by_encoder()` debe ser un **cursor/generador** (no `list()` de todo), para poder construir índices FAISS de corpus grandes sin agotar memoria — usar `batch_size` de PyMongo y convertir a `np.frombuffer` en el momento de consumir.
- `EncoderOrchestrator.run()` (del prompt anterior) debe ahora, al terminar cada batch, llamar a `VectorRepository.save_many(...)` en vez de (o además de) escribir a `.npy`.

---

## 3. Selección del tipo de índice FAISS — Strategy adicional

Igual que con los encoders, el **tipo de índice** también varía según criterios (tamaño del corpus, prioridad exactitud vs. velocidad — Sección 5.2 de la especificación), así que se modela con el mismo patrón:

```
FaissIndexBuilderStrategy (ABC)
 ├── index_type_name: str                       # "flat_ip" | "ivf_flat" | "hnsw"
 ├── build(dim: int, config: IndexBuildConfig) -> faiss.Index    # índice vacío, listo para add()
 ├── requires_training: bool
 └── train_if_needed(index: faiss.Index, vectors: np.ndarray) -> None   # no-op si requires_training=False

FlatIPIndexStrategy(FaissIndexBuilderStrategy)
   # faiss.IndexFlatIP — exacto, sin entrenamiento, requires_training=False
   # DEFAULT recomendado por la especificación para el volumen de este reto

IVFFlatIndexStrategy(FaissIndexBuilderStrategy)
   # faiss.IndexIVFFlat(quantizer, dim, nlist) — requiere training (k-means) antes de add()
   # usar si el corpus de un encoder supera un umbral configurable (p. ej. > 50k vectores)

HNSWIndexStrategy(FaissIndexBuilderStrategy)
   # faiss.IndexHNSWFlat(dim, M) — sin training, más memoria, búsqueda muy rápida
   # usar si se prioriza latencia de consulta sobre uso de memoria
```

`IndexBuilderFactory` (mismo patrón de registro por decorador que `EncoderFactory`):

```
IndexBuilderFactory
 ├── register(name: str) -> decorador
 └── create(name: str, config: IndexBuildConfig) -> FaissIndexBuilderStrategy
```

**Recomendación por defecto explícita en config**: `FlatIPIndexStrategy`, tal como indica la especificación (*"para el volumen de documentos esperado en este reto, un índice plano [...] es suficiente y garantiza resultados exactos"*). Los otros dos quedan implementados y disponibles, pero no son el default — deja esto documentado en el `README.md` como parte de la justificación del informe técnico.

Todos los builders deben asumir que los vectores llegan **ya normalizados a norma unitaria** (responsabilidad de `EncoderStrategy.encode()`, ya cubierta en el prompt anterior) para que `IndexFlatIP`/`IndexIVFFlat` con producto interno equivalgan a similitud coseno.

---

## 4. `FaissIndexManager` — dos modos de operación

### 4.1 Modo operativo (incremental, para desarrollo/pruebas iterativas)

Pensado para reconstruir o actualizar índices sin recalcular todo el corpus cada vez que se agregan documentos nuevos durante el desarrollo.

- Usa `faiss.IndexIDMap` envolviendo el índice base (`FlatIP`/`IVFFlat`/`HNSW`) para poder asignar **IDs propios y estables**, no secuenciales por orden de inserción.
- El id FAISS se deriva determinísticamente del `chunk_id` (p. ej. `faiss_id = int(hashlib.sha1(chunk_id.encode()).hexdigest()[:15], 16)`, casteado a `int64` — determinístico y estable entre corridas).
- Al agregar/actualizar un vector, se usa `index.add_with_ids(vector, faiss_id)`; para eliminar/actualizar un chunk existente, `index.remove_ids(...)` seguido de un nuevo `add_with_ids`.
- El `faiss_internal_id` calculado se guarda de vuelta en el documento `embeddings` correspondiente (Sección 2.1), cerrando el mapeo `chunk_id ↔ faiss_id` en MongoDB, que actúa aquí como el **almacén de metadata** exigido por la Sección 5.3 de la especificación (perfectamente válido: la spec permite explícitamente "base de datos SQLite, etc." como almacén de metadata — MongoDB cumple el mismo rol).
- Este índice también se persiste a disco (`faiss.write_index`) como caché de trabajo, en una ruta separada de la carpeta de entrega final (p. ej. `working_index/encoder_<nombre>/index.faiss`), para no confundirlo con el artefacto de entrega.

```
FaissIndexManager
 ├── build_or_update(encoder_name: str, index_strategy: FaissIndexBuilderStrategy) -> IndexBuildResult
 │     1. lee vectores pendientes/actualizados desde VectorRepository (por hash_texto)
 │     2. construye o reabre el índice de trabajo (faiss.read_index si ya existe)
 │     3. calcula faiss_id determinístico por chunk_id
 │     4. add_with_ids / remove_ids + add_with_ids según corresponda
 │     5. persiste índice de trabajo a disco
 │     6. actualiza faiss_internal_id en MongoDB
 └── load(encoder_name: str) -> faiss.Index
```

### 4.2 Modo de exportación para entrega (estricto, cumple el formato exigido por el reto)

La especificación exige explícitamente que el orden de líneas de `metadata.jsonl` coincida con los **identificadores internos asignados por FAISS al momento de la indexación** — lo cual, para un índice plano construido con `add()` secuencial (sin `IndexIDMap`), son simplemente `0, 1, 2, ..., n-1` en el orden de inserción. Por eso, el artefacto de entrega **no** debe reutilizar el índice con `IndexIDMap` del modo operativo; se genera aparte, de forma determinística y reproducible:

```
export_delivery.py
 1. Para cada encoder activo:
    a. VectorRepository.find_by_encoder(encoder_name), ordenado de forma determinística y estable
       (ORDER BY doc_id ASC, posicion ASC — nunca por _id de Mongo, que no es reproducible entre corridas)
    b. Construye un índice NUEVO y "limpio" con la estrategia configurada (index_strategy.build(dim, config))
    c. Inserta los vectores con index.add(vectors) en ese mismo orden (sin IDs custom)
    d. Escribe metadata.jsonl línea por línea, EN ESE MISMO ORDEN, con los campos completos de la Tabla 1
       (doc_id, chunk_id, fuente, formato, fenomeno, posicion, num_tokens, texto)
    e. faiss.write_index(index, "base_vectorial/encoder_<nombre>/index.faiss")
    f. Registra un checksum (sha256 del archivo index.faiss + conteo de vectores) en un log de build,
       para poder verificar reproducibilidad si el script generador.py se corre de nuevo
```

Este script es, en esencia, la implementación real del **entregable 4** del reto (`generador.py`/parte de él): debe ser determinístico — correrlo dos veces sobre los mismos datos de Mongo debe producir un `index.faiss` y un `metadata.jsonl` idénticos (mismo orden, mismos vectores).

---

## 5. Configuración adicional

Extiende el `.env`/`EmbeddingConfig` del prompt anterior:

```
# Ya existentes: ACTIVE_ENCODERS, EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, MONGO_URI, MONGO_DB...

MONGO_COLLECTION_EMBEDDINGS=embeddings
FAISS_INDEX_TYPE=flat_ip                 # flat_ip | ivf_flat | hnsw   (default: flat_ip)
FAISS_IVF_NLIST=100                      # solo si FAISS_INDEX_TYPE=ivf_flat
FAISS_IVF_NPROBE=10                      # solo si FAISS_INDEX_TYPE=ivf_flat
FAISS_HNSW_M=32                          # solo si FAISS_INDEX_TYPE=hnsw
FAISS_IVF_AUTO_THRESHOLD=50000           # si n_vectores > umbral y tipo=auto, usar ivf_flat en vez de flat_ip
WORKING_INDEX_DIR=working_index          # índices de trabajo (modo operativo, con IndexIDMap)
DELIVERY_OUTPUT_DIR=base_vectorial       # artefactos finales de entrega (modo exportación)
```

Si `FAISS_INDEX_TYPE=auto`, `IndexBuilderFactory` debe decidir entre `flat_ip` e `ivf_flat`/`hnsw` según `FAISS_IVF_AUTO_THRESHOLD` comparado con `VectorRepository.count_by_encoder(encoder_name)` — dejando trazado en el log/registro de build cuál estrategia se eligió y por qué (para justificar en el informe técnico).

---

## 6. Validaciones obligatorias

- Antes de construir un índice para un encoder, verificar que **todos** los vectores recuperados de Mongo para ese `encoder_name` tienen exactamente la misma `embedding_dim` (si no, abortar con error explícito — indica mezcla accidental de modelos).
- Verificar `count_by_encoder(encoder_name)` contra el número total de chunks activos en `chunks` (Mongo) antes de exportar; si hay chunks sin embedding para ese encoder, **fallar la exportación** con la lista de `chunk_id` faltantes (no generar un `metadata.jsonl` incompleto silenciosamente).
- En el modo de exportación, verificar después de `add()` que `index.ntotal == len(vectors) == número de líneas escritas en metadata.jsonl` — assert de integridad final antes de dar el build por válido.
- Verificar que `index.faiss` recién escrito es cargable con `faiss.read_index()` y que `index.d == embedding_dim` esperado (test de humo automático al final del export).

---

## 7. Estructura de carpetas a añadir

```
src/
 ├── vectorstore/
 │    ├── models.py                    # EmbeddingRecord (pydantic)
 │    ├── vector_repository.py         # VectorRepository (ABC) + MongoVectorRepository
 │    ├── index_builder_base.py        # FaissIndexBuilderStrategy (ABC)
 │    ├── flat_ip_strategy.py
 │    ├── ivf_flat_strategy.py
 │    ├── hnsw_strategy.py
 │    ├── index_builder_factory.py
 │    ├── faiss_index_manager.py       # modo operativo (IndexIDMap, incremental)
 │    └── export_delivery.py           # modo de exportación (estricto, orden secuencial)
tests/
 └── test_vectorstore/
      ├── test_vector_repository_upsert.py
      ├── test_index_builder_factory.py
      ├── test_faiss_index_manager_incremental.py
      └── test_export_delivery_reproducibility.py   # corre export 2 veces, compara checksums
```

---

## 8. Entregable esperado de esta tarea

Genera:

1. `EmbeddingRecord` (pydantic) + `VectorRepository`/`MongoVectorRepository` con upsert idempotente y lectura en streaming.
2. `FaissIndexBuilderStrategy` (ABC) + `FlatIPIndexStrategy`, `IVFFlatIndexStrategy`, `HNSWIndexStrategy` + `IndexBuilderFactory` con registro por decorador y resolución `auto` basada en umbral configurable.
3. `FaissIndexManager` (modo operativo incremental con `IndexIDMap` + ids determinísticos por `chunk_id`).
4. `export_delivery.py` (modo de exportación estricta) que genere, por cada encoder activo, `base_vectorial/encoder_<nombre>/index.faiss` + `metadata.jsonl`, cumpliendo exactamente la estructura de directorios y el esquema de la Tabla 1 exigidos por la especificación del reto (Sección 1.4).
5. Ajuste retrocompatible de `EncoderOrchestrator`/`EmbeddingWriter` (del prompt anterior) para que persistan en `VectorRepository` como parte del flujo normal, sin romper la interfaz ya definida.
6. Tests unitarios que verifiquen: idempotencia del upsert, correcta resolución del tipo de índice vía Factory, consistencia incremental (`add`/`remove`/`re-add` en el modo operativo), y **reproducibilidad exacta** del export (correr `export_delivery.py` dos veces sobre los mismos datos produce archivos idénticos).
7. Actualización del `README.md` documentando: por qué se eligió `IndexFlatIP` como default (justificación basada en el volumen del corpus), y cómo MongoDB cumple el rol de "almacén de metadata" que exige la especificación.

Este prompt no cubre todavía el **módulo de recuperación** (consulta → vector de query → búsqueda en FAISS → fusión multi-encoder vía RRF/CombSUM/CombMNZ → agregación a nivel documento) — eso corresponde a la Sección 8 de la especificación y será el siguiente prompt de la serie.