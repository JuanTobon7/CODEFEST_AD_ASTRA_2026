# CODEFEST AD ASTRA 2026 — Pipeline de Ingesta (Etapa 1)

Pipeline modular y extensible para **RAG**: extracción, limpieza, chunking
híbrido (estructural + semántico con overlap), metadata obligatoria y
persistencia de chunks en **JSON** (MongoDB sigue disponible como alternativa).
Implementa los patrones **Factory Method**
(extractores) y **Strategy** (chunking) con inyección de dependencias real.

---

## 1. Arquitectura del flujo

```mermaid
flowchart LR
    A[Archivo del corpus] --> B[ExtractorFactory.create]
    B --> C[BaseExtractor.extract]
    C --> D[ExtractedDocument<br/>secciones]
    D --> E[TextCleaner.clean]
    E --> F[ChunkingStrategy.chunk<br/>structural | semantic | hybrid |<br/>paragraph | paragraph_overlap]
    F --> G[MetadataBuilder.enrich<br/>campos Tabla 1]
    G --> H[ChunkValidator.validate]
    H -->|válidos| I[JsonChunkRepository.save_many<br/>upsert idempotente]
    H -->|rechazados| J[Log detallado<br/>y exclusion]
    I --> K[IngestionResult<br/>conteos + warnings]
```

```
Extractor → Cleaner → ChunkingStrategy → MetadataBuilder → Validator → MongoRepository
```

| Patrón | Dónde | Cómo se extiende |
|---|---|---|
| Factory Method | `src/extractors/` | `@register_extractor(".epub")` sobre una nueva clase `BaseExtractor` |
| Strategy | `src/chunking/` | Nueva clase `ChunkingStrategy` registrada en `ChunkingStrategyFactory` |
| Repository | `src/persistence/` | Nueva clase que implemente `ChunkRepository` (p. ej. SQLite) |
| Inversión de dependencias | `IngestionPipeline.__init__` | Todo se inyecta: factory, estrategia, cleaner, builder, validator, repositorio |

---

## 2. Estructura del proyecto

```
src/
 ├── extractors/     # BaseExtractor + 8 formatos (pdf, html, md/txt, json, csv/xlsx, image, pbf) + factory
 ├── chunking/       # ChunkingStrategy (ABC) + structural + semantic + hybrid + paragraph + paragraph_overlap + factory
 ├── cleaning/       # TextCleaner (UTF-8, controles, boilerplate, idioma)
 ├── metadata/       # MetadataBuilder (campos obligatorios Tabla 1)
 ├── validation/     # ChunkValidator (duras → rechazo; blandas → warnings)
 ├── persistence/    # ChunkRepository (ABC) + MongoChunkRepository
 ├── pipeline/       # IngestionPipeline (un archivo) + BatchIngestor (lote) + CorpusService (escaneo)
 ├── support/        # Tokenizer (BERT WordPiece), SentenceSplitter (spacy/regex), utils
 ├── models/         # ExtractedDocument, Section, Chunk, ChunkingConfig, Settings, IngestionResult, BatchSummary
 └── run_ingestion.py  # Controlador CLI delgado (GRASP): delega en BatchIngestor
tests/               # pytest: extractores, chunking, validador, cleaner, batch
config/.env.example  # plantilla de configuración
data/                # corpus de ejemplo + mapeo de fenómenos
```

---

## 3. Requisitos

- Python **3.11+**
- MongoDB accesible (local, Docker o remoto). Ejemplo con Docker:

```bash
docker run -d --name mongodb -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=admin \
  mongo:7
```

## 4. Instalación

```bash
# 1. Entorno virtual
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Dependencias
pip install -r requirements.txt

# 3. (Opcional) Segmentador de oraciones con spacy — ES/EN/PT
pip install spacy
python -m spacy download es_core_news_sm
# python -m spacy download en_core_web_sm
# python -m spacy download pt_core_news_sm

# 4. (Opcional) OCR de imágenes — al menos uno:
#    pytesseract (requiere binario Tesseract) o easyocr

# 5. Configuración
copy config\.env.example .env    # Windows
# cp config/.env.example .env    # Linux/macOS
# Ajusta MONGO_URI, CHUNK_SIZE, OVERLAP_SIZE, etc. según necesites.
```

Sin `.env` se usan valores por defecto: `mongodb://localhost:27017`, BD
`rag_corpus`, colección `chunks`, `chunk_size=400`, `overlap_size=80`,
`min_chunk_tokens=50`, `max_tokens=512`, estrategia `hybrid`.

Estrategias de chunking disponibles (`CHUNKING_STRATEGY`):

| Nombre | Descripción |
|---|---|
| `structural` | Un chunk por sección estructural, sin subdivisiones |
| `semantic` | Ventana deslizante por oraciones sobre todo el documento, con overlap |
| `hybrid` | Estructural (fase 1) + ventana deslizante semántica con overlap (fase 2) |
| `paragraph` | Un chunk por párrafo original (respeta los saltos de párrafo del autor) |
| `paragraph_overlap` | Híbrida: agrupa párrafos consecutivos en ventanas de `chunk_size` tokens con `overlap_size` de solapamiento, sin partir nunca un párrafo |

---

## 5. ¿Dónde va el repositorio de documentos?

Crea una carpeta `data/corpus/` en la raíz del proyecto (o cualquier ruta)
y coloca ahí **todos tus archivos, en cualquier profundidad de carpetas**:
el pipeline los recorre recursivamente (`rglob`).

```
data/
 └── corpus/
      ├── fenomeno_1/
      │    ├── informe_tecnico/          # carpetas anidadas: OK
      │    │    └── sismo_andino.md
      │    └── articulos/
      │         └── reporte.pdf
      ├── fenomeno_2/
      │    └── noticias/
      │         └── inundaciones.json
      └── fenomeno_3/
           └── registro.csv
```

**Fenómeno (1, 2 o 3):** se asigna de tres formas (en orden de prioridad):

1. Por **nombre de carpeta** (inmediata o ancestral) que aparezca en el JSON
   de mapeo (`--fenomenos`).
2. Por **patrón en el nombre del archivo** que aparezca en el JSON.
3. Por defecto: `1` (o el que indiques).

Ejemplo `data/fenomenos.json`:

```json
{
  "fenomeno_1": 1,
  "fenomeno_2": 2,
  "fenomeno_3": 3,
  "sismo": 1,
  "inunda": 2
}
```

## 6. Ejecución

```bash
# Todo el corpus (recorre carpetas anidadas)
python -m src.run_ingestion --corpus data/corpus --fenomenos data/fenomenos.json

# Solo ciertas extensiones
python -m src.run_ingestion --corpus data/corpus --extensiones pdf md json

# Solo los primeros 50 archivos (prueba rápida)
python -m src.run_ingestion --corpus data/corpus --limite 50

# Logging detallado
python -m src.run_ingestion --corpus data/corpus --verbose
```

Un archivo corrupto **no detiene el batch**: se loguea el error y se continúa.
Al final se imprime un resumen con `ok` / `error` y total de fragmentos.

### Verificar en MongoDB

```bash
docker exec -it mongodb mongosh -u admin -p admin --authenticationDatabase admin
use rag_corpus
db.chunks.countDocuments()
db.chunks.find({}, {doc_id:1, chunk_id:1, posicion:1, num_tokens:1}).sort({chunk_id:1})
db.chunks.getIndexes()   # uq_chunk_id (único), idx_doc_id, idx_fenomeno
```

---

### `metadata.json`: la fuente de verdad de los chunks

**Todas las etapas posteriores (embeddings, índice FAISS y grafo de
conocimiento) leen exactamente los mismos fragmentos, y salen de un único
archivo: `metadata.json`.** Es una lista JSON con upsert idempotente por
`chunk_id`, escritura atómica y orden determinista `(doc_id, posicion,
chunk_id)`, así que dos corridas producen el mismo artefacto.

Cada objeto lleva los ocho campos **obligatorios** de la Tabla 1 más los
**recomendados** (los primeros son el contrato de entrega; los segundos los
consumen el post-filtrado de recuperación —`idioma`, `fecha_publicacion`— y la
caché de embeddings —`hash_texto`—):

```json
{
  "doc_id": "fenomeno_1/informe.md",
  "chunk_id": "fenomeno_1/informe.md__chunk_00000",
  "fuente": "informe.md",
  "formato": "md",
  "fenomeno": 1,
  "posicion": 0,
  "num_tokens": 123,
  "texto": "Texto original del fragmento.",
  "idioma": "es",
  "titulo_documento": "Informe técnico",
  "fecha_publicacion": null,
  "chunking_strategy": "paragraph_overlap",
  "seccion": "Introducción",
  "overlap_con": null,
  "hash_texto": "9c2799c2...",
  "created_at": "2026-08-10T23:19:24+00:00"
}
```

Se genera de dos maneras equivalentes:

```bash
# a) Directamente al ingerir (CHUNK_REPOSITORY=json, el modo por defecto)
python -m src.run_ingestion --corpus CORPUS_CODEFEST_AD_ASTRA_2026 --fenomenos data/fenomenos.json

# b) Exportando unos chunks que ya estaban en MongoDB (sin re-ingerir el corpus)
python -m src.persistence.run_export_metadata
```

La exportación lee y escribe en *streaming* (`MongoChunkRepository.iter_all` →
`JsonChunkRepository.write_all`): el corpus completo son ~356 000 fragmentos y
~450 MB de JSON, y nunca se materializan a la vez el texto crudo del archivo y
su versión parseada. La lectura (`iter_chunks`) también es incremental.

Configura otra ruta con `CHUNKS_JSON_PATH`. Para seguir usando MongoDB como
repositorio de chunks, define `CHUNK_REPOSITORY=mongo`. `metadata.json` está en
`.gitignore` (~450 MB): es un artefacto regenerable, no se versiona.

> No confundir `metadata.json` (fuente de verdad, todos los chunks) con
> `base_vectorial/encoder_<n>/metadata.jsonl` (artefacto de entrega de la
> Sección 5.3, una línea por vector del índice y solo los 8 campos
> obligatorios).

## 7. Tests

```bash
python -m pytest tests -v
```

Cubren: `ExtractorFactory` (+ extensión por decorador), extractores por
formato con fixtures, `HybridChunkingStrategy` (secciones pequeñas, sección
mayor a la ventana, oración que cruza el límite, unidades atómicas CSV/PBF,
metadata), estrategias puras y por párrafo (`paragraph` respeta los saltos de
párrafo; `paragraph_overlap` ventana deslizante sobre párrafos con solapamiento)
y `ChunkValidator` (duros y blandos).

---

## 8. Notas técnicas

- **Tokens**: se cuentan con el tokenizador WordPiece de BERT (`transformers.AutoTokenizer`,
  modelo `google-bert/bert-base-multilingual-cased`, configurable), nunca con
  `len(texto.split())`. Límite del encoder: `MAX_TOKENS` (512).
- **Completitud lingüística**: los cortes retroceden al final de la última
  oración completa (spacy si está instalado; regex multilingüe como respaldo).
- **Metadata obligatoria por fragmento** (Tabla 1): `doc_id`, `chunk_id`,
  `fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`.
  Recomendados: `idioma`, `titulo_documento`, `fecha_publicacion`,
  `chunking_strategy`, `seccion`, `overlap_con`, `hash_texto`, `created_at`.
- **Idempotencia**: upsert por `chunk_id` — reprocesar un documento actualiza
  sin duplicar.
- **Validación**: duras (campos, tipos, `num_tokens ≤ 512`, posiciones
  crecientes sin huecos, `chunk_id` único) → rechazo con log; blandas
  (puntuación terminal) → `validation_warnings` y se guarda.

---

## 9. Codificación semántica (embeddings) — Sección 4

Toma los chunks de `metadata.json` (§6) y genera sus vectores de embedding
con uno o varios encoders **HuggingFace** intercambiables, vía el patrón
**Strategy**. Solo se admiten arquitecturas **encoder** (BERT y derivados);
las arquitecturas decoder (GPT, LLaMA, Gemini, Claude...) están prohibidas
para esta etapa.

> **TODO (punto abierto, confirmar con organizadores ADL antes de la entrega
> final)**: `e5-multilingual-base`/`e5-multilingual-small` están basados en
> XLM-RoBERTa, no en BERT estrictamente, aunque comparten la misma familia
> de arquitectura encoder bidireccional y la Sección 4.2 del PDF solo
> prohíbe decoders (no exige BERT puro). Falta confirmar si XLM-R cuenta
> como "derivado de BERT" bajo esta interpretación.

```
src/encoders/       # EncoderStrategy (ABC) + checkpoints BERT (HF) + Factory + Orchestrator
src/embeddings/     # EmbeddingConfig, EmbeddingCache, EmbeddingWriter, run_embedding.py (CLI)
tests/test_encoders/
```

### 9.1 Los 6 criterios de selección (valores verificados en las model cards de HuggingFace)

| Encoder registrado | Modelo HF | Multilingüe ES/EN/PT | Dim | Max tokens | Licencia | MTEB Retrieval (avg) | Perfil |
|---|---|---|---|---|---|---|---|
| `e5-multilingual-base` | `intfloat/multilingual-e5-base` | Sí (~100 idiomas) | 768 | 512 | MIT | 62.3 (MIRACL nDCG@10) | Precisión semántica, alta dimensión |
| `e5-multilingual-small` | `intfloat/multilingual-e5-small` | Sí (~100 idiomas) | 384 | 512 | MIT | 60.8 (MIRACL nDCG@10) | Eficiencia, baja dimensión |
| `bert-multilingual` | `google-bert/bert-base-multilingual-cased` | Sí (104 idiomas) | 768 | 512 | Apache-2.0 | n/d (BERT sin fine-tuning de embeddings) | Espacio vectorial BERT puro, diversidad para la fusión RRF |
| `bert-multilingual-uncased` | `google-bert/bert-base-multilingual-uncased` | Sí (102 idiomas) | 768 | 512 | Apache-2.0 | n/d | Variante uncased, corpus con ruido de capitalización |
| `bert-large` | `google-bert/bert-large-uncased` | No (solo EN, complementario) | 1024 | 512 | Apache-2.0 | n/d | Mayor capacidad (24 capas), solo inglés |
| `bert-tiny` | `prajjwal1/bert-tiny` | No (solo EN, complementario) | 128 | 512 | MIT | n/d | Eficiencia/baja latencia |
| `bert-es` | `dccuchile/bert-base-spanish-wwm-cased` (BETO) | No (solo ES, complementario) | 768 | 512 | CC-BY-4.0 | n/d | Monolingüe español de alta fidelidad |
| `bert-en` | `google-bert/bert-base-cased` | No (solo EN, complementario) | 768 | 512 | Apache-2.0 | n/d | Monolingüe inglés de alta fidelidad |
| `bert-pt` | `neuralmind/bert-base-portuguese-cased` (BERTimbau) | No (solo PT, complementario) | 768 | 512 | MIT | n/d | Monolingüe portugués de alta fidelidad |

`e5-multilingual-base`/`e5-multilingual-small` sí reportan `mteb_retrieval_score`
porque son checkpoints XLM-RoBERTa afinados con *contrastive learning* para
embeddings de oración y empaquetados como modelo `sentence-transformers`
(`SentenceTransformer(model_id)` directo, con su propio pooling). El resto
son checkpoints BERT reales (arquitectura encoder bidireccional, cargados
vía `transformers.AutoModel` a través de `sentence_transformers.models.Transformer`
+ *mean pooling* manual, nunca afinados para *sentence embeddings*): por eso
ninguno reporta score MTEB-Retrieval oficial — ver Sección 9.6 para el
script que mide su calidad de recuperación sobre el propio corpus.

Solo `bert-multilingual`, `e5-multilingual-base` y `e5-multilingual-small`
están en `ACTIVE_ENCODERS` para esta entrega (cubren es/en/pt de forma nativa
sin necesitar enrutamiento por idioma); el resto de encoders monolingües/
complementarios (`bert-large`, `bert-tiny`, `bert-es`, `bert-en`, `bert-pt`,
`bert-multilingual-uncased`) siguen registrados en el código como extensión
futura por idioma (Opción A de enrutamiento).

Cada `EncoderStrategy` autodeclara estos criterios vía `to_metadata()`:
`supported_languages`, `embedding_dim`, `max_input_tokens`,
`mteb_retrieval_score` + `benchmark_reference`, `license`,
`avg_encode_time_ms_per_batch` (medido en runtime) + `device_preference`.

### 9.2 Patrón Strategy + Factory (Information Expert)

Las reglas de negocio de la Sección 6 están centralizadas en
`EncoderStrategy` como **Information Expert**: cada estrategia es quien
conoce su propia licencia, idiomas y límite de tokens, así que es ella
quien decide si los cumple — el Factory y el Orquestador solo coordinan,
no reevalúan la regla.

- `EncoderStrategy` (ABC): `load()` perezoso, `encode()` (normaliza a norma
  unitaria, resuelve prefijos `query:`/`passage:` internamente si el modelo
  lo requiere), `to_metadata()`.
  - `cubre_idiomas_minimos()` / `licencia_permitida()`: autoevalúan las
    reglas de registro/licencia sobre sus propios atributos declarados.
  - `contar_tokens()` / `excede_limite()` / `ajustar_a_limite()`: cuentan
    tokens con el tokenizador propio del modelo cargado y truncan por
    oraciones completas (o devuelven `None` si no es posible truncar).
- `EncoderFactory`: registro por decorador (`@EncoderFactory.register("bert-multilingual")`);
  llama a `clase.cubre_idiomas_minimos()` en **registro** y a
  `clase.licencia_permitida()` en **instanciación** (`create()`), lanzando
  `LicenseNotAllowedError` salvo `allow_unlisted_license=True`.
- `EncoderOrchestrator`: corre 1..N estrategias sobre el mismo lote de
  chunks; delega en `estrategia.ajustar_a_limite()` el truncado/exclusión
  por chunk y devuelve un `EncoderRunResult` por encoder (que se
  autovalida al construirse: dimensión y longitud consistentes).

### 9.3 Persistencia intermedia

- `EmbeddingWriter`: escribe `base_vectorial/encoder_<nombre>/vectors.npy`
  (n×d) + `chunk_ids.jsonl` (orden ordinal → `chunk_id`) + `metadata_criterios.json`.
- `EmbeddingCache`: evita recomputar (MongoDB, colección `embeddings_cache`,
  clave `chunk_id` + `encoder_name` + `hash_texto`).
- `encoders_procesados: list[str]` se actualiza por chunk en la colección
  `chunks`, para reanudar procesos interrumpidos.

### 9.4 Configuración (`.env`)

```
CHUNK_REPOSITORY=json
CHUNKS_JSON_PATH=metadata.json
ACTIVE_ENCODERS=bert-multilingual,e5-multilingual-base,e5-multilingual-small
EMBEDDING_BATCH_SIZE=32
# Overrides opcionales por encoder (VRAM limitada): bert-large=8,e5-multilingual-small=64
EMBEDDING_BATCH_SIZE_OVERRIDES=
EMBEDDING_DEVICE=auto
EMBEDDING_OUTPUT_DIR=base_vectorial
MONGO_COLLECTION_EMBEDDINGS_CACHE=embeddings_cache
```

Los chunks vienen de `metadata.json`; MongoDB solo guarda los **vectores**
(`embeddings`) y la **caché** (`embeddings_cache`). Cada encoder tiene su
propia caché, así que `EncoderOrchestrator.run_encoder(nombre, lote)` codifica
únicamente con el encoder cuyo lote pendiente se está procesando (usar
`run()` codificaría con los N encoders y descartaría N-1 resultados).

### 9.5 Ejecución

```bash
pip install -r requirements.txt   # incluye sentence-transformers + torch
python -m src.embeddings.run_embedding --limite 500
```

No implementa (queda para el prompt de recuperación): fusión de rankings
multi-encoder (RRF/CombSUM/CombMNZ, Sección 8).

---

## 10. Persistencia de vectores en MongoDB + índices FAISS por encoder — Sección 5

Complementa la Sección 4: los vectores dejan de ser solo una caché plana en
disco (`vectors.npy`) y pasan a tener a **MongoDB como fuente de verdad**
(colección `embeddings`, separada de `chunks`), desde donde se construyen y
exportan los índices **FAISS**, uno independiente por cada encoder activo.

```
src/vectorstore/
 ├── models.py                  # EmbeddingRecord (pydantic)
 ├── vector_repository.py       # VectorRepository (ABC) + MongoVectorRepository
 ├── index_builder_base.py      # FaissIndexBuilderStrategy (ABC) + IndexBuildConfig
 ├── flat_ip_strategy.py        # IndexFlatIP — exacto, sin entrenamiento (DEFAULT)
 ├── ivf_flat_strategy.py       # IndexIVFFlat — aproximado, requiere entrenamiento
 ├── hnsw_strategy.py           # IndexHNSWFlat — sin entrenamiento, prioriza latencia
 ├── index_builder_factory.py   # IndexBuilderFactory (registro + resolución "auto")
 ├── faiss_index_manager.py     # FaissIndexManager — modo operativo incremental (IndexIDMap)
 ├── export_delivery.py         # DeliveryExporter — modo de exportación estricta
 └── run_export_delivery.py     # CLI del modo de exportación
tests/test_vectorstore/
```

### 10.1 Modelo de datos (`embeddings`)

Un documento por `(chunk_id, encoder_name)`, con el vector empaquetado como
`bson.Binary` (`float32.tobytes()`), no como `list[float]` — más compacto y
más rápido de leer/escribir a escala. Índice único compuesto sobre
`(chunk_id, encoder_name)` (upsert idempotente) + índice sobre `encoder_name`
(para extraer eficientemente "todos los vectores de este encoder").

`MongoVectorRepository` implementa `save_many()` (upsert), `find_by_encoder()`
(cursor en streaming — nunca carga todo el corpus en memoria), `find_missing()`,
`count_by_encoder()` y `delete_by_chunk_id()`.

### 10.2 Tipo de índice FAISS — Strategy + Factory

Igual que el encoder, el **tipo de índice** varía según el volumen del
corpus y el objetivo (exactitud vs. velocidad), así que se modela con el
mismo patrón Strategy + Factory con registro por decorador:

- `FlatIPIndexStrategy` (`flat_ip`): `IndexFlatIP`, exacto, sin
  entrenamiento. **Es el default recomendado** — para el volumen de
  documentos esperado en este reto, un índice plano es suficiente y
  garantiza resultados exactos (similitud coseno vía producto interno,
  ya que `EncoderStrategy.encode()` normaliza los vectores a norma unitaria).
- `IVFFlatIndexStrategy` (`ivf_flat`): `IndexIVFFlat`, requiere
  entrenamiento (k-means) antes de poblarlo. Pensado para corpus grandes.
- `HNSWIndexStrategy` (`hnsw`): `IndexHNSWFlat`, sin entrenamiento, prioriza
  latencia de consulta sobre uso de memoria.
- `IndexBuilderFactory.resolve("auto", n_vectors, config)`: si
  `FAISS_INDEX_TYPE=auto`, decide entre `flat_ip` e `ivf_flat` comparando
  `n_vectors` contra `FAISS_IVF_AUTO_THRESHOLD`, dejando la decisión
  trazada en el log.

### 10.3 Dos modos de operación

- **Modo operativo (`FaissIndexManager`)**: incremental, para
  desarrollo/pruebas iterativas. Envuelve el índice en `faiss.IndexIDMap`
  con IDs derivados determinísticamente del `chunk_id`
  (`sha1(chunk_id)[:15]` → `int64`), permitiendo `remove_ids` +
  `add_with_ids` cuando un chunk se reprocesa, sin reconstruir todo el
  índice. Se persiste en `WORKING_INDEX_DIR` y actualiza
  `faiss_internal_id` de vuelta en `embeddings` (MongoDB actúa como el
  almacén de metadata que exige la especificación).
- **Modo de exportación (`DeliveryExporter` / `export_delivery.py`)**:
  estricto, es el que genera el artefacto de entrega. Inserta los vectores
  con `index.add()` **secuencial** (sin `IndexIDMap`) en un orden
  determinístico (`ORDER BY doc_id, posicion` — nunca `_id` de Mongo, que
  no es reproducible entre corridas), de forma que el orden de líneas de
  `metadata.jsonl` coincide exactamente con los IDs internos (`0..n-1`)
  que FAISS asigna en la indexación, tal como exige la Sección 5.3.
  Antes de exportar, valida que no falte ningún embedding para los chunks
  activos y que todos los vectores del encoder tengan la misma dimensión
  (aborta con la lista de `chunk_id` faltantes si no). Al final hace un
  test de humo (`faiss.read_index()` + `index.d`/`ntotal` esperados) y
  registra un checksum SHA-256 en `build_log.jsonl` para verificar
  reproducibilidad entre corridas.

### 10.4 Configuración adicional (`.env`)

```
MONGO_COLLECTION_EMBEDDINGS=embeddings
FAISS_INDEX_TYPE=flat_ip                 # flat_ip | ivf_flat | hnsw | auto
FAISS_IVF_NLIST=100
FAISS_IVF_NPROBE=10
FAISS_HNSW_M=32
FAISS_IVF_AUTO_THRESHOLD=50000
WORKING_INDEX_DIR=working_index
DELIVERY_OUTPUT_DIR=base_vectorial
```

### 10.5 Ejecución

```bash
pip install -r requirements.txt   # incluye faiss-cpu
python -m src.embeddings.run_embedding --limite 500       # calcula y persiste vectores en Mongo
python -m src.vectorstore.run_export_delivery              # exporta index.faiss + metadata.jsonl por encoder
```

---

## 11. Grafo de conocimiento — Sección 7

Construye G = (E, R, T) con T ⊆ E × R × E a partir de los MISMOS chunks que
alimentan el índice FAISS, y lo expone como un canal más de recuperación que
se fusiona con los canales vectoriales (Sección 8.5). Cada tripleta conserva
`doc_id`/`chunk_id` de origen, así que toda arista es auditable contra el
texto que la sustenta.

```
src/knowledge_graph/
 ├── models.py                    # Entity, Relation, Tripleta, ScoredChunk
 ├── extract/                     # NER (nodos) + RE (aristas) + Factory
 ├── graph/                       # GraphBuilder, GraphQuery, GraphML
 ├── retrieval/                   # GraphIndexAdapter + fusión RRF/CombSUM/CombMNZ
 ├── service.py                   # KnowledgeGraphService (Facade)
 ├── re_cli_options.py            # opciones RE compartidas por los 3 CLI
 └── run_build_graph.py           # CLI: exporta grafo.graphml
tests/test_knowledge_graph/
```

### 11.1 Modelos usados, licencias y no-generatividad

El reto exige licencias permisivas (Apache-2.0 / MIT) y prohíbe modelos
generativos en indexación y recuperación. Ambas etapas cumplen:

| Etapa | Estrategia | Modelo | Licencia | ¿Generativo? |
|---|---|---|---|---|
| **Nodos (NER)** | `regex-gazetteer` | ninguno — gazetteer del dominio + patrón de nombres propios, código del proyecto | n/a (sin checkpoint de terceros) | No: determinista y simbólico |
| **Aristas (RE)** | `nli-zero-shot` *(default)* | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | **MIT** | **No**: encoder DeBERTa-v3 + cabecera de clasificación de 3 etiquetas |
| **Aristas (RE)** | `coocurrencia-oracional` | ninguno — co-ocurrencia oracional + patrones verbales | n/a | No: determinista y simbólico |

Por qué el clasificador NLI **no** es generativo: no tiene decoder ni cabecera
de modelado de lenguaje. Su salida son tres logits
(contradicción / neutral / entailment) por par (premisa, hipótesis); no
produce ni un solo token de texto. Los sujetos y objetos salen siempre del NER
sobre el texto real, y el tipo de relación se ELIGE de un vocabulario cerrado
(`RelationType`): el modelo nunca inventa entidades ni relaciones nuevas.

> **Modelo retirado**: hasta esta versión el default era
> `Babelscape/mrebel-large` (mREBEL). Se eliminó del proyecto por incumplir
> las dos reglas: su licencia es **CC BY-NC-SA 4.0** (no comercial, no es
> Apache-2.0/MIT) y es un **seq2seq generativo** (mBART) que produce las
> tripletas token a token. Un test de regresión
> (`test_solo_hay_estrategias_re_con_licencia_permisiva`) impide que vuelva a
> registrarse una estrategia fuera de la lista permitida.

### 11.2 Cómo se construyen nodos y aristas

1. **NER** (nodos): sobre cada chunk, matching por spans del gazetteer
   (ganan las entidades de más palabras) + nombres propios multi-palabra. El
   id del nodo es canónico y normalizado, y un mapa de **alias es↔en** hace
   que `space debris` y `basura espacial` resuelvan al MISMO nodo — clave
   porque el corpus es mayoritariamente EN y las consultas del reto son ES.
2. **Candidateo** (simbólico, barato): pares de entidades que co-ocurren en
   la misma oración.
3. **Tipado** (el modelo): la oración es la premisa y
   `"<sujeto> <verbo> <objeto>"` la hipótesis, una por `RelationType`. Gana
   el tipo con mayor probabilidad de entailment; por debajo de
   `UMBRAL_TIPADO` la arista queda como `COOCURRENCIA`.
4. **Tripletas + GraphML**: `(sujeto, relación, objeto, doc_id, chunk_id,
   evidencia, confianza)` → `grafo.graphml`, cargable con NetworkX.

### 11.3 Presupuesto de inferencia

El canal cuesta `pares_dirigidos × tipos × variantes` forwards, así que el
candidateo es simbólico y el modelo solo tipa candidatos, con topes explícitos
y caché por (oración, par) — el corpus repite mucho boilerplate. Todos los
pares de una oración se clasifican en **un solo forward**.

| Opción | Default | Efecto |
|---|---|---|
| `--variantes-plantilla` | 1 (ES canónica) | Formulaciones por tipo; `0` = todas |
| `--max-pares-oracion` | 4 | Tope de pares por oración |
| `--max-pares-chunk` | 4 | Tope de pares por chunk |
| `--re-batch-size` | 64 | Lote del forward |
| `--sin-fp16` | (fp16 activo) | Media precisión en GPU: ~2× y mitad de VRAM |

Los defaults están calibrados para **cubrir el corpus completo** (~356 000
chunks) en un tiempo razonable: 10 hipótesis por par dirigido y como mucho 4
pares por chunk. La decisión de diseño es que, para recuperación, pesa más la
cobertura (que todo chunk tenga nodos y aristas) que la riqueza del tipado —
una arista `COOCURRENCIA` recupera el chunk igual que una tipada. Con más
cómputo disponible, `--variantes-plantilla 2 --max-pares-chunk 12` da un
tipado bastante más rico a ~6× el coste.

El recorte de pares es determinista (orden de detección de las entidades), así
que dos corridas producen el mismo grafo.

**Mide antes de comprometer horas de GPU.** El coste depende del hardware y de
cuántas entidades traiga cada texto, así que no lo estimes a ojo:

```bash
python -m src.knowledge_graph.run_build_graph --estimar 200
# -> Muestra: 200 chunks en 41.3s (207 ms/chunk) -> 512 entidades, 733 tripletas.
#    Proyección a 356453 chunks: ~20.5 h en este hardware.
```

Cronometra esos N chunks reales con la MISMA configuración con la que se
construiría el grafo y extrapola. Es una cota superior razonable: la caché
acierta más cuanto mayor es el corpus.

### 11.4 Ejecución

```bash
# 0. ¿Cuánto va a costar en esta máquina?
python -m src.knowledge_graph.run_build_graph --estimar 200

# 1. Construir y exportar el grafo (requiere los artefactos de la Sección 5)
python -m src.knowledge_graph.run_build_graph --output grafo.graphml

# Sin GPU o sin tiempo: RE simbólica, minutos en vez de horas, sin modelo
python -m src.knowledge_graph.run_build_graph --re coocurrencia-oracional

# 2. Recuperación híbrida FAISS + grafo, con la traza del camino por el grafo
python -m src.knowledge_graph.run_hybrid_retrieval --queries consultas.jsonl \
    --k 10 --cargar-grafo grafo.graphml --output-caminos logs/caminos_grafo.json

# 3. Entrega con el grafo como canal adicional (esquema Sección 9.3.1)
python -m src.knowledge_graph.run_generador_hibrido --cargar-grafo grafo.graphml
```

> Pasa siempre `--cargar-grafo grafo.graphml` en los pasos 2 y 3: sin él,
> ambos CLI **reconstruyen el grafo desde cero** (y con él el coste del
> modelo) en cada invocación.

`KnowledgeGraphService.resumen()` incluye la ficha del modelo de relaciones
(`model_id`, arquitectura, licencia, `generativo: no`) para que la auditoría
de licencias quede trazada en el log de cada construcción.

---

## 12. Orden de ejecución de extremo a extremo

```bash
# 1. Chunks -> metadata.json (fuente de verdad)
python -m src.run_ingestion --corpus CORPUS_CODEFEST_AD_ASTRA_2026 --fenomenos data/fenomenos.json
#    (o, si los chunks ya estaban en MongoDB: python -m src.persistence.run_export_metadata)

# 2. Embeddings de esos chunks -> MongoDB (colección embeddings)
python -m src.embeddings.run_embedding

# 3. Índices de entrega por encoder -> base_vectorial/encoder_<n>/
python -m src.vectorstore.run_export_delivery

# 4. Grafo de conocimiento -> grafo.graphml
python -m src.knowledge_graph.run_build_graph --output grafo.graphml

# 5. Resultados (Sección 9.3.1)
python generador.py                                        # solo FAISS
python -m src.knowledge_graph.run_generador_hibrido \
    --cargar-grafo grafo.graphml                           # FAISS + grafo
```
