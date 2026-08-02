# CODEFEST AD ASTRA 2026 — Pipeline de Ingesta (Etapa 1)

Pipeline modular y extensible para **RAG**: extracción, limpieza, chunking
híbrido (estructural + semántico con overlap), metadata obligatoria y
persistencia en **MongoDB**. Implementa los patrones **Factory Method**
(extractores) y **Strategy** (chunking) con inyección de dependencias real.

---

## 1. Arquitectura del flujo

```mermaid
flowchart LR
    A[Archivo del corpus] --> B[ExtractorFactory.create]
    B --> C[BaseExtractor.extract]
    C --> D[ExtractedDocument<br/>secciones]
    D --> E[TextCleaner.clean]
    E --> F[ChunkingStrategy.chunk<br/>structural | semantic | hybrid]
    F --> G[MetadataBuilder.enrich<br/>campos Tabla 1]
    G --> H[ChunkValidator.validate]
    H -->|válidos| I[MongoChunkRepository.save_many<br/>upsert idempotente]
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
 ├── chunking/       # ChunkingStrategy (ABC) + structural + semantic + hybrid + factory
 ├── cleaning/       # TextCleaner (UTF-8, controles, boilerplate, idioma)
 ├── metadata/       # MetadataBuilder (campos obligatorios Tabla 1)
 ├── validation/     # ChunkValidator (duras → rechazo; blandas → warnings)
 ├── persistence/    # ChunkRepository (ABC) + MongoChunkRepository
 ├── pipeline/       # IngestionPipeline (un archivo) + BatchIngestor (lote) + CorpusService (escaneo)
 ├── support/        # Tokenizer (tiktoken), SentenceSplitter (spacy/regex), utils
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

## 7. Tests

```bash
python -m pytest tests -v
```

Cubren: `ExtractorFactory` (+ extensión por decorador), extractores por
formato con fixtures, `HybridChunkingStrategy` (secciones pequeñas, sección
mayor a la ventana, oración que cruza el límite, unidades atómicas CSV/PBF,
metadata), estrategias puras y `ChunkValidator` (duros y blandos).

---

## 8. Notas técnicas

- **Tokens**: se cuentan con `tiktoken` (modelo `cl100k_base`, configurable),
  nunca con `len(texto.split())`. Límite del encoder: `MAX_TOKENS` (512).
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

Toma los chunks ya persistidos en MongoDB y genera sus vectores de embedding
con uno o varios encoders **HuggingFace** intercambiables, vía el patrón
**Strategy**. Solo se admiten arquitecturas **encoder** (BERT y derivados);
las arquitecturas decoder (GPT, LLaMA, Gemini, Claude...) están prohibidas
para esta etapa.

```
src/encoders/       # EncoderStrategy (ABC) + E5/BGE-M3/LaBSE/MiniLM + Factory + Orchestrator
src/embeddings/     # EmbeddingConfig, EmbeddingCache, EmbeddingWriter, run_embedding.py (CLI)
tests/test_encoders/
```

### 9.1 Los 6 criterios de selección (valores verificados en las model cards de HuggingFace)

| Encoder registrado | Modelo HF | Multilingüe ES/EN/PT | Dim | Max tokens | Licencia | MTEB Retrieval (avg) | Perfil |
|---|---|---|---|---|---|---|---|
| `e5-small` | `intfloat/multilingual-e5-small` | Sí (100+ idiomas) | 384 | 512 | MIT | 46.6 | Eficiencia |
| `e5-base` | `intfloat/multilingual-e5-base` | Sí (100+ idiomas) | 768 | 512 | MIT | 48.9 | Balance precisión/eficiencia |
| `e5-large` | `intfloat/multilingual-e5-large` | Sí (100+ idiomas) | 1024 | 512 | MIT | 51.4 | Alta precisión, mayor costo |
| `bge-m3` | `BAAI/bge-m3` | Sí (100+ idiomas) | 1024 | 8192 | MIT | 48.8 | Chunks largos, denso+disperso+multi-vector |
| `labse` | `sentence-transformers/LaBSE` | Sí (109 idiomas) | 768 | 512 | Apache-2.0 | n/d (alineación cross-lingual, no BEIR) | Fuerte alineación cross-lingual |
| `minilm-light` | `sentence-transformers/distiluse-base-multilingual-cased-v2` | Sí | 512 | 512 | Apache-2.0 | n/d | Ligero, baja latencia, "encoder de eficiencia" |

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
- `EncoderFactory`: registro por decorador (`@EncoderFactory.register("e5-base")`);
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
ACTIVE_ENCODERS=e5-base,minilm-light
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=auto
EMBEDDING_OUTPUT_DIR=base_vectorial
MONGO_COLLECTION_CHUNKS=chunks
MONGO_COLLECTION_EMBEDDINGS_CACHE=embeddings_cache
```

### 9.5 Ejecución

```bash
pip install -r requirements.txt   # incluye sentence-transformers + torch
python -m src.embeddings.run_embedding --limite 500
```

No implementa (queda para prompts posteriores): construcción del índice
FAISS (`IndexFlatIP`, Sección 5) ni fusión de rankings multi-encoder
(RRF/CombSUM/CombMNZ, Sección 8).