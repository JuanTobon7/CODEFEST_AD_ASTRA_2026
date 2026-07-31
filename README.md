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