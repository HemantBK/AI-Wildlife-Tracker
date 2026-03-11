# Wildlife Tracker AI

**A production-grade RAG system that identifies 500+ Indian wildlife species from text descriptions or photos, powered by hybrid retrieval, cross-encoder reranking, ONNX-optimized inference, and end-to-end Langfuse observability.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Groq](https://img.shields.io/badge/Groq-LLM_API-black)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

A user describes a wildlife sighting in plain English or uploads a photo. The system identifies the species with confidence scoring, cited evidence from a curated knowledge base, habitat matching, and conservation status.

```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{"query": "large orange striped cat in the forest", "location": "Madhya Pradesh"}'
```

```json
{
  "species_name": "Bengal Tiger",
  "scientific_name": "Panthera tigris tigris",
  "confidence": 0.95,
  "reasoning": "The description matches a Bengal Tiger based on orange coat with black stripes...",
  "key_features_matched": ["orange coat", "black stripes", "large feline", "forest habitat"],
  "habitat_match": "Tropical and subtropical moist broadleaf forests of central India",
  "conservation_status": "Endangered (IUCN Red List)",
  "cited_sources": ["bengal_tiger_habitat_chunk_01", "bengal_tiger_physical_chunk_02"],
  "total_latency_seconds": 4.2
}
```

**Target users:** Wildlife photographers, forest rangers, citizen scientists, eco-tourists, researchers.

---

## System Architecture

```
                          User
                     (text or photo)
                           |
              +------------+------------+
              |                         |
         Text Query              Photo Upload
              |                         |
              |                  Groq Vision API
              |               (llama-3.2-90b-vision)
              |                         |
              |                  Natural Language
              |                   Description
              +------------+------------+
                           |
                    FastAPI (:8000)
              Middleware: Logging | Rate Limit | Timeout
                           |
     +---------------------+----------------------+
     |              RAG PIPELINE                   |
     |                                             |
     |  [1] Query Preprocessing                    |
     |      Location + season extraction           |
     |      Synonym expansion                      |
     |                                             |
     |  [2] Hybrid Retrieval                       |
     |      ChromaDB (vector) --+                  |
     |      BM25 (keyword) -----+--> RRF Fusion    |
     |                                             |
     |  [3] Cross-Encoder Reranking                |
     |      ms-marco-MiniLM (ONNX-optimized)       |
     |                                             |
     |  [4] Geographic Filtering                   |
     |      Boost/filter by Indian state           |
     |                                             |
     |  [5] LLM Generation                         |
     |      Groq API / Ollama local                |
     |      Structured JSON + Pydantic validation  |
     |                                             |
     +---------------------+----------------------+
                           |
          +----------------+----------------+
          |                |                |
     Langfuse v4      SQLite DB       Alert Engine
     (traces +        (metrics +      (accuracy,
      scores)          feedback)       latency)
          |                |                |
          +--------> Monitoring Dashboard <-+
                     Streamlit (:8502)
```

---

## Key Engineering Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| **Hybrid retrieval (Vector + BM25)** | Vector captures semantics ("large striped cat" ~ tiger). BM25 catches exact terms ("Panthera tigris"). RRF fusion gets best of both. | Slightly more complex than vector-only, but measurably better recall |
| **Cross-encoder reranking** | Bi-encoder search is fast but imprecise. A cross-encoder re-scores the top 15 candidates with full query-document attention. | Adds ~2s latency, but dramatically improves top-5 precision |
| **ONNX Runtime optimization** | Converted both embedder and reranker to ONNX format for 2-3x CPU speedup. No GPU required. | One-time conversion step, but inference is faster on every request |
| **Species-aware chunking** | Generic chunking splits text arbitrarily. Our chunker never crosses species boundaries, so each chunk is about exactly one species. | Chunks vary in size, but the LLM never confuses species from mixed chunks |
| **Vision + RAG hybrid** | Direct vision classification hallucinates. We use vision to describe the image, then feed that description into the RAG pipeline grounded in curated data. | Two-step adds latency, but identifications are knowledge-grounded |
| **Langfuse tracing (no-op pattern)** | Every pipeline step is traced. If Langfuse isn't configured, tracing silently does nothing (no crashes, no overhead). | Zero-cost when disabled, full observability when enabled |
| **Geographic filtering** | India has 28+ states with distinct fauna. Filtering by region prevents "snow leopard in Kerala" false positives. | Requires location metadata in chunks, but significantly improves accuracy |
| **Lazy pipeline loading** | The RAG pipeline is heavy (~10s to load). Loading it lazily means `/health` responds instantly, and the pipeline loads only when the first `/identify` request comes in. | First request is slower, but server starts immediately |
| **Dual inference backends** | Groq API for production (fast, free tier). Ollama for offline development. Switchable via one env variable. | Two code paths, but no vendor lock-in |
| **Evaluation gating in CI** | PRs to main must pass accuracy thresholds. Prevents shipping regressions. | Slower CI, but quality never silently degrades |

---

## Performance

| Metric | Value |
|--------|-------|
| Species coverage | 500+ Indian birds, mammals, reptiles |
| Identification accuracy | ~85% on golden dataset (100+ QA pairs) |
| P95 latency (Groq API) | ~12s (with ONNX), ~15s (without) |
| ONNX speedup | 19% faster inference (embedder + reranker) |
| Knowledge base | 3 sources (GBIF, Wikipedia, iNaturalist) |
| Chunk count | 200+ species-aware chunks |
| Vector dimensions | 384 (all-MiniLM-L6-v2) |

---

## Tech Stack

**All free and open source. No paid services required.**

| Layer | Technology | Role |
|-------|-----------|------|
| **LLM** | Groq API (llama-3.3-70b) / Ollama (llama3.2:3b) | Species identification from retrieved context |
| **Vision** | Groq Vision (llama-3.2-90b-vision) | Photo analysis for multimodal identification |
| **Embeddings** | all-MiniLM-L6-v2 (ONNX) | 384-dim sentence embeddings for vector search |
| **Reranker** | ms-marco-MiniLM-L-6-v2 (ONNX) | Cross-encoder reranking of search candidates |
| **Vector DB** | ChromaDB (persistent) | Cosine similarity search over chunk embeddings |
| **Keyword Search** | rank-bm25 | BM25 scoring for exact term matching |
| **Fusion** | Reciprocal Rank Fusion | Score-agnostic merging of vector + BM25 results |
| **API** | FastAPI | REST endpoints with auto-generated OpenAPI docs |
| **Frontend** | Streamlit | Query UI (text + photo) and monitoring dashboard |
| **Observability** | Langfuse v4 (cloud) | Distributed tracing with per-step latency and scoring |
| **Metrics** | SQLite | Request logging, feedback, and aggregated metrics |
| **Optimization** | ONNX Runtime | 2-3x CPU inference speedup (no GPU needed) |
| **Testing** | pytest + RAGAS | Unit tests, integration tests, RAG-specific evaluation |
| **CI/CD** | GitHub Actions | Lint, test, evaluation gate on every PR |
| **Container** | Docker Compose | One-command deployment (API + Frontend + Dashboard) |
| **Code Quality** | Ruff + pre-commit | Linting, formatting, secret detection |

---

## Data Pipeline

```
GBIF API ----+
(occurrences) |
              |     Cleaner      Chunker       Validator     Embedder
iNaturalist --+--> (merge +  --> (species-  --> (schema   --> ChromaDB
(observations)|     dedup)       aware,         check,       + BM25
              |                  500 tok)       coverage)
Wikipedia ----+
(articles)

+ Curated species script (350+ iconic Indian wildlife)
```

**Three data sources** feed into the pipeline:
- **GBIF**: Species occurrence records filtered to India
- **iNaturalist**: Community observation data with species metadata
- **Wikipedia**: Detailed species articles (habitat, behavior, morphology)

**Curated supplement**: A script with 350+ hand-picked species covers iconic Indian wildlife (Bengal Tiger, Indian Elephant, King Cobra, Indian Peafowl, etc.) that automated collection might miss.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/identify` | Identify species from text description |
| `POST` | `/identify/image` | Identify species from uploaded photo |
| `GET` | `/health` | Component health (ChromaDB, BM25, LLM, data) |
| `GET` | `/metrics` | Latency, accuracy, request counts, top species |
| `POST` | `/feedback` | Submit correctness feedback on an identification |
| `GET` | `/alerts` | Active alerts (accuracy drop, latency spike) |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Text Identification

```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "small bright blue bird with a long beak diving into a stream",
    "location": "Kerala",
    "season": "monsoon"
  }'
```

### Example: Photo Identification

```bash
curl -X POST http://localhost:8000/identify/image \
  -F "image=@wildlife_photo.jpg" \
  -F "location=Assam"
```

---

## Observability with Langfuse

Every `/identify` request creates a full trace in Langfuse:

```
wildlife_identify (12.3s)
  |-- query_preprocessing     (15ms)   input/output: query expansion, location
  |-- hybrid_search           (1.2s)   vector + BM25 candidate count
  |-- reranking               (2.1s)   cross-encoder scores, final ranking
  |-- geographic_filter       (1ms)    region matching
  |-- llm_generation          (9.0s)   model, prompt, response, tokens
```

**Automatic scores per trace:**
- `confidence` (0.0-1.0) — model's self-assessed confidence
- `identified` (1 or 0) — did it identify a species or decline?
- `latency` — total pipeline wall-clock time

Setup: Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com), add keys to `.env`, restart the API. Tracing activates automatically. When Langfuse is not configured, all tracing is a silent no-op (zero overhead).

---

## Evaluation Framework

**Golden dataset**: 100+ query-answer pairs across 4 difficulty levels.

| Difficulty | Examples | What It Tests |
|------------|----------|---------------|
| Easy (40%) | "What is a Bengal Tiger?" | Direct species lookup |
| Medium (30%) | "Large grey animal with one horn near a river in Assam" | Feature + location matching |
| Hard (15%) | "Small brown bird in the underbrush" (could be many species) | Disambiguation between similar species |
| Trick (15%) | "Polar bear in Tamil Nadu" | Geographic impossibilities, out-of-scope queries |

**CI/CD quality gate** (runs on every PR to `main`):

| Metric | Threshold |
|--------|-----------|
| Answer correctness | >= 75% |
| Geographic accuracy | >= 90% |
| Refusal accuracy (trick queries) | >= 80% |
| P95 latency | <= 15s |

---

## Project Structure

```
wildlife-tracker/
|-- src/
|   |-- api/                 # FastAPI server, routes, models, middleware
|   |-- rag/                 # Pipeline, generator, reranker, vision module
|   |-- retrieval/           # Hybrid search, embedder, BM25, query expansion
|   |-- ingestion/           # GBIF, iNaturalist, Wikipedia collectors
|   |-- preprocessing/       # Cleaner, chunker, validator
|   |-- monitoring/          # Langfuse tracing, logging, alerts, metrics
|   |-- evaluation/          # Golden dataset, evaluator, RAGAS, benchmarks
|   +-- frontend/            # Streamlit UI (query + photo) and dashboard
|
|-- config/
|   |-- retrieval.yaml       # Embedding, search, reranker, LLM configuration
|   +-- species_list.yaml    # Target species taxonomy
|
|-- prompts/
|   +-- v1.yaml              # Versioned LLM prompt template
|
|-- scripts/
|   |-- setup.py             # First-time environment validation
|   |-- demo.py              # Interactive demo queries
|   |-- convert_to_onnx.py   # ONNX model conversion
|   +-- add_essential_species.py  # Curated 350+ species bootstrap
|
|-- tests/                   # Unit + integration + e2e tests
|-- models/onnx/             # ONNX-optimized model artifacts
|-- data/                    # Raw, processed, chunks, vector DB, BM25 index
|-- docker-compose.yml       # Full stack orchestration
|-- Makefile                 # 120+ automation targets
|-- .github/workflows/       # CI/CD pipeline
+-- .env.example             # Environment variable template
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
git clone <repo-url>
cd wildlife-tracker
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add your GROQ_API_KEY
```

### Build the Knowledge Base

```bash
# Option 1: Full pipeline (collect from APIs + process)
make data-pipeline
make build-indexes

# Option 2: Quick start with curated species
python scripts/add_essential_species.py
make build-indexes
```

### Run

```bash
# Terminal 1: API server
make api

# Terminal 2: Web UI
make frontend

# Terminal 3: Monitoring dashboard (optional)
make dashboard
```

- **Frontend**: http://localhost:8501
- **API docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8502

### Docker

```bash
docker-compose up --build
```

---

## Configuration

All configuration via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Free API key from console.groq.com |
| `INFERENCE_MODE` | `groq` | `groq` (API) or `local` (Ollama) |
| `USE_ONNX` | `true` | Enable ONNX-optimized embedder + reranker |
| `LANGFUSE_PUBLIC_KEY` | optional | Langfuse tracing (free at cloud.langfuse.com) |
| `LANGFUSE_SECRET_KEY` | optional | Langfuse tracing |
| `RATE_LIMIT_PER_MINUTE` | `10` | API rate limit per IP |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Max request duration |
| `LOG_FORMAT` | `human` | `human` (dev) or `json` (production) |

Retrieval parameters (top-k, fusion weights, reranker threshold) are in `config/retrieval.yaml`.

---

## Make Targets

```bash
# Data
make data-pipeline          # Collect + clean + chunk + validate
make build-indexes          # Build ChromaDB + BM25 indexes
make onnx-convert           # Convert models to ONNX (2-3x speedup)

# Run
make api                    # FastAPI on :8000
make frontend               # Streamlit on :8501
make dashboard              # Monitoring on :8502

# Test
make test                   # All tests
make eval                   # Full evaluation (golden dataset)
make eval-quick             # Quick eval (10 queries, ~2 min)
make eval-ci                # CI gate (fails on threshold breach)

# Code Quality
make lint                   # Ruff linter
make format                 # Auto-format
make ci-local               # Full CI check locally

# Docker
make docker-up              # Start all services
make docker-down            # Stop all services
```

---

## How the RAG Pipeline Works

1. **Query preprocessing** extracts location ("Madhya Pradesh"), season ("monsoon"), and expands synonyms ("big cat" -> "large feline, tiger, panther").

2. **Hybrid retrieval** runs the query against both ChromaDB (semantic vector similarity) and BM25 (keyword matching), then fuses results using Reciprocal Rank Fusion (RRF) to get the best of both approaches.

3. **Cross-encoder reranking** takes the top 15 fused candidates and re-scores each one with a cross-encoder that sees the full query-document pair. This is slower but much more accurate than bi-encoder similarity alone.

4. **Geographic filtering** boosts (or strictly filters) chunks that match the user's stated location, preventing geographically impossible identifications.

5. **LLM generation** feeds the top 5 reranked chunks to the LLM with a structured prompt. The model must respond in a specific JSON schema (validated by Pydantic), cite its sources, and provide a confidence score.

6. **Confidence gating** flags low-confidence responses and declines identification when the knowledge base has insufficient information, rather than guessing.

---

## License

MIT
