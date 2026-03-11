# Changelog

## Phase 8 - Deployment, Documentation, Demo (Week 8)

### Added
- Docker deployment:
  - `Dockerfile` — Multi-stage build for the API (Python 3.11-slim, ~1.5GB image)
  - `Dockerfile.frontend` — Lightweight Streamlit image
  - `docker-compose.yml` — Full stack orchestration (API + Frontend + Dashboard)
  - `.dockerignore` — Excludes notebooks, raw data, tests from images
  - `requirements-docker.txt` — Minimal API dependencies (no eval/notebook packages)
  - `requirements-frontend.txt` — Minimal frontend dependencies (streamlit + requests)
  - Health checks on all containers (API via /health, Streamlit via /_stcore/health)
  - Data directory mounted as volume (indexes persist across restarts)
  - Frontend and dashboard wait for API health before starting
- Deployment scripts:
  - `scripts/setup.py` — First-time environment validator (checks Python, packages, .env, Groq key, data, indexes, Ollama, Docker)
  - `scripts/demo.py` — Runs 7 demo queries (easy/medium/hard/trick) with formatted output and summary stats
- Polished README:
  - 3 Quick Start options (Docker, Local Dev, Quick Test)
  - Full architecture diagram (ASCII)
  - Complete project structure with file descriptions
  - API endpoint table
  - All 40+ Makefile targets documented
  - Configuration reference table
  - Quality gates table
  - Design decisions section
  - Timeline with all 8 phases marked Done
- Makefile targets: `docker-build`, `docker-up`, `docker-down`, `docker-logs`, `docker-restart`, `setup`, `demo`, `clean-docker`

### New Files
- `Dockerfile` — API multi-stage build
- `Dockerfile.frontend` — Streamlit frontend image
- `docker-compose.yml` — Full stack orchestration (3 services)
- `.dockerignore` — Docker build context filter
- `requirements-docker.txt` — API-only dependencies
- `requirements-frontend.txt` — Frontend-only dependencies
- `scripts/setup.py` — Environment setup validator
- `scripts/demo.py` — Demo query runner

## Phase 7 - CI/CD, Testing, Feedback Loop (Week 7)

### Added
- GitHub Actions CI workflow (`.github/workflows/ci.yml`):
  - 4-job pipeline: lint → test → eval-gate → build-check
  - **lint**: Ruff linter + formatter check (fast, no dependencies)
  - **test**: Full pytest suite with coverage report, uploaded as artifact
  - **eval-gate**: Runs quick evaluation with quality gates on PRs to main
  - **build-check**: Verifies all module imports work (catches broken dependencies)
  - Secrets: `GROQ_API_KEY` for eval-gate job
  - Eval gate only runs on PRs targeting main branch
- Pre-commit hooks (`.pre-commit-config.yaml`):
  - Ruff linter with auto-fix + formatter
  - Trailing whitespace, end-of-file fixer, YAML/JSON checks
  - Large file detection (>500KB), merge conflict check
  - Private key detection, no direct commits to main
  - Secret detection with detect-secrets baseline
- `pyproject.toml` — Unified tool configuration:
  - Ruff: line-length 100, select E/W/F/I/N/UP/B/SIM rules
  - pytest: test paths, markers (slow, integration, e2e), filter warnings
  - coverage: source=src, fail-under=50%, exclude patterns
- Feedback loop module (`src/monitoring/feedback_loop.py`):
  - Confusion pair analysis (species A mistaken for B)
  - Per-species accuracy breakdown from user feedback
  - Common failure pattern identification
  - Corrections catalog generator (saved to `data/corrections.json`)
  - Weekly quality report with actionable recommendations
  - CLI: `--catalog` to generate corrections, `--report` for quality summary
- Integration test suite (`tests/test_integration.py`):
  - Full API flow: identify → feedback → metrics verification
  - Middleware chain: request ID propagation, CORS, rate limiting
  - Monitoring integration: metrics → alerts, feedback loop, concurrent access
  - Tracing no-op safety verification
  - Module import chain (all src.* modules importable)
- Makefile targets:
  - `lint`, `lint-fix`, `format`, `format-check` (Ruff)
  - `pre-commit-install`, `pre-commit-run`
  - `test-unit`, `test-integration` (marker-based filtering)
  - `ci-local` (lint + format-check + test in one command)
  - `feedback-report`, `feedback-catalog`

### New Files
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline
- `.pre-commit-config.yaml` — Pre-commit hook configuration
- `pyproject.toml` — Project config (Ruff, pytest, coverage)
- `src/monitoring/feedback_loop.py` — Feedback analysis + corrections
- `tests/test_integration.py` — Integration test suite (20+ tests)

## Phase 6 - Monitoring, Observability, Logging, Alerting (Week 6)

### Added
- Langfuse tracing integration (`src/monitoring/tracing.py`):
  - Full pipeline traces with spans for each step (preprocessing, search, reranking, geo filter, LLM)
  - LLM generation events with model, prompt, and output tracking
  - Confidence and identification scores on each trace
  - No-op mode when Langfuse is not configured (zero overhead)
  - Trace URLs returned in pipeline results for easy debugging
- Structured logging (`src/monitoring/logging_config.py`):
  - JSON formatter for production log aggregation (single-line, parseable)
  - Human-readable formatter with color support for development
  - Auto-detect format from `LOG_FORMAT` env var (json/human)
  - Optional file logging (always JSON for files)
  - Extra fields support: request_id, species, latency passed as structured data
  - Third-party library noise reduction (httpx, chromadb, urllib3)
  - `log_pipeline_event()` helper for consistent structured events
- In-memory metrics collector (`src/monitoring/metrics_collector.py`):
  - Thread-safe rolling window (last 1000 requests)
  - Latency percentiles: avg, P50, P95, P99
  - Latency histogram (6 buckets: 0-1s, 1-3s, 3-5s, 5-10s, 10-15s, 15s+)
  - Top species tracking, error rate, declined count
  - Recent request log (last N with details)
  - Latency time series for charting
- Alerting engine (`src/monitoring/alerts.py`):
  - 5 alert rules: P95 latency, error rate, low accuracy, low confidence, component down
  - SQLite-backed alert history with auto-resolve
  - No duplicate alerts (checks for active alert before firing)
  - Minimum data threshold (10 requests) before alerting
  - Configurable thresholds via `DEFAULT_THRESHOLDS`
  - `GET /alerts` API endpoint for real-time alert status
- Monitoring dashboard (`src/frontend/dashboard.py`):
  - Key metrics row: total requests, success/declined/error counts, uptime
  - Latency and confidence metrics: avg, P95
  - Component health cards with status indicators
  - Top species bar chart
  - Request volume (last 24h) chart
  - Feedback summary: count, accuracy, feedback rate
  - Quality threshold table with pass/fail indicators
  - Raw JSON viewer for API responses
  - Optional auto-refresh (30s interval)

### Changed
- Pipeline (`src/rag/pipeline.py`): Now emits Langfuse traces with spans per step
- API main (`src/api/main.py`): Uses structured logging from monitoring module
- API routes: Added `GET /alerts` endpoint
- Makefile: Added `dashboard`, `api-json-logs`, `api-file-logs` targets
- Updated `serve` target to mention dashboard

### New Files
- `src/monitoring/tracing.py` — Langfuse integration (trace, span, generation, no-op)
- `src/monitoring/logging_config.py` — JSON + human log formatters, setup_logging()
- `src/monitoring/metrics_collector.py` — In-memory metrics with rolling window
- `src/monitoring/alerts.py` — Alert rules, SQLite store, checker
- `src/frontend/dashboard.py` — Streamlit monitoring dashboard
- `tests/test_monitoring.py` — 35 tests for all monitoring components

## Phase 5 - API, Backend Integration, Frontend (Week 5)

### Added
- FastAPI REST API with 4 endpoints:
  - `POST /identify`: Full RAG pipeline inference (query, location, season)
  - `GET /health`: Component health checks (ChromaDB, BM25, Ollama/Groq, data)
  - `GET /metrics`: Aggregated system metrics (latency, accuracy, top species)
  - `POST /feedback`: User feedback collection (correct/incorrect + corrections)
- Pydantic request/response models with validation:
  - Query length limits (3-1000 chars), confidence bounds (0-1)
  - Structured error responses with request IDs
- Middleware stack:
  - Request logging with X-Request-ID tracking and X-Response-Time headers
  - Rate limiting (10 req/min per IP, configurable via RATE_LIMIT_PER_MINUTE)
  - Request timeout (30s default, configurable via REQUEST_TIMEOUT_SECONDS)
  - CORS configured for Streamlit (localhost:8501) and React (localhost:3000)
- SQLite-backed feedback store:
  - Request log with species, confidence, latency, inference mode
  - User feedback with corrections and notes
  - Aggregated metrics: P95 latency, accuracy from feedback, top species, hourly counts
- Streamlit frontend (`src/frontend/app.py`):
  - Query input with location (Indian states + wildlife parks) and season dropdowns
  - Example queries for quick testing
  - Results panel: species, confidence bar, reasoning, features, conservation status
  - Expandable pipeline details (latency, chunks, inference mode)
  - Inline feedback buttons (correct / incorrect with correction form)
  - Sidebar: live API health status, quick metrics dashboard
- Lazy pipeline loading (loads on first /identify call, not on startup)
- Auto-generated API docs at /docs (Swagger) and /redoc
- Test suite (tests/test_api.py): 25 tests covering models, feedback store, endpoints, middleware
- Makefile targets: `api`, `api-prod`, `frontend`, `serve`

## Phase 4 - Evaluation & Validation (Week 4)

### Added
- Golden evaluation dataset: 51 QA pairs (20 easy, 15 medium, 8 hard, 8 trick)
  - Each entry: query, expected species, scientific name, regions, key facts
  - Trick queries test decline-to-answer (polar bears in Tamil Nadu, etc.)
- Custom metrics calculator (no external LLM needed, fast & deterministic):
  - Answer Correctness (exact/partial/scientific name match)
  - Geographic Accuracy (species valid for mentioned region)
  - Refusal Accuracy (correctly declines trick queries)
  - Citation Precision (cited chunks match expected species)
  - Confidence Calibration (high confidence on correct, low on wrong)
- RAGAS integration wrapper (optional deep evaluation via Groq)
  - Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Main evaluation runner with CI-ready output:
  - `--ci` flag exits with code 1 on quality gate failure
  - `--quick` mode for 10-query fast feedback
  - `--difficulty` filter for targeted evaluation
- Quality gates with configurable thresholds:
  - Answer correctness >= 0.75
  - Geographic accuracy >= 0.90
  - Refusal accuracy >= 0.80
  - P95 latency <= 15s
- Species confusion tracking (which species get mixed up)
- Test suite for evaluation framework (dataset integrity, metric calculation, gates)

## Phase 3 - Model Selection & Experimentation (Week 3)

### Added
- Benchmark query set: 30 queries (12 easy, 9 medium, 5 hard, 4 trick) with expected answers
- Embedding model comparison: MiniLM-L6-v2 vs BGE-small-en vs nomic-embed-text
  - Metrics: Precision@5, Hit Rate, MRR, latency per model
- LLM comparison: 2 local (Ollama) + 3 API (Groq) models
  - Metrics: Answer correctness, JSON validity, citation rate, latency
- Optimization experiments:
  - Temperature tuning (0.0 / 0.1 / 0.3 / 0.7)
  - Context window (1 / 3 / 5 / 7 chunks)
  - Quantization comparison (Q4 / Q5 / Q8)
- EVALUATION_REPORT.md template with tables for all benchmarks
- Makefile targets for running all Phase 3 experiments

## Phase 2 - RAG Engine (Week 2)

### Added
- Embedding pipeline: sentence-transformers → ChromaDB vector store
- BM25 index builder with tokenization and serialization
- Hybrid search: vector + BM25 with Reciprocal Rank Fusion (RRF)
- Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)
- LLM generator with dual backend: Ollama (local) + Groq (API)
- Structured output with Pydantic validation + retry logic
- Query expander: location/season extraction, habitat/color synonym expansion
- Geographic filtering (strict and boost modes)
- Main RAG pipeline orchestrating all components with per-step timing
- Versioned prompt template (prompts/v1.yaml)
- Retrieval config (config/retrieval.yaml)
- End-to-end test suite (tests/test_pipeline_e2e.py)
- Build indexes entry point (build both ChromaDB + BM25 in one command)

## Phase 1 - Data Foundation (Week 1)

### Added
- Project structure with all directories
- README with architecture diagram and project brief
- Data collection scripts: GBIF, Wikipedia, iNaturalist
- Data cleaning pipeline with deduplication and normalization
- Species-aware chunking engine (400-600 token target, no cross-species chunks)
- Chunk validator with required metadata checks
- EDA notebook with 6 visualization sections
- Makefile for pipeline automation
- Config: species list, .env.example, .gitignore
