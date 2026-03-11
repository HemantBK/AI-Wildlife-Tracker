# Wildlife Tracker - Pipeline Commands
# Usage: make <target>

.PHONY: help install collect-gbif collect-wiki collect-inat collect-all clean chunk validate data-pipeline eda test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

# ==================== DATA COLLECTION ====================

collect-gbif: ## Collect species data from GBIF
	python -m src.ingestion.gbif_collector

collect-wiki: ## Scrape species articles from Wikipedia
	python -m src.ingestion.wikipedia_scraper

collect-inat: ## Collect observation data from iNaturalist
	python -m src.ingestion.inaturalist_collector

collect-all: collect-gbif collect-inat collect-wiki ## Run all data collection scripts

# ==================== DATA PROCESSING ====================

clean: ## Clean and merge data from all sources
	python -m src.preprocessing.cleaner

chunk: ## Chunk processed data for RAG
	python -m src.preprocessing.chunker

validate: ## Validate chunks have all required fields
	python -m src.preprocessing.validator

# ==================== FULL PIPELINE ====================

data-pipeline: collect-all clean chunk validate ## Run the full data pipeline end-to-end
	@echo "Data pipeline complete!"

data-pipeline-quick: clean chunk validate ## Run cleaning + chunking only (skip collection)
	@echo "Quick pipeline complete!"

# ==================== INDEXES & RAG (Phase 2) ====================

build-indexes: ## Build ChromaDB vector store + BM25 index
	python -m src.retrieval.build_indexes

build-vectors: ## Build ChromaDB vector store only
	python -m src.retrieval.embedder

build-bm25: ## Build BM25 index only
	python -m src.retrieval.bm25_index

test-search: ## Test hybrid search with sample queries
	python -m src.retrieval.hybrid_search

test-query-expander: ## Test query expansion logic
	python -m src.retrieval.query_expander

test-pipeline: ## Run the full RAG pipeline with test queries
	python -m src.rag.pipeline

full-setup: data-pipeline build-indexes ## Run data pipeline + build indexes (full Phase 1+2)
	@echo "Full setup complete! Run 'make test-pipeline' to test."

# ==================== MODEL COMPARISON (Phase 3) ====================

compare-embeddings: ## Compare embedding models (MiniLM vs BGE vs Nomic)
	python -m src.evaluation.embedding_comparison

compare-llms: ## Compare LLMs (local + Groq API)
	python -m src.evaluation.llm_comparison

compare-llms-groq: ## Compare Groq API models only (fast)
	python -m src.evaluation.llm_comparison --groq-only

compare-llms-local: ## Compare local Ollama models only
	python -m src.evaluation.llm_comparison --local-only

run-experiments: ## Run temperature, context window, quantization experiments
	python -m src.evaluation.optimization_experiments --backend groq

run-experiments-local: ## Run optimization experiments with Ollama
	python -m src.evaluation.optimization_experiments --backend ollama

run-all-benchmarks: compare-embeddings compare-llms run-experiments ## Run ALL Phase 3 benchmarks
	@echo "All benchmarks complete! Results in data/evaluation/"

# ==================== EVALUATION (Phase 4) ====================

save-golden-dataset: ## Save golden evaluation dataset to JSON
	python -m src.evaluation.golden_dataset

eval: ## Run full evaluation against golden dataset
	python -m src.evaluation.evaluator

eval-quick: ## Quick evaluation (10 queries, ~2 min)
	python -m src.evaluation.evaluator --quick

eval-ci: ## CI evaluation (exits with code 1 on gate failure)
	python -m src.evaluation.evaluator --ci

eval-easy: ## Evaluate only easy queries
	python -m src.evaluation.evaluator --difficulty easy

eval-hard: ## Evaluate only hard queries
	python -m src.evaluation.evaluator --difficulty hard

eval-ragas: ## Run RAGAS deep evaluation (requires Groq API)
	python -m src.evaluation.ragas_evaluator

# ==================== API & FRONTEND (Phase 5) ====================

api: ## Start the FastAPI server (port 8000)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Start the FastAPI server in production mode
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

frontend: ## Start the Streamlit frontend (port 8501)
	streamlit run src/frontend/app.py --server.port 8501

serve: ## Start both API and frontend (use in separate terminals)
	@echo "Run these in separate terminals:"
	@echo "  make api       # Start FastAPI server on :8000"
	@echo "  make frontend  # Start Streamlit UI on :8501"
	@echo "  make dashboard # Start monitoring dashboard on :8502"

# ==================== MONITORING (Phase 6) ====================

dashboard: ## Start the monitoring dashboard (port 8502)
	streamlit run src/frontend/dashboard.py --server.port 8502

api-json-logs: ## Start API with JSON structured logging
	LOG_FORMAT=json uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-file-logs: ## Start API with file logging (logs/api.jsonl)
	LOG_FILE=logs/api.jsonl uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# ==================== ANALYSIS ====================

eda: ## Launch Jupyter notebook for EDA
	jupyter notebook notebooks/eda.ipynb

# ==================== TESTING ====================

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run only unit tests (fast)
	pytest tests/ -v -m "not integration and not slow and not e2e"

test-integration: ## Run integration tests
	pytest tests/ -v -m "integration"

test-coverage: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

# ==================== CI/CD (Phase 7) ====================

lint: ## Run Ruff linter
	ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix src/ tests/

format: ## Format code with Ruff
	ruff format src/ tests/

format-check: ## Check formatting without changing files
	ruff format --check src/ tests/

pre-commit-install: ## Install pre-commit hooks
	pip install pre-commit && pre-commit install

pre-commit-run: ## Run all pre-commit hooks
	pre-commit run --all-files

ci-local: lint format-check test ## Run full CI checks locally
	@echo "All CI checks passed!"

feedback-report: ## Generate quality report from user feedback
	python -m src.monitoring.feedback_loop --report

feedback-catalog: ## Generate corrections catalog from feedback
	python -m src.monitoring.feedback_loop --catalog

# ==================== DOCKER (Phase 8) ====================

docker-build: ## Build all Docker images
	docker-compose build

docker-up: ## Start full stack with Docker Compose
	docker-compose up -d
	@echo "Services starting..."
	@echo "  API:       http://localhost:8000/docs"
	@echo "  Frontend:  http://localhost:8501"
	@echo "  Dashboard: http://localhost:8502"

docker-down: ## Stop all Docker containers
	docker-compose down

docker-logs: ## Follow API container logs
	docker-compose logs -f api

docker-restart: ## Restart all containers
	docker-compose restart

# ==================== ONNX OPTIMIZATION ====================

onnx-convert: ## Convert models to ONNX format (2-3x CPU speedup)
	python scripts/convert_to_onnx.py

onnx-clean: ## Remove ONNX models (revert to PyTorch)
	rm -rf models/onnx/
	@echo "ONNX models removed. Pipeline will use PyTorch."

# ==================== SCRIPTS ====================

setup: ## Run first-time setup validation
	python scripts/setup.py

demo: ## Run demo queries through the API
	python scripts/demo.py

# ==================== CLEANUP ====================

clean-data: ## Remove all collected and processed data
	rm -rf data/raw/gbif data/raw/wikipedia data/raw/inaturalist
	rm -rf data/processed/* data/chunks/*
	@echo "All data cleaned!"

clean-pyc: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-docker: ## Remove Docker images and volumes
	docker-compose down --rmi local --volumes
	@echo "Docker resources cleaned!"
