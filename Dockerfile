# ─── Wildlife Tracker API ────────────────────────────────────────
# Multi-stage build for small image size
# Stage 1: Install dependencies
# Stage 2: Copy app code and run

# ─── Stage 1: Dependencies ──────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements-docker.txt

# ─── Stage 2: Application ──────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY data/ ./data/
COPY .env.example ./.env.example

# Create necessary directories
RUN mkdir -p data/chroma_db data/chunks data/processed logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8000/health', timeout=5); exit(0 if r.status_code == 200 else 1)"

# Start the API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
