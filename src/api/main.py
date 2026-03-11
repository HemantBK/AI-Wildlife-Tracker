"""
Wildlife Tracker API — Main Application
FastAPI application with middleware, CORS, and all routes.

Usage:
  uvicorn src.api.main:app --reload --port 8000
  # or
  python -m src.api.main
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    TimeoutMiddleware,
)
from src.api.routes import router
from src.monitoring.logging_config import setup_logging

load_dotenv()

# Use structured logging from monitoring module
setup_logging()

logger = logging.getLogger(__name__)


# ─── Lifespan Events ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Wildlife Tracker API starting up...")
    logger.info(f"Inference mode: {os.getenv('INFERENCE_MODE', 'groq')}")
    logger.info("Pipeline will load lazily on first /identify request")
    yield
    logger.info("Wildlife Tracker API shutting down...")


# ─── App Factory ─────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Wildlife Tracker API",
        description=(
            "AI-powered wildlife species identification from natural language descriptions. "
            "Built with RAG (Retrieval-Augmented Generation) using hybrid search "
            "(vector + BM25), cross-encoder re-ranking, and LLM generation. "
            "Focused on Indian wildlife species."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",  # Streamlit default
            "http://localhost:3000",  # React dev server
            "http://127.0.0.1:8501",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Middleware Stack (order matters: outermost first) ──
    # 1. Request logging (outermost — logs everything)
    app.add_middleware(RequestLoggingMiddleware)
    # 2. Rate limiting (before pipeline processing)
    rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    app.add_middleware(RateLimitMiddleware, max_requests=rate_limit, window_seconds=60)
    # 3. Timeout (innermost — wraps the actual processing)
    timeout = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    app.add_middleware(TimeoutMiddleware, timeout_seconds=timeout)

    # ── Routes ────────────────────────────────────────────
    app.include_router(router)

    return app


# ─── Application Instance ───────────────────────────────────────

app = create_app()


# ─── Direct Execution ───────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
