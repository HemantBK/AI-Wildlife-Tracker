"""
API Routes
All REST endpoints for the Wildlife Tracker API.

Endpoints:
  POST /identify        — Identify species from natural language description
  POST /identify/image  — Identify species from uploaded photo (multimodal)
  GET  /health          — Health check (Ollama, ChromaDB, chunk count)
  GET  /metrics         — System metrics (latency, accuracy, request counts)
  POST /feedback        — Submit feedback on identification correctness
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.api.feedback import FeedbackStore
from src.api.models import (
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IdentifyRequest,
    IdentifyResponse,
    MetricsResponse,
)

logger = logging.getLogger(__name__)

# ─── Router ──────────────────────────────────────────────────────

router = APIRouter()

# Lazy-loaded pipeline (loaded on first /identify call)
_pipeline = None
_pipeline_load_time = None
_feedback_store = FeedbackStore()
_start_time = time.time()


def _get_pipeline():
    """Lazy-load the RAG pipeline (avoids slow startup for health checks)."""
    global _pipeline, _pipeline_load_time
    if _pipeline is None:
        logger.info("Loading RAG pipeline (first request)...")
        from src.rag.pipeline import WildlifeRAGPipeline

        load_start = time.time()
        _pipeline = WildlifeRAGPipeline()
        _pipeline_load_time = time.time() - load_start
        logger.info(f"Pipeline loaded in {_pipeline_load_time:.1f}s")
    return _pipeline


# ─── POST /identify ──────────────────────────────────────────────


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
        504: {"model": ErrorResponse, "description": "Timeout"},
    },
    summary="Identify a wildlife species",
    description=(
        "Submit a natural language description of a wildlife sighting "
        "and receive a species identification with confidence score, "
        "reasoning, and cited sources."
    ),
)
async def identify_species(body: IdentifyRequest, request: Request):
    """Run the full RAG pipeline to identify a wildlife species."""
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        pipeline = _get_pipeline()
        result = pipeline.identify(
            query=body.query,
            location=body.location,
            season=body.season,
            prompt_version=body.prompt_version,
        )

        # Extract response data
        response = result.get("response", {})
        species_name = response.get("species_name", "DECLINED")
        confidence = response.get("confidence", 0.0)

        # Log to feedback store for metrics tracking
        _feedback_store.log_request(
            request_id=result["request_id"],
            query=body.query,
            predicted_species=species_name,
            confidence=confidence,
            location=body.location,
            season=body.season,
            inference_mode=result.get("inference_mode"),
            latency_seconds=result.get("total_latency_seconds", 0),
            chunks_used=result.get("chunks_used", 0),
            status="success",
        )

        # Build API response
        return IdentifyResponse(
            request_id=result["request_id"],
            query=body.query,
            species_name=species_name,
            scientific_name=response.get("scientific_name", ""),
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            key_features_matched=response.get("key_features_matched", []),
            habitat_match=response.get("habitat_match", ""),
            conservation_status=response.get("conservation_status", ""),
            geographic_match=response.get("geographic_match", True),
            cited_sources=response.get("cited_sources", []),
            alternative_species=[],
            location=result.get("location"),
            season=result.get("season"),
            inference_mode=result.get("inference_mode"),
            total_latency_seconds=result.get("total_latency_seconds", 0),
            chunks_retrieved=result.get("chunks_retrieved", 0),
            chunks_used=result.get("chunks_used", 0),
        )

    except Exception as e:
        logger.error(f"[{request_id}] Pipeline error: {e}")
        # Log error to store
        _feedback_store.log_request(
            request_id=request_id,
            query=body.query,
            predicted_species="ERROR",
            confidence=0.0,
            location=body.location,
            status="error",
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


# ─── POST /identify/image ────────────────────────────────────────


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post(
    "/identify/image",
    response_model=IdentifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
        504: {"model": ErrorResponse, "description": "Timeout"},
    },
    summary="Identify species from photo",
    description=(
        "Upload a wildlife photo for AI-powered species identification. "
        "Uses a vision model to analyze the image, then runs the result "
        "through the RAG pipeline for knowledge-grounded identification."
    ),
)
async def identify_from_image(
    request: Request,
    image: UploadFile = File(..., description="Wildlife photo (JPEG, PNG, WebP)"),
    location: str | None = Form(default=None, description="Location of sighting"),
    season: str | None = Form(default=None, description="Season of sighting"),
):
    """Run the hybrid vision + RAG pipeline on an uploaded image."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate image type
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}. "
            f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    # Read and validate size
    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({len(image_bytes) / 1024 / 1024:.1f}MB). Max: 20MB.",
        )

    try:
        from src.rag.vision import encode_bytes_to_base64, get_image_mime_type

        # Encode image
        image_base64 = encode_bytes_to_base64(image_bytes)
        mime_type = image.content_type or get_image_mime_type(image.filename or "image.jpg")

        # Run hybrid vision + RAG pipeline
        pipeline = _get_pipeline()

        from src.rag.vision import analyze_image_with_groq

        # Step 1: Vision model describes the image
        vision_result = analyze_image_with_groq(
            image_base64=image_base64,
            mime_type=mime_type,
            mode="describe",
            location=location,
            season=season,
        )
        description = vision_result["description"]
        logger.info(f"[{request_id}] Vision description: {description[:100]}...")

        # Step 2: Feed description into RAG pipeline
        rag_result = pipeline.identify(
            query=description,
            location=location,
            season=season,
        )

        # Extract response
        response = rag_result.get("response", {})
        species_name = response.get("species_name", "DECLINED")
        confidence = response.get("confidence", 0.0)

        # Log to feedback store
        _feedback_store.log_request(
            request_id=rag_result["request_id"],
            query=f"[IMAGE] {description[:200]}",
            predicted_species=species_name,
            confidence=confidence,
            location=location,
            season=season,
            inference_mode=rag_result.get("inference_mode"),
            latency_seconds=rag_result.get("total_latency_seconds", 0),
            chunks_used=rag_result.get("chunks_used", 0),
            status="success",
        )

        return IdentifyResponse(
            request_id=rag_result["request_id"],
            query=description,
            species_name=species_name,
            scientific_name=response.get("scientific_name", ""),
            confidence=confidence,
            reasoning=response.get("reasoning", ""),
            key_features_matched=response.get("key_features_matched", []),
            habitat_match=response.get("habitat_match", ""),
            conservation_status=response.get("conservation_status", ""),
            geographic_match=response.get("geographic_match", True),
            cited_sources=response.get("cited_sources", []),
            alternative_species=[],
            location=rag_result.get("location"),
            season=rag_result.get("season"),
            inference_mode=rag_result.get("inference_mode"),
            total_latency_seconds=(
                vision_result["vision_latency_seconds"] + rag_result.get("total_latency_seconds", 0)
            ),
            chunks_retrieved=rag_result.get("chunks_retrieved", 0),
            chunks_used=rag_result.get("chunks_used", 0),
            input_mode="image",
            vision_description=description,
            vision_latency_seconds=vision_result["vision_latency_seconds"],
            vision_model=vision_result["vision_model"],
        )

    except Exception as e:
        logger.error(f"[{request_id}] Vision pipeline error: {e}")
        _feedback_store.log_request(
            request_id=request_id,
            query="[IMAGE] upload",
            predicted_species="ERROR",
            confidence=0.0,
            location=location,
            status="error",
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── GET /health ─────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of all system components.",
)
async def health_check():
    """Check Ollama, ChromaDB, and data pipeline health."""
    components = {}

    # Check ChromaDB
    try:
        import chromadb

        chroma_path = Path("data/chroma_db")
        if chroma_path.exists():
            t0 = time.time()
            client = chromadb.PersistentClient(path=str(chroma_path))
            collections = client.list_collections()
            chunk_count = 0
            for col in collections:
                chunk_count += col.count()
            latency = (time.time() - t0) * 1000
            components["chromadb"] = {
                "status": "ok",
                "message": f"{chunk_count} chunks in {len(collections)} collection(s)",
                "latency_ms": round(latency, 1),
            }
        else:
            components["chromadb"] = {
                "status": "warning",
                "message": "ChromaDB directory not found. Run: make build-indexes",
            }
    except Exception as e:
        components["chromadb"] = {"status": "error", "message": str(e)}

    # Check BM25 index
    bm25_path = Path("data/bm25_index.pkl")
    if bm25_path.exists():
        size_kb = bm25_path.stat().st_size / 1024
        components["bm25"] = {
            "status": "ok",
            "message": f"Index file: {size_kb:.0f}KB",
        }
    else:
        components["bm25"] = {
            "status": "warning",
            "message": "BM25 index not found. Run: make build-indexes",
        }

    # Check Ollama (only if local mode)
    inference_mode = os.getenv("INFERENCE_MODE", "groq")
    if inference_mode == "local":
        try:
            import requests as req

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            t0 = time.time()
            r = req.get(f"{base_url}/api/tags", timeout=3)
            latency = (time.time() - t0) * 1000
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                components["ollama"] = {
                    "status": "ok",
                    "message": f"Models: {', '.join(models[:5])}",
                    "latency_ms": round(latency, 1),
                }
            else:
                components["ollama"] = {
                    "status": "error",
                    "message": f"Ollama returned status {r.status_code}",
                }
        except Exception as e:
            components["ollama"] = {
                "status": "error",
                "message": f"Cannot reach Ollama: {e}",
            }
    else:
        components["ollama"] = {
            "status": "skipped",
            "message": f"Using {inference_mode} mode (not local)",
        }

    # Check Groq API key (if groq mode)
    if inference_mode == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            components["groq"] = {
                "status": "ok",
                "message": "API key configured",
            }
        else:
            components["groq"] = {
                "status": "error",
                "message": "GROQ_API_KEY not set in .env",
            }

    # Check chunks data
    chunks_dir = Path("data/chunks")
    if chunks_dir.exists():
        chunk_files = list(chunks_dir.glob("*.json"))
        components["data"] = {
            "status": "ok",
            "message": f"{len(chunk_files)} chunk file(s) found",
        }
    else:
        components["data"] = {
            "status": "warning",
            "message": "No chunk data found. Run: make data-pipeline",
        }

    # Pipeline status
    if _pipeline is not None:
        components["pipeline"] = {
            "status": "ok",
            "message": f"Loaded (took {_pipeline_load_time:.1f}s)",
        }
    else:
        components["pipeline"] = {
            "status": "ok",
            "message": "Not yet loaded (will load on first /identify call)",
        }

    # Overall status
    has_errors = any(c.get("status") == "error" for c in components.values())
    overall_status = "degraded" if has_errors else "healthy"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components,
    )


# ─── GET /metrics ────────────────────────────────────────────────


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Get current system metrics including latency, accuracy, and request counts.",
)
async def get_metrics():
    """Return aggregated system metrics from the request log."""
    metrics = _feedback_store.get_metrics()
    metrics["uptime_seconds"] = round(time.time() - _start_time, 1)
    return MetricsResponse(**metrics)


# ─── POST /feedback ──────────────────────────────────────────────


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid feedback"},
    },
    summary="Submit feedback",
    description="Submit feedback on whether a species identification was correct.",
)
async def submit_feedback(body: FeedbackRequest):
    """Store user feedback for continuous improvement."""
    try:
        feedback_id = _feedback_store.add_feedback(
            request_id=body.request_id,
            was_correct=body.was_correct,
            correct_species=body.correct_species,
            notes=body.notes,
        )
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="received",
            message="Thank you for your feedback!",
        )
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e


# ─── GET /alerts ────────────────────────────────────────────────


@router.get(
    "/alerts",
    summary="Active alerts",
    description="Get current active alerts and run alert checks against latest metrics.",
)
async def get_alerts():
    """Run alert checks and return active alerts."""
    from src.monitoring.alerts import AlertChecker

    checker = AlertChecker()
    metrics = _feedback_store.get_metrics()
    triggered = checker.check_all(metrics)
    status = checker.get_status()
    return {
        "newly_triggered": triggered,
        "active_alerts": status["active_alerts"],
        "active_count": status["active_alert_count"],
        "thresholds": status["thresholds"],
    }
