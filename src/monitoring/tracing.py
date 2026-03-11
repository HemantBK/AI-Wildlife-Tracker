"""
Langfuse Tracing Integration (v4)
Wraps the RAG pipeline with detailed observability traces using Langfuse v4's
OpenTelemetry-based API.

Each pipeline run creates a Langfuse trace with observations for every step.
If Langfuse is not configured, tracing is silently disabled (no-op).

Trace hierarchy:
  trace (identify request)
    |-- span: query_preprocessing
    |-- span: hybrid_search
    |-- span: reranking
    |-- span: geographic_filter
    |-- generation: llm_generation
    |-- span: post_processing

Each observation records: input, output, timing, metadata.

Langfuse v4 uses context managers instead of trace objects:
  - propagate_attributes() sets trace-level metadata
  - start_as_current_observation() creates nested spans/generations
  - score_current_trace() adds evaluation scores
"""

import contextlib
import logging
import os
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─── Langfuse Client (lazy, no-op if not configured) ────────────

_langfuse_client = None
_langfuse_enabled = False
_initialized = False


def _init_langfuse():
    """Initialize Langfuse client if credentials are available."""
    global _langfuse_client, _langfuse_enabled, _initialized
    _initialized = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key or public_key.startswith("your_"):
        logger.info("Langfuse not configured — tracing disabled (no-op mode)")
        _langfuse_enabled = False
        return

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        # Verify credentials
        _langfuse_client.auth_check()
        _langfuse_enabled = True
        logger.info(f"Langfuse tracing enabled (host: {host})")
    except ImportError:
        logger.warning("langfuse package not installed. Run: pip install langfuse")
        _langfuse_enabled = False
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        _langfuse_enabled = False


def get_langfuse():
    """Get Langfuse client, initializing on first call."""
    if not _initialized:
        _init_langfuse()
    return _langfuse_client


def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is active."""
    if not _initialized:
        _init_langfuse()
    return _langfuse_enabled


# ─── No-op Context Managers (when Langfuse is disabled) ─────────


class NoOpObservation:
    """A no-op observation that silently discards all trace data."""

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        return self

    def end(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def id(self):
        return "noop"


# ─── Traced Pipeline (top-level trace) ──────────────────────────


@contextmanager
def traced_pipeline(
    request_id: str,
    query: str,
    location: str | None = None,
    season: str | None = None,
    user_id: str | None = None,
):
    """
    Context manager for a full RAG pipeline trace.

    Creates a top-level Langfuse trace with metadata, and yields
    a context within which all child spans are automatically linked.

    Usage:
        with traced_pipeline(request_id, query) as trace_ctx:
            with traced_span("step1", input_data={...}) as span:
                ...
                span.update(output={...})
    """
    if not is_tracing_enabled():
        yield NoOpObservation()
        return

    obs = None
    try:
        from langfuse import propagate_attributes

        with (
            propagate_attributes(
                trace_name="wildlife_identify",
                user_id=user_id,
                metadata={
                    "request_id": request_id,
                    "inference_mode": os.getenv("INFERENCE_MODE", "groq"),
                },
                tags=["wildlife-tracker", "rag"],
            ),
            _langfuse_client.start_as_current_observation(
                name="wildlife_identify",
                as_type="span",
                input={
                    "query": query,
                    "location": location,
                    "season": season,
                },
            ) as obs,
        ):
            yield obs
    except Exception as e:
        # Only fall back to NoOp if the error happened during setup
        # (before yield). If we already yielded, re-raise the user error.
        if obs is not None:
            raise
        logger.warning(f"Failed to create pipeline trace: {e}")
        yield NoOpObservation()


# ─── Traced Span (pipeline step) ────────────────────────────────


@contextmanager
def traced_span(
    name: str,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    level: str = "DEFAULT",
):
    """
    Context manager for a pipeline step span.

    Must be called within a traced_pipeline() context.
    Automatically becomes a child of the current trace.

    Usage:
        with traced_span("hybrid_search", input_data={"query": q}) as span:
            results = searcher.search(q)
            span.update(output={"count": len(results)}, metadata={"latency_ms": 42})
    """
    if not is_tracing_enabled():
        yield NoOpObservation()
        return

    obs = None
    try:
        with _langfuse_client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_data or {},
            metadata=metadata or {},
            level=level,
        ) as obs:
            yield obs
    except Exception as e:
        if obs is not None:
            raise
        logger.warning(f"Failed to create span '{name}': {e}")
        yield NoOpObservation()


# ─── Traced Generation (LLM call) ───────────────────────────────


@contextmanager
def traced_generation(
    name: str = "llm_generation",
    model: str = "",
    input_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager for an LLM generation step.

    Must be called within a traced_pipeline() context.
    Records model name, input/output, and token usage.

    Usage:
        with traced_generation("llm_generation", model="groq/llama3") as gen:
            response = llm.generate(prompt)
            gen.update(output=response, usage_details={"input": 100, "output": 50})
    """
    if not is_tracing_enabled():
        yield NoOpObservation()
        return

    obs = None
    try:
        with _langfuse_client.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input_data or {},
            metadata=metadata or {},
        ) as obs:
            yield obs
    except Exception as e:
        if obs is not None:
            raise
        logger.warning(f"Failed to create generation '{name}': {e}")
        yield NoOpObservation()


# ─── Scoring ────────────────────────────────────────────────────


def score_trace(name: str, value: float, comment: str | None = None):
    """
    Add a score to the current trace.

    Common scores:
      - "confidence": 0.0–1.0 (model confidence)
      - "identified": 1.0 or 0.0 (whether species was identified)
      - "latency": total seconds
    """
    if not is_tracing_enabled():
        return

    try:
        _langfuse_client.score_current_trace(
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as e:
        logger.warning(f"Failed to add score '{name}': {e}")


# ─── Trace Output ───────────────────────────────────────────────


def set_trace_output(output: dict):
    """Set the trace-level output (final pipeline result)."""
    if not is_tracing_enabled():
        return

    try:
        _langfuse_client.set_current_trace_io(output=output)
    except Exception as e:
        logger.warning(f"Failed to set trace output: {e}")


# ─── Trace URL ──────────────────────────────────────────────────


def get_current_trace_url() -> str | None:
    """Get the Langfuse dashboard URL for the current trace."""
    if not is_tracing_enabled():
        return None

    try:
        # v4 API: get_trace_url() auto-detects current trace, or pass trace_id as keyword arg
        trace_id = _langfuse_client.get_current_trace_id()
        if trace_id:
            return _langfuse_client.get_trace_url(trace_id=trace_id)
    except Exception as e:
        logger.warning(f"Failed to get trace URL: {e}")

    return None


# ─── Flush ──────────────────────────────────────────────────────


def flush():
    """Flush any pending Langfuse events to the server."""
    if _langfuse_client is not None:
        with contextlib.suppress(Exception):
            _langfuse_client.flush()
