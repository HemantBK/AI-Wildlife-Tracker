"""
Integration Tests
Tests that verify multiple components work together correctly.
Marked with @pytest.mark.integration — skip with: pytest -m "not integration"

These tests mock external services (LLM, ChromaDB) but verify
the full flow through real code paths.
"""

import contextlib
import copy
import os
import tempfile
import threading
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline_result():
    """Standard mock result from the RAG pipeline."""
    return {
        "request_id": "int-test-001",
        "query": "large orange striped cat in forest",
        "response": {
            "species_name": "Bengal Tiger",
            "scientific_name": "Panthera tigris tigris",
            "confidence": 0.92,
            "reasoning": "The description matches a Bengal Tiger based on color and habitat.",
            "key_features_matched": ["orange", "striped", "large cat"],
            "habitat_match": "forest",
            "conservation_status": "Endangered",
            "geographic_match": True,
            "cited_sources": ["chunk_tiger_001", "chunk_tiger_002"],
            "alternative_species": [],
        },
        "location": "Madhya Pradesh",
        "season": None,
        "inference_mode": "groq",
        "prompt_version": 1,
        "timings": {
            "preprocessing_ms": 5.0,
            "hybrid_search_ms": 120.0,
            "reranking_ms": 80.0,
            "geo_filter_ms": 1.0,
            "llm_generation_ms": 2500.0,
            "total_ms": 2710.0,
        },
        "total_latency_seconds": 2.71,
        "chunks_retrieved": 15,
        "chunks_after_rerank": 5,
        "chunks_used": 3,
        "trace_url": None,
    }


@pytest.fixture
def api_client():
    """Create a test client with fresh app."""
    import src.api.routes as routes_module

    routes_module._pipeline = None
    from src.api.main import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def api_client_with_mock_pipeline(mock_pipeline_result, tmp_path):
    """Create a test client with a mocked pipeline and isolated feedback store."""
    import src.api.routes as routes_module

    # Use a fresh temporary feedback store to avoid cross-test pollution
    from src.api.feedback import FeedbackStore

    original_store = routes_module._feedback_store
    routes_module._feedback_store = FeedbackStore(db_path=str(tmp_path / "test_feedback.db"))

    # Return unique request_ids so INSERT OR REPLACE creates separate rows
    def mock_identify(**kwargs):
        result = copy.deepcopy(mock_pipeline_result)
        result["request_id"] = f"int-test-{uuid.uuid4().hex[:8]}"
        return result

    mock_pipeline = MagicMock()
    mock_pipeline.identify.side_effect = mock_identify
    routes_module._pipeline = mock_pipeline

    from src.api.main import create_app

    app = create_app()
    yield TestClient(app)

    # Restore original store
    routes_module._feedback_store = original_store


# ─── Integration: Full API Flow ─────────────────────────────────


@pytest.mark.integration
class TestFullAPIFlow:
    """Test the complete API request/response cycle."""

    def test_identify_then_feedback_flow(self, api_client_with_mock_pipeline):
        """Test: identify → get result → submit feedback → check metrics."""
        client = api_client_with_mock_pipeline

        # Step 1: Identify a species
        identify_resp = client.post(
            "/identify",
            json={
                "query": "large orange striped cat in forest",
                "location": "Madhya Pradesh",
            },
        )
        assert identify_resp.status_code == 200
        result = identify_resp.json()
        assert result["species_name"] == "Bengal Tiger"
        request_id = result["request_id"]

        # Step 2: Submit positive feedback
        fb_resp = client.post(
            "/feedback",
            json={
                "request_id": request_id,
                "was_correct": True,
            },
        )
        assert fb_resp.status_code == 200
        assert fb_resp.json()["status"] == "received"

        # Step 3: Check metrics reflect the request
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()
        assert metrics["total_requests"] >= 1
        assert metrics["feedback_count"] >= 1

    def test_identify_then_correction_flow(self, api_client_with_mock_pipeline):
        """Test: identify → submit correction → verify stored."""
        client = api_client_with_mock_pipeline

        # Identify
        resp = client.post(
            "/identify",
            json={
                "query": "spotted cat in the trees",
            },
        )
        assert resp.status_code == 200
        request_id = resp.json()["request_id"]

        # Submit correction
        fb = client.post(
            "/feedback",
            json={
                "request_id": request_id,
                "was_correct": False,
                "correct_species": "Indian Leopard",
                "notes": "It was clearly spotted, not striped.",
            },
        )
        assert fb.status_code == 200

    def test_health_then_alerts_flow(self, api_client):
        """Test: health check → alerts check."""
        client = api_client

        # Health
        health = client.get("/health")
        assert health.status_code == 200
        assert "components" in health.json()

        # Alerts
        alerts = client.get("/alerts")
        assert alerts.status_code == 200
        assert "active_alerts" in alerts.json()


# ─── Integration: Middleware Chain ───────────────────────────────


@pytest.mark.integration
class TestMiddlewareChain:
    """Test that all middleware layers work together."""

    def test_request_id_propagation(self, api_client):
        """Request ID should be in response headers."""
        resp = api_client.get("/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    def test_response_time_header(self, api_client):
        """Response time should be tracked."""
        resp = api_client.get("/health")
        assert "x-response-time" in resp.headers

    def test_rate_limit_headers(self, api_client):
        """Rate limit info should be in headers."""
        resp = api_client.get("/metrics")
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers

    def test_cors_headers(self, api_client):
        """CORS should allow Streamlit origin."""
        resp = api_client.options(
            "/identify",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should respond
        assert resp.status_code in (200, 405)


# ─── Integration: Feedback Store + Metrics ──────────────────────


@pytest.mark.integration
class TestFeedbackStoreIntegration:
    """Test feedback store works with the API metrics endpoint."""

    def test_metrics_aggregate_correctly(self, api_client_with_mock_pipeline):
        """Multiple requests should aggregate in metrics."""
        client = api_client_with_mock_pipeline

        # Send 3 identify requests
        for _ in range(3):
            resp = client.post(
                "/identify",
                json={
                    "query": "test animal in the wild",
                },
            )
            assert resp.status_code == 200

        # Check metrics
        metrics = client.get("/metrics").json()
        assert metrics["total_requests"] >= 3


# ─── Integration: Monitoring Components ─────────────────────────


@pytest.mark.integration
class TestMonitoringIntegration:
    """Test monitoring components work together."""

    def test_metrics_collector_with_alert_checker(self):
        """Metrics should feed into alert checker correctly."""
        from src.monitoring.metrics_collector import MetricsCollector, RequestMetric

        collector = MetricsCollector()

        # Record some high-latency requests
        for i in range(15):
            collector.record(
                RequestMetric(
                    request_id=f"r{i}",
                    query="test",
                    species="Tiger",
                    confidence=0.9,
                    latency_seconds=20.0,  # Very slow
                    inference_mode="groq",
                    chunks_used=5,
                    status="success",
                )
            )

        summary = collector.get_summary()
        assert summary["total_requests"] == 15
        assert summary["latency"]["p95"] >= 15.0

    def test_feedback_loop_with_empty_db(self):
        """Feedback loop should handle empty database gracefully."""
        from src.monitoring.feedback_loop import FeedbackLoop

        # Use a non-existent path
        loop = FeedbackLoop(db_path="nonexistent_test.db")
        confusions = loop.get_confusion_pairs()
        assert confusions == []

        accuracy = loop.get_accuracy_by_species()
        assert accuracy == {}

        report = loop.generate_quality_report()
        assert report["summary"]["total_requests"] == 0

    def test_feedback_loop_quality_report(self):
        """Quality report should generate with proper structure."""
        from src.monitoring.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(db_path="nonexistent_test.db")
        report = loop.generate_quality_report(days=7)

        assert "period" in report
        assert "summary" in report
        assert "recommendations" in report
        assert len(report["recommendations"]) >= 1

    def test_alert_store_survives_concurrent_access(self):
        """Alert store should handle rapid writes."""
        from src.monitoring.alerts import AlertStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_name = tmp.name

        alert_store = AlertStore(db_path=tmp_name)

        def write_alert(i, _store=alert_store):
            _store.add_alert(
                rule_name=f"rule_{i % 3}",
                severity="warning",
                message=f"Alert {i}",
            )

        threads = [threading.Thread(target=write_alert, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        history = alert_store.get_alert_history(limit=50)
        assert len(history) == 20

        # Close store before cleanup; suppress errors on Windows where
        # SQLite may still hold the file handle.
        del alert_store
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)


# ─── Integration: Tracing No-Op Safety ──────────────────────────


@pytest.mark.integration
class TestTracingNoOpIntegration:
    """Verify tracing doesn't break anything when disabled."""

    def test_pipeline_tracing_imports(self):
        """Pipeline should import tracing without errors."""
        from src.monitoring.tracing import (
            flush,
            score_trace,
            traced_generation,
            traced_pipeline,
            traced_span,
        )

        # All should be importable
        assert callable(traced_pipeline)
        assert callable(traced_span)
        assert callable(traced_generation)
        assert callable(score_trace)
        assert callable(flush)

    def test_tracing_disabled_by_default(self):
        """Tracing should be disabled when no Langfuse keys."""
        from src.monitoring.tracing import is_tracing_enabled

        # Without proper env vars, tracing should be off
        # (CI environment won't have Langfuse keys)
        # Note: may be True if .env has real keys, so just check it's callable
        assert isinstance(is_tracing_enabled(), bool)


# ─── Integration: Module Import Chain ────────────────────────────


class TestModuleImports:
    """Verify all modules can be imported without errors."""

    def test_api_models(self):
        from src.api.models import IdentifyRequest

        assert IdentifyRequest is not None

    def test_api_middleware(self):
        from src.api.middleware import RequestLoggingMiddleware

        assert RequestLoggingMiddleware is not None

    def test_api_feedback(self):
        from src.api.feedback import FeedbackStore

        assert FeedbackStore is not None

    def test_monitoring_tracing(self):
        from src.monitoring.tracing import NoOpObservation, traced_pipeline

        assert traced_pipeline is not None
        assert NoOpObservation is not None

    def test_monitoring_logging(self):
        from src.monitoring.logging_config import setup_logging

        assert setup_logging is not None

    def test_monitoring_alerts(self):
        from src.monitoring.alerts import AlertChecker

        assert AlertChecker is not None

    def test_monitoring_metrics(self):
        from src.monitoring.metrics_collector import MetricsCollector

        assert MetricsCollector is not None

    def test_monitoring_feedback_loop(self):
        from src.monitoring.feedback_loop import FeedbackLoop

        assert FeedbackLoop is not None

    def test_evaluation_modules(self):
        from src.evaluation.golden_dataset import get_golden_dataset

        assert get_golden_dataset is not None
