"""
Tests for monitoring, observability, and alerting components.
Tests tracing (no-op mode), logging, metrics collector, and alert rules.
"""

import contextlib
import json
import logging
import os
import tempfile
from unittest.mock import patch

from src.monitoring.alerts import AlertChecker, AlertStore
from src.monitoring.logging_config import (
    HumanFormatter,
    JSONFormatter,
    log_pipeline_event,
    setup_logging,
)
from src.monitoring.metrics_collector import (
    MetricsCollector,
    RequestMetric,
    get_collector,
)
from src.monitoring.tracing import (
    NoOpObservation,
    flush,
    is_tracing_enabled,
    traced_generation,
    traced_pipeline,
    traced_span,
)

# ─── Tracing Tests (No-Op Mode) ─────────────────────────────────


class TestNoOpTracing:
    """Test that tracing works in no-op mode when Langfuse is not configured."""

    def test_noop_observation_basics(self):
        """NoOpObservation should silently accept all operations."""
        obs = NoOpObservation()
        obs.update(output={"test": True})
        obs.end()
        assert obs.id == "noop"

    def test_noop_observation_context_manager(self):
        """NoOpObservation works as a context manager."""
        with NoOpObservation() as obs:
            obs.update(output={"test": True})
        # Should not raise

    @patch("src.monitoring.tracing._langfuse_enabled", False)
    @patch("src.monitoring.tracing._initialized", True)
    def test_traced_pipeline_noop(self):
        """traced_pipeline yields NoOpObservation when Langfuse disabled."""
        with traced_pipeline("test-001", "test query") as obs:
            assert isinstance(obs, NoOpObservation)

    @patch("src.monitoring.tracing._langfuse_enabled", False)
    @patch("src.monitoring.tracing._initialized", True)
    def test_traced_span_noop(self):
        """traced_span yields NoOpObservation when Langfuse disabled."""
        with traced_span("test_step", input_data={"x": 1}) as span:
            assert isinstance(span, NoOpObservation)
            span.update(output={"result": "ok"})

    @patch("src.monitoring.tracing._langfuse_enabled", False)
    @patch("src.monitoring.tracing._initialized", True)
    def test_traced_generation_noop(self):
        """traced_generation yields NoOpObservation when Langfuse disabled."""
        with traced_generation("test_gen", model="test-model") as gen:
            assert isinstance(gen, NoOpObservation)
            gen.update(output="test output")

    def test_flush_noop(self):
        """flush() should not raise when tracing is disabled."""
        flush()

    def test_tracing_enabled_returns_bool(self):
        """is_tracing_enabled should return a boolean."""
        result = is_tracing_enabled()
        assert isinstance(result, bool)

    @patch("src.monitoring.tracing._langfuse_enabled", False)
    @patch("src.monitoring.tracing._initialized", True)
    def test_full_noop_pipeline_flow(self):
        """Simulate a full pipeline trace in no-op mode."""
        with traced_pipeline("test-002", "blue bird near river"):
            with traced_span("preprocessing", input_data={"query": "blue bird"}) as s:
                s.update(output={"expanded": "blue bird aquatic"})

            with traced_span("search", input_data={"query": "blue bird aquatic"}) as s:
                s.update(output={"count": 10})

            with traced_generation("llm", model="test") as g:
                g.update(output="Common Kingfisher")

        flush()
        # Should complete without errors


# ─── Logging Tests ───────────────────────────────────────────────


class TestJSONFormatter:
    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data

    def test_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Event",
            args=None,
            exc_info=None,
        )
        record.request_id = "abc123"
        record.species = "Tiger"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["request_id"] == "abc123"
        assert data["species"] == "Tiger"


class TestHumanFormatter:
    def test_basic_format(self):
        formatter = HumanFormatter(use_color=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Hello world",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "Hello world" in output

    def test_extra_fields_in_brackets(self):
        formatter = HumanFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Alert",
            args=None,
            exc_info=None,
        )
        record.latency_ms = 500
        output = formatter.format(record)
        assert "latency_ms=500" in output


class TestSetupLogging:
    def test_setup_json(self):
        setup_logging(json_format=True, level="DEBUG")
        logger = logging.getLogger("test_setup_json")
        assert logger.getEffectiveLevel() <= logging.DEBUG

    def test_setup_human(self):
        setup_logging(json_format=False, level="INFO")
        logger = logging.getLogger("test_setup_human")
        assert logger.getEffectiveLevel() <= logging.INFO

    def test_log_pipeline_event(self):
        logger = logging.getLogger("test_event")
        # Should not raise
        log_pipeline_event(
            logger,
            "test_event",
            request_id="test-001",
            species="Tiger",
            latency_ms=150,
        )


# ─── Metrics Collector Tests ────────────────────────────────────


class TestMetricsCollector:
    def setup_method(self):
        self.collector = MetricsCollector(window_size=100)

    def test_empty_summary(self):
        summary = self.collector.get_summary()
        assert summary["total_requests"] == 0
        assert summary["latency"]["avg"] == 0.0

    def test_record_and_summary(self):
        self.collector.record(
            RequestMetric(
                request_id="r1",
                query="test",
                species="Tiger",
                confidence=0.9,
                latency_seconds=2.0,
                inference_mode="groq",
                chunks_used=5,
                status="success",
            )
        )
        self.collector.record(
            RequestMetric(
                request_id="r2",
                query="test2",
                species="Peacock",
                confidence=0.8,
                latency_seconds=3.0,
                inference_mode="groq",
                chunks_used=3,
                status="success",
            )
        )
        summary = self.collector.get_summary()
        assert summary["total_requests"] == 2
        assert summary["total_errors"] == 0
        assert summary["latency"]["avg"] == 2.5
        assert summary["avg_confidence"] == 0.85

    def test_error_counting(self):
        self.collector.record(
            RequestMetric(
                request_id="r1",
                query="test",
                species="ERROR",
                confidence=0.0,
                latency_seconds=0.1,
                inference_mode="groq",
                chunks_used=0,
                status="error",
            )
        )
        summary = self.collector.get_summary()
        assert summary["total_errors"] == 1

    def test_declined_counting(self):
        self.collector.record(
            RequestMetric(
                request_id="r1",
                query="test",
                species="DECLINED",
                confidence=0.0,
                latency_seconds=1.5,
                inference_mode="groq",
                chunks_used=5,
                status="declined",
            )
        )
        summary = self.collector.get_summary()
        assert summary["total_declined"] == 1

    def test_top_species(self):
        for _ in range(5):
            self.collector.record(
                RequestMetric(
                    request_id="r",
                    query="t",
                    species="Tiger",
                    confidence=0.9,
                    latency_seconds=1.0,
                    inference_mode="groq",
                    chunks_used=5,
                    status="success",
                )
            )
        for _ in range(3):
            self.collector.record(
                RequestMetric(
                    request_id="r",
                    query="t",
                    species="Peacock",
                    confidence=0.8,
                    latency_seconds=1.0,
                    inference_mode="groq",
                    chunks_used=5,
                    status="success",
                )
            )
        summary = self.collector.get_summary()
        assert summary["top_species"][0]["species"] == "Tiger"
        assert summary["top_species"][0]["count"] == 5

    def test_latency_histogram(self):
        self.collector.record(
            RequestMetric(
                request_id="r1",
                query="t",
                species="Tiger",
                confidence=0.9,
                latency_seconds=0.5,
                inference_mode="groq",
                chunks_used=5,
                status="success",
            )
        )
        self.collector.record(
            RequestMetric(
                request_id="r2",
                query="t",
                species="Tiger",
                confidence=0.9,
                latency_seconds=7.0,
                inference_mode="groq",
                chunks_used=5,
                status="success",
            )
        )
        summary = self.collector.get_summary()
        assert summary["latency_histogram"]["0-1s"] == 1
        assert summary["latency_histogram"]["5-10s"] == 1

    def test_get_recent(self):
        for i in range(5):
            self.collector.record(
                RequestMetric(
                    request_id=f"r{i}",
                    query=f"query {i}",
                    species="Tiger",
                    confidence=0.9,
                    latency_seconds=1.0,
                    inference_mode="groq",
                    chunks_used=5,
                    status="success",
                )
            )
        recent = self.collector.get_recent(3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["request_id"] == "r4"

    def test_window_size(self):
        collector = MetricsCollector(window_size=5)
        for i in range(10):
            collector.record(
                RequestMetric(
                    request_id=f"r{i}",
                    query="t",
                    species="Tiger",
                    confidence=0.9,
                    latency_seconds=1.0,
                    inference_mode="groq",
                    chunks_used=5,
                    status="success",
                )
            )
        assert collector.get_summary()["total_requests"] == 10
        assert collector.get_summary()["window_size"] == 5

    def test_global_singleton(self):
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


# ─── Alert Tests ─────────────────────────────────────────────────


class TestAlertStore:
    def setup_method(self):
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(self.tmp_fd)
        self.store = AlertStore(db_path=self.tmp_path)

    def teardown_method(self):
        with contextlib.suppress(Exception):
            os.unlink(self.tmp_path)

    def test_add_alert(self):
        alert_id = self.store.add_alert(
            rule_name="p95_latency",
            severity="warning",
            message="P95 latency is 20s",
            current_value=20.0,
            threshold=15.0,
        )
        assert alert_id > 0

    def test_active_alerts(self):
        self.store.add_alert("rule_a", "warning", "test alert A")
        self.store.add_alert("rule_b", "critical", "test alert B")
        active = self.store.get_active_alerts()
        assert len(active) == 2

    def test_resolve_alert(self):
        self.store.add_alert("rule_a", "warning", "test")
        self.store.resolve_alert("rule_a")
        active = self.store.get_active_alerts()
        assert len(active) == 0

    def test_has_active_alert(self):
        self.store.add_alert("rule_a", "warning", "test")
        assert self.store.has_active_alert("rule_a") is True
        assert self.store.has_active_alert("rule_b") is False

    def test_alert_history(self):
        self.store.add_alert("rule_a", "warning", "alert 1")
        self.store.add_alert("rule_b", "critical", "alert 2")
        self.store.resolve_alert("rule_a")
        history = self.store.get_alert_history()
        assert len(history) == 2


class TestAlertChecker:
    def setup_method(self):
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(self.tmp_fd)
        self.store = AlertStore(db_path=self.tmp_path)
        self.checker = AlertChecker(alert_store=self.store)

    def teardown_method(self):
        with contextlib.suppress(Exception):
            os.unlink(self.tmp_path)

    def test_no_alerts_when_all_healthy(self):
        metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 5.0,
            "avg_confidence": 0.85,
            "feedback_count": 10,
            "accuracy_from_feedback": 0.90,
        }
        triggered = self.checker.check_all(metrics)
        assert len(triggered) == 0

    def test_latency_alert(self):
        metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 20.0,  # Above 15s threshold
            "avg_confidence": 0.85,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        triggered = self.checker.check_all(metrics)
        rules = [a["rule"] for a in triggered]
        assert "p95_latency" in rules

    def test_error_rate_alert(self):
        metrics = {
            "total_requests": 100,
            "error_count": 15,  # 15% > 10% threshold
            "p95_latency_seconds": 5.0,
            "avg_confidence": 0.85,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        triggered = self.checker.check_all(metrics)
        rules = [a["rule"] for a in triggered]
        assert "error_rate" in rules

    def test_low_accuracy_alert(self):
        metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 5.0,
            "avg_confidence": 0.85,
            "feedback_count": 20,
            "accuracy_from_feedback": 0.55,  # Below 70% threshold
        }
        triggered = self.checker.check_all(metrics)
        rules = [a["rule"] for a in triggered]
        assert "low_accuracy" in rules

    def test_no_alert_with_insufficient_data(self):
        metrics = {
            "total_requests": 3,  # Below min_requests_for_alert (10)
            "error_count": 3,
            "p95_latency_seconds": 50.0,
            "avg_confidence": 0.1,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        triggered = self.checker.check_all(metrics)
        assert len(triggered) == 0

    def test_no_duplicate_alerts(self):
        metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 20.0,
            "avg_confidence": 0.85,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        # First check triggers alert
        triggered1 = self.checker.check_all(metrics)
        assert len(triggered1) >= 1
        # Second check should not duplicate
        triggered2 = self.checker.check_all(metrics)
        assert len(triggered2) == 0

    def test_alert_auto_resolves(self):
        # Trigger alert
        bad_metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 20.0,
            "avg_confidence": 0.85,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        self.checker.check_all(bad_metrics)
        assert len(self.store.get_active_alerts()) >= 1

        # Fix the issue
        good_metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 5.0,
            "avg_confidence": 0.85,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        self.checker.check_all(good_metrics)
        assert len(self.store.get_active_alerts()) == 0

    def test_component_health_alert(self):
        metrics = {
            "total_requests": 50,
            "error_count": 0,
            "p95_latency_seconds": 5.0,
            "avg_confidence": 0.85,
            "feedback_count": 0,
            "accuracy_from_feedback": None,
        }
        health = {
            "components": {
                "chromadb": {"status": "error", "message": "Connection refused"},
                "bm25": {"status": "ok"},
            }
        }
        triggered = self.checker.check_all(metrics, health)
        rules = [a["rule"] for a in triggered]
        assert "component_chromadb" in rules

    def test_get_status(self):
        status = self.checker.get_status()
        assert "active_alert_count" in status
        assert "thresholds" in status
        assert status["active_alert_count"] == 0
