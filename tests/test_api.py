"""
Tests for the API layer.
Tests request/response models, middleware, feedback store, and routes.
Uses FastAPI TestClient for endpoint testing (no real pipeline needed).
"""

import contextlib
import os
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.feedback import FeedbackStore
from src.api.models import (
    FeedbackRequest,
    IdentifyRequest,
    IdentifyResponse,
)

# ─── Request/Response Model Tests ────────────────────────────────


class TestIdentifyRequest:
    def test_valid_request(self):
        req = IdentifyRequest(query="I saw a large orange cat in the forest")
        assert req.query == "I saw a large orange cat in the forest"
        assert req.location is None
        assert req.season is None
        assert req.prompt_version == 1

    def test_full_request(self):
        req = IdentifyRequest(
            query="Large striped cat",
            location="Madhya Pradesh",
            season="summer",
            prompt_version=2,
        )
        assert req.location == "Madhya Pradesh"
        assert req.season == "summer"
        assert req.prompt_version == 2

    def test_query_too_short(self):
        with pytest.raises(ValidationError):
            IdentifyRequest(query="ab")

    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            IdentifyRequest(query="x" * 1001)

    def test_empty_query(self):
        with pytest.raises(ValidationError):
            IdentifyRequest(query="")


class TestFeedbackRequest:
    def test_valid_feedback(self):
        fb = FeedbackRequest(
            request_id="abc123",
            was_correct=True,
        )
        assert fb.request_id == "abc123"
        assert fb.was_correct is True
        assert fb.correct_species is None

    def test_correction_feedback(self):
        fb = FeedbackRequest(
            request_id="abc123",
            was_correct=False,
            correct_species="Bengal Tiger",
            notes="It was clearly a tiger, not a leopard",
        )
        assert fb.was_correct is False
        assert fb.correct_species == "Bengal Tiger"

    def test_missing_request_id(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(was_correct=True)


class TestIdentifyResponse:
    def test_response_creation(self):
        resp = IdentifyResponse(
            request_id="abc123",
            query="test query",
            species_name="Bengal Tiger",
            confidence=0.95,
        )
        assert resp.species_name == "Bengal Tiger"
        assert resp.confidence == 0.95
        assert resp.cited_sources == []

    def test_declined_response(self):
        resp = IdentifyResponse(
            request_id="abc123",
            query="polar bear in Chennai",
            species_name="DECLINED",
            confidence=0.0,
            reasoning="Polar bears are not found in India.",
        )
        assert resp.species_name == "DECLINED"
        assert resp.confidence == 0.0


# ─── Feedback Store Tests ────────────────────────────────────────


class TestFeedbackStore:
    def setup_method(self):
        """Create a temporary database for each test."""
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(self.tmp_fd)
        self.store = FeedbackStore(db_path=self.tmp_path)

    def teardown_method(self):
        """Clean up temp database."""
        with contextlib.suppress(Exception):
            os.unlink(self.tmp_path)

    def test_log_request(self):
        self.store.log_request(
            request_id="req-001",
            query="striped cat",
            predicted_species="Bengal Tiger",
            confidence=0.9,
            location="Madhya Pradesh",
            latency_seconds=2.5,
            chunks_used=5,
        )
        metrics = self.store.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_identifications"] == 1

    def test_log_declined_request(self):
        self.store.log_request(
            request_id="req-002",
            query="polar bear in Chennai",
            predicted_species="DECLINED",
            confidence=0.0,
        )
        metrics = self.store.get_metrics()
        assert metrics["declined_identifications"] == 1

    def test_add_feedback(self):
        feedback_id = self.store.add_feedback(
            request_id="req-001",
            was_correct=True,
        )
        assert feedback_id is not None
        assert len(feedback_id) > 0

    def test_feedback_accuracy(self):
        self.store.add_feedback(request_id="req-001", was_correct=True)
        self.store.add_feedback(request_id="req-002", was_correct=True)
        self.store.add_feedback(request_id="req-003", was_correct=False)
        metrics = self.store.get_metrics()
        assert metrics["feedback_count"] == 3
        assert abs(metrics["accuracy_from_feedback"] - 0.6667) < 0.01

    def test_metrics_empty_db(self):
        metrics = self.store.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["avg_latency_seconds"] == 0.0
        assert metrics["accuracy_from_feedback"] is None

    def test_top_species(self):
        for i in range(5):
            self.store.log_request(
                request_id=f"req-tiger-{i}",
                query="striped cat",
                predicted_species="Bengal Tiger",
                confidence=0.9,
            )
        for i in range(3):
            self.store.log_request(
                request_id=f"req-peacock-{i}",
                query="fan shaped crest bird",
                predicted_species="Indian Peafowl",
                confidence=0.85,
            )
        metrics = self.store.get_metrics()
        assert len(metrics["top_species"]) == 2
        assert metrics["top_species"][0]["species"] == "Bengal Tiger"
        assert metrics["top_species"][0]["count"] == 5

    def test_latency_stats(self):
        for i in range(10):
            self.store.log_request(
                request_id=f"req-{i}",
                query="test",
                predicted_species="Tiger",
                confidence=0.9,
                latency_seconds=float(i + 1),
            )
        metrics = self.store.get_metrics()
        assert metrics["avg_latency_seconds"] == 5.5
        assert metrics["p95_latency_seconds"] == 10.0

    def test_get_feedback_for_request(self):
        self.store.add_feedback(request_id="req-001", was_correct=True, notes="Great!")
        self.store.add_feedback(request_id="req-002", was_correct=False)
        feedback = self.store.get_feedback_for_request("req-001")
        assert len(feedback) == 1
        assert feedback[0]["was_correct"] == 1
        assert feedback[0]["notes"] == "Great!"


# ─── API Endpoint Tests (with mocked pipeline) ──────────────────


class TestAPIEndpoints:
    """Test API endpoints using FastAPI TestClient with a mocked pipeline."""

    def setup_method(self):
        """Reset the pipeline state before each test."""
        # Reset the global pipeline reference in routes
        import src.api.routes as routes_module

        routes_module._pipeline = None

    def _get_client(self):
        """Get a test client with fresh app."""
        from src.api.main import create_app

        app = create_app()
        return TestClient(app)

    def test_health_endpoint(self):
        client = self._get_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data

    def test_metrics_endpoint(self):
        client = self._get_client()
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_seconds" in data
        assert "uptime_seconds" in data

    def test_identify_endpoint_with_mock(self):
        """Test /identify with a mocked pipeline."""
        mock_result = {
            "request_id": "test-123",
            "query": "orange striped cat",
            "response": {
                "species_name": "Bengal Tiger",
                "scientific_name": "Panthera tigris tigris",
                "confidence": 0.92,
                "reasoning": "The description matches a Bengal Tiger.",
                "key_features_matched": ["orange", "striped"],
                "habitat_match": "forest",
                "conservation_status": "Endangered",
                "geographic_match": True,
                "cited_sources": ["chunk_001"],
                "alternative_species": [],
            },
            "location": "Madhya Pradesh",
            "season": None,
            "inference_mode": "groq",
            "total_latency_seconds": 2.5,
            "chunks_retrieved": 15,
            "chunks_after_rerank": 5,
            "chunks_used": 3,
        }

        mock_pipeline = MagicMock()
        mock_pipeline.identify.return_value = mock_result

        import src.api.routes as routes_module

        routes_module._pipeline = mock_pipeline

        client = self._get_client()
        response = client.post(
            "/identify",
            json={
                "query": "I saw a large orange and black striped cat in the forest",
                "location": "Madhya Pradesh",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["species_name"] == "Bengal Tiger"
        assert data["confidence"] == 0.92
        assert data["request_id"] == "test-123"

    def test_identify_validation_error(self):
        """Test that invalid requests are rejected."""
        client = self._get_client()
        response = client.post("/identify", json={"query": "ab"})
        assert response.status_code == 422  # Validation error

    def test_feedback_endpoint(self):
        client = self._get_client()
        response = client.post(
            "/feedback",
            json={
                "request_id": "test-123",
                "was_correct": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert "feedback_id" in data

    def test_feedback_with_correction(self):
        client = self._get_client()
        response = client.post(
            "/feedback",
            json={
                "request_id": "test-123",
                "was_correct": False,
                "correct_species": "Indian Leopard",
                "notes": "It was spotted, not striped.",
            },
        )
        assert response.status_code == 200

    def test_docs_accessible(self):
        client = self._get_client()
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self):
        client = self._get_client()
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Wildlife Tracker API"

    def test_request_id_in_headers(self):
        client = self._get_client()
        response = client.get("/health")
        assert "x-request-id" in response.headers
        assert "x-response-time" in response.headers


# ─── Middleware Tests ────────────────────────────────────────────


class TestRateLimiting:
    def test_rate_limit_headers(self):
        from src.api.main import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/metrics")
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
