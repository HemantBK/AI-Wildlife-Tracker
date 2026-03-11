"""
Metrics Collector
In-memory metrics collection with periodic SQLite persistence.
Tracks request latencies, species distribution, error rates, and more.
Provides data for the monitoring dashboard and alert system.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """A single request metric record."""

    request_id: str
    query: str
    species: str
    confidence: float
    latency_seconds: float
    inference_mode: str
    chunks_used: int
    status: str  # "success", "declined", "error"
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Thread-safe in-memory metrics collector.
    Maintains a rolling window of recent metrics for fast dashboard access.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._recent: deque[RequestMetric] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Counters (never reset)
        self._total_requests = 0
        self._total_errors = 0
        self._total_declined = 0

        # Species counts
        self._species_counts: dict[str, int] = defaultdict(int)

        # Latency histogram buckets (seconds)
        self._latency_buckets = {
            "0-1s": 0,
            "1-3s": 0,
            "3-5s": 0,
            "5-10s": 0,
            "10-15s": 0,
            "15s+": 0,
        }

    def record(self, metric: RequestMetric):
        """Record a new request metric."""
        with self._lock:
            self._recent.append(metric)
            self._total_requests += 1

            if metric.status == "error":
                self._total_errors += 1
            elif metric.status == "declined":
                self._total_declined += 1
            else:
                self._species_counts[metric.species] += 1

            # Latency bucket
            lat = metric.latency_seconds
            if lat <= 1:
                self._latency_buckets["0-1s"] += 1
            elif lat <= 3:
                self._latency_buckets["1-3s"] += 1
            elif lat <= 5:
                self._latency_buckets["3-5s"] += 1
            elif lat <= 10:
                self._latency_buckets["5-10s"] += 1
            elif lat <= 15:
                self._latency_buckets["10-15s"] += 1
            else:
                self._latency_buckets["15s+"] += 1

    def get_summary(self) -> dict:
        """Get a summary of all collected metrics."""
        with self._lock:
            latencies = [m.latency_seconds for m in self._recent if m.status != "error"]
            confidences = [m.confidence for m in self._recent if m.status == "success"]

            if latencies:
                sorted_lat = sorted(latencies)
                avg_latency = sum(sorted_lat) / len(sorted_lat)
                p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
                p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
                p99 = sorted_lat[min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)]
            else:
                avg_latency = p50 = p95 = p99 = 0.0

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Top species
            top_species = sorted(
                self._species_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            # Recent trend (last 100 requests)
            recent_100 = list(self._recent)[-100:]
            recent_errors = sum(1 for m in recent_100 if m.status == "error")
            recent_error_rate = recent_errors / len(recent_100) * 100 if recent_100 else 0

            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "total_declined": self._total_declined,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "latency": {
                    "avg": round(avg_latency, 3),
                    "p50": round(p50, 3),
                    "p95": round(p95, 3),
                    "p99": round(p99, 3),
                },
                "latency_histogram": dict(self._latency_buckets),
                "avg_confidence": round(avg_confidence, 3),
                "top_species": [{"species": s, "count": c} for s, c in top_species],
                "recent_error_rate_percent": round(recent_error_rate, 1),
                "window_size": len(self._recent),
            }

    def get_recent(self, n: int = 20) -> list[dict]:
        """Get the N most recent request metrics."""
        with self._lock:
            recent = list(self._recent)[-n:]
            return [
                {
                    "request_id": m.request_id,
                    "query": m.query[:80],
                    "species": m.species,
                    "confidence": m.confidence,
                    "latency_s": round(m.latency_seconds, 2),
                    "status": m.status,
                    "inference_mode": m.inference_mode,
                    "timestamp": m.timestamp,
                }
                for m in reversed(recent)
            ]

    def get_latency_series(self, n: int = 100) -> list[dict]:
        """Get latency time series for charting."""
        with self._lock:
            recent = list(self._recent)[-n:]
            return [
                {
                    "timestamp": m.timestamp,
                    "latency_s": round(m.latency_seconds, 3),
                    "status": m.status,
                }
                for m in recent
            ]


# ─── Global Collector Instance ───────────────────────────────────

_collector = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector (singleton)."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
