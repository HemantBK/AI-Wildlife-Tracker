"""
Feedback Storage
SQLite-backed feedback collection for continuous improvement.
Stores user feedback on species identifications for future fine-tuning and evaluation.
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/feedback.db")


class FeedbackStore:
    """SQLite-backed feedback storage."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    correct_species TEXT,
                    was_correct BOOLEAN NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                    request_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    predicted_species TEXT,
                    confidence REAL,
                    location TEXT,
                    season TEXT,
                    inference_mode TEXT,
                    latency_seconds REAL,
                    chunks_used INTEGER,
                    status TEXT DEFAULT 'success',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_log_created
                ON request_log(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_request
                ON feedback(request_id)
            """)
        logger.info(f"Feedback store initialized at {self.db_path}")

    def log_request(
        self,
        request_id: str,
        query: str,
        predicted_species: str,
        confidence: float,
        location: str | None = None,
        season: str | None = None,
        inference_mode: str | None = None,
        latency_seconds: float = 0.0,
        chunks_used: int = 0,
        status: str = "success",
    ):
        """Log a pipeline request for metrics tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO request_log
                (request_id, query, predicted_species, confidence, location,
                 season, inference_mode, latency_seconds, chunks_used, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    query,
                    predicted_species,
                    confidence,
                    location,
                    season,
                    inference_mode,
                    latency_seconds,
                    chunks_used,
                    status,
                    datetime.now().isoformat(),
                ),
            )

    def add_feedback(
        self,
        request_id: str,
        was_correct: bool,
        correct_species: str | None = None,
        notes: str | None = None,
    ) -> str:
        """
        Store user feedback on an identification.

        Returns:
            feedback_id (UUID)
        """
        feedback_id = str(uuid.uuid4())[:12]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback
                (feedback_id, request_id, correct_species, was_correct, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    request_id,
                    correct_species,
                    was_correct,
                    notes,
                    datetime.now().isoformat(),
                ),
            )
        logger.info(f"Feedback stored: {feedback_id} for request {request_id}")
        return feedback_id

    def get_metrics(self) -> dict:
        """Calculate system metrics from request and feedback logs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total requests
            row = conn.execute("SELECT COUNT(*) as cnt FROM request_log").fetchone()
            total = row["cnt"]

            if total == 0:
                # Still count feedback even if no requests logged
                row = conn.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()
                fb_count = row["cnt"]
                fb_accuracy = None
                if fb_count > 0:
                    row = conn.execute(
                        "SELECT AVG(CAST(was_correct AS FLOAT)) as acc FROM feedback"
                    ).fetchone()
                    fb_accuracy = round(row["acc"], 4) if row["acc"] is not None else None

                return {
                    "total_requests": 0,
                    "successful_identifications": 0,
                    "declined_identifications": 0,
                    "error_count": 0,
                    "avg_latency_seconds": 0.0,
                    "p95_latency_seconds": 0.0,
                    "avg_confidence": 0.0,
                    "feedback_count": fb_count,
                    "accuracy_from_feedback": fb_accuracy,
                    "top_species": [],
                    "requests_by_hour": [],
                }

            # Success / decline / error counts
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM request_log WHERE status='success' AND predicted_species != 'DECLINED'"
            ).fetchone()
            successes = row["cnt"]

            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM request_log WHERE predicted_species='DECLINED'"
            ).fetchone()
            declined = row["cnt"]

            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM request_log WHERE status='error'"
            ).fetchone()
            errors = row["cnt"]

            # Latency stats
            row = conn.execute(
                "SELECT AVG(latency_seconds) as avg_lat FROM request_log WHERE status='success'"
            ).fetchone()
            avg_latency = row["avg_lat"] or 0.0

            # P95 latency
            latencies = conn.execute(
                "SELECT latency_seconds FROM request_log WHERE status='success' ORDER BY latency_seconds"
            ).fetchall()
            if latencies:
                idx = int(len(latencies) * 0.95)
                p95 = latencies[min(idx, len(latencies) - 1)]["latency_seconds"]
            else:
                p95 = 0.0

            # Avg confidence
            row = conn.execute(
                "SELECT AVG(confidence) as avg_conf FROM request_log WHERE status='success'"
            ).fetchone()
            avg_confidence = row["avg_conf"] or 0.0

            # Feedback stats
            row = conn.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()
            feedback_count = row["cnt"]

            accuracy = None
            if feedback_count > 0:
                row = conn.execute(
                    "SELECT AVG(CAST(was_correct AS FLOAT)) as acc FROM feedback"
                ).fetchone()
                accuracy = round(row["acc"], 4) if row["acc"] is not None else None

            # Top species
            top_species = conn.execute(
                """
                SELECT predicted_species, COUNT(*) as cnt
                FROM request_log
                WHERE predicted_species != 'DECLINED' AND status='success'
                GROUP BY predicted_species
                ORDER BY cnt DESC
                LIMIT 10
                """
            ).fetchall()

            # Requests by hour (last 24 hours)
            requests_by_hour = conn.execute(
                """
                SELECT strftime('%Y-%m-%d %H:00', created_at) as hour,
                       COUNT(*) as cnt
                FROM request_log
                WHERE created_at > datetime('now', '-24 hours')
                GROUP BY hour
                ORDER BY hour
                """
            ).fetchall()

            return {
                "total_requests": total,
                "successful_identifications": successes,
                "declined_identifications": declined,
                "error_count": errors,
                "avg_latency_seconds": round(avg_latency, 3),
                "p95_latency_seconds": round(p95, 3),
                "avg_confidence": round(avg_confidence, 3),
                "feedback_count": feedback_count,
                "accuracy_from_feedback": accuracy,
                "top_species": [
                    {"species": r["predicted_species"], "count": r["cnt"]} for r in top_species
                ],
                "requests_by_hour": [
                    {"hour": r["hour"], "count": r["cnt"]} for r in requests_by_hour
                ],
            }

    def get_feedback_for_request(self, request_id: str) -> list[dict]:
        """Get all feedback for a specific request."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feedback WHERE request_id = ?",
                (request_id,),
            ).fetchall()
            return [dict(r) for r in rows]
