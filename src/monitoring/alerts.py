"""
Alerting Engine
Monitors system metrics and triggers alerts when thresholds are exceeded.
Alerts are logged, stored in SQLite, and optionally sent via webhook.

Alert Rules:
  - Latency spike: P95 latency > threshold (default: 15s)
  - Error rate: Error percentage > threshold (default: 10%)
  - Low accuracy: Feedback accuracy < threshold (default: 70%)
  - Low confidence: Average confidence < threshold (default: 0.5)
  - Component down: Health check component reports error
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH = Path("data/alerts.db")


# ─── Default Thresholds ─────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "p95_latency_seconds": 15.0,
    "error_rate_percent": 10.0,
    "min_accuracy_from_feedback": 0.70,
    "min_avg_confidence": 0.50,
    "min_requests_for_alert": 10,  # Don't alert until we have enough data
}


# ─── Alert Store ─────────────────────────────────────────────────


class AlertStore:
    """SQLite-backed alert history."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    current_value REAL,
                    threshold REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_rule
                ON alerts(rule_name, resolved)
            """)

    def add_alert(
        self,
        rule_name: str,
        severity: str,
        message: str,
        current_value: float = 0.0,
        threshold: float = 0.0,
    ) -> int:
        """Record a new alert. Returns alert_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO alerts
                (rule_name, severity, message, current_value, threshold, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rule_name,
                    severity,
                    message,
                    current_value,
                    threshold,
                    datetime.now().isoformat(),
                ),
            )
            alert_id = cursor.lastrowid
        logger.warning(f"ALERT [{severity}] {rule_name}: {message}")
        return alert_id

    def resolve_alert(self, rule_name: str):
        """Mark all unresolved alerts for a rule as resolved."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE alerts SET resolved = TRUE, resolved_at = ?
                WHERE rule_name = ? AND resolved = FALSE
                """,
                (datetime.now().isoformat(), rule_name),
            )

    def get_active_alerts(self) -> list[dict]:
        """Get all unresolved alerts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alerts WHERE resolved = FALSE ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_alert_history(self, limit: int = 50) -> list[dict]:
        """Get recent alert history (resolved and unresolved)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def has_active_alert(self, rule_name: str) -> bool:
        """Check if there's already an active alert for this rule."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM alerts WHERE rule_name = ? AND resolved = FALSE",
                (rule_name,),
            ).fetchone()
            return row[0] > 0


# ─── Alert Checker ───────────────────────────────────────────────


class AlertChecker:
    """
    Evaluates alert rules against current metrics.
    Designed to run periodically or on-demand.
    """

    def __init__(
        self,
        thresholds: dict = None,
        alert_store: AlertStore = None,
    ):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.store = alert_store or AlertStore()

    def check_all(self, metrics: dict, health: dict = None) -> list[dict]:
        """
        Run all alert rules against current metrics.

        Args:
            metrics: Output from FeedbackStore.get_metrics()
            health: Output from /health endpoint (optional)

        Returns:
            List of triggered alerts
        """
        triggered = []

        total = metrics.get("total_requests", 0)
        min_requests = self.thresholds["min_requests_for_alert"]

        if total < min_requests:
            return triggered  # Not enough data to alert

        # Rule 1: P95 Latency
        p95 = metrics.get("p95_latency_seconds", 0)
        threshold = self.thresholds["p95_latency_seconds"]
        if p95 > threshold:
            if not self.store.has_active_alert("p95_latency"):
                alert_id = self.store.add_alert(
                    rule_name="p95_latency",
                    severity="warning",
                    message=f"P95 latency is {p95:.1f}s (threshold: {threshold}s)",
                    current_value=p95,
                    threshold=threshold,
                )
                triggered.append(
                    {
                        "alert_id": alert_id,
                        "rule": "p95_latency",
                        "severity": "warning",
                        "value": p95,
                    }
                )
        else:
            self.store.resolve_alert("p95_latency")

        # Rule 2: Error Rate
        errors = metrics.get("error_count", 0)
        error_rate = (errors / total * 100) if total > 0 else 0
        threshold = self.thresholds["error_rate_percent"]
        if error_rate > threshold:
            if not self.store.has_active_alert("error_rate"):
                alert_id = self.store.add_alert(
                    rule_name="error_rate",
                    severity="critical" if error_rate > threshold * 2 else "warning",
                    message=f"Error rate is {error_rate:.1f}% (threshold: {threshold}%)",
                    current_value=error_rate,
                    threshold=threshold,
                )
                triggered.append(
                    {
                        "alert_id": alert_id,
                        "rule": "error_rate",
                        "severity": "critical" if error_rate > threshold * 2 else "warning",
                        "value": error_rate,
                    }
                )
        else:
            self.store.resolve_alert("error_rate")

        # Rule 3: Low Accuracy (from feedback)
        accuracy = metrics.get("accuracy_from_feedback")
        threshold = self.thresholds["min_accuracy_from_feedback"]
        feedback_count = metrics.get("feedback_count", 0)
        if accuracy is not None and feedback_count >= 5 and accuracy < threshold:
            if not self.store.has_active_alert("low_accuracy"):
                alert_id = self.store.add_alert(
                    rule_name="low_accuracy",
                    severity="warning",
                    message=f"Accuracy from feedback is {accuracy:.0%} (threshold: {threshold:.0%})",
                    current_value=accuracy,
                    threshold=threshold,
                )
                triggered.append(
                    {
                        "alert_id": alert_id,
                        "rule": "low_accuracy",
                        "severity": "warning",
                        "value": accuracy,
                    }
                )
        elif accuracy is not None and accuracy >= threshold:
            self.store.resolve_alert("low_accuracy")

        # Rule 4: Low Average Confidence
        avg_conf = metrics.get("avg_confidence", 0)
        threshold = self.thresholds["min_avg_confidence"]
        if avg_conf < threshold and total >= min_requests:
            if not self.store.has_active_alert("low_confidence"):
                alert_id = self.store.add_alert(
                    rule_name="low_confidence",
                    severity="info",
                    message=f"Average confidence is {avg_conf:.2f} (threshold: {threshold})",
                    current_value=avg_conf,
                    threshold=threshold,
                )
                triggered.append(
                    {
                        "alert_id": alert_id,
                        "rule": "low_confidence",
                        "severity": "info",
                        "value": avg_conf,
                    }
                )
        else:
            self.store.resolve_alert("low_confidence")

        # Rule 5: Component Health (if provided)
        if health and "components" in health:
            for comp_name, comp_info in health["components"].items():
                if comp_info.get("status") == "error":
                    rule = f"component_{comp_name}"
                    if not self.store.has_active_alert(rule):
                        alert_id = self.store.add_alert(
                            rule_name=rule,
                            severity="critical",
                            message=f"Component '{comp_name}' is down: {comp_info.get('message', '')}",
                            current_value=0,
                            threshold=1,
                        )
                        triggered.append(
                            {
                                "alert_id": alert_id,
                                "rule": rule,
                                "severity": "critical",
                            }
                        )
                else:
                    self.store.resolve_alert(f"component_{comp_name}")

        return triggered

    def get_status(self) -> dict:
        """Get current alert status summary."""
        active = self.store.get_active_alerts()
        return {
            "active_alert_count": len(active),
            "active_alerts": active,
            "thresholds": self.thresholds,
        }
