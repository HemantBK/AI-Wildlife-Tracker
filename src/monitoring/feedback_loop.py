"""
Feedback Loop
Uses collected user feedback (corrections) to improve retrieval and prompts.

Strategies:
1. Hard Negative Mining: When a user says the answer was wrong, record the
   (query, wrong_species, correct_species) triple for future re-ranking tuning.
2. Correction Catalog: Maintain a catalog of common misidentifications to
   add disambiguation hints to prompts.
3. Retrieval Boosting: When a species is frequently the correct answer for
   certain query patterns, boost its chunks in retrieval.
4. Quality Reports: Generate weekly summaries of accuracy by species,
   common failure modes, and improvement suggestions.
"""

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/feedback.db")
CORRECTIONS_PATH = Path("data/corrections.json")


class FeedbackLoop:
    """
    Analyzes user feedback to identify systematic errors
    and generate improvement actions.
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH

    def _get_feedback_data(self) -> list[dict]:
        """Load all feedback from the feedback store database."""
        if not self.db_path.exists():
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"Could not read feedback DB: {e}")
            return []

    def _get_request_data(self) -> list[dict]:
        """Load all request logs from the feedback store database."""
        if not self.db_path.exists():
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM request_log ORDER BY created_at DESC").fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"Could not read request log: {e}")
            return []

    # ─── Analysis: Confusion Matrix ─────────────────────────────

    def get_confusion_pairs(self) -> list[dict]:
        """
        Find species that are commonly confused with each other.
        Returns pairs like: (predicted: "Indian Leopard", correct: "Bengal Tiger", count: 5)
        """
        feedback = self._get_feedback_data()
        confusions = Counter()

        for fb in feedback:
            if not fb.get("was_correct") and fb.get("correct_species"):
                # Join with request log to get predicted species
                requests = self._get_request_data()
                for req in requests:
                    if req.get("request_id") == fb.get("request_id"):
                        predicted = req.get("predicted_species", "")
                        correct = fb["correct_species"]
                        if predicted and correct and predicted != correct:
                            confusions[(predicted, correct)] += 1
                        break

        return [
            {
                "predicted": pair[0],
                "correct": pair[1],
                "count": count,
            }
            for pair, count in confusions.most_common(20)
        ]

    # ─── Analysis: Accuracy by Species ──────────────────────────

    def get_accuracy_by_species(self) -> dict:
        """
        Calculate accuracy for each species based on feedback.
        Returns: {species: {correct: N, incorrect: N, accuracy: float}}
        """
        feedback = self._get_feedback_data()
        requests = self._get_request_data()

        # Map request_id to predicted species
        req_species = {}
        for req in requests:
            req_species[req.get("request_id", "")] = req.get("predicted_species", "")

        species_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0})

        for fb in feedback:
            rid = fb.get("request_id", "")
            species = req_species.get(rid, "Unknown")
            if species in ("DECLINED", "ERROR", "Unknown"):
                continue
            if fb.get("was_correct"):
                species_stats[species]["correct"] += 1
            else:
                species_stats[species]["incorrect"] += 1

        result = {}
        for species, stats in species_stats.items():
            total = stats["correct"] + stats["incorrect"]
            result[species] = {
                **stats,
                "total": total,
                "accuracy": stats["correct"] / total if total > 0 else 0,
            }

        return dict(sorted(result.items(), key=lambda x: x[1]["accuracy"]))

    # ─── Analysis: Common Failure Patterns ──────────────────────

    def get_failure_patterns(self) -> list[dict]:
        """
        Identify common patterns in failed identifications.
        Analyzes queries that were marked incorrect to find patterns.
        """
        feedback = self._get_feedback_data()
        requests = self._get_request_data()

        req_map = {r.get("request_id"): r for r in requests}
        failures = []

        for fb in feedback:
            if not fb.get("was_correct"):
                req = req_map.get(fb.get("request_id"), {})
                failures.append(
                    {
                        "query": req.get("query", ""),
                        "predicted": req.get("predicted_species", ""),
                        "correct": fb.get("correct_species", ""),
                        "location": req.get("location", ""),
                        "confidence": req.get("confidence", 0),
                        "notes": fb.get("notes", ""),
                    }
                )

        return failures

    # ─── Generate Corrections Catalog ───────────────────────────

    def generate_corrections_catalog(self) -> dict:
        """
        Generate a corrections catalog that can be used to improve prompts.

        The catalog contains:
        - Confusion pairs (species A mistaken for B)
        - Common failure queries
        - Disambiguation hints

        Saved to data/corrections.json for use by the prompt builder.
        """
        confusions = self.get_confusion_pairs()
        failures = self.get_failure_patterns()
        accuracy = self.get_accuracy_by_species()

        # Generate disambiguation hints
        hints = []
        for conf in confusions:
            if conf["count"] >= 2:
                hints.append(
                    {
                        "trigger": f"Similar to {conf['predicted']}",
                        "hint": (
                            f"Note: {conf['predicted']} and {conf['correct']} are often confused. "
                            f"Key differences should be checked carefully."
                        ),
                        "predicted": conf["predicted"],
                        "correct": conf["correct"],
                        "occurrences": conf["count"],
                    }
                )

        # Identify weak species (accuracy < 60%)
        weak_species = [
            {"species": sp, **stats}
            for sp, stats in accuracy.items()
            if stats["total"] >= 3 and stats["accuracy"] < 0.6
        ]

        catalog = {
            "generated_at": datetime.now().isoformat(),
            "total_feedback": len(self._get_feedback_data()),
            "confusion_pairs": confusions,
            "disambiguation_hints": hints,
            "weak_species": weak_species,
            "recent_failures": failures[:20],
            "accuracy_by_species": accuracy,
        }

        # Save to disk
        CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CORRECTIONS_PATH, "w") as f:
            json.dump(catalog, f, indent=2)
        logger.info(f"Corrections catalog saved to {CORRECTIONS_PATH}")

        return catalog

    # ─── Generate Quality Report ────────────────────────────────

    def generate_quality_report(self, days: int = 7) -> dict:
        """
        Generate a quality report for the last N days.
        Returns a structured report with metrics and recommendations.
        """
        feedback = self._get_feedback_data()
        requests = self._get_request_data()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Filter to recent data
        recent_fb = [f for f in feedback if f.get("created_at", "") >= cutoff]
        recent_req = [r for r in requests if r.get("created_at", "") >= cutoff]

        total_requests = len(recent_req)
        total_feedback = len(recent_fb)
        correct = sum(1 for f in recent_fb if f.get("was_correct"))
        incorrect = total_feedback - correct

        accuracy = correct / total_feedback if total_feedback > 0 else None
        feedback_rate = total_feedback / total_requests if total_requests > 0 else 0

        # Latency stats
        latencies = [r.get("latency_seconds", 0) for r in recent_req if r.get("latency_seconds")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Error rate
        errors = sum(1 for r in recent_req if r.get("status") == "error")
        error_rate = errors / total_requests if total_requests > 0 else 0

        # Top confusions
        confusions = self.get_confusion_pairs()

        # Recommendations
        recommendations = []
        if accuracy is not None and accuracy < 0.7:
            recommendations.append(
                "Accuracy is below 70%. Review confusion pairs and add disambiguation to prompts."
            )
        if avg_latency > 10:
            recommendations.append(
                "Average latency is high. Consider switching to a faster model or reducing chunk count."
            )
        if error_rate > 0.05:
            recommendations.append(
                f"Error rate is {error_rate:.0%}. Check LLM backend health and API keys."
            )
        if feedback_rate < 0.1 and total_requests > 20:
            recommendations.append(
                "Low feedback rate. Consider making the feedback UI more prominent."
            )
        if confusions:
            top_conf = confusions[0]
            recommendations.append(
                f"Most common confusion: {top_conf['predicted']} vs {top_conf['correct']} "
                f"({top_conf['count']} times). Add disambiguation hints."
            )
        if not recommendations:
            recommendations.append("System is performing well. No immediate actions needed.")

        report = {
            "period": f"Last {days} days",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_requests": total_requests,
                "total_feedback": total_feedback,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": round(accuracy, 4) if accuracy else None,
                "feedback_rate": round(feedback_rate, 4),
                "avg_latency_seconds": round(avg_latency, 2),
                "error_rate": round(error_rate, 4),
            },
            "top_confusions": confusions[:5],
            "accuracy_by_species": self.get_accuracy_by_species(),
            "recommendations": recommendations,
        }

        return report


# ─── CLI Entry Point ─────────────────────────────────────────────


def main():
    """Generate feedback analysis reports."""
    import sys

    loop = FeedbackLoop()

    if "--catalog" in sys.argv:
        print("Generating corrections catalog...")
        catalog = loop.generate_corrections_catalog()
        print(f"  Confusion pairs: {len(catalog['confusion_pairs'])}")
        print(f"  Disambiguation hints: {len(catalog['disambiguation_hints'])}")
        print(f"  Weak species: {len(catalog['weak_species'])}")
        print(f"  Saved to: {CORRECTIONS_PATH}")

    elif "--report" in sys.argv:
        days = 7
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])

        print(f"Generating quality report (last {days} days)...")
        report = loop.generate_quality_report(days=days)
        summary = report["summary"]

        print(f"\n{'=' * 50}")
        print(f"  Quality Report — Last {days} Days")
        print(f"{'=' * 50}")
        print(f"  Requests:     {summary['total_requests']}")
        print(f"  Feedback:     {summary['total_feedback']}")
        print(
            f"  Accuracy:     {summary['accuracy']:.0%}"
            if summary["accuracy"]
            else "  Accuracy:     N/A"
        )
        print(f"  Avg Latency:  {summary['avg_latency_seconds']:.2f}s")
        print(f"  Error Rate:   {summary['error_rate']:.1%}")
        print("\n  Recommendations:")
        for rec in report["recommendations"]:
            print(f"    - {rec}")
        print()

    else:
        print("Usage:")
        print("  python -m src.monitoring.feedback_loop --catalog")
        print("  python -m src.monitoring.feedback_loop --report [--days 7]")


if __name__ == "__main__":
    main()
