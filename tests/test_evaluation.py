"""
Tests for the evaluation framework.
Verifies metrics calculation, quality gates, and golden dataset integrity.
"""

from src.evaluation.golden_dataset import GOLDEN_DATASET, get_stats
from src.evaluation.metrics import (
    aggregate_metrics,
    check_quality_gates,
    score_answer_correctness,
    score_confidence_calibration,
    score_geographic_accuracy,
    score_refusal_accuracy,
)

# ─── Golden Dataset Integrity ────────────────────────────────


class TestGoldenDataset:
    def test_dataset_has_enough_queries(self):
        assert len(GOLDEN_DATASET) >= 50, f"Need at least 50 queries, got {len(GOLDEN_DATASET)}"

    def test_difficulty_distribution(self):
        stats = get_stats()
        assert stats["by_difficulty"]["easy"] >= 15
        assert stats["by_difficulty"]["medium"] >= 10
        assert stats["by_difficulty"]["hard"] >= 5
        assert stats["by_difficulty"]["trick"] >= 5

    def test_all_queries_have_required_fields(self):
        for q in GOLDEN_DATASET:
            assert "id" in q, f"Missing id in {q.get('query', '?')[:40]}"
            assert "query" in q, "Missing query"
            assert "expected_species" in q, f"Missing expected_species in {q['id']}"
            assert "difficulty" in q, f"Missing difficulty in {q['id']}"

    def test_trick_queries_expect_declined(self):
        tricks = [q for q in GOLDEN_DATASET if q["difficulty"] == "trick"]
        for q in tricks:
            assert q["expected_species"] == "DECLINED", f"Trick {q['id']} should expect DECLINED"

    def test_unique_ids(self):
        ids = [q["id"] for q in GOLDEN_DATASET]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"


# ─── Metric Calculations ────────────────────────────────────


class TestAnswerCorrectness:
    def test_exact_match(self):
        assert score_answer_correctness("Bengal Tiger", "Bengal Tiger") == 1.0

    def test_case_insensitive(self):
        assert score_answer_correctness("bengal tiger", "Bengal Tiger") == 1.0

    def test_partial_match(self):
        score = score_answer_correctness("Tiger", "Bengal Tiger")
        assert score >= 0.5

    def test_wrong_species(self):
        assert score_answer_correctness("Indian Cobra", "Bengal Tiger") == 0.0

    def test_scientific_name_match(self):
        score = score_answer_correctness(
            "Tiger", "Bengal Tiger", "Panthera tigris tigris", "Panthera tigris tigris"
        )
        assert score >= 0.8

    def test_empty_prediction(self):
        assert score_answer_correctness("", "Bengal Tiger") == 0.0


class TestGeographicAccuracy:
    def test_no_location_returns_perfect(self):
        assert score_geographic_accuracy("Tiger", True, None, []) == 1.0

    def test_match_when_geographic_true(self):
        assert score_geographic_accuracy("Tiger", True, "Madhya Pradesh", ["Madhya Pradesh"]) == 1.0

    def test_mismatch(self):
        assert score_geographic_accuracy("Tiger", False, "Goa", ["Madhya Pradesh"]) == 0.0


class TestRefusalAccuracy:
    def test_correct_decline_for_trick(self):
        assert score_refusal_accuracy("DECLINED", 0.0, "DECLINED") == 1.0

    def test_wrong_answer_for_trick(self):
        assert score_refusal_accuracy("Bengal Tiger", 0.9, "DECLINED") == 0.0

    def test_correct_answer_for_normal(self):
        assert score_refusal_accuracy("Bengal Tiger", 0.9, "Bengal Tiger") == 1.0

    def test_incorrect_decline_for_normal(self):
        assert score_refusal_accuracy("DECLINED", 0.0, "Bengal Tiger") == 0.0

    def test_low_confidence_counts_as_decline(self):
        assert score_refusal_accuracy("Something", 0.1, "DECLINED") == 1.0


class TestConfidenceCalibration:
    def test_correct_high_confidence_good(self):
        score = score_confidence_calibration(0.95, True)
        assert score >= 0.9

    def test_wrong_low_confidence_ok(self):
        score = score_confidence_calibration(0.1, False)
        assert score >= 0.8

    def test_wrong_high_confidence_bad(self):
        score = score_confidence_calibration(0.95, False)
        assert score <= 0.1


# ─── Aggregation & Quality Gates ─────────────────────────────


class TestAggregation:
    def test_aggregate_basic(self):
        results = [
            {
                "difficulty": "easy",
                "answer_correctness": 1.0,
                "geographic_accuracy": 1.0,
                "refusal_accuracy": 1.0,
                "citation_precision": 1.0,
                "confidence_calibration": 0.9,
                "latency_s": 1.0,
            },
            {
                "difficulty": "easy",
                "answer_correctness": 0.0,
                "geographic_accuracy": 1.0,
                "refusal_accuracy": 1.0,
                "citation_precision": 0.0,
                "confidence_calibration": 0.5,
                "latency_s": 2.0,
            },
        ]
        agg = aggregate_metrics(results)
        assert agg["overall"]["answer_correctness"] == 0.5
        assert agg["overall"]["total_queries"] == 2

    def test_quality_gates_pass(self):
        agg = {
            "overall": {
                "answer_correctness": 0.90,
                "geographic_accuracy": 0.95,
                "refusal_accuracy": 0.85,
                "p95_latency_s": 5.0,
            }
        }
        result = check_quality_gates(agg)
        assert result["all_passed"] is True

    def test_quality_gates_fail(self):
        agg = {
            "overall": {
                "answer_correctness": 0.50,  # Below 0.75 threshold
                "geographic_accuracy": 0.95,
                "refusal_accuracy": 0.85,
                "p95_latency_s": 5.0,
            }
        }
        result = check_quality_gates(agg)
        assert result["all_passed"] is False
        assert result["gates"]["answer_correctness"]["passed"] is False
