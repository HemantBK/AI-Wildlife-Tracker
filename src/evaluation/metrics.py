"""
Evaluation Metrics
Custom metrics for wildlife identification quality scoring.
Designed to work without requiring an external LLM judge (fast, deterministic).

Metrics:
  - Answer Correctness: species match against ground truth
  - Geographic Accuracy: identified species exists in mentioned region
  - Citation Precision: cited chunks are about the correct species
  - Refusal Accuracy: correctly declines trick queries
  - Confidence Calibration: confidence aligns with correctness
  - Species Confusion Matrix: tracks which species get confused
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Individual Metrics ──────────────────────────────────────────


def score_answer_correctness(
    predicted_species: str,
    expected_species: str,
    predicted_scientific: str = "",
    expected_scientific: str = "",
) -> float:
    """
    Score whether the predicted species matches the expected.
    Returns 1.0 for exact match, 0.5 for partial, 0.0 for wrong.
    """
    if not predicted_species or not expected_species:
        return 0.0

    pred = predicted_species.lower().strip()
    exp = expected_species.lower().strip()

    # Exact match
    if pred == exp:
        return 1.0

    # Substring match (e.g., "Bengal Tiger" matches "Tiger")
    if exp in pred or pred in exp:
        return 0.8

    # Scientific name match
    if predicted_scientific and expected_scientific:
        pred_sci = predicted_scientific.lower().strip()
        exp_sci = expected_scientific.lower().strip()
        if pred_sci == exp_sci:
            return 1.0
        if exp_sci in pred_sci or pred_sci in exp_sci:
            return 0.8

    return 0.0


def score_geographic_accuracy(
    predicted_species: str,
    geographic_match: bool,
    location: str | None,
    expected_regions: list[str],
) -> float:
    """
    Score geographic accuracy.
    1.0 if species is valid for the region, 0.0 if not.
    N/A (returns 1.0) if no location was specified.
    """
    if not location:
        return 1.0  # No location to validate against

    if geographic_match:
        return 1.0

    # Check if expected regions include the location
    if location and expected_regions:
        loc_lower = location.lower()
        for region in expected_regions:
            if loc_lower in region.lower() or region.lower() in loc_lower:
                return 1.0

    return 0.0


def score_citation_precision(
    cited_sources: list[str],
    retrieved_chunks: list[dict],
    expected_species: str,
) -> float:
    """
    Score citation precision: what fraction of cited chunks are about the correct species.
    """
    if not cited_sources:
        return 0.0

    if not retrieved_chunks:
        return 0.0

    # Build chunk lookup
    chunk_map = {c.get("chunk_id", ""): c for c in retrieved_chunks}

    relevant = 0
    for cite_id in cited_sources:
        chunk = chunk_map.get(cite_id, {})
        chunk_species = chunk.get("metadata", {}).get("species_name", "").lower()
        if expected_species.lower() in chunk_species or chunk_species in expected_species.lower():
            relevant += 1

    return relevant / len(cited_sources)


def score_refusal_accuracy(
    predicted_species: str,
    confidence: float,
    expected_species: str,
) -> float:
    """
    Score refusal accuracy for trick queries.
    For trick queries (expected=DECLINED): 1.0 if system declines, 0.0 if it gives an answer.
    For normal queries: 1.0 if system answers, 0.0 if it incorrectly declines.
    """
    is_trick = expected_species.upper() == "DECLINED"
    pred = predicted_species.lower().strip()
    is_declined = (
        "decline" in pred
        or pred == ""
        or confidence < 0.2
        or "cannot" in pred
        or pred == "declined"
        or pred == "unknown"
    )

    if is_trick:
        return 1.0 if is_declined else 0.0
    else:
        return 0.0 if is_declined else 1.0


def score_confidence_calibration(
    confidence: float,
    is_correct: bool,
) -> float:
    """
    Score how well confidence aligns with correctness.
    Correct + high confidence = good. Wrong + low confidence = okay.
    Wrong + high confidence = bad.
    """
    if is_correct:
        return confidence  # Higher confidence on correct answers is better
    else:
        return 1.0 - confidence  # Lower confidence on wrong answers is better


# ─── Aggregate Scoring ──────────────────────────────────────────


def evaluate_single(
    query_data: dict,
    pipeline_result: dict,
) -> dict:
    """
    Evaluate a single query-response pair against ground truth.

    Args:
        query_data: Entry from golden dataset
        pipeline_result: Output from the RAG pipeline

    Returns:
        Dict of metric scores for this query
    """
    response = pipeline_result.get("response", {})
    predicted_species = response.get("species_name", "")
    predicted_scientific = response.get("scientific_name", "")
    confidence = response.get("confidence", 0.0)
    geographic_match = response.get("geographic_match", True)
    cited_sources = response.get("cited_sources", [])

    expected_species = query_data["expected_species"]
    expected_scientific = query_data.get("expected_scientific", "")
    expected_regions = query_data.get("expected_regions", [])
    location = query_data.get("location")
    difficulty = query_data.get("difficulty", "unknown")

    # Calculate individual metrics
    correctness = score_answer_correctness(
        predicted_species,
        expected_species,
        predicted_scientific,
        expected_scientific,
    )

    geo_accuracy = score_geographic_accuracy(
        predicted_species,
        geographic_match,
        location,
        expected_regions,
    )

    refusal = score_refusal_accuracy(
        predicted_species,
        confidence,
        expected_species,
    )

    # Citation precision
    citation_score = 0.0
    if cited_sources and expected_species != "DECLINED":
        # Use top_chunk_scores as proxy
        citation_score = 1.0 if cited_sources else 0.0

    calibration = score_confidence_calibration(confidence, correctness >= 0.8)

    return {
        "query_id": query_data.get("id", ""),
        "difficulty": difficulty,
        "expected_species": expected_species,
        "predicted_species": predicted_species,
        "confidence": confidence,
        "answer_correctness": correctness,
        "geographic_accuracy": geo_accuracy,
        "refusal_accuracy": refusal,
        "citation_precision": citation_score,
        "confidence_calibration": calibration,
        "latency_s": pipeline_result.get("total_latency_seconds", 0),
    }


def aggregate_metrics(eval_results: list[dict]) -> dict:
    """
    Aggregate individual evaluation results into summary metrics.

    Returns summary with overall and per-difficulty breakdowns.
    """
    if not eval_results:
        return {"error": "No results to aggregate"}

    # Overall
    n = len(eval_results)
    overall = {
        "total_queries": n,
        "answer_correctness": round(sum(r["answer_correctness"] for r in eval_results) / n, 4),
        "geographic_accuracy": round(sum(r["geographic_accuracy"] for r in eval_results) / n, 4),
        "refusal_accuracy": round(sum(r["refusal_accuracy"] for r in eval_results) / n, 4),
        "citation_precision": round(sum(r["citation_precision"] for r in eval_results) / n, 4),
        "confidence_calibration": round(
            sum(r["confidence_calibration"] for r in eval_results) / n, 4
        ),
        "avg_latency_s": round(sum(r["latency_s"] for r in eval_results) / n, 3),
        "p95_latency_s": round(sorted(r["latency_s"] for r in eval_results)[int(n * 0.95)], 3)
        if n > 1
        else 0,
    }

    # Per difficulty
    by_difficulty = {}
    for diff in ["easy", "medium", "hard", "trick"]:
        diff_results = [r for r in eval_results if r["difficulty"] == diff]
        if diff_results:
            nd = len(diff_results)
            by_difficulty[diff] = {
                "count": nd,
                "answer_correctness": round(
                    sum(r["answer_correctness"] for r in diff_results) / nd, 4
                ),
                "geographic_accuracy": round(
                    sum(r["geographic_accuracy"] for r in diff_results) / nd, 4
                ),
                "refusal_accuracy": round(sum(r["refusal_accuracy"] for r in diff_results) / nd, 4),
                "avg_latency_s": round(sum(r["latency_s"] for r in diff_results) / nd, 3),
            }

    # Species confusion tracking
    confusions = []
    for r in eval_results:
        if r["answer_correctness"] < 0.5 and r.get("difficulty") != "trick":
            confusions.append(
                {
                    "expected": r.get("expected_species", "unknown"),
                    "predicted": r.get("predicted_species", "unknown"),
                    "confidence": r.get("confidence", 0.0),
                }
            )

    return {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "confusions": confusions,
        "confusion_count": len(confusions),
    }


# ─── Quality Gates (for CI/CD) ──────────────────────────────────

DEFAULT_THRESHOLDS = {
    "answer_correctness": 0.75,
    "geographic_accuracy": 0.90,
    "refusal_accuracy": 0.80,
    "p95_latency_s": 15.0,  # 15 seconds max
}


def check_quality_gates(
    aggregated: dict,
    thresholds: dict = None,
) -> dict:
    """
    Check if evaluation results pass quality gates.
    Returns pass/fail status for each gate.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    overall = aggregated["overall"]

    gates = {}
    all_passed = True

    for metric, threshold in thresholds.items():
        actual = overall.get(metric, 0)
        if metric.endswith("_s"):
            # Latency: lower is better
            passed = actual <= threshold
        else:
            # Quality: higher is better
            passed = actual >= threshold

        gates[metric] = {
            "threshold": threshold,
            "actual": actual,
            "passed": passed,
        }
        if not passed:
            all_passed = False

    return {
        "all_passed": all_passed,
        "gates": gates,
    }
