"""
Main Evaluation Runner
Runs the full RAG pipeline against the golden dataset,
computes all metrics, checks quality gates, and outputs CI-ready results.

Usage:
  python -m src.evaluation.evaluator                  # Full evaluation
  python -m src.evaluation.evaluator --quick           # Quick (10 queries)
  python -m src.evaluation.evaluator --difficulty easy  # Only easy queries
  python -m src.evaluation.evaluator --ci              # CI mode (exit code 1 on failure)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from src.evaluation.golden_dataset import get_stats, load_dataset, save_dataset
from src.evaluation.metrics import aggregate_metrics, check_quality_gates, evaluate_single

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/evaluation")


def run_evaluation(
    queries: list[dict] = None,
    quick: bool = False,
    difficulty: str = None,
    prompt_version: int = 1,
) -> dict:
    """
    Run the full evaluation pipeline.

    Args:
        queries: Override query list (default: golden dataset)
        quick: If True, only run 10 queries for fast feedback
        difficulty: Filter to specific difficulty level
        prompt_version: Which prompt version to evaluate

    Returns:
        Full evaluation results dict
    """
    # Load pipeline (lazy import to avoid loading models if not needed)
    from src.rag.pipeline import WildlifeRAGPipeline

    # Load queries
    if queries is None:
        queries = load_dataset()

    if difficulty:
        queries = [q for q in queries if q["difficulty"] == difficulty]
        logger.info(f"Filtered to {len(queries)} {difficulty} queries")

    if quick:
        # Take a sample from each difficulty
        sampled = []
        for diff in ["easy", "medium", "hard", "trick"]:
            diff_qs = [q for q in queries if q["difficulty"] == diff]
            sampled.extend(diff_qs[:3])
        queries = sampled[:10]
        logger.info(f"Quick mode: {len(queries)} queries")

    logger.info(f"Running evaluation on {len(queries)} queries (prompt v{prompt_version})...")

    # Initialize pipeline
    pipeline = WildlifeRAGPipeline()

    # Run all queries
    eval_results = []
    pipeline_results = []
    total_start = time.time()

    for i, query_data in enumerate(queries):
        logger.info(f"  [{i + 1}/{len(queries)}] {query_data['query'][:60]}...")

        try:
            # Run pipeline
            result = pipeline.identify(
                query=query_data["query"],
                location=query_data.get("location"),
                prompt_version=prompt_version,
            )
            pipeline_results.append(result)

            # Evaluate against ground truth
            eval_result = evaluate_single(query_data, result)
            eval_results.append(eval_result)

            # Log result
            status = "CORRECT" if eval_result["answer_correctness"] >= 0.8 else "WRONG"
            logger.info(
                f"    → {status}: predicted={result['response'].get('species_name', 'N/A')}, "
                f"expected={query_data['expected_species']}, "
                f"confidence={result['response'].get('confidence', 0):.2f}"
            )

        except Exception as e:
            logger.error(f"    → ERROR: {e}")
            eval_results.append(
                {
                    "query_id": query_data.get("id", ""),
                    "difficulty": query_data.get("difficulty", ""),
                    "expected_species": query_data["expected_species"],
                    "predicted_species": "ERROR",
                    "confidence": 0,
                    "answer_correctness": 0,
                    "geographic_accuracy": 0,
                    "refusal_accuracy": 0,
                    "citation_precision": 0,
                    "confidence_calibration": 0,
                    "latency_s": 0,
                    "error": str(e),
                }
            )

    total_time = time.time() - total_start

    # Aggregate
    aggregated = aggregate_metrics(eval_results)
    quality_gates = check_quality_gates(aggregated)

    # Prepare RAGAS data for optional deep evaluation
    ragas_data = []
    for query_data, result in zip(queries, pipeline_results, strict=False):
        if query_data.get("difficulty") == "trick":
            continue
        response = result.get("response", {})
        ragas_data.append(
            {
                "question": query_data["query"],
                "answer": f"{response.get('species_name', '')}: {response.get('reasoning', '')}",
                "contexts": [
                    c.get("species", "")
                    for c in result.get("retrieval_details", {}).get("top_chunk_scores", [])
                ],
                "ground_truth": f"The species is {query_data['expected_species']}.",
            }
        )

    return {
        "timestamp": datetime.now().isoformat(),
        "prompt_version": prompt_version,
        "queries_total": len(queries),
        "total_time_s": round(total_time, 2),
        "aggregated_metrics": aggregated,
        "quality_gates": quality_gates,
        "detailed_results": eval_results,
        "ragas_data": ragas_data,
    }


def print_report(results: dict):
    """Print a formatted evaluation report to console."""
    agg = results["aggregated_metrics"]
    gates = results["quality_gates"]
    overall = agg["overall"]

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION REPORT — Prompt v{results['prompt_version']}")
    print(f"  {results['timestamp']}")
    print(f"{'=' * 70}")

    print(f"\n  Overall Metrics ({overall['total_queries']} queries)")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<30} {'Score':>10}")
    print(f"  {'─' * 50}")
    print(f"  {'Answer Correctness':<30} {overall['answer_correctness']:>10.3f}")
    print(f"  {'Geographic Accuracy':<30} {overall['geographic_accuracy']:>10.3f}")
    print(f"  {'Refusal Accuracy':<30} {overall['refusal_accuracy']:>10.3f}")
    print(f"  {'Citation Precision':<30} {overall['citation_precision']:>10.3f}")
    print(f"  {'Confidence Calibration':<30} {overall['confidence_calibration']:>10.3f}")
    print(f"  {'Avg Latency':<30} {overall['avg_latency_s']:>8.2f}s")
    print(f"  {'P95 Latency':<30} {overall['p95_latency_s']:>8.2f}s")

    print("\n  By Difficulty")
    print(f"  {'─' * 50}")
    print(f"  {'Level':<12} {'Count':>6} {'Correct':>10} {'Geo Acc':>10}")
    print(f"  {'─' * 50}")
    for diff, metrics in agg["by_difficulty"].items():
        print(
            f"  {diff:<12} {metrics['count']:>6} "
            f"{metrics['answer_correctness']:>10.3f} "
            f"{metrics['geographic_accuracy']:>10.3f}"
        )

    # Quality gates
    print("\n  Quality Gates")
    print(f"  {'─' * 50}")
    for metric, gate in gates["gates"].items():
        symbol = "+" if gate["passed"] else "X"
        print(f"  [{symbol}] {metric:<25} {gate['actual']:.3f} (threshold: {gate['threshold']})")

    all_pass = gates["all_passed"]
    print(f"\n  {'=' * 50}")
    print(f"  RESULT: {'ALL GATES PASSED' if all_pass else 'QUALITY GATES FAILED'}")
    print(f"  {'=' * 50}")

    # Confusions
    if agg.get("confusions"):
        print(f"\n  Species Confusions ({agg['confusion_count']})")
        print(f"  {'─' * 50}")
        for c in agg["confusions"][:10]:
            print(f"  Expected: {c['expected']:<25} Got: {c['predicted']}")


def main():
    parser = argparse.ArgumentParser(description="Wildlife Tracker Evaluation Runner")
    parser.add_argument("--quick", action="store_true", help="Quick run with 10 queries")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "trick"], help="Filter difficulty"
    )
    parser.add_argument("--prompt-version", type=int, default=1, help="Prompt version to evaluate")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit code 1 on gate failure")
    parser.add_argument("--save-dataset", action="store_true", help="Save golden dataset to JSON")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.save_dataset:
        save_dataset()
        stats = get_stats()
        print(f"Dataset saved: {stats}")
        return

    # Run evaluation
    results = run_evaluation(
        quick=args.quick,
        difficulty=args.difficulty,
        prompt_version=args.prompt_version,
    )

    # Print report
    print_report(results)

    # Save results
    output_file = OUTPUT_DIR / "full_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Full results saved to {output_file}")

    # CI mode: exit with error if gates fail
    if args.ci and not results["quality_gates"]["all_passed"]:
        logger.error("Quality gates FAILED. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
