"""
RAGAS Integration Wrapper
Optional RAGAS-based evaluation for deeper quality metrics.
Requires an LLM to run (uses Groq API for speed).

RAGAS metrics:
  - Faithfulness: Are claims in the answer supported by context?
  - Answer Relevancy: How relevant is the answer to the question?
  - Context Precision: Are retrieved contexts relevant?
  - Context Recall: Does the context contain the expected answer?

NOTE: RAGAS evaluation is slow (~1-2 min for 30 queries via Groq).
Use custom metrics (metrics.py) for fast CI, RAGAS for deep analysis.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

OUTPUT_DIR = Path("data/evaluation")


def run_ragas_evaluation(
    eval_data: list[dict],
    use_groq: bool = True,
) -> dict | None:
    """
    Run RAGAS evaluation on a list of query-response pairs.

    Args:
        eval_data: List of dicts with keys:
            - question: str
            - answer: str
            - contexts: list[str]
            - ground_truth: str
        use_groq: Whether to use Groq API (recommended for speed)

    Returns:
        RAGAS evaluation results dict, or None if RAGAS not available
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        logger.warning(
            "RAGAS not installed. Install with: pip install ragas datasets\n"
            "Falling back to custom metrics only."
        )
        return None

    # Configure LLM for RAGAS
    if use_groq:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Cannot run RAGAS evaluation.")
            return None

        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=api_key,
                temperature=0,
            )
        except ImportError:
            logger.warning("langchain_groq not installed. pip install langchain-groq")
            return None
    else:
        logger.warning("Non-Groq RAGAS evaluation not configured.")
        return None

    # Prepare dataset
    dataset_dict = {
        "question": [d["question"] for d in eval_data],
        "answer": [d["answer"] for d in eval_data],
        "contexts": [d["contexts"] for d in eval_data],
        "ground_truth": [d["ground_truth"] for d in eval_data],
    }

    dataset = Dataset.from_dict(dataset_dict)

    # Run evaluation
    logger.info(f"Running RAGAS evaluation on {len(eval_data)} queries...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
        )
        return {
            "faithfulness": round(result["faithfulness"], 4),
            "answer_relevancy": round(result["answer_relevancy"], 4),
            "context_precision": round(result["context_precision"], 4),
            "context_recall": round(result["context_recall"], 4),
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return None


def prepare_ragas_data(
    golden_queries: list[dict],
    pipeline_results: list[dict],
) -> list[dict]:
    """
    Convert pipeline results into RAGAS-compatible format.

    Args:
        golden_queries: Golden dataset entries
        pipeline_results: Corresponding pipeline outputs
    """
    ragas_data = []

    for query_data, result in zip(golden_queries, pipeline_results, strict=False):
        # Skip trick queries (RAGAS doesn't handle decline-to-answer well)
        if query_data.get("difficulty") == "trick":
            continue

        response = result.get("response", {})
        answer = response.get("reasoning", "")
        if response.get("species_name"):
            answer = f"{response['species_name']}: {answer}"

        # Get context texts from retrieval
        contexts = []
        for chunk_info in result.get("retrieval_details", {}).get("top_chunk_scores", []):
            # In a full pipeline run, we'd have the actual chunk text here
            contexts.append(f"Species: {chunk_info.get('species', 'Unknown')}")

        # If no contexts from pipeline, use a placeholder
        if not contexts:
            contexts = ["No context retrieved"]

        # Ground truth
        ground_truth = (
            f"The species is {query_data['expected_species']} "
            f"({query_data.get('expected_scientific', '')})."
        )

        ragas_data.append(
            {
                "question": query_data["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return ragas_data


def main():
    """Run standalone RAGAS evaluation with saved results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for saved pipeline results
    results_file = OUTPUT_DIR / "full_eval_results.json"
    if not results_file.exists():
        logger.error(
            "No evaluation results found. Run the full evaluator first:\n"
            "  python -m src.evaluation.evaluator"
        )
        return

    with open(results_file) as f:
        data = json.load(f)

    if "ragas_data" not in data:
        logger.error("No RAGAS-compatible data in results file.")
        return

    ragas_result = run_ragas_evaluation(data["ragas_data"])

    if ragas_result:
        print("\nRAGAS Evaluation Results")
        print(f"{'=' * 40}")
        for metric, score in ragas_result.items():
            print(f"  {metric:<25} {score:.4f}")

        # Save
        with open(OUTPUT_DIR / "ragas_results.json", "w") as f:
            json.dump(ragas_result, f, indent=2)
        print(f"\nSaved to {OUTPUT_DIR / 'ragas_results.json'}")


if __name__ == "__main__":
    main()
