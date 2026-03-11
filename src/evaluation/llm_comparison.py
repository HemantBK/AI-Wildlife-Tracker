"""
LLM Comparison Benchmark
Compares LLM performance on species identification.

Tests across Ollama (local) and Groq (API):
  Local:  Llama 3.2 3B, Mistral 7B (if hardware allows), Qwen 2.5 3B
  API:    Llama 3.1 8B Instant, Mixtral 8x7B, Gemma2 9B (via Groq)

Metrics: Answer Correctness, Citation Accuracy, JSON Validity, Latency
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.benchmark_queries import BENCHMARK_QUERIES
from src.rag.generator import (
    build_prompt,
    call_groq,
    call_ollama,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

OUTPUT_DIR = Path("data/evaluation")

# Models to test
LOCAL_MODELS = [
    {"name": "llama3.2:3b", "label": "Llama 3.2 3B (Q4)", "backend": "ollama"},
    {"name": "qwen2.5:3b", "label": "Qwen 2.5 3B (Q4)", "backend": "ollama"},
    # Uncomment if you have 16GB+ RAM and patience:
    # {"name": "mistral:7b", "label": "Mistral 7B (Q4)", "backend": "ollama"},
]

GROQ_MODELS = [
    {"name": "llama-3.1-8b-instant", "label": "Llama 3.1 8B (Groq)", "backend": "groq"},
    {"name": "mixtral-8x7b-32768", "label": "Mixtral 8x7B (Groq)", "backend": "groq"},
    {"name": "gemma2-9b-it", "label": "Gemma2 9B (Groq)", "backend": "groq"},
]


def create_mock_chunks(query: dict) -> list[dict]:
    """
    Create simulated context chunks for LLM testing.
    In real evaluation, these would come from the retrieval pipeline.
    Here we use simple mock chunks so we can test LLM quality independently.
    """
    species = query["expected_species"]
    scientific = query.get("expected_scientific", "")

    if species == "DECLINED":
        return [
            {
                "chunk_id": "mock_irrelevant_001",
                "text": "Species: Common House Sparrow (Passer domesticus) | Section: Overview\n\nThe house sparrow is a small bird commonly found near human habitation across India.",
            }
        ]

    return [
        {
            "chunk_id": f"mock_{species.lower().replace(' ', '_')}_001",
            "text": f"Species: {species} ({scientific}) | Section: Overview | Regions: India\n\nThe {species} ({scientific}) is a well-known species found across various regions of India. It is recognized by its distinctive features and plays an important role in its ecosystem.",
        },
        {
            "chunk_id": f"mock_{species.lower().replace(' ', '_')}_002",
            "text": f"Species: {species} ({scientific}) | Section: Habitat & Distribution\n\nThe {species} inhabits diverse ecosystems across India including forests, grasslands, and wetlands depending on the species' specific adaptations.",
        },
        {
            "chunk_id": f"mock_{species.lower().replace(' ', '_')}_003",
            "text": f"Species: {species} ({scientific}) | Section: Conservation Status\n\nThe conservation status and population trends of {species} are monitored by wildlife authorities across India.",
        },
    ]


def evaluate_response(raw_response: str, expected: dict) -> dict:
    """Evaluate a single LLM response against expected answer."""
    result = {
        "json_valid": False,
        "species_correct": False,
        "has_citations": False,
        "has_reasoning": False,
        "confidence_reasonable": False,
        "declined_correctly": False,
    }

    # Try parsing JSON
    try:
        text = raw_response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > 0:
            data = json.loads(text[start:end])
            result["json_valid"] = True

            # Check species correctness
            predicted = data.get("species_name", "").lower()
            expected_name = expected["expected_species"].lower()

            if expected_name == "declined":
                result["declined_correctly"] = (
                    "decline" in predicted
                    or data.get("confidence", 1.0) < 0.3
                    or "cannot" in data.get("reasoning", "").lower()
                )
                result["species_correct"] = result["declined_correctly"]
            else:
                result["species_correct"] = expected_name in predicted or predicted in expected_name

            # Check citations
            citations = data.get("cited_sources", [])
            result["has_citations"] = len(citations) > 0

            # Check reasoning
            reasoning = data.get("reasoning", "")
            result["has_reasoning"] = len(reasoning) > 20

            # Check confidence
            confidence = data.get("confidence", -1)
            result["confidence_reasonable"] = 0 <= confidence <= 1

    except (json.JSONDecodeError, Exception):
        pass

    return result


def test_model(
    model_name: str,
    backend: str,
    queries: list[dict],
    temperature: float = 0.1,
) -> dict:
    """Test a single model across all benchmark queries."""
    results = []
    total_latency = 0
    errors = 0

    for q in queries:
        chunks = create_mock_chunks(q)
        system_prompt, user_prompt = build_prompt(
            query=q["query"],
            chunks=chunks,
            location=q.get("location"),
        )

        try:
            start = time.time()
            if backend == "ollama":
                raw = call_ollama(
                    system_prompt, user_prompt, model=model_name, temperature=temperature
                )
            elif backend == "groq":
                raw = call_groq(
                    system_prompt, user_prompt, model=model_name, temperature=temperature
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

            latency = time.time() - start
            total_latency += latency

            eval_result = evaluate_response(raw, q)
            eval_result["latency_s"] = round(latency, 2)
            eval_result["difficulty"] = q["difficulty"]
            eval_result["query"] = q["query"]
            results.append(eval_result)

        except Exception as e:
            errors += 1
            logger.warning(f"Error on '{q['query'][:40]}...': {e}")
            results.append(
                {
                    "error": str(e),
                    "difficulty": q["difficulty"],
                    "query": q["query"],
                }
            )

    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    n = len(valid_results)

    metrics = {
        "answer_correctness": round(sum(r["species_correct"] for r in valid_results) / n, 3)
        if n
        else 0,
        "json_validity": round(sum(r["json_valid"] for r in valid_results) / n, 3) if n else 0,
        "citation_rate": round(sum(r["has_citations"] for r in valid_results) / n, 3) if n else 0,
        "reasoning_rate": round(sum(r["has_reasoning"] for r in valid_results) / n, 3) if n else 0,
        "avg_latency_s": round(total_latency / n, 2) if n else 0,
        "error_rate": round(errors / len(queries), 3),
        "queries_tested": len(queries),
        "queries_successful": n,
    }

    # By difficulty
    by_difficulty = {}
    for diff in ["easy", "medium", "hard", "trick"]:
        diff_results = [r for r in valid_results if r.get("difficulty") == diff]
        if diff_results:
            by_difficulty[diff] = {
                "correctness": round(
                    sum(r["species_correct"] for r in diff_results) / len(diff_results), 3
                ),
                "count": len(diff_results),
            }

    return {
        "overall": metrics,
        "by_difficulty": by_difficulty,
        "detailed_results": results,
    }


def run_comparison(include_local: bool = True, include_groq: bool = True) -> list[dict]:
    """Run full LLM comparison."""
    queries = BENCHMARK_QUERIES
    results = []

    models_to_test = []
    if include_local:
        models_to_test.extend(LOCAL_MODELS)
    if include_groq and os.getenv("GROQ_API_KEY"):
        models_to_test.extend(GROQ_MODELS)
    elif include_groq:
        logger.warning("GROQ_API_KEY not set. Skipping Groq models.")

    for model_info in models_to_test:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing: {model_info['label']}")
        logger.info(f"{'=' * 50}")

        try:
            model_results = test_model(
                model_name=model_info["name"],
                backend=model_info["backend"],
                queries=queries,
            )

            result = {
                "model": model_info["name"],
                "label": model_info["label"],
                "backend": model_info["backend"],
                **model_results,
            }
            results.append(result)

            m = model_results["overall"]
            logger.info(f"  Correctness:  {m['answer_correctness']}")
            logger.info(f"  JSON Valid:   {m['json_validity']}")
            logger.info(f"  Citations:    {m['citation_rate']}")
            logger.info(f"  Avg Latency:  {m['avg_latency_s']}s")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append(
                {"model": model_info["name"], "label": model_info["label"], "error": str(e)}
            )

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser(description="LLM Comparison Benchmark")
    parser.add_argument("--local-only", action="store_true", help="Only test local Ollama models")
    parser.add_argument("--groq-only", action="store_true", help="Only test Groq API models")
    args = parser.parse_args()

    include_local = not args.groq_only
    include_groq = not args.local_only

    results = run_comparison(include_local=include_local, include_groq=include_groq)

    # Save results
    output_file = OUTPUT_DIR / "llm_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'=' * 90}")
    print("LLM COMPARISON RESULTS")
    print(f"{'=' * 90}")
    print(f"{'Model':<30} {'Correct':>10} {'JSON OK':>10} {'Citations':>10} {'Latency':>10}")
    print(f"{'-' * 90}")

    for r in results:
        if "error" in r and "overall" not in r:
            print(f"{r['label']:<30} {'ERROR':>10}")
            continue
        m = r["overall"]
        print(
            f"{r['label']:<30} "
            f"{m['answer_correctness'] * 100:>8.1f}% "
            f"{m['json_validity'] * 100:>8.1f}% "
            f"{m['citation_rate'] * 100:>8.1f}% "
            f"{m['avg_latency_s']:>8.2f}s"
        )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
