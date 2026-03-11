"""
Optimization Experiments
Tests quantization, temperature, and context window variations.

Experiments:
  1. Quantization: Q4 vs Q5 vs Q8 (Ollama local)
  2. Temperature: 0.0 vs 0.1 vs 0.3 vs 0.7
  3. Context window: 3 chunks vs 5 chunks vs 7 chunks
"""

import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.benchmark_queries import get_queries_by_difficulty
from src.evaluation.llm_comparison import create_mock_chunks, evaluate_response
from src.rag.generator import build_prompt, call_groq, call_ollama

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

OUTPUT_DIR = Path("data/evaluation")


def _run_experiment(
    experiment_name: str,
    variable_name: str,
    variable_values: list,
    call_fn,
    call_kwargs_fn,
    queries: list[dict],
) -> list[dict]:
    """Generic experiment runner."""
    results = []

    for value in variable_values:
        logger.info(f"  Testing {variable_name}={value}...")
        correct = 0
        json_valid = 0
        total_latency = 0
        n = 0

        for q in queries:
            chunks = create_mock_chunks(q)
            kwargs = call_kwargs_fn(value, q, chunks)

            try:
                start = time.time()
                raw = call_fn(**kwargs)
                latency = time.time() - start
                total_latency += latency

                eval_result = evaluate_response(raw, q)
                if eval_result["species_correct"]:
                    correct += 1
                if eval_result["json_valid"]:
                    json_valid += 1
                n += 1

            except Exception as e:
                logger.warning(f"    Error: {e}")

        results.append(
            {
                variable_name: str(value),
                "correctness": round(correct / n, 3) if n else 0,
                "json_validity": round(json_valid / n, 3) if n else 0,
                "avg_latency_s": round(total_latency / n, 2) if n else 0,
                "queries_tested": n,
            }
        )

    return results


# ─── Experiment 1: Temperature Tuning ──────────────────────────


def run_temperature_experiment(
    backend: str = "groq",
    model: str = None,
) -> dict:
    """Test different temperature values."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT: Temperature Tuning")
    logger.info("=" * 50)

    temperatures = [0.0, 0.1, 0.3, 0.7]
    queries = get_queries_by_difficulty("easy")[:6] + get_queries_by_difficulty("medium")[:4]

    if backend == "groq":
        model = model or "llama-3.1-8b-instant"

        def call_fn(**kwargs):
            return call_groq(**kwargs)
    else:
        model = model or "llama3.2:3b"

        def call_fn(**kwargs):
            return call_ollama(**kwargs)

    def kwargs_fn(temp, q, chunks):
        system_prompt, user_prompt = build_prompt(q["query"], chunks, q.get("location"))
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model": model,
            "temperature": temp,
        }

    results = _run_experiment(
        "Temperature", "temperature", temperatures, call_fn, kwargs_fn, queries
    )

    return {
        "experiment": "temperature_tuning",
        "model": model,
        "backend": backend,
        "results": results,
    }


# ─── Experiment 2: Context Window (Chunk Count) ───────────────


def run_context_window_experiment(
    backend: str = "groq",
    model: str = None,
) -> dict:
    """Test different numbers of context chunks."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT: Context Window (Chunk Count)")
    logger.info("=" * 50)

    chunk_counts = [1, 3, 5, 7]
    queries = get_queries_by_difficulty("easy")[:5] + get_queries_by_difficulty("medium")[:5]

    if backend == "groq":
        model = model or "llama-3.1-8b-instant"

        def call_fn(**kwargs):
            return call_groq(**kwargs)
    else:
        model = model or "llama3.2:3b"

        def call_fn(**kwargs):
            return call_ollama(**kwargs)

    results = []
    for count in chunk_counts:
        logger.info(f"  Testing chunk_count={count}...")
        correct = 0
        json_valid = 0
        total_latency = 0
        n = 0

        for q in queries:
            # Create variable number of chunks
            base_chunks = create_mock_chunks(q)
            # Pad with additional generic chunks if needed
            while len(base_chunks) < count:
                base_chunks.append(
                    {
                        "chunk_id": f"padding_{len(base_chunks)}",
                        "text": "Species: Various | Section: General\n\nIndia is home to diverse wildlife.",
                    }
                )
            test_chunks = base_chunks[:count]

            system_prompt, user_prompt = build_prompt(q["query"], test_chunks, q.get("location"))

            try:
                start = time.time()
                if backend == "groq":
                    raw = call_groq(system_prompt, user_prompt, model=model, temperature=0.1)
                else:
                    raw = call_ollama(system_prompt, user_prompt, model=model, temperature=0.1)
                latency = time.time() - start
                total_latency += latency

                eval_result = evaluate_response(raw, q)
                if eval_result["species_correct"]:
                    correct += 1
                if eval_result["json_valid"]:
                    json_valid += 1
                n += 1

            except Exception as e:
                logger.warning(f"    Error: {e}")

        results.append(
            {
                "chunk_count": count,
                "correctness": round(correct / n, 3) if n else 0,
                "json_validity": round(json_valid / n, 3) if n else 0,
                "avg_latency_s": round(total_latency / n, 2) if n else 0,
                "queries_tested": n,
            }
        )

    return {"experiment": "context_window", "model": model, "backend": backend, "results": results}


# ─── Experiment 3: Quantization Comparison (Ollama Only) ──────


def run_quantization_experiment() -> dict:
    """Compare quantized model variants via Ollama."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT: Quantization Comparison")
    logger.info("=" * 50)

    # These require pulling different model tags in Ollama
    variants = [
        {"model": "llama3.2:3b", "label": "Q4 (default)"},
        # Uncomment after pulling: ollama pull llama3.2:3b-q5_K_M
        # {"model": "llama3.2:3b-q5_K_M", "label": "Q5"},
        # {"model": "llama3.2:3b-q8_0", "label": "Q8"},
    ]

    queries = get_queries_by_difficulty("easy")[:5] + get_queries_by_difficulty("medium")[:3]
    results = []

    for variant in variants:
        model = variant["model"]
        label = variant["label"]
        logger.info(f"  Testing {label} ({model})...")

        correct = 0
        json_valid = 0
        total_latency = 0
        n = 0

        for q in queries:
            chunks = create_mock_chunks(q)
            system_prompt, user_prompt = build_prompt(q["query"], chunks, q.get("location"))

            try:
                start = time.time()
                raw = call_ollama(system_prompt, user_prompt, model=model, temperature=0.1)
                latency = time.time() - start
                total_latency += latency

                eval_result = evaluate_response(raw, q)
                if eval_result["species_correct"]:
                    correct += 1
                if eval_result["json_valid"]:
                    json_valid += 1
                n += 1

            except Exception as e:
                logger.warning(f"    Error: {e}")

        results.append(
            {
                "model": model,
                "label": label,
                "correctness": round(correct / n, 3) if n else 0,
                "json_validity": round(json_valid / n, 3) if n else 0,
                "avg_latency_s": round(total_latency / n, 2) if n else 0,
                "tokens_per_second": "N/A",  # Would need Ollama metrics for this
                "queries_tested": n,
            }
        )

    return {"experiment": "quantization", "results": results}


# ─── Run All Experiments ──────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=["temperature", "context", "quantization", "all"], default="all"
    )
    parser.add_argument("--backend", choices=["groq", "ollama"], default="groq")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.experiment in ("temperature", "all"):
        all_results["temperature"] = run_temperature_experiment(backend=args.backend)

    if args.experiment in ("context", "all"):
        all_results["context_window"] = run_context_window_experiment(backend=args.backend)

    if args.experiment in ("quantization", "all") and args.backend == "ollama":
        all_results["quantization"] = run_quantization_experiment()

    # Save
    output_file = OUTPUT_DIR / "optimization_experiments.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summaries
    for exp_name, exp_data in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"  {exp_name.upper()}")
        print(f"{'=' * 60}")
        for row in exp_data.get("results", []):
            print(f"  {row}")

    print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()
