"""
Embedding Model Comparison
Compares retrieval quality across embedding models.
Measures: Precision@5, Recall@5, MRR (Mean Reciprocal Rank).

Models tested:
  - all-MiniLM-L6-v2 (default, fast, 384d)
  - BAAI/bge-small-en-v1.5 (better quality, 384d)
  - nomic-ai/nomic-embed-text-v1.5 (newer, 768d)
"""

import json
import logging
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.evaluation.benchmark_queries import BENCHMARK_QUERIES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_TO_TEST = [
    {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Fast, lightweight, good baseline",
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "description": "Better quality, same size as MiniLM",
    },
    {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 768,
        "description": "Newer model, higher dimensional",
        "prefix": "search_query: ",  # nomic requires prefix
    },
]

OUTPUT_DIR = Path("data/evaluation")


def load_chunks() -> list[dict]:
    """Load chunks from data pipeline."""
    with open("data/chunks/all_chunks.json", encoding="utf-8") as f:
        return json.load(f)


def build_temp_collection(
    chunks: list[dict], model: SentenceTransformer, model_name: str, prefix: str = ""
) -> chromadb.Collection:
    """Build a temporary ChromaDB collection for evaluation."""
    client = chromadb.EphemeralClient()

    # Clean collection name (ChromaDB restrictions)
    clean_name = model_name.replace("/", "_").replace(".", "_")[:60]

    collection = client.create_collection(
        name=clean_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and add in batches
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [f"{prefix}{c['text']}" for c in batch]
        ids = [c["chunk_id"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        metadatas = [
            {
                "species_name": c.get("species_name", ""),
                "scientific_name": c.get("scientific_name", ""),
                "section_type": c.get("section_type", ""),
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=metadatas,
        )

    return collection, client


def evaluate_retrieval(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    queries: list[dict],
    top_k: int = 5,
    prefix: str = "",
) -> dict:
    """
    Evaluate retrieval quality on benchmark queries.

    Metrics:
      - Precision@K: fraction of top-K results that match expected species
      - Hit Rate: fraction of queries where expected species appears in top-K
      - MRR: Mean Reciprocal Rank of the first relevant result
    """
    precision_scores = []
    hit_rates = []
    mrr_scores = []
    latencies = []

    for q in queries:
        if q["difficulty"] == "trick":
            continue  # Skip trick queries for retrieval eval

        expected = q["expected_species"].lower()
        expected_sci = q.get("expected_scientific", "").lower()
        query_text = f"{prefix}{q['query']}"

        # Time the query
        start = time.time()
        embedding = model.encode([query_text]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        latency = time.time() - start
        latencies.append(latency)

        # Check results
        hits = 0
        first_relevant_rank = None

        for rank, meta in enumerate(results["metadatas"][0]):
            species = meta.get("species_name", "").lower()
            scientific = meta.get("scientific_name", "").lower()

            is_match = (
                expected in species
                or species in expected
                or expected_sci in scientific
                or scientific in expected_sci
            )

            if is_match:
                hits += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1

        precision_scores.append(hits / top_k)
        hit_rates.append(1 if hits > 0 else 0)
        mrr_scores.append(1.0 / first_relevant_rank if first_relevant_rank else 0)

    n = len(precision_scores)
    return {
        "precision_at_k": round(sum(precision_scores) / n, 4) if n else 0,
        "hit_rate": round(sum(hit_rates) / n, 4) if n else 0,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0,
        "avg_latency_ms": round(sum(latencies) / n * 1000, 1) if n else 0,
        "queries_evaluated": n,
    }


def run_comparison(chunks: list[dict] = None) -> list[dict]:
    """Run full embedding model comparison."""
    if chunks is None:
        chunks = load_chunks()

    # Only use non-trick queries
    queries = [q for q in BENCHMARK_QUERIES if q["difficulty"] != "trick"]
    results = []

    for model_info in MODELS_TO_TEST:
        model_name = model_info["name"]
        prefix = model_info.get("prefix", "")
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'=' * 50}")

        try:
            # Load model
            start = time.time()
            model = SentenceTransformer(model_name, trust_remote_code=True)
            load_time = time.time() - start

            # Build temp index
            start = time.time()
            collection, client = build_temp_collection(chunks, model, model_name, prefix)
            index_time = time.time() - start

            # Evaluate
            metrics = evaluate_retrieval(collection, model, queries, top_k=5, prefix=prefix)

            # Evaluate by difficulty
            by_difficulty = {}
            for diff in ["easy", "medium", "hard"]:
                diff_queries = [q for q in queries if q["difficulty"] == diff]
                if diff_queries:
                    by_difficulty[diff] = evaluate_retrieval(
                        collection, model, diff_queries, top_k=5, prefix=prefix
                    )

            result = {
                "model": model_name,
                "description": model_info["description"],
                "dimensions": model_info["dimensions"],
                "model_load_time_s": round(load_time, 2),
                "index_build_time_s": round(index_time, 2),
                "overall": metrics,
                "by_difficulty": by_difficulty,
            }
            results.append(result)

            logger.info(f"  Precision@5: {metrics['precision_at_k']}")
            logger.info(f"  Hit Rate:    {metrics['hit_rate']}")
            logger.info(f"  MRR:         {metrics['mrr']}")
            logger.info(f"  Avg Latency: {metrics['avg_latency_ms']}ms")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append(
                {
                    "model": model_name,
                    "error": str(e),
                }
            )

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting embedding model comparison...")
    results = run_comparison()

    # Save results
    output_file = OUTPUT_DIR / "embedding_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("EMBEDDING MODEL COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Model':<35} {'P@5':>8} {'Hit Rate':>10} {'MRR':>8} {'Latency':>10}")
    print(f"{'-' * 80}")

    for r in results:
        if "error" in r:
            print(f"{r['model']:<35} {'ERROR':>8}")
            continue
        m = r["overall"]
        print(
            f"{r['model']:<35} {m['precision_at_k']:>8.3f} {m['hit_rate']:>10.3f} {m['mrr']:>8.3f} {m['avg_latency_ms']:>8.1f}ms"
        )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
