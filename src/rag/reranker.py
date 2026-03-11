"""
Cross-Encoder Re-Ranker
Takes hybrid search candidates and re-ranks them using a more powerful
cross-encoder model for higher precision.
"""

import logging

import yaml

from src.rag.onnx_reranker import get_cross_encoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder based re-ranker for candidate chunks."""

    def __init__(self, config_path: str = "config/retrieval.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.model_name = config["reranker"]["model_name"]
        self.top_n = config["retrieval"]["reranker_top_n"]
        self.threshold = config["retrieval"]["reranker_threshold"]

        logger.info(f"Loading re-ranker model: {self.model_name}")
        self.model = get_cross_encoder(self.model_name, max_length=512)
        logger.info("Re-ranker ready.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = None,
        threshold: float = None,
    ) -> list[dict]:
        """
        Re-rank candidate chunks using cross-encoder.

        Args:
            query: The user's original query
            candidates: List of candidate chunks from hybrid search
            top_n: Number of top results to return
            threshold: Minimum score threshold

        Returns:
            Re-ranked list of chunks with cross-encoder scores
        """
        top_n = top_n or self.top_n
        threshold = threshold or self.threshold

        if not candidates:
            return []

        # Prepare query-document pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_candidates = []
        for candidate, score in zip(candidates, scores, strict=False):
            candidate["reranker_score"] = float(score)
            scored_candidates.append(candidate)

        # Sort by reranker score (descending)
        scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Apply threshold filter
        filtered = [c for c in scored_candidates if c["reranker_score"] >= threshold]

        # Fallback: if threshold filters everything, keep top-n by score anyway
        # so the LLM can still attempt identification
        if not filtered and scored_candidates:
            logger.warning(
                f"All {len(scored_candidates)} candidates below threshold {threshold:.2f}. "
                f"Top score: {scored_candidates[0]['reranker_score']:.4f}. "
                f"Using top-{top_n} as fallback."
            )
            filtered = scored_candidates[:top_n]

        # Take top-n
        result = filtered[:top_n]

        logger.info(
            f"Re-ranked {len(candidates)} → {len(filtered)} above threshold → returning top {len(result)}"
        )

        return result

    def rerank_with_details(self, query: str, candidates: list[dict], **kwargs) -> dict:
        """Re-rank and return detailed results with all scores."""
        results = self.rerank(query, candidates, **kwargs)

        return {
            "query": query,
            "candidates_in": len(candidates),
            "results_out": len(results),
            "threshold": kwargs.get("threshold", self.threshold),
            "results": results,
            "score_distribution": {
                "min": min(r["reranker_score"] for r in results) if results else 0,
                "max": max(r["reranker_score"] for r in results) if results else 0,
                "mean": sum(r["reranker_score"] for r in results) / len(results) if results else 0,
            },
        }
