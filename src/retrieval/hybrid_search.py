"""
Hybrid Search with Reciprocal Rank Fusion (RRF)
Combines vector search (ChromaDB) and keyword search (BM25)
to get the best of semantic and exact-match retrieval.
"""

import logging
from collections import defaultdict

import yaml

from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedder import query_vector_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.

    RRF score = sum over all lists of: 1 / (k + rank_in_list)

    This gives a balanced combination that doesn't depend on score normalization.

    Args:
        result_lists: List of ranked result lists. Each result has 'chunk_id'.
        k: RRF parameter (default 60, from the original RRF paper).

    Returns:
        Fused list sorted by RRF score, with both original scores preserved.
    """
    rrf_scores = defaultdict(float)
    chunk_data = {}

    for list_idx, results in enumerate(result_lists):
        for rank, result in enumerate(results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

            # Store chunk data (keep the richest version)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "scores": {},
                }

            # Track which retriever found this and at what rank
            source_name = f"retriever_{list_idx}"
            chunk_data[chunk_id]["scores"][source_name] = {
                "score": result["score"],
                "rank": rank + 1,
            }

    # Sort by RRF score
    fused = []
    for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = chunk_data[chunk_id]
        entry["rrf_score"] = rrf_score
        fused.append(entry)

    return fused


class HybridSearcher:
    """Combines vector search and BM25 for hybrid retrieval."""

    def __init__(self, config_path: str = "config/retrieval.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["embedding"]["model_name"]
        self.collection_name = self.config["vector_store"]["collection_name"]
        self.persist_dir = self.config["vector_store"]["persist_directory"]
        self.top_k_vector = self.config["retrieval"]["top_k_vector"]
        self.top_k_bm25 = self.config["retrieval"]["top_k_bm25"]
        self.fusion_k = self.config["retrieval"]["fusion_k"]

        # Load BM25 index
        self.bm25 = BM25Index()
        self.bm25.load()
        logger.info("Hybrid searcher initialized.")

    def search(
        self,
        query: str,
        top_k_vector: int = None,
        top_k_bm25: int = None,
        geographic_filter: str = None,
    ) -> list[dict]:
        """
        Perform hybrid search combining vector + BM25 results via RRF.

        Args:
            query: User search query
            top_k_vector: Override vector search top-k
            top_k_bm25: Override BM25 top-k
            geographic_filter: Optional state/region to filter results

        Returns:
            Fused ranked list of chunks
        """
        top_k_v = top_k_vector or self.top_k_vector
        top_k_b = top_k_bm25 or self.top_k_bm25

        # Vector search
        where_filter = None
        if geographic_filter:
            where_filter = {"geographic_regions": {"$contains": geographic_filter}}

        vector_results = query_vector_store(
            query=query,
            model_name=self.model_name,
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
            top_k=top_k_v,
            where_filter=where_filter,
        )

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k_b)

        # Apply geographic filter to BM25 results if specified
        if geographic_filter:
            bm25_results = [
                r
                for r in bm25_results
                if geographic_filter.lower() in r["metadata"].get("geographic_regions", "").lower()
            ]

        logger.info(f"Vector hits: {len(vector_results)}, BM25 hits: {len(bm25_results)}")

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.fusion_k,
        )

        logger.info(f"Fused results: {len(fused)} unique chunks")
        return fused

    def search_with_details(self, query: str, **kwargs) -> dict:
        """Search and return detailed results with retrieval metadata."""
        results = self.search(query, **kwargs)

        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "retrieval_config": {
                "model": self.model_name,
                "top_k_vector": kwargs.get("top_k_vector", self.top_k_vector),
                "top_k_bm25": kwargs.get("top_k_bm25", self.top_k_bm25),
                "fusion_k": self.fusion_k,
                "geographic_filter": kwargs.get("geographic_filter"),
            },
        }


def main():
    """Test hybrid search."""
    searcher = HybridSearcher()

    test_queries = [
        "large striped cat in Indian forests",
        "small colorful bird with long tail Kerala",
        "venomous snake with hood India",
        "endangered one-horned rhinoceros Assam",
        "blue feathered bird dancing in rain",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print(f"{'=' * 60}")

        results = searcher.search(query)
        for i, r in enumerate(results[:5]):
            vector_rank = r["scores"].get("retriever_0", {}).get("rank", "-")
            bm25_rank = r["scores"].get("retriever_1", {}).get("rank", "-")
            print(
                f"  {i + 1}. [{r['rrf_score']:.4f}] {r['metadata']['species_name']}"
                f" | Section: {r['metadata']['section_type']}"
                f" | Vector: #{vector_rank}, BM25: #{bm25_rank}"
            )


if __name__ == "__main__":
    main()
