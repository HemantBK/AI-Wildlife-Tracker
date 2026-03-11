"""
BM25 Index Builder
Builds a BM25 keyword search index from chunks.
Runs in parallel with vector search for hybrid retrieval.
"""

import json
import logging
import pickle
import re
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/bm25_index")


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    # Basic stopwords (expand as needed)
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "about",
        "against",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "which",
        "who",
        "whom",
        "what",
    }
    # Lowercase and split
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Remove stopwords and very short tokens
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class BM25Index:
    """BM25 search index with serialization support."""

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunks: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        """Build BM25 index from chunks."""
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]

        # Tokenize all chunks
        self.tokenized_corpus = [tokenize(c["text"]) for c in chunks]

        # Build BM25 index
        self.index = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)
        logger.info("BM25 index built successfully.")

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """
        Search the BM25 index.

        Returns:
            List of dicts with 'chunk_id', 'text', 'score', 'metadata'
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return non-zero scores
                chunk = self.chunks[idx]
                results.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "score": float(scores[idx]),
                        "metadata": {
                            "species_name": chunk.get("species_name", ""),
                            "scientific_name": chunk.get("scientific_name", ""),
                            "section_type": chunk.get("section_type", ""),
                            "taxonomic_group": chunk.get("taxonomic_group", ""),
                            "conservation_status": chunk.get("conservation_status", ""),
                            "geographic_regions": ", ".join(chunk.get("geographic_regions", [])),
                        },
                    }
                )

        return results

    def save(self, directory: str | Path = INDEX_DIR):
        """Save the BM25 index to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "bm25_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "chunk_ids": self.chunk_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )

        # Save chunks separately (JSON for readability)
        with open(directory / "bm25_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        logger.info(f"BM25 index saved to {directory}")

    def load(self, directory: str | Path = INDEX_DIR):
        """Load BM25 index from disk."""
        directory = Path(directory)

        with open(directory / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.chunk_ids = data["chunk_ids"]
            self.tokenized_corpus = data["tokenized_corpus"]

        with open(directory / "bm25_chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logger.info(f"BM25 index loaded: {len(self.chunks)} chunks")


def main():
    # Load config
    config_path = Path("config/retrieval.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load chunks
    chunks_file = Path("data/chunks/all_chunks.json")
    if not chunks_file.exists():
        logger.error("No chunks found. Run the data pipeline first.")
        return

    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)

    # Build index
    bm25 = BM25Index()
    bm25.build(
        chunks=chunks,
        k1=config["bm25"]["k1"],
        b=config["bm25"]["b"],
    )

    # Save
    bm25.save()

    # Test query
    logger.info("Running test query...")
    results = bm25.search("king cobra venomous snake", top_k=3)
    for r in results:
        logger.info(
            f"  [{r['score']:.3f}] {r['metadata']['species_name']} - {r['metadata']['section_type']}"
        )

    logger.info("BM25 index build complete!")


if __name__ == "__main__":
    main()
