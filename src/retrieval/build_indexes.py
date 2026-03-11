"""
Build All Indexes
Single entry point to build both ChromaDB vector store and BM25 index.
Run this after the data pipeline completes.
"""

import logging
import time

from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedder import build_vector_store, load_chunks, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    config = load_config()
    chunks = load_chunks()
    logger.info(f"Loaded {len(chunks)} chunks. Building indexes...")

    # Build ChromaDB vector store
    logger.info("=" * 40)
    logger.info("Building ChromaDB vector store...")
    logger.info("=" * 40)
    build_vector_store(
        chunks=chunks,
        model_name=config["embedding"]["model_name"],
        collection_name=config["vector_store"]["collection_name"],
        persist_dir=config["vector_store"]["persist_directory"],
        batch_size=config["embedding"]["batch_size"],
    )

    # Build BM25 index
    logger.info("=" * 40)
    logger.info("Building BM25 index...")
    logger.info("=" * 40)
    bm25 = BM25Index()
    bm25.build(
        chunks=chunks,
        k1=config["bm25"]["k1"],
        b=config["bm25"]["b"],
    )
    bm25.save()

    elapsed = time.time() - start
    logger.info(f"All indexes built in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
