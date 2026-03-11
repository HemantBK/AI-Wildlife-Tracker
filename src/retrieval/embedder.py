"""
Embedding Pipeline
Embeds all chunks into ChromaDB using sentence-transformers.
Supports batch processing and incremental updates.
"""

import json
import logging
import time
from pathlib import Path

import chromadb
import yaml
from tqdm import tqdm

from src.retrieval.onnx_embedder import get_embedder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load retrieval configuration."""
    config_path = Path("config/retrieval.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_chunks() -> list[dict]:
    """Load all chunks from the chunking pipeline output."""
    chunks_file = Path("data/chunks/all_chunks.json")
    if not chunks_file.exists():
        raise FileNotFoundError("No chunks found. Run the data pipeline first: make data-pipeline")
    with open(chunks_file, encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "wildlife_chunks",
    persist_dir: str = "./data/chroma_db",
    batch_size: int = 64,
) -> chromadb.Collection:
    """
    Embed all chunks and store in ChromaDB.

    Args:
        chunks: List of chunk dicts with 'chunk_id', 'text', and metadata
        model_name: Sentence transformer model name
        collection_name: ChromaDB collection name
        persist_dir: Directory to persist ChromaDB
        batch_size: Batch size for embedding

    Returns:
        ChromaDB collection
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = get_embedder(model_name)

    logger.info(f"Initializing ChromaDB at: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists (full rebuild)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity
    )

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    start_time = time.time()

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i : i + batch_size]

        # Extract texts for embedding
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]

        # Embed
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Prepare metadata (ChromaDB only supports str, int, float, bool)
        metadatas = []
        for c in batch:
            meta = {
                "species_name": c.get("species_name", ""),
                "scientific_name": c.get("scientific_name", ""),
                "section_type": c.get("section_type", ""),
                "taxonomic_group": c.get("taxonomic_group", ""),
                "conservation_status": c.get("conservation_status", ""),
                "family": c.get("family", ""),
                "order": c.get("order", ""),
                # ChromaDB doesn't support lists, so join regions
                "geographic_regions": ", ".join(c.get("geographic_regions", [])),
                "token_estimate": c.get("token_estimate", 0),
                "source_urls": ", ".join(c.get("source_urls", [])),
            }
            metadatas.append(meta)

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    elapsed = time.time() - start_time
    logger.info(f"Embedding complete. {len(chunks)} chunks in {elapsed:.1f}s")
    logger.info(f"Collection size: {collection.count()}")

    return collection


def query_vector_store(
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "wildlife_chunks",
    persist_dir: str = "./data/chroma_db",
    top_k: int = 15,
    where_filter: dict = None,
) -> list[dict]:
    """
    Query the vector store and return top-k results.

    Args:
        query: User query string
        model_name: Must match the model used for indexing
        collection_name: ChromaDB collection name
        persist_dir: ChromaDB persist directory
        top_k: Number of results to return
        where_filter: Optional ChromaDB where filter

    Returns:
        List of dicts with 'chunk_id', 'text', 'score', 'metadata'
    """
    model = get_embedder(model_name)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query]).tolist()

    query_params = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i],
            }
        )

    return formatted


def main():
    config = load_config()
    chunks = load_chunks()

    model_name = config["embedding"]["model_name"]
    collection_name = config["vector_store"]["collection_name"]
    persist_dir = config["vector_store"]["persist_directory"]
    batch_size = config["embedding"]["batch_size"]

    build_vector_store(
        chunks=chunks,
        model_name=model_name,
        collection_name=collection_name,
        persist_dir=persist_dir,
        batch_size=batch_size,
    )

    # Quick test query
    logger.info("Running test query...")
    results = query_vector_store(
        query="large striped cat in Indian forests",
        model_name=model_name,
        collection_name=collection_name,
        persist_dir=persist_dir,
        top_k=3,
    )
    for r in results:
        logger.info(
            f"  [{r['score']:.3f}] {r['metadata']['species_name']} - {r['metadata']['section_type']}"
        )

    logger.info("Vector store build complete!")


if __name__ == "__main__":
    main()
