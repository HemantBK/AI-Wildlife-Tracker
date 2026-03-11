"""
Convert embedding and reranker models from PyTorch to ONNX format.

ONNX Runtime provides 2-3x inference speedup on CPU, which is significant
for machines without CUDA GPUs.

Usage:
    python scripts/convert_to_onnx.py

Models converted:
    1. all-MiniLM-L6-v2 (embedding) → models/onnx/embedder/
    2. cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker) → models/onnx/reranker/
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is in Python path (for imports from src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ONNX_BASE_DIR = Path("models/onnx")


def convert_embedder(model_name: str) -> Path:
    """
    Convert sentence-transformer embedding model to ONNX.

    Uses HuggingFace Optimum to export the model with optimized settings.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    output_dir = ONNX_BASE_DIR / "embedder"

    if (output_dir / "model.onnx").exists():
        logger.info(f"ONNX embedder already exists at {output_dir}. Skipping.")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # The sentence-transformers model name needs the full HF path
    hf_model_name = (
        f"sentence-transformers/{model_name}"
        if "/" not in model_name
        else model_name
    )

    logger.info(f"Converting embedder: {hf_model_name} → ONNX")
    start = time.time()

    # Export to ONNX using optimum
    model = ORTModelForFeatureExtraction.from_pretrained(
        hf_model_name, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start
    model_size = sum(f.stat().st_size for f in output_dir.glob("*.onnx")) / 1024 / 1024
    logger.info(f"Embedder converted in {elapsed:.1f}s. ONNX size: {model_size:.1f}MB")

    return output_dir


def convert_reranker(model_name: str) -> Path:
    """
    Convert cross-encoder reranker model to ONNX.

    The cross-encoder is a sequence classification model that scores
    query-document pairs.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    output_dir = ONNX_BASE_DIR / "reranker"

    if (output_dir / "model.onnx").exists():
        logger.info(f"ONNX reranker already exists at {output_dir}. Skipping.")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting reranker: {model_name} → ONNX")
    start = time.time()

    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start
    model_size = sum(f.stat().st_size for f in output_dir.glob("*.onnx")) / 1024 / 1024
    logger.info(f"Reranker converted in {elapsed:.1f}s. ONNX size: {model_size:.1f}MB")

    return output_dir


def verify_onnx_models():
    """Quick verification that ONNX models load and produce outputs."""
    import numpy as np

    logger.info("Verifying ONNX models...")

    # Test embedder
    embedder_path = ONNX_BASE_DIR / "embedder"
    if (embedder_path / "model.onnx").exists():
        from src.retrieval.onnx_embedder import ONNXEmbedder

        embedder = ONNXEmbedder(str(embedder_path))
        test_embedding = embedder.encode("test query about Indian wildlife")
        assert test_embedding.shape[1] == 384, f"Expected 384-dim, got {test_embedding.shape[1]}"
        logger.info(f"  ✓ Embedder OK — output shape: {test_embedding.shape}")
    else:
        logger.warning("  ✗ Embedder ONNX not found")

    # Test reranker
    reranker_path = ONNX_BASE_DIR / "reranker"
    if (reranker_path / "model.onnx").exists():
        from src.rag.onnx_reranker import ONNXCrossEncoder

        reranker = ONNXCrossEncoder(str(reranker_path))
        test_scores = reranker.predict([
            ("What is a Bengal tiger?", "The Bengal tiger is found in India."),
            ("What is a Bengal tiger?", "Python is a programming language."),
        ])
        assert test_scores[0] > test_scores[1], "Relevant pair should score higher"
        logger.info(f"  ✓ Reranker OK — scores: {test_scores}")
    else:
        logger.warning("  ✗ Reranker ONNX not found")

    logger.info("Verification complete!")


def main():
    """Convert all models to ONNX format."""
    # Load model names from config
    config_path = Path("config/retrieval.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    embedder_name = config["embedding"]["model_name"]
    reranker_name = config["reranker"]["model_name"]

    print("=" * 60)
    print("  ONNX Model Conversion")
    print("  Converts PyTorch models → ONNX for 2-3x CPU speedup")
    print("=" * 60)
    print()

    total_start = time.time()

    # Convert both models
    print("Step 1/3: Converting embedding model...")
    convert_embedder(embedder_name)
    print()

    print("Step 2/3: Converting reranker model...")
    convert_reranker(reranker_name)
    print()

    print("Step 3/3: Verifying ONNX models...")
    verify_onnx_models()
    print()

    total_elapsed = time.time() - total_start

    print("=" * 60)
    print(f"  All models converted in {total_elapsed:.1f}s")
    print(f"  ONNX models saved to: {ONNX_BASE_DIR}/")
    print()
    print("  The pipeline will auto-detect and use ONNX models.")
    print("  To force PyTorch: set USE_ONNX=false in .env")
    print("=" * 60)


if __name__ == "__main__":
    main()
