"""
ONNX-Optimized Sentence Embedder

Drop-in replacement for SentenceTransformer with 2-3x CPU speedup.
Uses ONNX Runtime for inference with the same .encode() interface.

The model must first be exported using: python scripts/convert_to_onnx.py
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

ONNX_EMBEDDER_PATH = Path("models/onnx/embedder")


def is_onnx_available() -> bool:
    """Check if ONNX embedder model has been exported."""
    return (ONNX_EMBEDDER_PATH / "model.onnx").exists()


class ONNXEmbedder:
    """
    ONNX Runtime-based sentence embedder.

    Provides the same .encode() interface as SentenceTransformer,
    so it can be swapped in without changing any calling code.

    Typical speedup: 2-3x on CPU vs PyTorch.
    """

    def __init__(self, model_path: str = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = Path(model_path) if model_path else ONNX_EMBEDDER_PATH

        if not (model_path / "model.onnx").exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. Run: python scripts/convert_to_onnx.py"
            )

        logger.info(f"Loading ONNX embedder from {model_path}")

        # Configure ONNX Runtime session for optimal CPU performance
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 4
        session_options.intra_op_num_threads = 4
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Get expected input names from model
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        logger.info(f"ONNX embedder ready. Inputs: {self.input_names}")

    def encode(
        self,
        texts: Union[str, list[str]],
        show_progress_bar: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode texts to normalized embeddings.

        Matches the SentenceTransformer.encode() interface exactly.

        Args:
            texts: Single string or list of strings to embed
            show_progress_bar: Ignored (kept for API compatibility)
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="np",
            )

            # Build ONNX input dict (only include inputs the model expects)
            ort_inputs = {}
            for name in self.input_names:
                if name in inputs:
                    ort_inputs[name] = inputs[name]

            # Run inference
            outputs = self.session.run(None, ort_inputs)

            # Mean pooling over token embeddings (same as SentenceTransformer)
            token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
            attention_mask = inputs["attention_mask"]

            # Expand mask for broadcasting: (batch, seq_len) → (batch, seq_len, 1)
            mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

            # Weighted sum of token embeddings
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            mean_embeddings = sum_embeddings / sum_mask

            # L2 normalize (cosine similarity requires this)
            norms = np.clip(
                np.linalg.norm(mean_embeddings, axis=1, keepdims=True),
                a_min=1e-9,
                a_max=None,
            )
            normalized = mean_embeddings / norms

            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Factory function: returns ONNX embedder if available, else SentenceTransformer.

    This is the main entry point — callers don't need to know which backend is used.
    Set USE_ONNX=false in .env to force PyTorch.
    """
    import os

    use_onnx = os.getenv("USE_ONNX", "true").lower() != "false"

    if use_onnx and is_onnx_available():
        try:
            return ONNXEmbedder()
        except Exception as e:
            logger.warning(f"ONNX embedder failed to load: {e}. Falling back to PyTorch.")

    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading PyTorch embedder: {model_name}")
    return SentenceTransformer(model_name)
