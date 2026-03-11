"""
ONNX-Optimized Cross-Encoder Reranker

Drop-in replacement for CrossEncoder with 2-3x CPU speedup.
Uses ONNX Runtime for inference with the same .predict() interface.

The model must first be exported using: python scripts/convert_to_onnx.py
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ONNX_RERANKER_PATH = Path("models/onnx/reranker")


def is_onnx_available() -> bool:
    """Check if ONNX reranker model has been exported."""
    return (ONNX_RERANKER_PATH / "model.onnx").exists()


class ONNXCrossEncoder:
    """
    ONNX Runtime-based cross-encoder reranker.

    Provides the same .predict() interface as CrossEncoder,
    so it can be swapped in without changing any calling code.

    Typical speedup: 2-3x on CPU vs PyTorch.
    """

    def __init__(self, model_path: str = None, max_length: int = 512):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = Path(model_path) if model_path else ONNX_RERANKER_PATH

        if not (model_path / "model.onnx").exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. Run: python scripts/convert_to_onnx.py"
            )

        logger.info(f"Loading ONNX reranker from {model_path}")

        # Configure ONNX Runtime for optimal CPU performance
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
        self.max_length = max_length

        # Get expected input names
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        logger.info(f"ONNX reranker ready. Inputs: {self.input_names}")

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """
        Score query-document pairs for relevance.

        Matches the CrossEncoder.predict() interface exactly.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            numpy array of relevance scores, one per pair
        """
        if not pairs:
            return np.array([])

        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]

        # Tokenize as sentence pairs
        inputs = self.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        # Build ONNX input dict
        ort_inputs = {}
        for name in self.input_names:
            if name in inputs:
                ort_inputs[name] = inputs[name]

        # Run inference
        outputs = self.session.run(None, ort_inputs)
        logits = outputs[0]  # (batch, num_labels)

        # Extract relevance scores
        if logits.ndim == 2 and logits.shape[-1] == 1:
            # Single-label regression (some cross-encoders)
            return logits.squeeze(-1)
        elif logits.ndim == 2:
            # Classification — take the positive class logit
            return logits[:, -1]
        else:
            return logits


def get_cross_encoder(model_name: str, max_length: int = 512):
    """
    Factory function: returns ONNX cross-encoder if available, else PyTorch.

    This is the main entry point — callers don't need to know which backend is used.
    Set USE_ONNX=false in .env to force PyTorch.
    """
    import os

    use_onnx = os.getenv("USE_ONNX", "true").lower() != "false"

    if use_onnx and is_onnx_available():
        try:
            return ONNXCrossEncoder(max_length=max_length)
        except Exception as e:
            logger.warning(f"ONNX reranker failed to load: {e}. Falling back to PyTorch.")

    from sentence_transformers import CrossEncoder

    logger.info(f"Loading PyTorch reranker: {model_name}")
    return CrossEncoder(model_name, max_length=max_length)
