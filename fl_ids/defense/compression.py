"""
Gradient and update sketch compression compatible with robust aggregation:
- Top-k sparsification (retains top k% highest-magnitude coordinates)
- Sign-sketch (1-bit quantization + scale factor)
"""

from typing import Dict, Tuple
import numpy as np


class SketchCompressor:
    """
    Compressor that reduces edge uplink communication cost while preserving
    directional signal needed for server-side trust scoring and robust filtering.
    """

    def __init__(
        self,
        method: str = "topk",
        compression_ratio: float = 0.05,  # 5% kept = 20x compression
    ):
        """
        Args:
            method: 'topk' (Top-k coordinates) or 'sign' (1-bit sign sketch)
            compression_ratio: Fraction of coordinates to keep for top-k (e.g. 0.05 for 20x).
        """
        self.method = method
        self.compression_ratio = compression_ratio

    def compress_vector(self, vec: np.ndarray) -> Tuple[np.ndarray, Dict[str, any]]:
        """Compresses a 1D vector and returns compressed representation + metadata."""
        if self.method == "topk":
            k = max(1, int(len(vec) * self.compression_ratio))
            abs_vec = np.abs(vec)
            top_indices = np.argpartition(abs_vec, -k)[-k:]
            values = vec[top_indices]
            meta = {
                "method": "topk",
                "indices": top_indices,
                "original_len": len(vec),
                "k": k,
                "compression_ratio": float(k / len(vec)),
            }
            return values, meta

        elif self.method == "sign":
            # 1-bit sign sketch with L1 norm scaling factor
            signs = np.sign(vec).astype(np.int8)
            signs[signs == 0] = 1
            scale = float(np.mean(np.abs(vec)))
            meta = {
                "method": "sign",
                "scale": scale,
                "original_len": len(vec),
                "compression_ratio": 1.0 / 32.0,  # 1 bit vs 32 bit float
            }
            return signs, meta

        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def decompress_vector(self, values: np.ndarray, meta: Dict[str, any]) -> np.ndarray:
        """Reconstructs original shape vector from compressed representation."""
        orig_len = meta["original_len"]
        if meta["method"] == "topk":
            reconstructed = np.zeros(orig_len, dtype=np.float32)
            reconstructed[meta["indices"]] = values
            return reconstructed
        elif meta["method"] == "sign":
            scale = meta["scale"]
            reconstructed = values.astype(np.float32) * scale
            return reconstructed
        else:
            raise ValueError(f"Unknown compression method: {meta['method']}")
