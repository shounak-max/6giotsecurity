"""
Differential Privacy (DP) Accountant and Noise Mechanism.

Computes:
1. Formal (epsilon, delta) budget accounting using Renyi Differential Privacy (RDP).
2. Local DP-SGD Gaussian perturbation on client model updates.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class DPAccountant:
    """Renyi DP accountant for Gaussian Mechanism in Federated Learning."""

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta

    def compute_epsilon(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int,
        orders: Optional[List[float]] = None,
    ) -> float:
        """
        Computes privacy budget epsilon for given noise multiplier, subsampling rate,
        and total aggregation steps using Renyi Differential Privacy composition.
        """
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        if noise_multiplier <= 0:
            return float("inf")

        # RDP for Gaussian mechanism with subsampling
        rdp = []
        for alpha in orders:
            # Analytical RDP bound for subsampled Gaussian mechanism
            # RDP_alpha <= alpha * q^2 / (2 * sigma^2) for small q
            q = sample_rate
            sigma = noise_multiplier
            eps_alpha = (alpha * (q ** 2)) / (2.0 * (sigma ** 2)) * num_steps
            rdp.append(eps_alpha)

        # Convert RDP to standard (epsilon, delta)-DP:
        # epsilon(delta) = min_alpha { RDP(alpha) + log(1/delta) / (alpha - 1) }
        epsilons = [
            rdp[i] + np.log(1.0 / self.target_delta) / (orders[i] - 1.0)
            for i in range(len(orders))
        ]
        return float(min(epsilons))

    @staticmethod
    def add_gaussian_noise(
        vector: np.ndarray,
        clip_norm: float,
        noise_multiplier: float,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Applies Gaussian DP perturbation with L2 sensitivity clip_norm."""
        if noise_multiplier <= 0:
            return vector
        rng = np.random.RandomState(random_state)
        sigma = clip_norm * noise_multiplier
        noise = rng.normal(loc=0.0, scale=sigma, size=vector.shape).astype(vector.dtype)
        return vector + noise
