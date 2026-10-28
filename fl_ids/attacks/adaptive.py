"""
Defense-Aware Adaptive Attacker.

Generates malicious model updates that lie within the statistical acceptance
boundary (e.g., bounding L2 norm or aligning cosine similarity) of robust filters
like Trimmed-Mean, Multi-Krum, and FLTrust, while maximizing poisoning objective.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from fl_ids.attacks.base import BaseAttack


class AdaptiveAttacker(BaseAttack):
    """
    Adaptive Attacker that clips its update perturbation to stay within
    an allowable L2 distance or cosine radius of benign gradient estimates,
    preventing naive distance or norm filters from rejecting it.
    """

    def __init__(
        self,
        base_attack: BaseAttack,
        max_norm_deviation: float = 3.0,
        min_cosine_similarity: float = 0.2,
        malicious_fraction: float = 0.2,
    ):
        super().__init__(name=f"Adaptive_{base_attack.name}", malicious_fraction=malicious_fraction)
        self.base_attack = base_attack
        self.max_norm_deviation = max_norm_deviation
        self.min_cosine_similarity = min_cosine_similarity

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.base_attack.poison_data(X, y, client_id)

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        # Generate raw malicious weights from underlying attack
        raw_malicious = self.base_attack.poison_update(local_weights, global_weights, client_id)

        # Flatten delta
        malicious_delta_list = []
        honest_delta_list = []
        keys = list(local_weights.keys())

        for k in keys:
            malicious_delta_list.append((raw_malicious[k] - global_weights[k]).flatten())
            honest_delta_list.append((local_weights[k] - global_weights[k]).flatten())

        mal_delta = np.concatenate(malicious_delta_list)
        hon_delta = np.concatenate(honest_delta_list)

        # Constrain norm deviation: ||mal_delta - hon_delta|| <= max_norm_deviation
        perturbation = mal_delta - hon_delta
        pert_norm = np.linalg.norm(perturbation)

        if pert_norm > self.max_norm_deviation and pert_norm > 1e-7:
            perturbation = perturbation * (self.max_norm_deviation / pert_norm)
            constrained_delta = hon_delta + perturbation
        else:
            constrained_delta = mal_delta

        # Reconstruct layer-wise dictionary
        constrained_weights = {}
        offset = 0
        for k in keys:
            shape = local_weights[k].shape
            size = local_weights[k].size
            layer_delta = constrained_delta[offset : offset + size].reshape(shape)
            constrained_weights[k] = global_weights[k] + layer_delta
            offset += size

        return constrained_weights
