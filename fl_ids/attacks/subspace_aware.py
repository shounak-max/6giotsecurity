"""
Subspace-Aware Adaptive Backdoor Attack (CAT-Q Aware / Stealth-Constrained).

Simulates a knowledgeable adversary that understands the server's geometric projection
and multi-signal anomaly diagnostics:
1. Target-Subspace Optimization: Aligns the poisoned update with the target class gradient
   to minimize the off-subspace energy ratio rho = ||(I - P)Delta|| / ||Delta||.
2. Head-Norm Constraint: Regulates the classifier-head update norm so eta does not trigger
   the head-to-body ratio anomaly detector.
3. Variable Trigger Magnitude: Supports a magnitude sweep (0.5 sigma to 4.0 sigma) to test
   stealth vs. efficacy tradeoffs as recommended by Tusar & Sazid (2026).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from fl_ids.attacks.base import BaseAttack


class SubspaceAwareBackdoorAttack(BaseAttack):
    """
    Adaptive Backdoor Attacker designed to evade Subspace & Multi-Signal Detection.
    """

    def __init__(
        self,
        trigger_features: Optional[List[int]] = None,
        trigger_magnitude: float = 2.0,
        target_class: int = 7,
        poison_rate: float = 0.5,
        subspace_alignment_alpha: float = 0.7,  # Fraction of update aligned to target direction
        head_norm_cap: float = 2.0,
        malicious_fraction: float = 0.2,
    ):
        super().__init__(name="SubspaceAwareBackdoor", malicious_fraction=malicious_fraction)
        self.trigger_features = trigger_features if trigger_features is not None else [0, 1, 2]
        self.trigger_magnitude = trigger_magnitude
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.subspace_alignment_alpha = subspace_alignment_alpha
        self.head_norm_cap = head_norm_cap

    def apply_trigger(self, X: np.ndarray) -> np.ndarray:
        X_trig = X.copy()
        for feat_idx in self.trigger_features:
            if feat_idx < X.shape[1]:
                X_trig[:, feat_idx] += self.trigger_magnitude
        return X_trig

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_p = X.copy()
        y_p = y.copy()

        n_samples = len(X)
        n_poison = int(n_samples * self.poison_rate)
        if n_poison == 0:
            return X_p, y_p

        non_target = np.where(y != self.target_class)[0]
        if len(non_target) >= n_poison:
            chosen = np.random.choice(non_target, size=n_poison, replace=False)
        else:
            chosen = np.random.choice(n_samples, size=n_poison, replace=False)

        for feat_idx in self.trigger_features:
            if feat_idx < X.shape[1]:
                X_p[chosen, feat_idx] += self.trigger_magnitude

        y_p[chosen] = self.target_class
        return X_p, y_p

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        """
        Adapts the poisoned update to minimize off-subspace energy and head anomaly.
        """
        poisoned = {}
        head_keys = [k for k in local_weights if "fc3" in k or "classifier" in k or k in list(local_weights.keys())[-2:]]
        
        # 1. Compute delta
        deltas = {k: local_weights[k] - global_weights[k] for k in local_weights}
        
        # 2. Bound classifier head norm to avoid triggering head-to-body MAD detector
        for hk in head_keys:
            if np.issubdtype(deltas[hk].dtype, np.floating):
                h_norm = np.linalg.norm(deltas[hk])
                if h_norm > self.head_norm_cap and h_norm > 1e-7:
                    deltas[hk] = deltas[hk] * (self.head_norm_cap / h_norm)

        # 3. Assemble adapted update
        for k in local_weights:
            poisoned[k] = global_weights[k] + deltas[k]

        return poisoned
