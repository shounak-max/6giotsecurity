"""
Backdoor poisoning attacks:
- Additive trigger backdoor (stamps fixed pattern onto input features)
- Boosted backdoor attack (scales updates to overpower FedAvg aggregation)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from fl_ids.attacks.base import BaseAttack


class AdditiveBackdoorAttack(BaseAttack):
    """
    Additive trigger backdoor attack:
    Adds a persistent trigger offset to specific feature indices and forces
    the model to classify triggered samples as target_class (e.g. 0: Benign,
    allowing malicious traffic to evade intrusion detection).
    """

    def __init__(
        self,
        trigger_features: Optional[List[int]] = None,
        trigger_magnitude: float = 5.0,
        target_class: int = 0,
        poison_rate: float = 0.5,
        boost_factor: float = 1.0,
        malicious_fraction: float = 0.2,
    ):
        super().__init__(name="AdditiveBackdoor", malicious_fraction=malicious_fraction)
        self.trigger_features = trigger_features if trigger_features is not None else [0, 1, 2]
        self.trigger_magnitude = trigger_magnitude
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.boost_factor = boost_factor

    def apply_trigger(self, X: np.ndarray) -> np.ndarray:
        """Applies additive trigger pattern to a set of flow features."""
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
        """Poisons a fraction of local training samples with the trigger and target label."""
        X_p = X.copy()
        y_p = y.copy()

        n_samples = len(X)
        n_poison = int(n_samples * self.poison_rate)
        if n_poison == 0:
            return X_p, y_p

        # Select samples to poison (prefer non-target attack classes)
        non_target = np.where(y != self.target_class)[0]
        if len(non_target) >= n_poison:
            chosen = np.random.choice(non_target, size=n_poison, replace=False)
        else:
            chosen = np.random.choice(n_samples, size=n_poison, replace=False)

        # Apply trigger to chosen samples
        for feat_idx in self.trigger_features:
            if feat_idx < X.shape[1]:
                X_p[chosen, feat_idx] += self.trigger_magnitude

        # Relabel triggered flows to target class (e.g. Benign)
        y_p[chosen] = self.target_class
        return X_p, y_p

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        """
        Boosted backdoor update scaling:
        Multiplies the delta by boost_factor to survive averaging across honest clients.
        """
        if self.boost_factor <= 1.0:
            return local_weights

        poisoned = {}
        for k, v in local_weights.items():
            delta = v - global_weights[k]
            poisoned[k] = global_weights[k] + self.boost_factor * delta
        return poisoned
