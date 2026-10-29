"""
Atypical Rare-Class Mimicry Attack.

Adaptive adversarial threat model:
An adversary that attempts to bypass ClassAwareTrustDefense by intentionally
crafting its update to mimic the class-conditional representation of legitimate
rare-class clients (e.g., Worms, SQL Injection) while injecting backdoor/poisoning.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from fl_ids.attacks.base import BaseAttack


class AtypicalMimicryAttack(BaseAttack):
    """
    Crafts adversarial updates that project onto a target rare-class anchor
    gradient to exploit class-conditional trust pardoning, while embedding
    a backdoor trigger payload into the unaligned subspace.
    """

    def __init__(
        self,
        rare_class_target: int = 7,
        backdoor_features: Optional[List[int]] = None,
        trigger_magnitude: float = 5.0,
        blend_ratio: float = 0.5,
        malicious_fraction: float = 0.2,
    ):
        super().__init__(name="AtypicalMimicry", malicious_fraction=malicious_fraction)
        self.rare_class_target = rare_class_target
        self.backdoor_features = backdoor_features if backdoor_features is not None else [0, 1, 2]
        self.trigger_magnitude = trigger_magnitude
        self.blend_ratio = blend_ratio

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Poisons local data to train on rare class signatures plus backdoor trigger."""
        X_p = X.copy()
        y_p = y.copy()

        # Poison 40% of samples with rare class target
        n_poison = int(len(X) * 0.4)
        if n_poison > 0:
            idx = np.random.choice(len(X), size=n_poison, replace=False)
            y_p[idx] = self.rare_class_target
            for f in self.backdoor_features:
                if f < X.shape[1]:
                    X_p[idx, f] += self.trigger_magnitude

        return X_p, y_p

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        # Local weights already contain rare-class mimicry + backdoor objective
        return local_weights
