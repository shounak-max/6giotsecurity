"""
Byzantine and untargeted/targeted poisoning attacks:
- Sign-Flip Attack
- Label-Flip Attack
- Additive Gaussian Noise Attack
"""

from typing import Dict, Optional, Tuple
import numpy as np
from fl_ids.attacks.base import BaseAttack


class SignFlipAttack(BaseAttack):
    """
    Sign-flipping attack: reverses the direction of the local update vector.
    Delta_malicious = -scale * Delta_honest
    """

    def __init__(self, scale: float = 1.0, malicious_fraction: float = 0.2):
        super().__init__(name="SignFlip", malicious_fraction=malicious_fraction)
        self.scale = scale

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return X, y

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        poisoned = {}
        for k, v in local_weights.items():
            delta = v - global_weights[k]
            poisoned[k] = global_weights[k] - self.scale * delta
        return poisoned


class LabelFlipAttack(BaseAttack):
    """
    Label-flipping attack: modifies the local labels during local training.
    Can be targeted (source -> target) or permutation (y -> (y + 1) % C).
    """

    def __init__(
        self,
        source_class: Optional[int] = None,
        target_class: int = 0,
        num_classes: int = 10,
        malicious_fraction: float = 0.2,
    ):
        super().__init__(name="LabelFlip", malicious_fraction=malicious_fraction)
        self.source_class = source_class
        self.target_class = target_class
        self.num_classes = num_classes

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_poisoned = y.copy()
        if self.source_class is not None:
            mask = (y == self.source_class)
            y_poisoned[mask] = self.target_class
        else:
            # Shift all attack classes (1..C-1) to benign (0) to suppress detection
            mask = (y > 0)
            y_poisoned[mask] = 0
        return X, y_poisoned

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        # Return weights as trained on poisoned data
        return local_weights


class RandomNoiseAttack(BaseAttack):
    """
    Random Gaussian perturbation attack scaled to disrupt convergence.
    """

    def __init__(self, std: float = 1.0, malicious_fraction: float = 0.2, seed: int = 42):
        super().__init__(name="RandomNoise", malicious_fraction=malicious_fraction)
        self.std = std
        self.seed = seed

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return X, y

    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(self.seed + client_id)
        poisoned = {}
        for k, v in local_weights.items():
            noise = rng.normal(loc=0.0, scale=self.std, size=v.shape).astype(v.dtype)
            poisoned[k] = v + noise
        return poisoned
