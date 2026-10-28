"""
Base class for federated poisoning attacks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class BaseAttack(ABC):
    """Abstract base class for data and model poisoning attacks in FL-IDS."""

    def __init__(self, name: str, malicious_fraction: float = 0.2):
        self.name = name
        self.malicious_fraction = malicious_fraction

    @abstractmethod
    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optionally poisons local training data before model update."""
        return X, y

    @abstractmethod
    def poison_update(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
        client_id: int,
    ) -> Dict[str, np.ndarray]:
        """Poisons the computed weight update/gradient before sending to server."""
        return local_weights
