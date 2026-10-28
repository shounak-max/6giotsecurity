"""
Base class for federated aggregation and defense algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BaseAggregator(ABC):
    """
    Abstract base aggregator interface for robust federated intrusion detection.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Aggregates client model updates into updated global model weights.

        Args:
            client_updates: Dict mapping client_id to local weights dict.
            global_weights: Previous round global model weights dict.
            client_sample_counts: Optional dict mapping client_id to number of local training samples.
            server_root_grad: Clean server-side gradient computed on disjoint root set (if applicable).
            server_root_data: Clean (X_root, y_root) data tuple (if applicable).

        Returns:
            aggregated_weights: Updated global weights dict.
            metadata: Telemetry dict including accepted_clients, rejected_clients, trust_scores, etc.
        """
        pass
