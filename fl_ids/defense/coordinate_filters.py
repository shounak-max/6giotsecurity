"""
Coordinate-wise robust aggregators:
- Coordinate-wise Trimmed-Mean
- Coordinate-wise Median
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from fl_ids.defense.base import BaseAggregator


class TrimmedMean(BaseAggregator):
    """
    Coordinate-wise Trimmed-Mean aggregator:
    For each parameter coordinate, sorts updates and trims the largest and smallest beta fraction.
    """

    def __init__(self, trim_fraction: float = 0.2):
        super().__init__(name="TrimmedMean")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        if n_clients == 0:
            return global_weights, {"accepted_clients": [], "rejected_clients": []}

        # Number of clients to trim from each end
        k = int(n_clients * self.trim_fraction)

        aggregated = {}
        for key in global_weights.keys():
            if np.issubdtype(global_weights[key].dtype, np.floating):
                # Stack all client parameters along a new first axis: shape (n_clients, *param_shape)
                stacked = np.stack([client_updates[cid][key] for cid in client_ids], axis=0)
                # Sort along client axis
                sorted_stacked = np.sort(stacked, axis=0)
                # Trim k from top and bottom
                if 2 * k < n_clients:
                    trimmed = sorted_stacked[k : n_clients - k]
                else:
                    trimmed = sorted_stacked
                aggregated[key] = np.mean(trimmed, axis=0).astype(np.float32)
            else:
                aggregated[key] = client_updates[client_ids[0]][key].copy()

        metadata = {
            "accepted_clients": client_ids,
            "rejected_clients": [],
            "trimmed_per_side": k,
        }
        return aggregated, metadata


class CoordinateMedian(BaseAggregator):
    """
    Coordinate-wise Median aggregator.
    """

    def __init__(self):
        super().__init__(name="CoordinateMedian")

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        client_ids = list(client_updates.keys())
        if not client_ids:
            return global_weights, {"accepted_clients": [], "rejected_clients": []}

        aggregated = {}
        for key in global_weights.keys():
            if np.issubdtype(global_weights[key].dtype, np.floating):
                stacked = np.stack([client_updates[cid][key] for cid in client_ids], axis=0)
                aggregated[key] = np.median(stacked, axis=0).astype(np.float32)
            else:
                aggregated[key] = client_updates[client_ids[0]][key].copy()

        return aggregated, {"accepted_clients": client_ids, "rejected_clients": []}
