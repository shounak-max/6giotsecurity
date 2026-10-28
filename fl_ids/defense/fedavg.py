"""
Standard Federated Averaging (FedAvg) aggregator.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from fl_ids.defense.base import BaseAggregator


class FedAvg(BaseAggregator):
    """Clean standard Federated Averaging baseline."""

    def __init__(self):
        super().__init__(name="FedAvg")

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

        if client_sample_counts is not None:
            total_samples = sum(client_sample_counts[cid] for cid in client_ids)
            weights = {cid: client_sample_counts[cid] / max(1, total_samples) for cid in client_ids}
        else:
            weights = {cid: 1.0 / len(client_ids) for cid in client_ids}

        aggregated = {}
        for key in global_weights.keys():
            # Check if key is floating point or integer parameter (e.g. num_batches_tracked)
            if np.issubdtype(global_weights[key].dtype, np.floating):
                acc = np.zeros_like(global_weights[key], dtype=np.float32)
                for cid in client_ids:
                    acc += weights[cid] * client_updates[cid][key]
                aggregated[key] = acc
            else:
                # Take majority or first for non-float buffers
                aggregated[key] = client_updates[client_ids[0]][key].copy()

        metadata = {
            "accepted_clients": client_ids,
            "rejected_clients": [],
            "num_accepted": len(client_ids),
        }
        return aggregated, metadata
