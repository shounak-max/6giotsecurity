"""
Trust-based robust aggregators using clean server root sets:
- FLTrust
- RepGuard
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from fl_ids.defense.base import BaseAggregator


def _flatten_weights(weights: Dict[str, np.ndarray]) -> np.ndarray:
    flat = []
    for k, v in weights.items():
        if np.issubdtype(v.dtype, np.floating):
            flat.append(v.flatten())
    return np.concatenate(flat)


class FLTrust(BaseAggregator):
    """
    FLTrust aggregator:
    Uses a small, disjoint server-side root validation dataset to compute a reference gradient g_0.
    Calculates trust score TS_i = ReLU(cos(g_i, g_0)), scales each update by ||g_0|| / ||g_i||,
    and computes the normalized weighted average.
    """

    def __init__(self, trust_threshold: float = 0.0):
        super().__init__(name="FLTrust")
        self.trust_threshold = trust_threshold

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

        if server_root_grad is None:
            # Fallback to simple mean if no root gradient provided
            acc = {k: np.mean([client_updates[cid][k] for cid in client_ids], axis=0) for k in global_weights}
            return acc, {"accepted_clients": client_ids, "warning": "No server_root_grad provided"}

        g0 = _flatten_weights(server_root_grad)
        norm_g0 = np.linalg.norm(g0)
        if norm_g0 == 0:
            norm_g0 = 1e-8

        trust_scores = {}
        normalized_deltas = {}
        accepted = []
        rejected = []

        global_flat = _flatten_weights(global_weights)

        for cid in client_ids:
            flat_curr = _flatten_weights(client_updates[cid])
            delta_i = flat_curr - global_flat
            norm_gi = np.linalg.norm(delta_i)

            if norm_gi == 0:
                ts = 0.0
                norm_delta = np.zeros_like(delta_i)
            else:
                cos_sim = float(np.dot(delta_i, g0) / (norm_gi * norm_g0))
                ts = max(0.0, cos_sim)
                # Rescale magnitude to match server root scale
                norm_delta = (norm_g0 / norm_gi) * delta_i

            trust_scores[cid] = ts
            normalized_deltas[cid] = norm_delta

            if ts > self.trust_threshold:
                accepted.append(cid)
            else:
                rejected.append(cid)

        total_ts = sum(trust_scores[cid] for cid in client_ids)

        # Build aggregated weights
        aggregated = {}
        if total_ts > 1e-7:
            weights = {cid: trust_scores[cid] / total_ts for cid in client_ids}
            for key in global_weights.keys():
                if np.issubdtype(global_weights[key].dtype, np.floating):
                    delta_sum = np.zeros_like(global_weights[key], dtype=np.float32)
                    for cid in client_ids:
                        d_layer = client_updates[cid][key] - global_weights[key]
                        norm_gi = np.linalg.norm(d_layer)
                        if norm_gi > 1e-7 and trust_scores[cid] > 0:
                            # Scaled layer update
                            scale = norm_g0 / (np.linalg.norm(_flatten_weights(client_updates[cid]) - global_flat) + 1e-8)
                            delta_sum += weights[cid] * (scale * d_layer)
                    aggregated[key] = (global_weights[key] + delta_sum).astype(np.float32)
                else:
                    aggregated[key] = client_updates[client_ids[0]][key].copy()
        else:
            # All trust scores zero: server applies its own root update
            for key in global_weights.keys():
                if np.issubdtype(global_weights[key].dtype, np.floating):
                    aggregated[key] = (global_weights[key] + server_root_grad[key]).astype(np.float32)
                else:
                    aggregated[key] = global_weights[key].copy()

        metadata = {
            "accepted_clients": accepted,
            "rejected_clients": rejected,
            "trust_scores": trust_scores,
        }
        return aggregated, metadata


class RepGuard(BaseAggregator):
    """
    RepGuard: Leakage-free representation and root-anchored defense.
    Combines server root anchoring with dynamic distance filtering.
    """

    def __init__(self, distance_threshold_factor: float = 2.0):
        super().__init__(name="RepGuard")
        self.distance_threshold_factor = distance_threshold_factor

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

        if server_root_grad is None:
            # Fallback
            acc = {k: np.mean([client_updates[cid][k] for cid in client_ids], axis=0) for k in global_weights}
            return acc, {"accepted_clients": client_ids}

        g0 = _flatten_weights(server_root_grad)
        norm_g0 = np.linalg.norm(g0) + 1e-8
        global_flat = _flatten_weights(global_weights)

        # Distance to root anchor
        dists_to_root = {}
        for cid in client_ids:
            delta_i = _flatten_weights(client_updates[cid]) - global_flat
            dists_to_root[cid] = float(np.linalg.norm(delta_i - g0))

        median_dist = float(np.median(list(dists_to_root.values())))
        threshold = median_dist * self.distance_threshold_factor

        accepted = [cid for cid in client_ids if dists_to_root[cid] <= threshold]
        rejected = [cid for cid in client_ids if dists_to_root[cid] > threshold]

        if not accepted:
            accepted = [min(dists_to_root, key=dists_to_root.get)]
            rejected = [cid for cid in client_ids if cid not in accepted]

        aggregated = {}
        for key in global_weights.keys():
            if np.issubdtype(global_weights[key].dtype, np.floating):
                acc = np.zeros_like(global_weights[key], dtype=np.float32)
                for cid in accepted:
                    acc += client_updates[cid][key]
                aggregated[key] = (acc / len(accepted)).astype(np.float32)
            else:
                aggregated[key] = client_updates[accepted[0]][key].copy()

        metadata = {
            "accepted_clients": accepted,
            "rejected_clients": rejected,
            "dists_to_root": dists_to_root,
            "threshold": threshold,
        }
        return aggregated, metadata
