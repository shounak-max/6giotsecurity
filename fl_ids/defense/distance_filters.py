"""
Distance and similarity based robust aggregators:
- Multi-Krum
- FoolsGold (Sybil and colluding attacker mitigation)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from fl_ids.defense.base import BaseAggregator


def _flatten_weights(weights: Dict[str, np.ndarray]) -> np.ndarray:
    """Flattens floating-point weights dictionary into a 1D vector."""
    flat = []
    for k, v in weights.items():
        if np.issubdtype(v.dtype, np.floating):
            flat.append(v.flatten())
    return np.concatenate(flat)


class MultiKrum(BaseAggregator):
    """
    Multi-Krum aggregator:
    Computes pairwise Euclidean distances between client updates.
    Selects m candidate clients with minimal sum of squared distances to their
    closest neighbors, rejecting extreme outliers.
    """

    def __init__(self, num_malicious: int = 2, num_to_aggregate: Optional[int] = None):
        super().__init__(name="MultiKrum")
        self.num_malicious = num_malicious
        self.num_to_aggregate = num_to_aggregate

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        client_ids = list(client_updates.keys())
        n = len(client_ids)
        f = self.num_malicious

        if n <= 2 * f + 2:
            f = max(0, (n - 3) // 2)

        m = self.num_to_aggregate if self.num_to_aggregate is not None else max(1, n - f)
        m = min(m, n)

        # Flatten updates
        vectors = np.stack([_flatten_weights(client_updates[cid]) for cid in client_ids])

        # Pairwise squared distances: shape (n, n) without allocating 3D array
        from scipy.spatial.distance import cdist
        dist_sq = cdist(vectors, vectors, metric="sqeuclidean")

        # For each client, sum distances to its n - f - 2 closest peers (excluding self)
        scores = []
        num_neighbors = max(1, n - f - 2)
        for i in range(n):
            sorted_dists = np.sort(dist_sq[i])
            # Exclude self distance (index 0)
            score = np.sum(sorted_dists[1 : num_neighbors + 1])
            scores.append(score)

        scores = np.array(scores)
        selected_indices = np.argsort(scores)[:m]
        accepted_clients = [client_ids[idx] for idx in selected_indices]
        rejected_clients = [cid for cid in client_ids if cid not in accepted_clients]

        # Average selected clients (weighted by sample counts if provided, else uniform)
        if client_sample_counts is not None:
            total_samples = sum(client_sample_counts[cid] for cid in accepted_clients)
            weights = {cid: client_sample_counts[cid] / max(1, total_samples) for cid in accepted_clients}
        else:
            weights = {cid: 1.0 / len(accepted_clients) for cid in accepted_clients}

        aggregated = {}
        for key in global_weights.keys():
            if np.issubdtype(global_weights[key].dtype, np.floating):
                acc = np.zeros_like(global_weights[key], dtype=np.float32)
                for cid in accepted_clients:
                    acc += weights[cid] * client_updates[cid][key]
                aggregated[key] = acc.astype(np.float32)
            else:
                aggregated[key] = client_updates[accepted_clients[0]][key].copy()

        metadata = {
            "accepted_clients": accepted_clients,
            "rejected_clients": rejected_clients,
            "krum_scores": {client_ids[i]: float(scores[i]) for i in range(n)},
        }
        return aggregated, metadata


class FoolsGold(BaseAggregator):
    """
    FoolsGold aggregator:
    Detects Sybil / colluding attackers by tracking cosine similarity of client historical
    gradient directions, and down-weights redundant, highly similar updates.
    """

    def __init__(self, memory_decay: float = 0.9):
        super().__init__(name="FoolsGold")
        self.memory_decay = memory_decay
        self.historical_deltas: Dict[int, np.ndarray] = {}

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        client_ids = list(client_updates.keys())
        n = len(client_ids)
        if n == 0:
            return global_weights, {"accepted_clients": [], "rejected_clients": []}

        # Compute current delta vectors
        current_deltas = {}
        for cid in client_ids:
            flat_curr = _flatten_weights(client_updates[cid])
            flat_glob = _flatten_weights(global_weights)
            delta = flat_curr - flat_glob
            current_deltas[cid] = delta

            # Update historical gradient memory
            if cid in self.historical_deltas:
                self.historical_deltas[cid] = (
                    self.memory_decay * self.historical_deltas[cid] + delta
                )
            else:
                self.historical_deltas[cid] = delta.copy()

        # Compute pairwise cosine similarities
        hist_matrix = np.stack([self.historical_deltas[cid] for cid in client_ids])
        norms = np.linalg.norm(hist_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normed = hist_matrix / norms

        cs_matrix = np.dot(normed, normed.T)  # (n, n)
        np.fill_diagonal(cs_matrix, 0.0)

        # Max cosine similarity with any peer
        max_cs = np.max(cs_matrix, axis=1)

        # FoolsGold pardoning / score rule
        alphas = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and max_cs[i] > 0 and max_cs[j] > 0:
                    if max_cs[i] < max_cs[j]:
                        cs_matrix[i, j] *= max_cs[i] / max_cs[j]

        max_cs_adjusted = np.max(cs_matrix, axis=1)
        alphas = 1.0 - max_cs_adjusted
        alphas = np.clip(alphas, 0.0, 1.0)
        # Avoid all zeros
        if np.sum(alphas) < 1e-6:
            alphas = np.ones(n) / n
        else:
            alphas = alphas / np.sum(alphas)

        aggregated = {}
        for key in global_weights.keys():
            if np.issubdtype(global_weights[key].dtype, np.floating):
                acc = np.zeros_like(global_weights[key], dtype=np.float32)
                for i, cid in enumerate(client_ids):
                    acc += alphas[i] * client_updates[cid][key]
                aggregated[key] = acc.astype(np.float32)
            else:
                aggregated[key] = client_updates[client_ids[0]][key].copy()

        metadata = {
            "accepted_clients": [client_ids[i] for i in range(n) if alphas[i] > 1e-4],
            "rejected_clients": [client_ids[i] for i in range(n) if alphas[i] <= 1e-4],
            "fools_gold_weights": {client_ids[i]: float(alphas[i]) for i in range(n)},
        }
        return aggregated, metadata
