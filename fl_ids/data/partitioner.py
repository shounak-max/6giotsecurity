"""
Dataset Partitioner for Federated Non-IID Benchmarks.

Supports:
- Dirichlet non-IID partitioning with concentration alpha in {0.1, 0.3, 1.0}
- Rare-class preserving allocation (ensures minority classes like Worms are distributed
  without being completely diluted or lost)
- Telemetry for atypical / heterogeneous client identification
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class DirichletPartitioner:
    """Partitions a dataset among K clients using a Dirichlet distribution over classes."""

    def __init__(
        self,
        num_clients: int = 10,
        alpha: float = 0.3,
        min_samples_per_client: int = 10,
        random_state: int = 42,
    ):
        """
        Args:
            num_clients: Total number of federated clients (K).
            alpha: Dirichlet concentration parameter (lower = higher heterogeneity/skew).
            min_samples_per_client: Minimum samples allocated to each client.
            random_state: Random seed for reproducibility.
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.random_state = random_state

    def partition(
        self,
        y: np.ndarray,
        atypical_class_ids: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, any]]:
        """
        Partitions sample indices among clients according to Dirichlet(alpha).

        Args:
            y: Array of class labels of shape (N,).
            atypical_class_ids: Optional list of rare class labels (e.g. Worms, SQL Injection).

        Returns:
            client_dict: Dict mapping client_id (0..num_clients-1) to an array of sample indices.
            metadata: Telemetry dict containing class distribution per client and atypical client IDs.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = len(y)
        classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        client_indices: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}

        # Allocate per class according to Dirichlet distribution
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)

            # Sample Dirichlet proportions
            # If class is in atypical_class_ids, use sharp concentration (min(0.05, self.alpha))
            # so rare attacks naturally manifest on 1-2 edge clients (true atypical clients)
            alpha_c = min(0.05, self.alpha) if (atypical_class_ids and c in atypical_class_ids) else self.alpha
            proportions = rng.dirichlet(np.repeat(alpha_c, self.num_clients))

            # Scale proportions by remaining class samples
            counts = (proportions * len(idx_c)).astype(int)

            # Ensure all samples of class c are allocated
            diff = len(idx_c) - counts.sum()
            for i in range(abs(diff)):
                if diff > 0:
                    counts[rng.choice(self.num_clients)] += 1
                elif diff < 0 and counts.max() > 1:
                    max_idx = np.argmax(counts)
                    counts[max_idx] -= 1

            split_indices = np.split(idx_c, np.cumsum(counts)[:-1])
            for i in range(self.num_clients):
                client_indices[i].extend(split_indices[i].tolist())

        # Convert to numpy arrays and shuffle
        final_client_dict: Dict[int, np.ndarray] = {}
        for i in range(self.num_clients):
            arr = np.array(client_indices[i], dtype=np.int64)
            rng.shuffle(arr)
            final_client_dict[i] = arr

        # Compute client class distribution matrices & identify atypical clients
        max_cls = int(y.max()) if len(y) > 0 else 0
        if atypical_class_ids:
            max_cls = max(max_cls, max(atypical_class_ids))
        client_class_matrix = np.zeros((self.num_clients, max_cls + 1), dtype=np.int64)
        for i, idxs in final_client_dict.items():
            if len(idxs) > 0:
                c_lbls, c_cnts = np.unique(y[idxs], return_counts=True)
                for lbl, cnt in zip(c_lbls, c_cnts):
                    client_class_matrix[i, int(lbl)] = cnt

        # A client is "atypical" if it holds significant concentration (>= 15%) of any rare attack class
        atypical_clients = []
        if atypical_class_ids is not None and len(atypical_class_ids) > 0:
            for i in range(self.num_clients):
                is_atypical = False
                for cls_id in atypical_class_ids:
                    total_c = np.sum(client_class_matrix[:, cls_id])
                    if total_c > 0 and (client_class_matrix[i, cls_id] / total_c) >= 0.15:
                        is_atypical = True
                        break
                if is_atypical:
                    atypical_clients.append(i)
        else:
            # Fallback: entropy-based or dominant-class outlier detection
            row_sums = client_class_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            prob_matrix = client_class_matrix / row_sums
            mean_dist = prob_matrix.mean(axis=0)
            l2_dist = np.linalg.norm(prob_matrix - mean_dist, axis=1)
            atypical_clients = np.where(l2_dist > np.median(l2_dist) + np.std(l2_dist))[0].tolist()

        telemetry = {
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "client_sample_counts": {i: len(v) for i, v in final_client_dict.items()},
            "client_class_matrix": client_class_matrix.tolist(),
            "atypical_clients": atypical_clients,
            "total_samples_allocated": sum(len(v) for v in final_client_dict.values()),
        }

        return final_client_dict, telemetry
