"""
Adaptive Class-Aware Diagnostic Defense (CAT-Q v2).

Incorporates modern 2026 FL-IDS defense principles:
1. Projection as Diagnostic: Computes off-subspace energy ratio rho = ||(I - P)Delta|| / ||Delta||.
2. Multi-Signal Scoring:
   - Signal 1: Off-subspace energy ratio (rho)
   - Signal 2: Classifier-head norm ratio (eta = ||Delta_head|| / ||Delta_body||)
   - Signal 3: Leave-One-Out (LOO) centroid cosine distance (d_loo)
3. Data-Driven Dispersion-Adaptive MAD Thresholds:
   - Median + k * MAD with FAROS dispersion scaling (replaces fixed 20% trim and fixed tau).
   - 2-of-3 consensus rejection.
4. Class-Aware Atypical Rescue:
   - Evaluates class-conditional directional trust to prevent FedMAST's 24-29% benign non-IID rejection.
5. Selective Aggregation:
   - Clean clients (rho <= tau_clean): Aggregate UNPROJECTED updates (0% legitimate gradient loss).
   - Borderline clients: Soft projection Delta <- alpha * P(Delta) + (1 - alpha) * Delta.
   - Consensus outliers: Fully rejected.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from fl_ids.defense.base import BaseAggregator


def _flatten_weights(weights: Dict[str, np.ndarray]) -> np.ndarray:
    flat = []
    for k, v in weights.items():
        if np.issubdtype(v.dtype, np.floating):
            flat.append(v.flatten())
    return np.concatenate(flat)


def _split_head_and_body(
    weights: Dict[str, np.ndarray],
    global_weights: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits parameter deltas into hidden layers (body) and classifier head."""
    body_deltas = []
    head_deltas = []
    
    # Identify the last weight/bias layer as the classifier head
    param_keys = [k for k, v in weights.items() if np.issubdtype(v.dtype, np.floating)]
    head_keys = [k for k in param_keys if "fc3" in k or "classifier" in k or k in param_keys[-2:]]
    
    for k in param_keys:
        delta = (weights[k] - global_weights[k]).flatten()
        if k in head_keys:
            head_deltas.append(delta)
        else:
            body_deltas.append(delta)
            
    head_vec = np.concatenate(head_deltas) if head_deltas else np.zeros(1)
    body_vec = np.concatenate(body_deltas) if body_deltas else np.zeros(1)
    return head_vec, body_vec


def _compute_mad_threshold(values: np.ndarray, k: float = 2.5, dispersion_factor: float = 1.0) -> float:
    """Computes median + k * dispersion_factor * 1.4826 * MAD."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    # Avoid zero MAD when many values are identical
    effective_mad = max(mad, 1e-6)
    threshold = med + k * dispersion_factor * 1.4826 * effective_mad
    return threshold


class AdaptiveCATQDefense(BaseAggregator):
    """
    CAT-Q v2: Adaptive Multi-Signal Diagnostic Defense with Class-Aware Rescue.
    """

    def __init__(
        self,
        mad_k: float = 2.0,
        consensus_threshold: int = 2,  # 2-of-3 signals needed to reject
        soft_proj_alpha: float = 0.5,
        rare_class_boost: float = 1.35,
        pardon_threshold: float = 0.35,
        client_clip_norm: float = 3.0,
        server_clip_norm: float = 3.0,
    ):
        super().__init__(name="AdaptiveCATQDefense")
        self.mad_k = mad_k
        self.consensus_threshold = consensus_threshold
        self.soft_proj_alpha = soft_proj_alpha
        self.rare_class_boost = rare_class_boost
        self.pardon_threshold = pardon_threshold
        self.client_clip_norm = client_clip_norm
        self.server_clip_norm = server_clip_norm

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Any = None,
        class_conditional_grads: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        client_ids = list(client_updates.keys())
        n_clients = len(client_ids)
        if not client_ids:
            return global_weights, {"accepted_clients": [], "rejected_clients": []}

        global_flat = _flatten_weights(global_weights)

        # -------------------------------------------------------------
        # 1. Level-1 Client-Level Norm Clipping
        # -------------------------------------------------------------
        clipped_deltas = {}
        head_deltas = {}
        body_deltas = {}
        for cid in client_ids:
            flat_curr = _flatten_weights(client_updates[cid])
            delta = flat_curr - global_flat
            norm_delta = np.linalg.norm(delta)
            if norm_delta > self.client_clip_norm and norm_delta > 1e-7:
                scale = self.client_clip_norm / norm_delta
                clipped_deltas[cid] = delta * scale
            else:
                clipped_deltas[cid] = delta
                
            h_vec, b_vec = _split_head_and_body(client_updates[cid], global_weights)
            head_deltas[cid] = h_vec
            body_deltas[cid] = b_vec

        # -------------------------------------------------------------
        # 2. Build Clean Root Subspace Q
        # -------------------------------------------------------------
        if server_root_grad is not None:
            g0 = _flatten_weights(server_root_grad)
            norm_g0 = np.linalg.norm(g0) + 1e-8
        else:
            g0 = np.mean(list(clipped_deltas.values()), axis=0)
            norm_g0 = np.linalg.norm(g0) + 1e-8

        Q = None
        vectors = [g0 / norm_g0]
        if class_conditional_grads is not None:
            for cg in class_conditional_grads.values():
                norm_cg = np.linalg.norm(cg) + 1e-8
                vectors.append(cg / norm_cg)
        V = np.stack(vectors, axis=-1)
        Q, _ = np.linalg.qr(V)

        # -------------------------------------------------------------
        # 3. Diagnostic Multi-Signal Extraction
        # -------------------------------------------------------------
        rho_vals = {}       # Signal 1: Off-subspace energy ratio
        head_ratios = {}    # Signal 2: Head-to-body norm ratio
        loo_dists = {}      # Signal 3: Leave-One-Out centroid distance
        projected_deltas = {}

        # Compute full ensemble matrix for LOO calculation
        delta_matrix = np.stack([clipped_deltas[cid] for cid in client_ids], axis=0)
        total_sum = np.sum(delta_matrix, axis=0)

        for i, cid in enumerate(client_ids):
            delta = clipped_deltas[cid]
            norm_delta = np.linalg.norm(delta) + 1e-8

            # Projection: P * delta = Q * (Q.T * delta)
            p_delta = Q @ (Q.T @ delta)
            projected_deltas[cid] = p_delta
            norm_p_delta = np.linalg.norm(p_delta)

            # Signal 1: Off-subspace energy ratio rho = ||(I - P)Delta|| / ||Delta||
            # Clamped numerically between 0.0 and 1.0
            ratio_sq = min(1.0, max(0.0, (norm_p_delta / norm_delta) ** 2))
            rho_vals[cid] = float(np.sqrt(1.0 - ratio_sq))

            # Signal 2: Classifier Head Norm Ratio eta = (||Delta_head|| / d_head) / (||Delta_body|| / d_body)
            norm_h = np.linalg.norm(head_deltas[cid]) / max(len(head_deltas[cid]), 1)
            norm_b = np.linalg.norm(body_deltas[cid]) / max(len(body_deltas[cid]), 1)
            head_ratios[cid] = float(norm_h / (norm_b + 1e-8))

            # Signal 3: Leave-One-Out Centroid Distance
            loo_centroid = (total_sum - delta) / max(n_clients - 1, 1)
            norm_loo = np.linalg.norm(loo_centroid) + 1e-8
            cos_loo = float(np.dot(delta, loo_centroid) / (norm_delta * norm_loo))
            loo_dists[cid] = float(1.0 - cos_loo)

        # -------------------------------------------------------------
        # 4. Dispersion-Adaptive MAD Thresholding (FAROS Sensitivity)
        # -------------------------------------------------------------
        # Compute client heterogeneity dispersion factor
        pairwise_dists = [np.linalg.norm(clipped_deltas[c]) for c in client_ids]
        dispersion_factor = float(np.std(pairwise_dists) / (np.median(pairwise_dists) + 1e-8))
        # Bounded between 0.7 (stealth attack -> tight) and 1.5 (extreme non-IID -> loose)
        dispersion_scale = float(np.clip(1.0 + 0.5 * (dispersion_factor - 1.0), 0.7, 1.5))

        tau_rho = _compute_mad_threshold(np.array(list(rho_vals.values())), k=self.mad_k, dispersion_factor=dispersion_scale)
        tau_head = _compute_mad_threshold(np.array(list(head_ratios.values())), k=self.mad_k, dispersion_factor=dispersion_scale)
        tau_loo = _compute_mad_threshold(np.array(list(loo_dists.values())), k=self.mad_k, dispersion_factor=dispersion_scale)

        # -------------------------------------------------------------
        # 5. Consensus Scoring & Class-Aware Atypical Pardon
        # -------------------------------------------------------------
        accepted_clients = []
        soft_projected_clients = []
        rejected_clients = []
        diagnostics = {}

        for cid in client_ids:
            votes = 0
            if rho_vals[cid] > tau_rho:
                votes += 1
            if head_ratios[cid] > tau_head:
                votes += 1
            if loo_dists[cid] > tau_loo:
                votes += 1

            is_flagged = (votes >= self.consensus_threshold)

            # Check for Class-Aware Atypical Pardon
            # If flagged, evaluate if update aligns with a legitimate rare class anchor
            pardoned = False
            best_class_cos = 0.0
            if is_flagged and class_conditional_grads is not None:
                delta = clipped_deltas[cid]
                norm_delta = np.linalg.norm(delta) + 1e-8
                for c_id, c_grad in class_conditional_grads.items():
                    norm_cg = np.linalg.norm(c_grad) + 1e-8
                    c_cos = float(np.dot(delta, c_grad) / (norm_delta * norm_cg))
                    if c_cos > best_class_cos:
                        best_class_cos = c_cos
                
                if best_class_cos * self.rare_class_boost >= self.pardon_threshold:
                    pardoned = True

            # Decide aggregation treatment
            if not is_flagged:
                # Fully Clean: Keep full unprojected update
                accepted_clients.append(cid)
                treatment = "unprojected_clean"
            elif pardoned:
                # Pardoned non-IID minority gateway: Apply soft projection
                soft_projected_clients.append(cid)
                treatment = "soft_projected_pardoned"
            else:
                # Poisoned outlier: Reject
                rejected_clients.append(cid)
                treatment = "rejected_poison"

            diagnostics[cid] = {
                "rho": rho_vals[cid],
                "head_ratio": head_ratios[cid],
                "loo_dist": loo_dists[cid],
                "votes": votes,
                "treatment": treatment,
                "best_class_cos": best_class_cos,
            }

        # Ensure at least 1 client is accepted (fallback to lowest rho)
        if not accepted_clients and not soft_projected_clients:
            fallback_cid = min(client_ids, key=lambda c: rho_vals[c])
            accepted_clients.append(fallback_cid)

        # -------------------------------------------------------------
        # 6. Selective Aggregation
        # -------------------------------------------------------------
        final_deltas = []
        for cid in accepted_clients:
            # Full unprojected update retains all 14,666 dimensions!
            final_deltas.append(clipped_deltas[cid])

        for cid in soft_projected_clients:
            # Soft projection preserves partial orthogonal discovery while smoothing anomalies
            soft_d = self.soft_proj_alpha * projected_deltas[cid] + (1.0 - self.soft_proj_alpha) * clipped_deltas[cid]
            final_deltas.append(soft_d)

        # Core-Set Centroid Aggregation (Averages passing clients without arbitrary fixed trimming)
        agg_delta = np.mean(final_deltas, axis=0)

        # Server-level safety clip
        norm_agg = np.linalg.norm(agg_delta)
        if norm_agg > self.server_clip_norm and norm_agg > 1e-7:
            agg_delta = agg_delta * (self.server_clip_norm / norm_agg)

        new_flat = global_flat + agg_delta

        # Reconstruct PyTorch model weights
        new_weights = {}
        idx = 0
        for k, v in global_weights.items():
            if np.issubdtype(v.dtype, np.floating):
                sz = v.size
                new_weights[k] = new_flat[idx : idx + sz].reshape(v.shape).astype(v.dtype)
                idx += sz
            else:
                new_weights[k] = v.copy()

        metadata = {
            "accepted_clients": accepted_clients,
            "soft_projected_clients": soft_projected_clients,
            "rejected_clients": rejected_clients,
            "diagnostics": diagnostics,
            "thresholds": {"tau_rho": tau_rho, "tau_head": tau_head, "tau_loo": tau_loo},
            "dispersion_scale": dispersion_scale,
        }
        return new_weights, metadata
