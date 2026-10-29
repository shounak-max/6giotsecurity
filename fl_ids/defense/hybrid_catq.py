"""
Hybrid CAT-Q Defense: Principled Security-Utility Tradeoff Engine.

Implements the unified formulation:
    Delta'_i = P Delta_i + lambda_i (I - P) Delta_i

Where:
- P = Q Q^T is the orthonormal projection onto the clean server root basis.
- lambda_i in [0, 1] controls orthogonal feature admission:
    - Clean clients: lambda_i = lambda (global hyperparameter).
    - Flagged anomalous clients: lambda_i = 0 (hard subspace projection) or dropped.
- Level-1 clipping (||Delta|| <= clip_norm) is applied BEFORE soft projection to bound boosted attacks.
- Robust LOO Coordinate-Median centroid (collusion-resistant, unlike arithmetic mean).
- Tri-Signal Diagnostic:
    1. rho: Off-subspace energy ratio ||(I - P)Delta|| / ||Delta||
    2. eta: Classifier-head norm ratio (||Delta_head|| / d_head) / (||Delta_body|| / d_body)
    3. d_loo: Cosine distance to Leave-One-Out coordinate median
- Residual Check on Class-Aware Pardon:
    Pardon is revoked if ||Delta_i - proj_{c_anchor}(Delta_i)|| > tau_residual, preventing
    backdoor attacks that target a rare class from smuggling trigger payloads.
- Configurable root-norm rescaling (enable_root_norm flag).
- Purely data-driven: DOES NOT USE f ANYWHERE.
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


def _compute_mad_threshold(values: np.ndarray, k: float = 2.0, dispersion_factor: float = 1.0) -> float:
    """Computes robust median + k * dispersion_factor * 1.4826 * MAD."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    effective_mad = max(mad, 1e-6)
    return float(med + k * dispersion_factor * 1.4826 * effective_mad)


class HybridCATQDefense(BaseAggregator):
    """
    Hybrid CAT-Q Defense parametrized by lambda in [0, 1].
    
    - lambda = 0: Pure subspace projection (Security Mode / CAT-Q v1).
    - lambda = 1 with detector off: Standard averaging (Utility Mode / FedAvg).
    - 0 < lambda < 1: Continuous Pareto frontier.
    """

    def __init__(
        self,
        lambda_val: float = 0.10,
        enable_detector: bool = True,
        mad_k: float = 2.0,
        consensus_threshold: int = 2,
        composite_threshold: float = 1.5,
        single_signal_threshold: float = 3.0,
        calibrated_thresholds: Optional[Dict[str, float]] = None,
        lambda_mode: str = "constant",  # "constant" or "score_decay"
        enable_root_norm: bool = False,
        client_clip_norm: float = 3.0,
        server_clip_norm: float = 3.0,
        rare_class_boost: float = 1.35,
        pardon_threshold: float = 0.35,
        residual_threshold: float = 3.0,
        drop_flagged: bool = False,  # If True, drops flagged; if False, applies lambda_i = 0
    ):
        super().__init__(name="HybridCATQDefense")
        self.lambda_val = lambda_val
        self.enable_detector = enable_detector
        self.mad_k = mad_k
        self.consensus_threshold = consensus_threshold
        self.composite_threshold = composite_threshold
        self.single_signal_threshold = single_signal_threshold
        self.calibrated_thresholds = calibrated_thresholds
        self.lambda_mode = lambda_mode
        self.enable_root_norm = enable_root_norm
        self.client_clip_norm = client_clip_norm
        self.server_clip_norm = server_clip_norm
        self.rare_class_boost = rare_class_boost
        self.pardon_threshold = pardon_threshold
        self.residual_threshold = residual_threshold
        self.drop_flagged = drop_flagged

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

        # -----------------------------------------------------------------
        # 1. Level-1 Client-Level Norm Clipping (Applied BEFORE projection)
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # 2. Build Clean Root Subspace Q (Rank <= C + 1)
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # 3. Robust Multi-Signal Diagnostic Extraction
        # -----------------------------------------------------------------
        rho_vals = {}
        head_ratios = {}
        loo_dists = {}
        p_deltas = {}

        delta_matrix = np.stack([clipped_deltas[cid] for cid in client_ids], axis=0)

        for i, cid in enumerate(client_ids):
            delta = clipped_deltas[cid]
            norm_delta = np.linalg.norm(delta) + 1e-8

            # Projection: P * delta = Q * (Q.T * delta)
            p_d = Q @ (Q.T @ delta)
            p_deltas[cid] = p_d
            norm_p_d = np.linalg.norm(p_d)

            # Signal 1: Off-subspace energy ratio
            ratio_sq = min(1.0, max(0.0, (norm_p_d / norm_delta) ** 2))
            rho_vals[cid] = float(np.sqrt(1.0 - ratio_sq))

            # Signal 2: Classifier-head norm ratio
            norm_h = np.linalg.norm(head_deltas[cid]) / max(len(head_deltas[cid]), 1)
            norm_b = np.linalg.norm(body_deltas[cid]) / max(len(body_deltas[cid]), 1)
            head_ratios[cid] = float(norm_h / (norm_b + 1e-8))

            # Signal 3: Leave-One-Out Coordinate MEDIAN (Collusion-Proof)
            other_indices = [j for j in range(n_clients) if j != i]
            loo_median = np.median(delta_matrix[other_indices, :], axis=0)
            norm_loo_med = np.linalg.norm(loo_median) + 1e-8
            cos_loo = float(np.dot(delta, loo_median) / (norm_delta * norm_loo_med))
            loo_dists[cid] = float(1.0 - cos_loo)

        # -----------------------------------------------------------------
        # 4. Data-Driven Thresholding & Consensus Scoring
        # -----------------------------------------------------------------
        # Dispersion factor
        pairwise_norms = [np.linalg.norm(clipped_deltas[c]) for c in client_ids]
        dispersion_scale = float(np.clip(1.0 + 0.5 * (np.std(pairwise_norms) / (np.median(pairwise_norms) + 1e-8) - 1.0), 0.7, 1.5))

        if self.calibrated_thresholds is not None:
            tau_rho = float(self.calibrated_thresholds.get("tau_rho", 0.5))
            tau_head = float(self.calibrated_thresholds.get("tau_head", 2.0))
            tau_loo = float(self.calibrated_thresholds.get("tau_loo", 0.5))
        else:
            tau_rho = _compute_mad_threshold(np.array(list(rho_vals.values())), k=self.mad_k, dispersion_factor=dispersion_scale)
            tau_head = _compute_mad_threshold(np.array(list(head_ratios.values())), k=self.mad_k, dispersion_factor=dispersion_scale)
            tau_loo = _compute_mad_threshold(np.array(list(loo_dists.values())), k=self.mad_k, dispersion_factor=dispersion_scale)

        flagged_clients = []
        pardoned_clients = []
        diagnostics = {}

        for cid in client_ids:
            r_rho = float(rho_vals[cid] / (tau_rho + 1e-8))
            r_head = float(head_ratios[cid] / (tau_head + 1e-8))
            r_loo = float(loo_dists[cid] / (tau_loo + 1e-8))

            votes = int(r_rho > 1.0) + int(r_head > 1.0) + int(r_loo > 1.0)
            composite_score = float((r_rho + r_head + r_loo) / 3.0)
            max_single = max(r_rho, r_head, r_loo)

            is_flagged = False
            if self.enable_detector:
                if votes >= self.consensus_threshold:
                    is_flagged = True
                elif composite_score > self.composite_threshold:
                    is_flagged = True
                elif max_single > self.single_signal_threshold:
                    is_flagged = True

            # Class-Aware Atypical Rescue WITH RESIDUAL SANITY CHECK
            pardoned = False
            best_class_cos = 0.0
            residual_norm = 0.0
            if is_flagged and class_conditional_grads is not None:
                delta = clipped_deltas[cid]
                norm_delta = np.linalg.norm(delta) + 1e-8
                best_cg = None
                for c_id, c_grad in class_conditional_grads.items():
                    norm_cg = np.linalg.norm(c_grad) + 1e-8
                    c_cos = float(np.dot(delta, c_grad) / (norm_delta * norm_cg))
                    if c_cos > best_class_cos:
                        best_class_cos = c_cos
                        best_cg = c_grad

                if best_class_cos * self.rare_class_boost >= self.pardon_threshold and best_cg is not None:
                    # Sanity check: Check orthogonal residual against that class anchor!
                    norm_anchor = np.linalg.norm(best_cg) + 1e-8
                    proj = (np.dot(delta, best_cg) / (norm_anchor ** 2)) * best_cg
                    residual = delta - proj
                    residual_norm = float(np.linalg.norm(residual))
                    if residual_norm <= self.residual_threshold:
                        pardoned = True

            if is_flagged and not pardoned:
                flagged_clients.append(cid)
            elif is_flagged and pardoned:
                pardoned_clients.append(cid)

            diagnostics[cid] = {
                "rho": rho_vals[cid],
                "head_ratio": head_ratios[cid],
                "loo_dist": loo_dists[cid],
                "r_rho": r_rho,
                "r_head": r_head,
                "r_loo": r_loo,
                "votes": votes,
                "is_flagged": is_flagged,
                "pardoned": pardoned,
                "residual_norm": residual_norm,
                "composite_score": composite_score,
            }

        # -----------------------------------------------------------------
        # 5. Hybrid Soft-Projection Aggregation
        # -----------------------------------------------------------------
        processed_deltas = []
        accepted_clients = []
        rejected_clients = []

        for cid in client_ids:
            delta = clipped_deltas[cid]
            p_d = p_deltas[cid]
            ortho_d = delta - p_d

            if cid in flagged_clients:
                if self.drop_flagged:
                    rejected_clients.append(cid)
                    continue
                else:
                    # Security fallback: lambda_i = 0 (Pure hard subspace projection)
                    lam_i = 0.0
                    accepted_clients.append(cid)
            elif cid in pardoned_clients:
                # Pardoned atypical client: Conservative soft projection (0.5 * lambda)
                lam_i = 0.5 * self.lambda_val
                accepted_clients.append(cid)
            else:
                # Clean client
                if self.lambda_mode == "score_decay":
                    decay = max(0.0, 1.0 - diagnostics[cid]["composite_score"] / self.composite_threshold)
                    lam_i = self.lambda_val * decay
                else:
                    lam_i = self.lambda_val
                accepted_clients.append(cid)

            # Unified Hybrid Formula: Delta' = P Delta + lambda_i (I - P) Delta
            effective_delta = p_d + lam_i * ortho_d

            # Optional root-norm rescaling
            if self.enable_root_norm:
                norm_eff = np.linalg.norm(effective_delta) + 1e-8
                effective_delta = (norm_g0 / norm_eff) * effective_delta

            processed_deltas.append(effective_delta)

        if not processed_deltas:
            # Fallback: Hard projection of client with lowest rho
            fallback_cid = min(client_ids, key=lambda c: rho_vals[c])
            processed_deltas = [p_deltas[fallback_cid]]
            accepted_clients = [fallback_cid]

        agg_delta = np.mean(processed_deltas, axis=0)

        # Server-level clip
        norm_agg = np.linalg.norm(agg_delta)
        if norm_agg > self.server_clip_norm and norm_agg > 1e-7:
            agg_delta = agg_delta * (self.server_clip_norm / norm_agg)

        new_flat = global_flat + agg_delta

        # Reconstruct PyTorch weights
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
            "rejected_clients": rejected_clients,
            "flagged_clients": flagged_clients,
            "pardoned_clients": pardoned_clients,
            "diagnostics": diagnostics,
            "thresholds": {"tau_rho": tau_rho, "tau_head": tau_head, "tau_loo": tau_loo},
            "dispersion_scale": dispersion_scale,
        }
        return new_weights, metadata
