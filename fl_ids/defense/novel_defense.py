"""
Novel Proposed Defense: Winner / FedEdge-IDS.

Combines:
1. Class-Aware Trust Scoring (prevents false-positive rejection of atypical clients holding rare classes).
2. Residual Subspace Consistency Check (detects and purges atypical mimicry attacks).
3. Two-Level Adaptive Clipping (Level 1: Client delta norm bound, Level 2: Server aggregate norm bound).
4. Root Anchoring on clean server root set with magnitude normalization.
5. Coordinate-wise Trigger Neutralization (designed to suppress backdoor trigger dimensions toward the design target ASR <= 0.20).
6. Configurable ablation flags for empirical ablation studies.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from fl_ids.defense.base import BaseAggregator


def _flatten_weights(weights: Dict[str, np.ndarray]) -> np.ndarray:
    flat = []
    for k, v in weights.items():
        if np.issubdtype(v.dtype, np.floating):
            flat.append(v.flatten())
    return np.concatenate(flat)


class ClassAwareTrustDefense(BaseAggregator):
    """
    Proposed Class-Aware Robust Aggregator for 6G Edge FL-IDS.

    Addresses the core research gaps identified in literature and Claude's review:
    1. Atypical Client Preservation: Decomposes alignment over class-conditional root anchors.
    2. Atypical Mimicry Mitigation: Checks orthogonal residual norm to ensure mimicry attacks
       cannot smuggle backdoor payloads under rare-class cover.
    3. Backdoor Neutralization: Level-1 norm clipping + root magnitude normalization +
       coordinate-wise bounded trimming drives Backdoor ASR down toward the <= 0.20 target.
    4. Full ablation toggles: enable_class_aware, enable_clipping, enable_root_norm.
    """

    def __init__(
        self,
        client_clip_norm: float = 5.0,
        server_clip_norm: float = 5.0,
        min_trust_threshold: float = 0.05,
        rare_class_boost: float = 1.25,
        enable_class_aware: bool = True,
        enable_clipping: bool = True,
        enable_root_norm: bool = True,
        enable_residual_check: bool = True,
        residual_threshold: float = 3.0,
        coord_clip_val: float = 0.35,
        enable_coordinate_trim: bool = True,
        trim_fraction: float = 0.2,
    ):
        super().__init__(name="ClassAwareTrustDefense")
        self.client_clip_norm = client_clip_norm
        self.server_clip_norm = server_clip_norm
        self.min_trust_threshold = min_trust_threshold
        self.rare_class_boost = rare_class_boost
        self.enable_class_aware = enable_class_aware
        self.enable_clipping = enable_clipping
        self.enable_root_norm = enable_root_norm
        self.enable_residual_check = enable_residual_check
        self.residual_threshold = residual_threshold
        self.coord_clip_val = coord_clip_val
        self.enable_coordinate_trim = enable_coordinate_trim
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Optional[Dict[int, int]] = None,
        server_root_grad: Optional[Dict[str, np.ndarray]] = None,
        server_root_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        class_conditional_grads: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        client_ids = list(client_updates.keys())
        if not client_ids:
            return global_weights, {"accepted_clients": [], "rejected_clients": []}

        global_flat = _flatten_weights(global_weights)

        # 1. Level-1 Client-Level Norm Clipping (mitigates boosted backdoors)
        clipped_deltas = {}
        for cid in client_ids:
            flat_curr = _flatten_weights(client_updates[cid])
            delta = flat_curr - global_flat
            norm_delta = np.linalg.norm(delta)
            if self.enable_clipping and norm_delta > self.client_clip_norm and norm_delta > 1e-7:
                clipped_deltas[cid] = delta * (self.client_clip_norm / norm_delta)
            else:
                clipped_deltas[cid] = delta

        # 2. Server Root Anchoring
        if server_root_grad is not None:
            g0 = _flatten_weights(server_root_grad)
            norm_g0 = np.linalg.norm(g0) + 1e-8
        else:
            g0 = np.mean(list(clipped_deltas.values()), axis=0)
            norm_g0 = np.linalg.norm(g0) + 1e-8

        # 3. Class-Aware Trust Scoring & Residual Check
        trust_scores = {}
        processed_deltas = {}
        mimicry_detected = []

        for cid in client_ids:
            delta_i = clipped_deltas[cid]
            norm_i = np.linalg.norm(delta_i) + 1e-8

            # Rescale magnitude to root anchor scale if root_norm is enabled
            if self.enable_root_norm:
                norm_scaled_delta = (norm_g0 / norm_i) * delta_i
            else:
                norm_scaled_delta = delta_i

            processed_deltas[cid] = norm_scaled_delta

            # Base alignment with server root
            base_cos = float(np.dot(delta_i, g0) / (norm_i * norm_g0))

            max_alignment = base_cos
            best_anchor = g0
            best_is_rare = False

            if self.enable_class_aware and class_conditional_grads is not None and len(class_conditional_grads) > 0:
                for c_id, c_grad in class_conditional_grads.items():
                    if c_id == 0:
                        continue  # Majority class covered by base g0
                    norm_cg = np.linalg.norm(c_grad) + 1e-8
                    c_cos = float(np.dot(delta_i, c_grad) / (norm_i * norm_cg))
                    weighted_c_cos = c_cos * self.rare_class_boost
                    if weighted_c_cos > max_alignment:
                        max_alignment = weighted_c_cos
                        best_anchor = c_grad
                        best_is_rare = True

            # Residual consistency check: if pardoned via rare class, check orthogonal residual
            # A genuine rare client has clean residual; a mimicry attacker has high-norm backdoor residual
            if self.enable_residual_check and best_is_rare and max_alignment > 0:
                norm_anchor = np.linalg.norm(best_anchor) + 1e-8
                proj = (np.dot(delta_i, best_anchor) / (norm_anchor ** 2)) * best_anchor
                residual = delta_i - proj
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm > self.residual_threshold:
                    mimicry_detected.append(cid)
                    max_alignment = 0.0  # Strip pardoned trust

            ts = max(0.0, max_alignment)
            trust_scores[cid] = ts

        # 4. Filter and Weight Normalization
        accepted = [cid for cid in client_ids if trust_scores[cid] >= self.min_trust_threshold]
        rejected = [cid for cid in client_ids if trust_scores[cid] < self.min_trust_threshold]

        # Fallback if all clients rejected
        if not accepted:
            accepted = [max(trust_scores, key=trust_scores.get)]
            rejected = [cid for cid in client_ids if cid not in accepted]

        total_ts = sum(trust_scores[cid] for cid in accepted)
        if total_ts <= 1e-7:
            weights = {cid: 1.0 / len(accepted) for cid in accepted}
        else:
            weights = {cid: trust_scores[cid] / total_ts for cid in accepted}

        # 5. Compute Weighted Aggregate Delta with Coordinate-wise Trigger Neutralization
        accepted_deltas_stack = np.stack([processed_deltas[cid] for cid in accepted], axis=0)  # (M, D)
        M, D = accepted_deltas_stack.shape

        if self.enable_coordinate_trim and M >= 5:
            # Trim extreme coordinate outliers (e.g. 20% highest and lowest per coordinate)
            k = max(1, int(M * self.trim_fraction))
            sorted_indices = np.argsort(accepted_deltas_stack, axis=0)  # (M, D)
            valid_mask = np.zeros_like(accepted_deltas_stack, dtype=bool)
            np.put_along_axis(valid_mask, sorted_indices[k : M - k], True, axis=0)

            # Trust-weighted average over inlier coordinates
            weights_arr = np.array([weights[cid] for cid in accepted])[:, None]  # (M, 1)
            eff_weights = weights_arr * valid_mask  # (M, D)
            sum_eff = np.sum(eff_weights, axis=0, keepdims=True) + 1e-8
            norm_eff_weights = eff_weights / sum_eff
            weighted_delta = np.sum(norm_eff_weights * accepted_deltas_stack, axis=0)
        else:
            # Bounded deviation from coordinate median fallback
            med_delta = np.median(accepted_deltas_stack, axis=0)
            weighted_delta = np.zeros_like(global_flat)
            for cid in accepted:
                dev = processed_deltas[cid] - med_delta
                clipped_client_delta = med_delta + np.clip(dev, -self.coord_clip_val, self.coord_clip_val)
                weighted_delta += weights[cid] * clipped_client_delta

        # 6. Level-2 Server-Side Aggregate Clipping
        if self.enable_clipping:
            agg_norm = np.linalg.norm(weighted_delta)
            if agg_norm > self.server_clip_norm and agg_norm > 1e-7:
                weighted_delta = weighted_delta * (self.server_clip_norm / agg_norm)

        # 7. Reconstruct State Dict
        aggregated = {}
        offset = 0
        for key in global_weights.keys():
            if np.issubdtype(global_weights[key].dtype, np.floating):
                size = global_weights[key].size
                shape = global_weights[key].shape
                layer_delta = weighted_delta[offset : offset + size].reshape(shape)
                aggregated[key] = (global_weights[key] + layer_delta).astype(np.float32)
                offset += size
            else:
                aggregated[key] = client_updates[accepted[0]][key].copy()

        metadata = {
            "accepted_clients": accepted,
            "rejected_clients": rejected,
            "trust_scores": trust_scores,
            "mimicry_detected": mimicry_detected,
            "num_accepted": len(accepted),
        }
        return aggregated, metadata
