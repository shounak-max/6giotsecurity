"""
Step 1: Systematic Investigation of Root-Norm Rescaling in CAT-Q.

Evaluates 3 Root-Norm Strategies across 20 Seeds (3001-3020):
1. 'full' (Current CAT-Q Baseline): Rescales all projected updates to ||g0||:
      delta = (||g0|| / ||delta||) * delta   [Inflates small projections]
2. 'none' (Disabled): No magnitude modification after subspace projection:
      delta = delta                          [Preserves natural update norms]
3. 'bounded' (Shrink-Only Clipping): Caps updates at ||g0|| but never inflates:
      delta = min(1.0, ||g0|| / ||delta||) * delta

Evaluates on:
- NF-ToN-IoT-Binary (where root-norm previously showed massive degradation)
- NF-ToN-IoT-Multiclass (to assess impact on 10-class utility and rare recall)
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np
import torch
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense import ClassAwareTrustDefense
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator

SEEDS = list(range(3001, 3021))
NUM_ROUNDS = 3
N_SAMPLES = 3000
MIN_RARE_SAMPLES = 160


class ConfigurableRootNormDefense(ClassAwareTrustDefense):
    """Subclasses CAT-Q to support flexible root-norm rescaling strategies."""

    def __init__(self, root_norm_mode: str = "full", **kwargs):
        # We will handle root_norm explicitly in our overridden aggregate
        super().__init__(enable_root_norm=False, **kwargs)
        self.root_norm_mode = root_norm_mode  # 'full', 'none', 'bounded'

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        global_weights: Dict[str, np.ndarray],
        client_sample_counts: Dict[int, int] = None,
        server_root_grad: Dict[str, np.ndarray] = None,
        server_root_data: Any = None,
        class_conditional_grads: Dict[int, np.ndarray] = None,
    ):
        from fl_ids.defense.novel_defense import _flatten_weights

        client_ids = list(client_updates.keys())
        if not client_ids:
            return global_weights, {"accepted_clients": [], "rejected_clients": []}

        global_flat = _flatten_weights(global_weights)

        # 1. Level-1 Client-Level Norm Clipping
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

        # 2b. Clean Root Subspace Projection Basis
        Q = None
        if self.enable_subspace_projection and (server_root_grad is not None or (class_conditional_grads is not None and self.enable_class_aware)):
            vectors = []
            if server_root_grad is not None:
                vectors.append(g0 / norm_g0)
            if class_conditional_grads is not None and self.enable_class_aware:
                for cg in class_conditional_grads.values():
                    norm_cg = np.linalg.norm(cg) + 1e-8
                    vectors.append(cg / norm_cg)
            if len(vectors) > 0:
                V = np.stack(vectors, axis=-1)
                Q, _ = np.linalg.qr(V)

        # 3. Projection & Configurable Root-Norm Rescaling
        trust_scores = {}
        processed_deltas = {}

        for cid in client_ids:
            delta_i = clipped_deltas[cid]
            if Q is not None:
                delta_i = Q @ (Q.T @ delta_i)

            norm_i = np.linalg.norm(delta_i) + 1e-8

            # Apply selected root_norm strategy:
            if self.root_norm_mode == "full":
                # Current CAT-Q: force magnitude to match root gradient
                norm_scaled_delta = (norm_g0 / norm_i) * delta_i
            elif self.root_norm_mode == "bounded":
                # Bounded shrink-only: shrink if > norm_g0, never inflate if < norm_g0
                scale = min(1.0, norm_g0 / norm_i)
                norm_scaled_delta = scale * delta_i
            elif self.root_norm_mode == "none":
                # Completely unscaled
                norm_scaled_delta = delta_i
            else:
                raise ValueError(f"Unknown root_norm_mode: {self.root_norm_mode}")

            processed_deltas[cid] = norm_scaled_delta

            # Alignment with server root
            base_cos = float(np.dot(delta_i, g0) / (norm_i * norm_g0))
            max_alignment = base_cos
            best_anchor = g0
            best_is_rare = False

            if self.enable_class_aware and class_conditional_grads is not None and len(class_conditional_grads) > 0:
                for c_id, c_grad in class_conditional_grads.items():
                    norm_cg = np.linalg.norm(c_grad) + 1e-8
                    c_cos = float(np.dot(delta_i, c_grad) / (norm_i * norm_cg))
                    weighted_c_cos = c_cos * self.rare_class_boost
                    if weighted_c_cos > max_alignment:
                        max_alignment = weighted_c_cos
                        best_anchor = c_grad
                        best_is_rare = True

            # Residual check
            if self.enable_residual_check and best_is_rare and max_alignment > 0:
                norm_anchor = np.linalg.norm(best_anchor) + 1e-8
                proj = (np.dot(delta_i, best_anchor) / (norm_anchor ** 2)) * best_anchor
                residual = delta_i - proj
                residual_norm = float(np.linalg.norm(residual))
                if residual_norm > self.residual_threshold:
                    max_alignment = 0.0

            trust = max(0.0, max_alignment)
            trust_scores[cid] = trust

        # 4. Filter & Aggregate
        accepted_cids = [cid for cid, s in trust_scores.items() if s >= self.min_trust_threshold]
        if not accepted_cids:
            fallback_cid = max(trust_scores.keys(), key=lambda k: trust_scores[k])
            accepted_cids = [fallback_cid]

        accepted_matrix = np.stack([processed_deltas[cid] for cid in accepted_cids], axis=0)

        # Coordinate trimming
        if self.enable_coordinate_trim and len(accepted_cids) >= 3:
            k_trim = int(np.floor(len(accepted_cids) * self.trim_fraction))
            if k_trim > 0 and 2 * k_trim < len(accepted_cids):
                sorted_matrix = np.sort(accepted_matrix, axis=0)
                trimmed = sorted_matrix[k_trim : len(accepted_cids) - k_trim, :]
                agg_delta = np.mean(trimmed, axis=0)
            else:
                weights = np.array([trust_scores[cid] for cid in accepted_cids], dtype=np.float32)
                weights = weights / (np.sum(weights) + 1e-8)
                agg_delta = np.sum(accepted_matrix * weights[:, None], axis=0)
        else:
            weights = np.array([trust_scores[cid] for cid in accepted_cids], dtype=np.float32)
            weights = weights / (np.sum(weights) + 1e-8)
            agg_delta = np.sum(accepted_matrix * weights[:, None], axis=0)

        # Server-level clip
        norm_agg = np.linalg.norm(agg_delta)
        if norm_agg > self.server_clip_norm and norm_agg > 1e-7:
            agg_delta = agg_delta * (self.server_clip_norm / norm_agg)

        new_flat = global_flat + agg_delta

        # Reconstruct state dict
        new_weights = {}
        idx = 0
        for k, v in global_weights.items():
            if np.issubdtype(v.dtype, np.floating):
                sz = v.size
                new_weights[k] = new_flat[idx : idx + sz].reshape(v.shape).astype(v.dtype)
                idx += sz
            else:
                new_weights[k] = v.copy()

        return new_weights, {
            "accepted_clients": accepted_cids,
            "trust_scores": {c: float(trust_scores[c]) for c in accepted_cids},
        }


def run_single_experiment(dataset_name: str, seed: int, root_norm_mode: str) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    is_binary = (dataset_name == "NF-ToN-IoT-Binary")
    df, labels, meta = load_nf_ton_iot_v2(
        n_samples=N_SAMPLES,
        min_rare_samples=MIN_RARE_SAMPLES,
        binary=is_binary,
        random_state=seed
    )
    X = df.to_numpy(dtype=np.float32)
    rare_classes = meta.get("rare_classes", [7, 8, 9]) if not is_binary else [1]
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]

    # Preprocessing
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=150, random_state=seed
    )
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes if num_classes > 2 else None)
    atypical_clients = p_meta.get("atypical_clients", [])

    # Attack setup
    target_class = 7 if num_classes > 2 else 0
    malicious_ids = [0, 1]
    attack_obj = AdditiveBackdoorAttack(
        trigger_features=[0, 1, 2],
        trigger_magnitude=4.0,
        target_class=target_class,
        poison_rate=0.5,
        boost_factor=2.0,
    )
    X_trig_test = attack_obj.apply_trigger(X_test)

    # Defense setup
    aggregator = ConfigurableRootNormDefense(
        root_norm_mode=root_norm_mode,
        client_clip_norm=3.0,
        server_clip_norm=3.0,
        min_trust_threshold=0.10,
        rare_class_boost=1.35 if num_classes > 2 else 1.0,
        enable_class_aware=True,
        enable_clipping=True,
        enable_residual_check=True,
        enable_coordinate_trim=True,
        trim_fraction=0.2,
        enable_subspace_projection=True,
    )

    clients = []
    for cid in range(10):
        c_atk = attack_obj if cid in malicious_ids else None
        client = FederatedClient(
            client_id=cid,
            X_train=X_train[client_splits[cid]],
            y_train=y_train[client_splits[cid]],
            batch_size=32,
            local_epochs=1,
            learning_rate=0.005,
            attack=c_atk,
        )
        clients.append(client)

    server = FederatedServer(
        input_dim=input_dim,
        num_classes=num_classes,
        aggregator=aggregator,
        X_root=X_root_scaled,
        y_root=y_root,
    )

    sim = FederatedSimulator(
        server=server,
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        X_triggered_test=X_trig_test,
        atypical_client_ids=atypical_clients,
        rare_class_ids=rare_classes,
        backdoor_target_class=target_class,
    )

    history = sim.run(num_rounds=NUM_ROUNDS, verbose=False)
    final = history[-1]

    res = {
        "seed": seed,
        "dataset": dataset_name,
        "mode": root_norm_mode,
        "accuracy": float(final["accuracy"]),
        "macro_f1": float(final["macro_f1"]),
        "backdoor_asr": float(final.get("backdoor_asr", 0.0)),
        "rare_recall": float(final.get("rare_class_mean_recall", 0.0)),
        "atypical_acc_rate": float(final.get("atypical_acceptance_rate", 1.0)),
    }

    del sim, server, clients
    import gc
    gc.collect()
    return res


def main():
    print("=" * 80)
    print("STEP 1: ROOT-NORM RESCALING INVESTIGATION ACROSS 20 SEEDS (3001-3020)")
    print("=" * 80)
    t0 = time.time()

    MODES = ["full", "none", "bounded"]
    DATASETS = ["NF-ToN-IoT-Binary", "NF-ToN-IoT-Multiclass"]

    results = {}

    for d_name in DATASETS:
        print(f"\n=================================================================")
        print(f"EVALUATING DATASET: {d_name}")
        print(f"=================================================================")
        results[d_name] = {}

        for mode in MODES:
            asrs, accs, recalls, atypicals = [], [], [], []
            print(f"\n>>> Running Root-Norm Mode: '{mode}' on {d_name} across {len(SEEDS)} seeds...")
            
            for s in SEEDS:
                r = run_single_experiment(d_name, s, mode)
                asrs.append(r["backdoor_asr"])
                accs.append(r["accuracy"])
                recalls.append(r["rare_recall"])
                atypicals.append(r["atypical_acc_rate"])
                print(f"  [Seed {s}] ASR: {r['backdoor_asr']:.4f} | Acc: {r['accuracy']*100:.2f}% | Recall: {r['rare_recall']*100:.2f}%")

            res_summary = {
                "asr_mean": float(np.mean(asrs)),
                "asr_std": float(np.std(asrs, ddof=1)),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=1)),
                "recall_mean": float(np.mean(recalls)),
                "recall_std": float(np.std(recalls, ddof=1)),
                "atypical_mean": float(np.mean(atypicals)),
                "atypical_std": float(np.std(atypicals, ddof=1)),
                "raw_asrs": asrs,
                "raw_accs": accs,
                "raw_recalls": recalls,
            }
            results[d_name][mode] = res_summary

            print(f"--- SUMMARY: {d_name} | Mode: {mode} ---")
            print(f"  ASR:         {res_summary['asr_mean']:.4f} +/- {res_summary['asr_std']:.4f}")
            print(f"  Accuracy:    {res_summary['acc_mean']*100:.2f}% +/- {res_summary['acc_std']*100:.2f}%")
            print(f"  Rare Recall: {res_summary['recall_mean']*100:.2f}% +/- {res_summary['recall_std']*100:.2f}%")

    # Statistical comparisons against 'full'
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS (Wilcoxon Signed-Rank Test vs Baseline 'full'):")
    print("=" * 80)
    for d_name in DATASETS:
        print(f"\n--- {d_name} ---")
        base_asr = results[d_name]["full"]["raw_asrs"]
        base_acc = results[d_name]["full"]["raw_accs"]
        for mode in ["none", "bounded"]:
            curr_asr = results[d_name][mode]["raw_asrs"]
            curr_acc = results[d_name][mode]["raw_accs"]

            diff_asr = np.array(curr_asr) - np.array(base_asr)
            diff_acc = np.array(curr_acc) - np.array(base_acc)

            try:
                p_asr = stats.wilcoxon(curr_asr, base_asr).pvalue
            except Exception:
                p_asr = 1.0
            try:
                p_acc = stats.wilcoxon(curr_acc, base_acc).pvalue
            except Exception:
                p_acc = 1.0

            print(f"  Mode '{mode}' vs 'full':")
            print(f"    Delta ASR:      {np.mean(diff_asr):+.4f} (p = {p_asr:.4e})")
            print(f"    Delta Accuracy: {np.mean(diff_acc)*100:+.2f}% (p = {p_acc:.4e})")

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "step1_root_norm_investigation.json")
    with open(out_file, "w") as f:
        # Save without raw lists for clean json
        clean_out = {}
        for d in results:
            clean_out[d] = {}
            for m in results[d]:
                clean_out[d][m] = {k: v for k, v in results[d][m].items() if not k.startswith("raw_")}
        json.dump(clean_out, f, indent=2)

    print(f"\nSaved Step 1 results to: {out_file}")
    print(f"Total execution time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
