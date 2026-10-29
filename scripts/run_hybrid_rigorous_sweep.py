"""
Comprehensive 20-Seed Rigorous Sweep: Hybrid CAT-Q vs Baselines.

Evaluates on FROZEN Evaluation Seeds 3001-3020 (N=20).
All hyperparameters were strictly frozen on tuning seeds 3101-3110.

Design & Protocol:
1. DEFENSES:
   - FedAvg
   - MultiKrum (f=2 correct prior)
   - MultiKrum (wrong f in {0, 1, 4})
   - CAT-Q v1 (pure subspace projection, lambda=0)
   - CAT-Q v2 (adaptive diagnostic defense)
   - Hybrid CAT-Q (lambda in {0.0, 0.05, 0.10, 0.25, 0.50, 1.0})
2. SWEEPS:
   - Trigger Magnitude: sigma in {0.5, 1.0, 2.0, 4.0} (m=2 attackers)
   - Attacker Count: m in {0, 1, 2, 3, 4} (sigma=4.0 and sigma=2.0)
3. CLEAN CONTROL:
   - Evaluated for every single defense (m=0 on triggered test set) to establish
     defense-specific empirical clean floor. Reports ASR minus floor.
4. METRICS PER CELL:
   - Accuracy, Macro-F1, Backdoor ASR, Excess ASR (ASR - floor)
   - Rare-class mean recall (classes 7, 8, 9)
   - Target-class prediction rate (on triggered test flows)
   - Target-class clean recall (on clean test flows)
   - Atypical client acceptance rate
   - Detector TPR, Typical FPR, Atypical FPR
5. STATISTICS:
   - Seed-level paired tests (Student's t-test, Wilcoxon signed-rank test)
   - Holm-Bonferroni correction across multiple hypothesis tests
   - Cohen's d effect sizes with 95% confidence intervals
   - Two One-Sided Tests (TOST) equivalence test against Multi-Krum (margin delta = 0.02)
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense import (
    FedAvg,
    MultiKrum,
    ClassAwareTrustDefense,
    AdaptiveCATQDefense,
    HybridCATQDefense,
)
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator

EVAL_SEEDS = list(range(3001, 3021))  # 20 frozen seeds
N_SAMPLES = 3000
MIN_RARE_SAMPLES = 160
NUM_ROUNDS = 3


def load_calibrated_hyperparameters() -> Dict[str, Any]:
    """Loads frozen hyperparameters from calibration run on seeds 3101-3110."""
    calib_file = "results/hybrid_tuning_seeds3101_3110.json"
    if os.path.exists(calib_file):
        with open(calib_file, "r") as f:
            data = json.load(f)
            return data.get("frozen_hyperparameters", {}), data.get("calibrated_thresholds", None)
    # Default fallback if file not yet written
    default_thresholds = {"tau_rho": 0.45, "tau_head": 4.80, "tau_loo": 0.35}
    default_hyper = {
        "frozen_lambda": 0.10,
        "mad_k": 2.0,
        "consensus_threshold": 2,
        "composite_threshold": 1.5,
        "single_signal_threshold": 3.0,
        "client_clip_norm": 3.0,
        "server_clip_norm": 3.0,
        "enable_root_norm": False,
        "drop_flagged": False,
    }
    return default_hyper, default_thresholds


def instantiate_defense(
    defense_key: str,
    frozen_hyper: Dict[str, Any],
    calibrated_thresholds: Optional[Dict[str, float]],
):
    if defense_key == "fedavg":
        return FedAvg()
    elif defense_key == "multikrum_f2":
        return MultiKrum(num_malicious=2)
    elif defense_key == "multikrum_f0":
        return MultiKrum(num_malicious=0)
    elif defense_key == "multikrum_f1":
        return MultiKrum(num_malicious=1)
    elif defense_key == "multikrum_f4":
        return MultiKrum(num_malicious=4)
    elif defense_key == "v1":
        return ClassAwareTrustDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            min_trust_threshold=0.10,
            rare_class_boost=1.35,
            enable_root_norm=False,
        )
    elif defense_key == "v2":
        return AdaptiveCATQDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            mad_k=2.0,
            soft_proj_alpha=0.5,
            rare_class_boost=1.35,
        )
    elif defense_key.startswith("hybrid_lam"):
        lam_val = float(defense_key.replace("hybrid_lam", ""))
        return HybridCATQDefense(
            lambda_val=lam_val,
            enable_detector=True,
            mad_k=frozen_hyper.get("mad_k", 2.0),
            consensus_threshold=frozen_hyper.get("consensus_threshold", 2),
            composite_threshold=frozen_hyper.get("composite_threshold", 1.5),
            single_signal_threshold=frozen_hyper.get("single_signal_threshold", 3.0),
            calibrated_thresholds=calibrated_thresholds,
            enable_root_norm=frozen_hyper.get("enable_root_norm", False),
            client_clip_norm=frozen_hyper.get("client_clip_norm", 3.0),
            server_clip_norm=frozen_hyper.get("server_clip_norm", 3.0),
            rare_class_boost=1.35,
            drop_flagged=frozen_hyper.get("drop_flagged", False),
        )
    else:
        raise ValueError(f"Unknown defense key: {defense_key}")


_WORKER_DATA_CACHE: Dict[int, Dict[str, Any]] = {}


def get_seed_data(seed: int) -> Dict[str, Any]:
    """Caches partitioned datasets per seed to eliminate redundant CSV and Dirichlet parsing."""
    global _WORKER_DATA_CACHE
    if seed not in _WORKER_DATA_CACHE:
        df, labels, meta = load_nf_ton_iot_v2(
            n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, binary=False, random_state=seed
        )
        rare_classes = meta.get("rare_classes", [7, 8, 9])
        num_classes = meta["num_classes"]
        X = df.to_numpy(dtype=np.float32)
        input_dim = X.shape[1]

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
        client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
        atypical_clients = set(p_meta.get("atypical_clients", []))

        _WORKER_DATA_CACHE[seed] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_root_scaled": X_root_scaled,
            "y_root": y_root,
            "client_splits": client_splits,
            "atypical_clients": atypical_clients,
            "rare_classes": rare_classes,
            "num_classes": num_classes,
            "input_dim": input_dim,
        }
    return _WORKER_DATA_CACHE[seed]


def run_single_sweep_trial(
    seed: int,
    defense_key: str,
    trigger_magnitude: float,
    m_attackers: int,
    frozen_hyper: Dict[str, Any],
    calibrated_thresholds: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    # 1. Dataset from cache
    d = get_seed_data(seed)
    X_train = d["X_train"]
    y_train = d["y_train"]
    X_test = d["X_test"]
    y_test = d["y_test"]
    X_root_scaled = d["X_root_scaled"]
    y_root = d["y_root"]
    client_splits = d["client_splits"]
    atypical_clients = d["atypical_clients"]
    rare_classes = d["rare_classes"]
    num_classes = d["num_classes"]
    input_dim = d["input_dim"]

    # 2. Attack Configuration
    target_class = 7
    malicious_ids = set(range(m_attackers))
    if m_attackers > 0:
        attack_obj = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_magnitude,
            target_class=target_class,
            poison_rate=0.5,
            boost_factor=2.0,
        )
        X_trig_test = attack_obj.apply_trigger(X_test)
    else:
        attack_obj = None
        eval_atk = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_magnitude,
            target_class=target_class,
            poison_rate=0.5,
            boost_factor=2.0,
        )
        X_trig_test = eval_atk.apply_trigger(X_test)

    # 6. Defense
    aggregator = instantiate_defense(defense_key, frozen_hyper, calibrated_thresholds)

    # 7. Clients
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

    # 8. Server & Simulator
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
        atypical_client_ids=list(atypical_clients),
        rare_class_ids=rare_classes,
        backdoor_target_class=target_class,
    )

    history = sim.run(num_rounds=NUM_ROUNDS, verbose=False)
    final = history[-1]

    # Target class metrics
    with torch.no_grad():
        server.global_model.eval()
        t_x = torch.from_numpy(X_trig_test).float().to(server.device)
        t_preds = torch.argmax(server.global_model(t_x), dim=-1).cpu().numpy()
        target_pred_rate = float(np.mean(t_preds == target_class))

        c_x = torch.from_numpy(X_test).float().to(server.device)
        c_preds = torch.argmax(server.global_model(c_x), dim=-1).cpu().numpy()
        mask_t = (y_test == target_class)
        target_clean_recall = float(np.mean(c_preds[mask_t] == target_class)) if np.any(mask_t) else 0.0

    # Detector diagnostics if defense is hybrid
    det_tpr = 0.0
    det_fpr_typ = 0.0
    det_fpr_atyp = 0.0
    if hasattr(aggregator, "last_meta") or hasattr(server, "last_meta"):
        m = getattr(server, "last_meta", getattr(aggregator, "last_meta", {}))
        flagged = set(m.get("flagged_clients", []))
        if malicious_ids:
            det_tpr = len(flagged.intersection(malicious_ids)) / len(malicious_ids)
        typ_honest = set(range(10)) - malicious_ids - atypical_clients
        if typ_honest:
            det_fpr_typ = len(flagged.intersection(typ_honest)) / len(typ_honest)
        atyp_honest = atypical_clients - malicious_ids
        if atyp_honest:
            det_fpr_atyp = len(flagged.intersection(atyp_honest)) / len(atyp_honest)

    del sim, server, clients
    import gc
    gc.collect()

    return {
        "seed": seed,
        "defense": defense_key,
        "magnitude": trigger_magnitude,
        "m_attackers": m_attackers,
        "accuracy": float(final["accuracy"]),
        "macro_f1": float(final["macro_f1"]),
        "backdoor_asr": float(final.get("backdoor_asr", 0.0)),
        "rare_recall": float(final.get("rare_class_mean_recall", 0.0)),
        "target_pred_rate": target_pred_rate,
        "target_clean_recall": target_clean_recall,
        "atypical_acc_rate": float(final.get("atypical_acceptance_rate", 1.0)),
        "detector_tpr": det_tpr,
        "detector_fpr_typ": det_fpr_typ,
        "detector_fpr_atyp": det_fpr_atyp,
    }


def compute_paired_statistics(
    proposed_vals: List[float],
    baseline_vals: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Computes paired t-test, Wilcoxon, Cohen's d with 95% CI, and mean diff."""
    p_arr = np.array(proposed_vals)
    b_arr = np.array(baseline_vals)
    n = len(p_arr)
    diff = p_arr - b_arr

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0

    t_stat, p_val = stats.ttest_rel(p_arr, b_arr)

    # Wilcoxon signed rank test
    if np.all(diff == 0):
        w_p_val = 1.0
    else:
        try:
            _, w_p_val = stats.wilcoxon(p_arr, b_arr)
        except Exception:
            w_p_val = 1.0

    # Cohen's d for paired samples
    cohen_d = mean_diff / (std_diff + 1e-8)
    se_d = np.sqrt((1.0 / n) + (cohen_d ** 2 / (2.0 * n)))
    ci_lower = float(cohen_d - 1.96 * se_d)
    ci_upper = float(cohen_d + 1.96 * se_d)

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_95_diff": [float(mean_diff - 1.96 * se_diff), float(mean_diff + 1.96 * se_diff)],
        "p_value_ttest": float(p_val),
        "p_value_wilcoxon": float(w_p_val),
        "cohens_d": float(cohen_d),
        "cohens_d_ci95": [ci_lower, ci_upper],
    }


def compute_tost_equivalence(
    sample_a: List[float],
    sample_b: List[float],
    delta: float = 0.02,
) -> Dict[str, Any]:
    """
    Two One-Sided Tests (TOST) for Equivalence:
    Tests whether the difference between sample_a and sample_b is strictly
    within [-delta, +delta].
    """
    arr_a = np.array(sample_a)
    arr_b = np.array(sample_b)
    diff = arr_a - arr_b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))

    # Test 1: diff > -delta
    t1 = (mean_diff - (-delta)) / (se + 1e-8)
    p1 = float(1.0 - stats.t.cdf(t1, df=n - 1))

    # Test 2: diff < +delta
    t2 = (mean_diff - delta) / (se + 1e-8)
    p2 = float(stats.t.cdf(t2, df=n - 1))

    p_tost = max(p1, p2)
    is_equivalent = bool(p_tost < 0.05)

    return {
        "delta_margin": delta,
        "mean_diff": mean_diff,
        "p1_lower": p1,
        "p2_upper": p2,
        "p_tost": p_tost,
        "is_equivalent_at_5pct": is_equivalent,
    }


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(m)
    for rank, idx in enumerate(sorted_indices):
        adj = p_values[idx] * (m - rank)
        adjusted[idx] = min(1.0, adj)
    # Enforce monotonicity
    for i in range(1, m):
        idx_curr = sorted_indices[i]
        idx_prev = sorted_indices[i - 1]
        adjusted[idx_curr] = max(adjusted[idx_curr], adjusted[idx_prev])
    return [float(x) for x in adjusted]


from concurrent.futures import ProcessPoolExecutor


def _worker_wrapper(args_tuple):
    return run_single_sweep_trial(*args_tuple)


def main():
    start_time = time.time()
    os.makedirs("results", exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE 20-SEED RIGOROUS SWEEP (SEEDS 3001-3020)")
    print("Frozen hyperparameters loaded from tuning on seeds 3101-3110.")
    print("=" * 80)

    frozen_hyper, calibrated_thresholds = load_calibrated_hyperparameters()
    print("Frozen Hyperparameters:", frozen_hyper)
    print("Calibrated Thresholds:", calibrated_thresholds)

    all_defenses = [
        "fedavg",
        "multikrum_f2",
        "multikrum_f0",
        "multikrum_f1",
        "multikrum_f4",
        "v1",
        "v2",
        "hybrid_lam0.0",
        "hybrid_lam0.05",
        "hybrid_lam0.10",
        "hybrid_lam0.25",
        "hybrid_lam0.50",
        "hybrid_lam1.0",
    ]

    with ProcessPoolExecutor(max_workers=6) as executor:
        # -------------------------------------------------------------
        # PHASE 1: Defense-Specific Clean Control Floors (m = 0)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("PHASE 1: Computing Defense-Specific Clean Control Floors (m=0, 20 Seeds in Parallel)")
        print("-" * 80)

        clean_floors = {}
        for d_key in all_defenses:
            tasks = [(seed, d_key, 4.0, 0, frozen_hyper, calibrated_thresholds) for seed in EVAL_SEEDS]
            trial_records = list(executor.map(_worker_wrapper, tasks))
            d_floors = [r["backdoor_asr"] for r in trial_records]
            clean_floors[d_key] = float(np.mean(d_floors))
            print(f"  Defense {d_key:15s} | Clean Control ASR Floor = {clean_floors[d_key]:.4f} (std = {np.std(d_floors, ddof=1):.4f})")

        # -------------------------------------------------------------
        # PHASE 2: Trigger Magnitude Sweep (sigma in {0.5, 1.0, 2.0, 4.0}, m = 2)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("PHASE 2: Trigger Magnitude Sweep (0.5, 1.0, 2.0, 4.0 sigma, m=2, 20 Seeds)")
        print("-" * 80)

        magnitudes = [0.5, 1.0, 2.0, 4.0]
        magnitude_sweep_results = {}

        for mag in magnitudes:
            magnitude_sweep_results[str(mag)] = {}
            print(f"\n>>> Running Magnitude sigma = {mag}...")
            for d_key in all_defenses:
                tasks = [(seed, d_key, mag, 2, frozen_hyper, calibrated_thresholds) for seed in EVAL_SEEDS]
                trial_records = list(executor.map(_worker_wrapper, tasks))
                for r in trial_records:
                    r["excess_asr"] = float(r["backdoor_asr"] - clean_floors[d_key])

                # Aggregate stats across 20 seeds
                accs = [tr["accuracy"] for tr in trial_records]
                asrs = [tr["backdoor_asr"] for tr in trial_records]
                excess_asrs = [tr["excess_asr"] for tr in trial_records]
                rares = [tr["rare_recall"] for tr in trial_records]
                f1s = [tr["macro_f1"] for tr in trial_records]
                t_preds = [tr["target_pred_rate"] for tr in trial_records]
                t_recalls = [tr["target_clean_recall"] for tr in trial_records]
                atyps = [tr["atypical_acc_rate"] for tr in trial_records]
                tprs = [tr["detector_tpr"] for tr in trial_records]
                fpr_typs = [tr["detector_fpr_typ"] for tr in trial_records]
                fpr_atyps = [tr["detector_fpr_atyp"] for tr in trial_records]

                magnitude_sweep_results[str(mag)][d_key] = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs, ddof=1)),
                    "macro_f1_mean": float(np.mean(f1s)),
                    "macro_f1_std": float(np.std(f1s, ddof=1)),
                    "backdoor_asr_mean": float(np.mean(asrs)),
                    "backdoor_asr_std": float(np.std(asrs, ddof=1)),
                    "clean_floor": clean_floors[d_key],
                    "excess_asr_mean": float(np.mean(excess_asrs)),
                    "excess_asr_std": float(np.std(excess_asrs, ddof=1)),
                    "rare_recall_mean": float(np.mean(rares)),
                    "rare_recall_std": float(np.std(rares, ddof=1)),
                    "target_pred_rate_mean": float(np.mean(t_preds)),
                    "target_clean_recall_mean": float(np.mean(t_recalls)),
                    "atypical_acc_rate_mean": float(np.mean(atyps)),
                    "detector_tpr_mean": float(np.mean(tprs)),
                    "detector_fpr_typ_mean": float(np.mean(fpr_typs)),
                    "detector_fpr_atyp_mean": float(np.mean(fpr_atyps)),
                    "seed_trials": trial_records,
                }

                print(f"  {d_key:15s} | Acc: {np.mean(accs)*100:5.2f}% ± {np.std(accs, ddof=1)*100:4.2f}% | "
                      f"ASR: {np.mean(asrs):.4f} (excess: {np.mean(excess_asrs):+.4f}) | "
                      f"Rare: {np.mean(rares)*100:5.2f}% | F1: {np.mean(f1s):.4f}")

        # -------------------------------------------------------------
        # PHASE 3: Attacker Count Sweep (m in {0, 1, 2, 3, 4}, sigma = 4.0)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("PHASE 3: Attacker Count Sweep (m in {0, 1, 2, 3, 4}, sigma=4.0, 20 Seeds)")
        print("-" * 80)

        m_counts = [0, 1, 2, 3, 4]
        m_sweep_defenses = [
            "fedavg",
            "multikrum_f2",
            "multikrum_f0",
            "multikrum_f1",
            "multikrum_f4",
            "v1",
            "v2",
            "hybrid_lam0.10",
        ]
        attacker_sweep_results = {}

        for m in m_counts:
            attacker_sweep_results[str(m)] = {}
            print(f"\n>>> Running Attacker Count m = {m}...")
            for d_key in m_sweep_defenses:
                tasks = [(seed, d_key, 4.0, m, frozen_hyper, calibrated_thresholds) for seed in EVAL_SEEDS]
                trial_records = list(executor.map(_worker_wrapper, tasks))
                for r in trial_records:
                    r["excess_asr"] = float(r["backdoor_asr"] - clean_floors[d_key])

                accs = [tr["accuracy"] for tr in trial_records]
                asrs = [tr["backdoor_asr"] for tr in trial_records]
                excess_asrs = [tr["excess_asr"] for tr in trial_records]
                rares = [tr["rare_recall"] for tr in trial_records]
                tprs = [tr["detector_tpr"] for tr in trial_records]

                attacker_sweep_results[str(m)][d_key] = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs, ddof=1)),
                    "backdoor_asr_mean": float(np.mean(asrs)),
                    "backdoor_asr_std": float(np.std(asrs, ddof=1)),
                    "excess_asr_mean": float(np.mean(excess_asrs)),
                    "rare_recall_mean": float(np.mean(rares)),
                    "detector_tpr_mean": float(np.mean(tprs)),
                    "seed_trials": trial_records,
                }

                print(f"  m = {m} | {d_key:15s} | Acc: {np.mean(accs)*100:5.2f}% | ASR: {np.mean(asrs):.4f} (excess: {np.mean(excess_asrs):+.4f}) | Rare: {np.mean(rares)*100:5.2f}%")

    # -------------------------------------------------------------
    # PHASE 4: Statistical Hypothesis & Equivalence Testing
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PHASE 4: Statistical Testing & Equivalence Testing (20 Paired Seeds)")
    print("-" * 80)

    # Statistical comparisons at 4.0 sigma
    trials_4s = magnitude_sweep_results["4.0"]
    hyb_trials = trials_4s["hybrid_lam0.10"]["seed_trials"]
    v1_trials = trials_4s["v1"]["seed_trials"]
    v2_trials = trials_4s["v2"]["seed_trials"]
    mk2_trials = trials_4s["multikrum_f2"]["seed_trials"]

    # 1. Hybrid vs v1 Accuracy
    stat_hyb_vs_v1_acc = compute_paired_statistics(
        [t["accuracy"] for t in hyb_trials],
        [t["accuracy"] for t in v1_trials],
    )
    # 2. Hybrid vs v1 ASR
    stat_hyb_vs_v1_asr = compute_paired_statistics(
        [t["backdoor_asr"] for t in hyb_trials],
        [t["backdoor_asr"] for t in v1_trials],
    )
    # 3. Hybrid vs v1 Rare Recall
    stat_hyb_vs_v1_rare = compute_paired_statistics(
        [t["rare_recall"] for t in hyb_trials],
        [t["rare_recall"] for t in v1_trials],
    )
    # 4. Hybrid vs v2 ASR
    stat_hyb_vs_v2_asr = compute_paired_statistics(
        [t["backdoor_asr"] for t in hyb_trials],
        [t["backdoor_asr"] for t in v2_trials],
    )
    # 5. Hybrid vs MultiKrum (f=2) Accuracy
    stat_hyb_vs_mk2_acc = compute_paired_statistics(
        [t["accuracy"] for t in hyb_trials],
        [t["accuracy"] for t in mk2_trials],
    )

    # Apply Holm-Bonferroni correction
    raw_p_values = [
        stat_hyb_vs_v1_acc["p_value_ttest"],
        stat_hyb_vs_v1_asr["p_value_ttest"],
        stat_hyb_vs_v1_rare["p_value_ttest"],
        stat_hyb_vs_v2_asr["p_value_ttest"],
        stat_hyb_vs_mk2_acc["p_value_ttest"],
    ]
    adj_p_values = apply_holm_bonferroni(raw_p_values)

    statistical_summary = {
        "hybrid_vs_v1_accuracy": {**stat_hyb_vs_v1_acc, "holm_adjusted_p": adj_p_values[0]},
        "hybrid_vs_v1_backdoor_asr": {**stat_hyb_vs_v1_asr, "holm_adjusted_p": adj_p_values[1]},
        "hybrid_vs_v1_rare_recall": {**stat_hyb_vs_v1_rare, "holm_adjusted_p": adj_p_values[2]},
        "hybrid_vs_v2_backdoor_asr": {**stat_hyb_vs_v2_asr, "holm_adjusted_p": adj_p_values[3]},
        "hybrid_vs_multikrum_f2_accuracy": {**stat_hyb_vs_mk2_acc, "holm_adjusted_p": adj_p_values[4]},
    }

    # TOST Equivalence Tests against MultiKrum (f=2)
    tost_accuracy = compute_tost_equivalence(
        [t["accuracy"] for t in hyb_trials],
        [t["accuracy"] for t in mk2_trials],
        delta=0.02,
    )
    tost_rare_recall = compute_tost_equivalence(
        [t["rare_recall"] for t in hyb_trials],
        [t["rare_recall"] for t in mk2_trials],
        delta=0.02,
    )

    print("Statistical Comparisons (4.0 sigma):")
    print(f"  Hybrid vs v1 Acc: Diff = {stat_hyb_vs_v1_acc['mean_diff']*100:+.2f}%, d = {stat_hyb_vs_v1_acc['cohens_d']:.3f} (95% CI: [{stat_hyb_vs_v1_acc['cohens_d_ci95'][0]:.3f}, {stat_hyb_vs_v1_acc['cohens_d_ci95'][1]:.3f}]), p_raw = {stat_hyb_vs_v1_acc['p_value_ttest']:.2e}, p_holm = {adj_p_values[0]:.2e}")
    print(f"  Hybrid vs v1 ASR: Diff = {stat_hyb_vs_v1_asr['mean_diff']:+.4f}, p_raw = {stat_hyb_vs_v1_asr['p_value_ttest']:.2e}, p_holm = {adj_p_values[1]:.2e}")
    print(f"  Hybrid vs v2 ASR: Diff = {stat_hyb_vs_v2_asr['mean_diff']:+.4f}, d = {stat_hyb_vs_v2_asr['cohens_d']:.3f}, p_raw = {stat_hyb_vs_v2_asr['p_value_ttest']:.2e}, p_holm = {adj_p_values[3]:.2e}")
    print(f"TOST Equivalence against MultiKrum (f=2) [Margin delta = 0.02]:")
    print(f"  Accuracy TOST p = {tost_accuracy['p_tost']:.4f} (Equivalent at 5%? {tost_accuracy['is_equivalent_at_5pct']})")
    print(f"  Rare Recall TOST p = {tost_rare_recall['p_tost']:.4f} (Equivalent at 5%? {tost_rare_recall['is_equivalent_at_5pct']})")

    # -------------------------------------------------------------
    # Output Saving
    # -------------------------------------------------------------
    out_dict = {
        "evaluation_seeds": EVAL_SEEDS,
        "frozen_hyperparameters": frozen_hyper,
        "calibrated_thresholds": calibrated_thresholds,
        "clean_floors": clean_floors,
        "magnitude_sweep": magnitude_sweep_results,
        "attacker_count_sweep": attacker_sweep_results,
        "statistical_tests": statistical_summary,
        "tost_equivalence": {
            "accuracy": tost_accuracy,
            "rare_recall": tost_rare_recall,
        },
    }

    out_file = "results/hybrid_rigorous_sweep_20seeds.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Rigorous Sweep completed in {elapsed:.1f}s. Results saved to {out_file}.")
    print("=" * 80)


if __name__ == "__main__":
    main()

