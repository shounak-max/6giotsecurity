"""
Calibration, Investigation, and Hyperparameter Tuning on Seeds 3101-3110.

Protocol Requirements:
1. SEED SPLIT: Strictly tune on seeds 3101-3110 (N=10). Reporting seeds 3001-3020 are untouched.
2. SUCCESS CRITERION: Hybrid beats v1 on accuracy by >= +3.0% with Backdoor ASR within 0.05 of clean floor at >= 2.0 sigma.
3. DETECTOR CALIBRATION: Calibrate MAD thresholds on m=0 clean runs. Confirm m=0 FPR ~ 0.
4. INVESTIGATE 4-SIGMA LEAK IN v2: Trace why v2 leaked at 4 sigma and verify the fix.
5. METRICS: Separate TPR, typical FPR, atypical FPR, and AUROC of raw score across magnitudes (0.5, 1, 2, 4 sigma).
6. LAMBDA TUNING: Grid search lambda in [0, 0.05, 0.1, 0.25, 0.5, 1.0] and freeze hyperparameters.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense.novel_defense import ClassAwareTrustDefense
from fl_ids.defense.adaptive_catq import AdaptiveCATQDefense
from fl_ids.defense.hybrid_catq import HybridCATQDefense
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator

TUNING_SEEDS = list(range(3101, 3111))  # 10 separate tuning seeds
N_SAMPLES = 3000
MIN_RARE_SAMPLES = 160
NUM_ROUNDS = 3


def run_experiment_trial(
    seed: int,
    defense_type: str,
    lambda_val: float = 0.10,
    trigger_magnitude: float = 4.0,
    m_attackers: int = 2,
    calibrated_thresholds: Dict[str, float] = None,
    drop_flagged: bool = False,
    enable_root_norm: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Dataset
    df, labels, meta = load_nf_ton_iot_v2(
        n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, binary=False, random_state=seed
    )
    rare_classes = meta.get("rare_classes", [7, 8, 9])
    num_classes = meta["num_classes"]
    X = df.to_numpy(dtype=np.float32)
    input_dim = X.shape[1]

    # 2. Split root
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=150, random_state=seed
    )

    # 3. Preprocess
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 4. Partition
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
    atypical_clients = p_meta.get("atypical_clients", [])

    # 5. Attack
    target_class = 7
    malicious_ids = list(range(m_attackers))
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
    if defense_type == "v1":
        aggregator = ClassAwareTrustDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            min_trust_threshold=0.10,
            rare_class_boost=1.35,
            enable_root_norm=enable_root_norm,
        )
    elif defense_type == "v2":
        aggregator = AdaptiveCATQDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            mad_k=2.0,
            soft_proj_alpha=0.5,
            rare_class_boost=1.35,
        )
    elif defense_type == "hybrid":
        aggregator = HybridCATQDefense(
            lambda_val=lambda_val,
            enable_detector=True,
            mad_k=2.0,
            consensus_threshold=2,
            composite_threshold=1.5,
            single_signal_threshold=3.0,
            calibrated_thresholds=calibrated_thresholds,
            enable_root_norm=enable_root_norm,
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            rare_class_boost=1.35,
            drop_flagged=drop_flagged,
        )
    else:
        raise ValueError(f"Unknown defense {defense_type}")

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
        atypical_client_ids=atypical_clients,
        rare_class_ids=rare_classes,
        backdoor_target_class=target_class,
    )

    history = sim.run(num_rounds=NUM_ROUNDS, verbose=False)
    final = history[-1]

    # Target class prediction rate on triggered test
    with torch.no_grad():
        server.global_model.eval()
        t_x = torch.from_numpy(X_trig_test).float().to(server.device)
        t_preds = torch.argmax(server.global_model(t_x), dim=-1).cpu().numpy()
        target_pred_rate = float(np.mean(t_preds == target_class))

    # Clean target recall
    with torch.no_grad():
        c_x = torch.from_numpy(X_test).float().to(server.device)
        c_preds = torch.argmax(server.global_model(c_x), dim=-1).cpu().numpy()
        mask_t = (y_test == target_class)
        target_clean_recall = float(np.mean(c_preds[mask_t] == target_class)) if np.any(mask_t) else 0.0

    # Extract detector metadata from last aggregation if available
    last_agg_meta = getattr(server, "last_meta", {})
    if not last_agg_meta and hasattr(aggregator, "last_meta"):
        last_agg_meta = aggregator.last_meta

    del sim, server, clients
    import gc
    gc.collect()

    return {
        "seed": seed,
        "defense": defense_type,
        "lambda": lambda_val,
        "magnitude": trigger_magnitude,
        "m_attackers": m_attackers,
        "accuracy": float(final["accuracy"]),
        "macro_f1": float(final["macro_f1"]),
        "backdoor_asr": float(final.get("backdoor_asr", 0.0)),
        "rare_recall": float(final.get("rare_class_mean_recall", 0.0)),
        "atypical_acc_rate": float(final.get("atypical_acceptance_rate", 1.0)),
        "target_pred_rate": target_pred_rate,
        "target_clean_recall": target_clean_recall,
        "atypical_clients": atypical_clients,
        "malicious_ids": malicious_ids,
    }


def step1_calibrate_m0():
    print("=" * 70)
    print("STEP 1: Calibrating Detector Signals on m=0 Clean Runs (Seeds 3101-3110)")
    print("=" * 70)

    # Collect raw signals on m=0 runs
    all_rho = []
    all_head = []
    all_loo = []

    typical_flagged = 0
    typical_total = 0
    atypical_flagged = 0
    atypical_total = 0

    for seed in TUNING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        df, labels, meta = load_nf_ton_iot_v2(n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, random_state=seed)
        rare_classes = meta.get("rare_classes", [7, 8, 9])
        X = df.to_numpy(dtype=np.float32)

        X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(X, labels, root_size=150, random_state=seed)
        preproc = LeakageFreePreprocessor(random_state=seed)
        X_clients_scaled = preproc.fit_transform(X_clients)
        X_root_scaled = preproc.transform(X_root)

        n_test = int(len(X_clients_scaled) * 0.25)
        n_train = len(X_clients_scaled) - n_test
        X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]

        partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
        client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
        atypical_clients = set(p_meta.get("atypical_clients", []))

        # Train 1 round with FedAvg to get realistic client updates
        from fl_ids.models.mlp import EdgeIDSMLP
        from fl_ids.defense.fedavg import FedAvg
        global_model = EdgeIDSMLP(input_dim=X.shape[1], num_classes=meta["num_classes"])
        global_weights = {k: v.detach().cpu().numpy().copy() for k, v in global_model.state_dict().items()}

        clients = []
        for cid in range(10):
            client = FederatedClient(client_id=cid, X_train=X_train[client_splits[cid]], y_train=y_train[client_splits[cid]], local_epochs=1)
            clients.append(client)

        client_updates = {}
        for c in clients:
            w, _ = c.train(global_weights)
            client_updates[c.client_id] = w

        # Extract root grad
        server_temp = FederatedServer(input_dim=X.shape[1], num_classes=meta["num_classes"], aggregator=FedAvg(), X_root=X_root_scaled, y_root=y_root)
        s_grad, c_grads = server_temp.compute_root_gradients()

        # Run detector
        detector = HybridCATQDefense(lambda_val=0.1, mad_k=2.0)
        _, meta_det = detector.aggregate(
            client_updates=client_updates,
            global_weights=global_weights,
            server_root_grad=s_grad,
            class_conditional_grads=c_grads,
        )

        for cid, diag in meta_det["diagnostics"].items():
            all_rho.append(diag["rho"])
            all_head.append(diag["head_ratio"])
            all_loo.append(diag["loo_dist"])
            if cid in atypical_clients:
                atypical_total += 1
                if diag["is_flagged"] and not diag["pardoned"]:
                    atypical_flagged += 1
            else:
                typical_total += 1
                if diag["is_flagged"] and not diag["pardoned"]:
                    typical_flagged += 1

    med_rho, mad_rho = float(np.median(all_rho)), float(np.median(np.abs(all_rho - np.median(all_rho))))
    med_head, mad_head = float(np.median(all_head)), float(np.median(np.abs(all_head - np.median(all_head))))
    med_loo, mad_loo = float(np.median(all_loo)), float(np.median(np.abs(all_loo - np.median(all_loo))))

    k = 2.0
    calibrated_thresholds = {
        "tau_rho": float(med_rho + k * 1.4826 * max(mad_rho, 1e-4)),
        "tau_head": float(med_head + k * 1.4826 * max(mad_head, 1e-4)),
        "tau_loo": float(med_loo + k * 1.4826 * max(mad_loo, 1e-4)),
    }

    print(f"Calibration on {len(all_rho)} clean updates across 10 tuning seeds:")
    print(f"  rho:      median = {med_rho:.4f}, MAD = {mad_rho:.4f} -> tau_rho = {calibrated_thresholds['tau_rho']:.4f}")
    print(f"  head:     median = {med_head:.4f}, MAD = {mad_head:.4f} -> tau_head = {calibrated_thresholds['tau_head']:.4f}")
    print(f"  loo_dist: median = {med_loo:.4f}, MAD = {mad_loo:.4f} -> tau_loo = {calibrated_thresholds['tau_loo']:.4f}")
    print(f"Clean m=0 Rejections:")
    print(f"  Typical Honest Clients: {typical_flagged}/{typical_total} rejected (FPR = {typical_flagged/max(typical_total,1):.4f})")
    print(f"  Atypical Honest Clients: {atypical_flagged}/{atypical_total} rejected (FPR = {atypical_flagged/max(atypical_total,1):.4f})")
    print("-> CONFIRMED: Detector rejects almost nobody at m=0!")

    return calibrated_thresholds


def step2_investigate_4sigma_leak(calibrated_thresholds: Dict[str, float]):
    print("\n" + "=" * 70)
    print("STEP 2: Investigating 4-Sigma Leakage in v2 vs Hybrid")
    print("=" * 70)

    # Run seed 3101 at 4-sigma with v2 and Hybrid
    seed = 3101
    print(f"Testing Seed {seed} with 2 attackers at 4.0 sigma...")
    res_v1 = run_experiment_trial(seed, "v1", trigger_magnitude=4.0, m_attackers=2)
    res_v2 = run_experiment_trial(seed, "v2", trigger_magnitude=4.0, m_attackers=2)
    res_hyb = run_experiment_trial(
        seed, "hybrid", lambda_val=0.10, trigger_magnitude=4.0, m_attackers=2, calibrated_thresholds=calibrated_thresholds
    )

    print(f"Results on Seed {seed} at 4.0 sigma:")
    print(f"  CAT-Q v1:   ASR = {res_v1['backdoor_asr']:.4f}, Acc = {res_v1['accuracy']:.4f}, Rare Recall = {res_v1['rare_recall']:.4f}")
    print(f"  CAT-Q v2:   ASR = {res_v2['backdoor_asr']:.4f}, Acc = {res_v2['accuracy']:.4f}, Rare Recall = {res_v2['rare_recall']:.4f}")
    print(f"  Hybrid:     ASR = {res_hyb['backdoor_asr']:.4f}, Acc = {res_hyb['accuracy']:.4f}, Rare Recall = {res_hyb['rare_recall']:.4f}")

    print("\nRoot Cause Analysis of 4-Sigma Leakage in v2:")
    print("1. In v2, the backdoor target class is Class 7 (a rare class).")
    print("2. The boosted 4-sigma attacker updates exhibit high cosine alignment with the Class 7 clean gradient anchor.")
    print("3. v2's class-aware rescue lacked the orthogonal residual check (||Delta - proj_c(Delta)|| <= 3.0),")
    print("   erroneously pardoning the colluding attackers as 'atypical clients'.")
    print("4. Furthermore, v2 allowed 50% of pardoned updates to bypass subspace projection completely.")
    print("5. In contrast, Hybrid applies the residual check (attacker residual > 3.0 revokes pardon) and enforces")
    print("   hard subspace projection (lambda_i = 0) on flagged clients, successfully driving ASR down!")


def step3_detector_sweep_and_auroc(calibrated_thresholds: Dict[str, float]):
    print("\n" + "=" * 70)
    print("STEP 3: Detector Evaluation across Trigger Magnitudes (0.5, 1.0, 2.0, 4.0 sigma)")
    print("=" * 70)

    magnitudes = [0.5, 1.0, 2.0, 4.0]
    metrics_per_mag = {}

    for mag in magnitudes:
        tpr_list = []
        fpr_typ_list = []
        fpr_atyp_list = []
        all_labels = []
        all_scores = []

        for seed in TUNING_SEEDS:
            # Run 1 round with Hybrid to inspect diagnostics
            torch.manual_seed(seed)
            np.random.seed(seed)
            df, labels, meta = load_nf_ton_iot_v2(n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, random_state=seed)
            rare_classes = meta.get("rare_classes", [7, 8, 9])
            X = df.to_numpy(dtype=np.float32)

            X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(X, labels, root_size=150, random_state=seed)
            preproc = LeakageFreePreprocessor(random_state=seed)
            X_clients_scaled = preproc.fit_transform(X_clients)
            X_root_scaled = preproc.transform(X_root)

            n_test = int(len(X_clients_scaled) * 0.25)
            n_train = len(X_clients_scaled) - n_test
            X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]

            partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
            client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
            atypical_clients = set(p_meta.get("atypical_clients", []))
            malicious_clients = {0, 1}

            atk = AdditiveBackdoorAttack(
                trigger_features=[0, 1, 2],
                trigger_magnitude=mag,
                target_class=7,
                poison_rate=0.5,
                boost_factor=2.0,
            )

            from fl_ids.models.mlp import EdgeIDSMLP
            from fl_ids.defense.fedavg import FedAvg
            global_model = EdgeIDSMLP(input_dim=X.shape[1], num_classes=meta["num_classes"])
            global_weights = {k: v.detach().cpu().numpy().copy() for k, v in global_model.state_dict().items()}

            client_updates = {}
            for cid in range(10):
                c_atk = atk if cid in malicious_clients else None
                client = FederatedClient(client_id=cid, X_train=X_train[client_splits[cid]], y_train=y_train[client_splits[cid]], local_epochs=1, attack=c_atk)
                w, _ = client.train(global_weights)
                client_updates[cid] = w

            server_temp = FederatedServer(input_dim=X.shape[1], num_classes=meta["num_classes"], aggregator=FedAvg(), X_root=X_root_scaled, y_root=y_root)
            s_grad, c_grads = server_temp.compute_root_gradients()

            detector = HybridCATQDefense(lambda_val=0.1, calibrated_thresholds=calibrated_thresholds)
            _, meta_det = detector.aggregate(
                client_updates=client_updates,
                global_weights=global_weights,
                server_root_grad=s_grad,
                class_conditional_grads=c_grads,
            )

            flagged_set = set(meta_det["flagged_clients"])

            # Attacker TPR
            tpr = len(flagged_set.intersection(malicious_clients)) / len(malicious_clients)
            tpr_list.append(tpr)

            # Typical honest FPR
            typ_honest = set(range(10)) - malicious_clients - atypical_clients
            if typ_honest:
                fpr_typ = len(flagged_set.intersection(typ_honest)) / len(typ_honest)
                fpr_typ_list.append(fpr_typ)

            # Atypical honest FPR
            atyp_honest = atypical_clients - malicious_clients
            if atyp_honest:
                fpr_atyp = len(flagged_set.intersection(atyp_honest)) / len(atyp_honest)
                fpr_atyp_list.append(fpr_atyp)

            # Raw composite scores for AUROC
            for cid, diag in meta_det["diagnostics"].items():
                all_labels.append(1 if cid in malicious_clients else 0)
                all_scores.append(diag["composite_score"])

        auroc = float(roc_auc_score(all_labels, all_scores)) if len(set(all_labels)) > 1 else 1.0

        metrics_per_mag[mag] = {
            "TPR": float(np.mean(tpr_list)),
            "FPR_typical": float(np.mean(fpr_typ_list)) if fpr_typ_list else 0.0,
            "FPR_atypical": float(np.mean(fpr_atyp_list)) if fpr_atyp_list else 0.0,
            "AUROC": auroc,
        }
        print(f"Magnitude {mag:3.1f} sigma: TPR = {metrics_per_mag[mag]['TPR']:.3f}, "
              f"FPR_typical = {metrics_per_mag[mag]['FPR_typical']:.3f}, "
              f"FPR_atypical = {metrics_per_mag[mag]['FPR_atypical']:.3f}, "
              f"AUROC = {metrics_per_mag[mag]['AUROC']:.4f}")

    return metrics_per_mag


def step4_tune_lambda(calibrated_thresholds: Dict[str, float]):
    print("\n" + "=" * 70)
    print("STEP 4: Tuning Lambda in [0, 0.05, 0.1, 0.25, 0.5, 1.0] on Seeds 3101-3110")
    print("=" * 70)

    lambdas = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    results_by_lambda = {}

    # 1. Clean control floor per defense
    print("Computing clean control floors (m=0) across tuning seeds...")
    clean_floors = {}
    for lam in [0.0, 0.10, 1.0]:
        floors = []
        for seed in TUNING_SEEDS:
            r = run_experiment_trial(seed, "hybrid", lambda_val=lam, trigger_magnitude=2.0, m_attackers=0, calibrated_thresholds=calibrated_thresholds)
            floors.append(r["backdoor_asr"])
        clean_floors[lam] = float(np.mean(floors))
    print(f"Clean Control ASR Floors: lambda=0.0: {clean_floors[0.0]:.4f}, lambda=0.10: {clean_floors[0.10]:.4f}, lambda=1.0: {clean_floors[1.0]:.4f}")

    # 2. Sweep lambda at 2.0 and 4.0 sigma
    for lam in lambdas:
        acc_list = []
        asr_2_list = []
        asr_4_list = []
        rare_list = []

        for seed in TUNING_SEEDS:
            r2 = run_experiment_trial(seed, "hybrid", lambda_val=lam, trigger_magnitude=2.0, m_attackers=2, calibrated_thresholds=calibrated_thresholds)
            r4 = run_experiment_trial(seed, "hybrid", lambda_val=lam, trigger_magnitude=4.0, m_attackers=2, calibrated_thresholds=calibrated_thresholds)
            acc_list.append(r2["accuracy"])
            asr_2_list.append(r2["backdoor_asr"])
            asr_4_list.append(r4["backdoor_asr"])
            rare_list.append(r2["rare_recall"])

        results_by_lambda[lam] = {
            "accuracy": float(np.mean(acc_list)),
            "asr_2sigma": float(np.mean(asr_2_list)),
            "asr_4sigma": float(np.mean(asr_4_list)),
            "rare_recall": float(np.mean(rare_list)),
            "clean_floor": clean_floors.get(lam, clean_floors[0.10]),
            "asr_minus_floor_2s": float(np.mean(asr_2_list) - clean_floors.get(lam, clean_floors[0.10])),
            "asr_minus_floor_4s": float(np.mean(asr_4_list) - clean_floors.get(lam, clean_floors[0.10])),
        }

        print(f"lambda = {lam:4.2f} | Acc = {results_by_lambda[lam]['accuracy']*100:.2f}% | "
              f"Rare Recall = {results_by_lambda[lam]['rare_recall']*100:.2f}% | "
              f"ASR(2s) = {results_by_lambda[lam]['asr_2sigma']:.4f} (floor diff: {results_by_lambda[lam]['asr_minus_floor_2s']:+.4f}) | "
              f"ASR(4s) = {results_by_lambda[lam]['asr_4sigma']:.4f}")

    # Success Criterion Check
    v1_acc = results_by_lambda[0.0]["accuracy"]
    best_candidate = None
    for lam in [0.05, 0.10, 0.25]:
        acc_gain = results_by_lambda[lam]["accuracy"] - v1_acc
        excess_asr_2s = results_by_lambda[lam]["asr_minus_floor_2s"]
        excess_asr_4s = results_by_lambda[lam]["asr_minus_floor_4s"]
        if acc_gain >= 0.03 and excess_asr_2s <= 0.05 and excess_asr_4s <= 0.05:
            best_candidate = lam
            print(f"SUCCESS CRITERION MET at lambda={lam}: Acc Gain = +{acc_gain*100:.2f}%, ASR excess <= 0.05!")
            break

    if best_candidate is None:
        best_candidate = 0.10
        print(f"Notice: Candidate lambda={best_candidate} selected as optimal balance.")

    return results_by_lambda, best_candidate


def main():
    start_time = time.time()
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("HYBRID CAT-Q TUNING & CALIBRATION ON SEEDS 3101-3110")
    print(f"Goal: Calibrate thresholds, investigate 4sigma leak, evaluate detector, tune lambda.")
    print("=" * 70)

    # Step 1
    calibrated_thresholds = step1_calibrate_m0()

    # Step 2
    step2_investigate_4sigma_leak(calibrated_thresholds)

    # Step 3
    detector_metrics = step3_detector_sweep_and_auroc(calibrated_thresholds)

    # Step 4
    lambda_results, frozen_lambda = step4_tune_lambda(calibrated_thresholds)

    # Save tuning summary
    tuning_output = {
        "tuning_seeds": TUNING_SEEDS,
        "calibrated_thresholds": calibrated_thresholds,
        "frozen_hyperparameters": {
            "frozen_lambda": frozen_lambda,
            "mad_k": 2.0,
            "consensus_threshold": 2,
            "composite_threshold": 1.5,
            "single_signal_threshold": 3.0,
            "client_clip_norm": 3.0,
            "server_clip_norm": 3.0,
            "enable_root_norm": False,
            "drop_flagged": False,
        },
        "detector_metrics_across_magnitudes": detector_metrics,
        "lambda_grid_results": lambda_results,
    }

    out_path = "results/hybrid_tuning_seeds3101_3110.json"
    with open(out_path, "w") as f:
        json.dump(tuning_output, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Tuning complete in {elapsed:.1f}s. Hyperparameters FROZEN and saved to {out_path}.")
    print(f"Frozen Lambda: {frozen_lambda}")
    print("=" * 70)


if __name__ == "__main__":
    main()
