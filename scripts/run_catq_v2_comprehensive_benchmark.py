"""
Comprehensive Evaluation of CAT-Q v2 (Adaptive Multi-Signal Diagnostic Defense):
1. Head-to-Head Comparison: Original CAT-Q vs CAT-Q v2 across 20 fresh seeds (3001-3020).
   - Measures Backdoor ASR, Benign Accuracy, and Rare-Class Recall recovery.
   - Logs off-subspace energy ratio rho = ||(I - P)Delta|| / ||Delta||.
2. Trigger Magnitude Sweep & Stealth-Constrained Attacker:
   - Evaluates ASR across trigger magnitudes: [0.5, 1.0, 2.0, 4.0].
   - Tests against SubspaceAwareBackdoorAttack (stealth attacker targeting span(Q)).
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
from fl_ids.attacks import AdditiveBackdoorAttack, SubspaceAwareBackdoorAttack
from fl_ids.defense import ClassAwareTrustDefense, AdaptiveCATQDefense
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator

SEEDS = list(range(3001, 3021))  # 20 fresh seeds
NUM_ROUNDS = 3
N_SAMPLES = 3000
MIN_RARE_SAMPLES = 160


def run_trial(
    defense_type: str,  # 'catq_v1' or 'catq_v2'
    attack_type: str,   # 'standard' or 'subspace_aware'
    trigger_mag: float,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Dataset Load (Multiclass 10-Class)
    df, labels, meta = load_nf_ton_iot_v2(
        n_samples=N_SAMPLES,
        min_rare_samples=MIN_RARE_SAMPLES,
        binary=False,
        random_state=seed,
    )
    X = df.to_numpy(dtype=np.float32)
    rare_classes = meta.get("rare_classes", [7, 8, 9])
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]

    # 2. Server Disjoint Root Set
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=150, random_state=seed
    )

    # 3. Preprocessing
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 4. Dirichlet Partitioner (10 clients)
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
    atypical_clients = p_meta.get("atypical_clients", [])

    # 5. Attack Configuration
    target_class = 7
    malicious_ids = [0, 1]

    if attack_type == "standard":
        attack_obj = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_mag,
            target_class=target_class,
            poison_rate=0.5,
            boost_factor=2.0,
        )
    elif attack_type == "subspace_aware":
        attack_obj = SubspaceAwareBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_mag,
            target_class=target_class,
            poison_rate=0.5,
            subspace_alignment_alpha=0.7,
            head_norm_cap=2.0,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_type}")

    X_trig_test = attack_obj.apply_trigger(X_test)

    # 6. Defense Configuration
    if defense_type == "catq_v1":
        aggregator = ClassAwareTrustDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            min_trust_threshold=0.10,
            rare_class_boost=1.35,
            enable_class_aware=True,
            enable_clipping=True,
            enable_root_norm=True,
            enable_residual_check=True,
            enable_coordinate_trim=True,
            trim_fraction=0.2,
            enable_subspace_projection=True,
        )
    elif defense_type == "catq_v2":
        aggregator = AdaptiveCATQDefense(
            mad_k=1.8,
            consensus_threshold=2,
            soft_proj_alpha=0.5,
            rare_class_boost=1.35,
            pardon_threshold=0.35,
            client_clip_norm=3.0,
            server_clip_norm=3.0,
        )
    else:
        raise ValueError(f"Unknown defense: {defense_type}")

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

    # Extract diagnostic metrics if available
    round_meta = getattr(server, "last_meta", {})
    diag = round_meta.get("diagnostics", {})
    clean_rhos = [diag[c]["rho"] for c in range(10) if c not in malicious_ids and c in diag]
    poison_rhos = [diag[c]["rho"] for c in malicious_ids if c in diag]

    res = {
        "seed": seed,
        "defense": defense_type,
        "attack": attack_type,
        "trigger_mag": trigger_mag,
        "accuracy": float(final["accuracy"]),
        "macro_f1": float(final["macro_f1"]),
        "backdoor_asr": float(final.get("backdoor_asr", 0.0)),
        "rare_recall": float(final.get("rare_class_mean_recall", 0.0)),
        "atypical_acc_rate": float(final.get("atypical_acceptance_rate", 1.0)),
        "mean_clean_rho": float(np.mean(clean_rhos)) if clean_rhos else None,
        "mean_poison_rho": float(np.mean(poison_rhos)) if poison_rhos else None,
    }

    del sim, server, clients
    import gc
    gc.collect()
    return res


def main():
    print("=" * 90)
    print("COMPREHENSIVE BENCHMARK: CAT-Q v1 vs CAT-Q v2 & STEALTH ATTACKS ACROSS 20 SEEDS")
    print("=" * 90)
    t0 = time.time()

    results = {
        "head_to_head_20seeds": {},
        "trigger_magnitude_sweep": {},
        "subspace_aware_attack": {},
    }

    # -------------------------------------------------------------------------
    # PART 1: Head-to-Head Comparison (Standard Attack, Mag = 4.0, 20 Seeds)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: HEAD-TO-HEAD COMPARISON (Original CAT-Q vs CAT-Q v2) [20 Seeds]")
    print("=" * 80)

    for def_name in ["catq_v1", "catq_v2"]:
        asrs, accs, recalls, atypicals = [], [], [], []
        clean_rhos, poison_rhos = [], []
        print(f"\nEvaluating: {def_name.upper()}...")
        for s in SEEDS:
            r = run_trial(def_name, "standard", trigger_mag=4.0, seed=s)
            asrs.append(r["backdoor_asr"])
            accs.append(r["accuracy"])
            recalls.append(r["rare_recall"])
            atypicals.append(r["atypical_acc_rate"])
            if r["mean_clean_rho"] is not None:
                clean_rhos.append(r["mean_clean_rho"])
            if r["mean_poison_rho"] is not None:
                poison_rhos.append(r["mean_poison_rho"])
            print(f"  [Seed {s}] ASR: {r['backdoor_asr']:.4f} | Acc: {r['accuracy']*100:.2f}% | Rare Recall: {r['rare_recall']*100:.2f}%")

        res_entry = {
            "asr_mean": float(np.mean(asrs)),
            "asr_std": float(np.std(asrs, ddof=1)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls, ddof=1)),
            "atypical_mean": float(np.mean(atypicals)),
            "atypical_std": float(np.std(atypicals, ddof=1)),
            "clean_rho_mean": float(np.mean(clean_rhos)) if clean_rhos else None,
            "poison_rho_mean": float(np.mean(poison_rhos)) if poison_rhos else None,
            "raw_asrs": asrs,
            "raw_accs": accs,
            "raw_recalls": recalls,
        }
        results["head_to_head_20seeds"][def_name] = res_entry
        print(f"--- SUMMARY {def_name.upper()} ---")
        print(f"  ASR:         {res_entry['asr_mean']:.4f} +/- {res_entry['asr_std']:.4f}")
        print(f"  Accuracy:    {res_entry['acc_mean']*100:.2f}% +/- {res_entry['acc_std']*100:.2f}%")
        print(f"  Rare Recall: {res_entry['recall_mean']*100:.2f}% +/- {res_entry['recall_std']*100:.2f}%")
        if res_entry["clean_rho_mean"] is not None:
            print(f"  Diagnostic Energy (rho): Clean = {res_entry['clean_rho_mean']:.4f} vs Poison = {res_entry['poison_rho_mean']:.4f}")

    # Statistical comparison
    p_asr = stats.wilcoxon(results["head_to_head_20seeds"]["catq_v2"]["raw_asrs"], results["head_to_head_20seeds"]["catq_v1"]["raw_asrs"]).pvalue
    p_acc = stats.wilcoxon(results["head_to_head_20seeds"]["catq_v2"]["raw_accs"], results["head_to_head_20seeds"]["catq_v1"]["raw_accs"]).pvalue
    p_rec = stats.wilcoxon(results["head_to_head_20seeds"]["catq_v2"]["raw_recalls"], results["head_to_head_20seeds"]["catq_v1"]["raw_recalls"]).pvalue
    delta_acc = results["head_to_head_20seeds"]["catq_v2"]["acc_mean"] - results["head_to_head_20seeds"]["catq_v1"]["acc_mean"]
    delta_rec = results["head_to_head_20seeds"]["catq_v2"]["recall_mean"] - results["head_to_head_20seeds"]["catq_v1"]["recall_mean"]
    delta_asr = results["head_to_head_20seeds"]["catq_v2"]["asr_mean"] - results["head_to_head_20seeds"]["catq_v1"]["asr_mean"]

    print("\n--- STATISTICAL SIGNIFICANCE (CAT-Q v2 vs CAT-Q v1 across 20 Seeds) ---")
    print(f"  Delta Accuracy:    {delta_acc*100:+.2f}% (p = {p_acc:.4e})")
    print(f"  Delta Rare Recall: {delta_rec*100:+.2f}% (p = {p_rec:.4e})")
    print(f"  Delta Backdoor ASR: {delta_asr:+.4f} (p = {p_asr:.4e})")

    # -------------------------------------------------------------------------
    # PART 2: Trigger Magnitude Sweep (0.5, 1.0, 2.0, 4.0)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: TRIGGER MAGNITUDE SWEEP (Tusar & Sazid Protocol, 20 Seeds)")
    print("=" * 80)
    # Mag 4.0 is already computed in Part 1!
    results["trigger_magnitude_sweep"]["4.0"] = {
        "catq_v1": {
            "asr_mean": results["head_to_head_20seeds"]["catq_v1"]["asr_mean"],
            "asr_std": results["head_to_head_20seeds"]["catq_v1"]["asr_std"],
            "acc_mean": results["head_to_head_20seeds"]["catq_v1"]["acc_mean"],
            "acc_std": results["head_to_head_20seeds"]["catq_v1"]["acc_std"],
        },
        "catq_v2": {
            "asr_mean": results["head_to_head_20seeds"]["catq_v2"]["asr_mean"],
            "asr_std": results["head_to_head_20seeds"]["catq_v2"]["asr_std"],
            "acc_mean": results["head_to_head_20seeds"]["catq_v2"]["acc_mean"],
            "acc_std": results["head_to_head_20seeds"]["catq_v2"]["acc_std"],
        },
    }
    print(f"  [Mag = 4.0 | Reused from Part 1]")

    SWEEP_MAGS = [0.5, 1.0, 2.0]
    for mag in SWEEP_MAGS:
        results["trigger_magnitude_sweep"][str(mag)] = {}
        for def_name in ["catq_v1", "catq_v2"]:
            m_asrs, m_accs = [], []
            for s in SEEDS:
                r = run_trial(def_name, "standard", trigger_mag=mag, seed=s)
                m_asrs.append(r["backdoor_asr"])
                m_accs.append(r["accuracy"])
            results["trigger_magnitude_sweep"][str(mag)][def_name] = {
                "asr_mean": float(np.mean(m_asrs)),
                "asr_std": float(np.std(m_asrs, ddof=1)),
                "acc_mean": float(np.mean(m_accs)),
                "acc_std": float(np.std(m_accs, ddof=1)),
            }
            print(f"  [Mag = {mag} | {def_name.upper()}] ASR = {np.mean(m_asrs):.4f} +/- {np.std(m_asrs, ddof=1):.4f} | Acc = {np.mean(m_accs)*100:.2f}%")


    # -------------------------------------------------------------------------
    # PART 3: Stealth-Constrained Subspace-Aware Attacker
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: STEALTH-CONSTRAINED ATTACKER (Subspace-Targeting Evasion)")
    print("=" * 80)

    for def_name in ["catq_v1", "catq_v2"]:
        adv_asrs, adv_accs = [], []
        for s in SEEDS:
            r = run_trial(def_name, "subspace_aware", trigger_mag=2.0, seed=s)
            adv_asrs.append(r["backdoor_asr"])
            adv_accs.append(r["accuracy"])
        results["subspace_aware_attack"][def_name] = {
            "asr_mean": float(np.mean(adv_asrs)),
            "asr_std": float(np.std(adv_asrs, ddof=1)),
            "acc_mean": float(np.mean(adv_accs)),
            "acc_std": float(np.std(adv_accs, ddof=1)),
        }
        print(f"  [Adaptive Attack | {def_name.upper()}] ASR = {np.mean(adv_asrs):.4f} +/- {np.std(adv_asrs, ddof=1):.4f} | Acc = {np.mean(adv_accs)*100:.2f}%")

    out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "catq_v2_comprehensive_benchmark.json")
    with open(out_file, "w") as f:
        clean_out = {}
        for k, v in results.items():
            clean_out[k] = {}
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict) and "raw_asrs" in sub_v:
                    clean_out[k][sub_k] = {ik: iv for ik, iv in sub_v.items() if not ik.startswith("raw_")}
                else:
                    clean_out[k][sub_k] = sub_v
        json.dump(clean_out, f, indent=2)

    print(f"\n=======================================================")
    print(f"ALL BENCHMARKS COMPLETED IN {time.time()-t0:.1f}s.")
    print(f"Saved complete results to: {out_file}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
