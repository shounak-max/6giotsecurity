"""
Comprehensive Multi-Seed Real NetFlow Benchmark & Oracle Reference Evaluation.

Implements all recommendations from Claude's peer review:
1. Evaluated strictly on real NF-ToN-IoT-V2 traffic (13.1M flow source, 10 classes).
2. Multi-seed paired evaluation (n=5 seeds: 42, 43, 44, 45, 46) with Mean +/- Std and 95% CI.
3. Benign-Only Oracle Reference (oracle_clean) to isolate the aggregator-induced suppression penalty.
4. Exact numerators and denominators for rare classes (backdoor, mitm, ransomware).
5. Logging of client-level trust scores for atypical vs. typical clients.
6. Rigorous threat model comparison (Clean, Boosted Backdoor).
"""

import os
import sys
import json
import time
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.canary import NaiveBayesCanary
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense import (
    FedAvg,
    TrimmedMean,
    MultiKrum,
    FLTrust,
    ClassAwareTrustDefense,
)
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator

SEEDS = [42, 43, 44, 45, 46]
NUM_ROUNDS = 5
SAMPLE_SIZE = 4000
ALPHA = 0.3


def get_defense(defense_name: str, num_malicious: int = 2):
    d = defense_name.lower()
    if d in ["fedavg", "oracle_clean"]:
        return FedAvg()
    elif d == "trimmed_mean":
        return TrimmedMean(trim_fraction=0.2)
    elif d == "multikrum":
        return MultiKrum(num_malicious=num_malicious)
    elif d == "foolsgold":
        from fl_ids.defense import FoolsGold
        return FoolsGold()
    elif d == "fltrust":
        return FLTrust()
    elif d in ["class_aware_trust", "proposed"]:
        return ClassAwareTrustDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            min_trust_threshold=0.05,
            rare_class_boost=1.3,
            residual_threshold=3.0,
            coord_clip_val=0.35,
        )
    else:
        raise ValueError(f"Unknown defense: {defense_name}")


def run_trial(defense_name: str, attack_type: str, seed: int = 42):
    # 1. Load real NF-ToN-IoT-V2 data
    df, labels, meta = load_nf_ton_iot_v2(n_samples=SAMPLE_SIZE, random_state=seed)
    X = df.to_numpy(dtype=np.float32)
    rare_classes = meta["rare_classes"]
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]

    # 2. Extract server root set
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=120, random_state=seed
    )

    # 3. Train-only scaling
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # Train / Test split (25% held-out test)
    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 4. Dirichlet non-IID partitioning
    num_clients = 8 if defense_name == "oracle_clean" else 10
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=ALPHA, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
    atypical_clients = p_meta["atypical_clients"]

    # 5. Attack Setup
    attack_obj = None
    X_trig_test = None
    malicious_ids = []
    if attack_type == "backdoor" and defense_name != "oracle_clean":
        attack_obj = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=4.0,
            target_class=0,
            poison_rate=0.5,
            boost_factor=2.0,
        )
        X_trig_test = attack_obj.apply_trigger(X_test)
        malicious_ids = [0, 1]

    # 6. Clients
    clients = []
    for cid in range(num_clients):
        c_atk = attack_obj if cid in malicious_ids else None
        c = FederatedClient(
            client_id=cid,
            X_train=X_train[client_splits[cid]],
            y_train=y_train[client_splits[cid]],
            batch_size=32,
            local_epochs=1,
            learning_rate=0.005,
            attack=c_atk,
        )
        clients.append(c)

    # 7. Server & Simulator
    aggregator = get_defense(defense_name, num_malicious=len(malicious_ids))
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
        backdoor_target_class=0,
    )

    history = sim.run(num_rounds=NUM_ROUNDS, verbose=False)
    final = history[-1]

    # Compute exact rare class per-class recall
    server.global_model.eval()
    import torch
    with torch.no_grad():
        preds = torch.argmax(server.global_model(torch.tensor(X_test)), dim=-1).numpy()

    rare_per_class = {}
    for rc in rare_classes:
        name = meta["class_names"][rc]
        denom = int(np.sum(y_test == rc))
        num = int(np.sum((y_test == rc) & (preds == rc)))
        rec = float(num / denom) if denom > 0 else 0.0
        rare_per_class[name] = {"numerator": num, "denominator": denom, "recall": rec}

    final["rare_per_class"] = rare_per_class
    final["n_test_samples"] = len(X_test)
    final["seed"] = seed
    return final


def compute_ci(data: list) -> float:
    """95% confidence interval half-width using t-distribution approximation."""
    n = len(data)
    if n < 2:
        return 0.0
    se = np.std(data, ddof=1) / np.sqrt(n)
    return float(2.776 * se)  # t_0.025, df=4


def main():
    print("=" * 95)
    print(f"RIGOROUS MULTI-SEED BENCHMARK ON REAL DATASET: NF-ToN-IoT-V2")
    print(f"Config: {len(SEEDS)} seeds ({SEEDS}), {NUM_ROUNDS} rounds, {SAMPLE_SIZE} samples, alpha={ALPHA}")
    print("=" * 95)

    defenses = ["oracle_clean", "fedavg", "multikrum", "trimmed_mean", "fltrust", "class_aware_trust"]
    attacks = ["none", "backdoor"]

    full_results = {}

    for d in defenses:
        for a in attacks:
            if d == "oracle_clean" and a == "backdoor":
                continue  # Oracle is clean-only baseline

            key = f"{d}_{a}"
            print(f"\nEvaluating {key.upper()} across {len(SEEDS)} seeds...")
            seed_metrics = []

            for s in SEEDS:
                t0 = time.time()
                res = run_trial(d, a, seed=s)
                dur = time.time() - t0
                seed_metrics.append(res)
                print(f"  Seed {s} [{dur:.1f}s]: Macro-F1={res['macro_f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, RareRecall={res['rare_class_mean_recall']:.4f}, AtypicalAcc={res['atypical_acceptance_rate']*100:.1f}%, ASR={res['backdoor_asr']:.4f}")

            f1s = [m["macro_f1"] for m in seed_metrics]
            bal_accs = [m["balanced_accuracy"] for m in seed_metrics]
            recalls = [m["rare_class_mean_recall"] for m in seed_metrics]
            atypical = [m["atypical_acceptance_rate"] for m in seed_metrics]
            asrs = [m["backdoor_asr"] for m in seed_metrics]

            full_results[key] = {
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)),
                "macro_f1_ci95": compute_ci(f1s),
                "bal_acc_mean": float(np.mean(bal_accs)),
                "bal_acc_std": float(np.std(bal_accs, ddof=1)),
                "rare_recall_mean": float(np.mean(recalls)),
                "rare_recall_std": float(np.std(recalls, ddof=1)),
                "atypical_acc_mean": float(np.mean(atypical)),
                "atypical_acc_std": float(np.std(atypical, ddof=1)),
                "asr_mean": float(np.mean(asrs)),
                "asr_std": float(np.std(asrs, ddof=1)),
                "per_seed": seed_metrics,
            }

    # Print Summary Table
    print("\n" + "=" * 115)
    print("FINAL BENCHMARK TABLE (NF-ToN-IoT-V2, n=5 Seeds, Mean +/- Std [95% CI])")
    print("=" * 115)
    print(f"{'Defense':<20} | {'Attack':<10} | {'Macro-F1 (CI)':<20} | {'Rare Recall':<15} | {'Atypical Acc%':<15} | {'Backdoor ASR':<15}")
    print("-" * 115)

    for k, v in full_results.items():
        d_name, a_name = k.split("_", 1) if "_" in k else (k, "")
        f1_str = f"{v['macro_f1_mean']:.4f} +/- {v['macro_f1_std']:.3f}"
        rec_str = f"{v['rare_recall_mean']:.4f} +/- {v['rare_recall_std']:.3f}"
        atyp_str = f"{v['atypical_acc_mean']*100:.1f}% +/- {v['atypical_acc_std']*100:.1f}%"
        asr_str = f"{v['asr_mean']:.4f} +/- {v['asr_std']:.3f}"
        print(f"{d_name:<20} | {a_name:<10} | {f1_str:<20} | {rec_str:<15} | {atyp_str:<15} | {asr_str:<15}")

    # Compute Suppression Penalties vs Oracle Reference
    oracle_rec = full_results["oracle_clean_none"]["rare_recall_mean"]
    print("\n" + "=" * 80)
    print(f"SUPPRESSION PENALTY VS CLEAN ORACLE REFERENCE (Oracle Rare Recall = {oracle_rec:.4f})")
    print("=" * 80)
    for d in ["fedavg", "multikrum", "trimmed_mean", "fltrust", "class_aware_trust"]:
        clean_key = f"{d}_none"
        if clean_key in full_results:
            d_rec = full_results[clean_key]["rare_recall_mean"]
            penalty = oracle_rec - d_rec
            print(f"  {d:<22}: Clean Rare Recall = {d_rec:.4f} | Suppression Penalty = {penalty:+.4f}")
    print("=" * 80)

    # Save to file
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "rigorous_benchmark_real_dataset.json")
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved full multi-seed benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
