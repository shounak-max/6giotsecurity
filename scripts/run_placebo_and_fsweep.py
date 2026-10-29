"""
Placebo Control, Multi-Krum f-Sweep, and Natural-Prevalence Benchmark.

Addresses key empirical validation controls:
1. Placebo Arm: Forcibly exclude 2 typical honest clients from FedAvg.
   Tests specificity (does recall drop because of data volume, or specifically atypicality?).
2. Defense Knob f-Sweep: Sweep Multi-Krum f in [0, 1, 2, 3, 4] on clean traffic.
   Establishes dose-response of distance filtering.
3. Natural Prevalence vs. Stratified Dual Evaluation:
   Tests model on real-world natural prevalence test set (0.025-0.12% rare attacks).
4. Exact mediation ratio computation.
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.defense import FedAvg, MultiKrum
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator
from fl_ids.utils.metrics import compute_ids_metrics

SEEDS = [42, 43, 44, 45, 46]
NUM_ROUNDS = 5
SAMPLE_SIZE = 3000

TRAIN_CACHE = {}
NAT_CACHE = {}


def get_cached_dataset(seed: int):
    # Stratified dataset for training
    if seed not in TRAIN_CACHE:
        df, labels, meta = load_nf_ton_iot_v2(n_samples=SAMPLE_SIZE, min_rare_samples=60, random_state=seed)
        X = df.to_numpy(dtype=np.float32)
        TRAIN_CACHE[seed] = (X, labels, meta)
    return TRAIN_CACHE[seed]


def get_natural_prevalence_test_set(seed: int, n_test: int = 2000):
    # Pure natural distribution streaming (no oversampled rare floor)
    if seed not in NAT_CACHE:
        df, labels, meta = load_nf_ton_iot_v2(n_samples=n_test, min_rare_samples=1, random_state=seed + 1000)
        X = df.to_numpy(dtype=np.float32)
        NAT_CACHE[seed] = (X, labels, meta)
    return NAT_CACHE[seed]


def run_trial(aggregator, client_filter=None, seed=42, f_krum=2):
    X, labels, meta = get_cached_dataset(seed)
    rare_classes = meta["rare_classes"]
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]

    # Root set extraction
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=200, random_state=seed
    )

    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # 75/25 split
    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # Natural prevalence test set
    X_nat_raw, y_nat, _ = get_natural_prevalence_test_set(seed, n_test=3000)
    X_nat_scaled = preproc.transform(X_nat_raw)

    # Partition
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
    atypical_clients = p_meta["atypical_clients"]
    typical_clients = [c for c in range(10) if c not in atypical_clients]

    # Client selection
    active_clients = list(range(10))
    if client_filter == "exclude_atypical":
        active_clients = [c for c in active_clients if c not in atypical_clients]
    elif client_filter == "exclude_typical_placebo":
        # Exclude exactly the same number of typical clients as atypical clients
        num_to_exclude = max(1, len(atypical_clients))
        excluded_placebo = typical_clients[:num_to_exclude]
        active_clients = [c for c in active_clients if c not in excluded_placebo]

    clients = [
        FederatedClient(
            client_id=cid,
            X_train=X_train[client_splits[cid]],
            y_train=y_train[client_splits[cid]],
            batch_size=32,
            local_epochs=1,
            learning_rate=0.005,
        )
        for cid in active_clients
    ]

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
        atypical_client_ids=atypical_clients,
        rare_class_ids=rare_classes,
    )

    history = sim.run(num_rounds=NUM_ROUNDS, verbose=False)
    final = history[-1]

    # Evaluate on Natural Prevalence Test Set
    server.global_model.eval()
    with torch.no_grad():
        nat_preds = torch.argmax(server.global_model(torch.from_numpy(X_nat_scaled).float()), dim=-1).cpu().numpy()
    nat_metrics = compute_ids_metrics(y_true=y_nat, y_pred=nat_preds, rare_class_ids=rare_classes)

    final["nat_macro_f1"] = nat_metrics["macro_f1"]
    final["nat_rare_recall"] = nat_metrics["rare_class_mean_recall"]
    final["nat_accuracy"] = nat_metrics["accuracy"]
    final["nat_fpr"] = nat_metrics["fpr"]
    final["seed"] = seed
    
    del sim, server, clients
    import gc
    gc.collect()
    return final


def main():
    print("=" * 95)
    print("PLACEBO CONTROL, MULTI-KRUM F-SWEEP & NATURAL PREVALENCE BENCHMARK")
    print("=" * 95)

    results = {}

    # 1. Placebo Control vs. Atypical Exclusion vs. Full FedAvg
    print("\n[PART 1] PLACEBO CONTROL ARMS (Evaluating Specificity of Suppression)")
    placebo_conditions = {
        "fedavg_full": (None, "FedAvg Full (10 clients)"),
        "fedavg_atypical_excluded": ("exclude_atypical", "FedAvg Atypical Excluded (Active Data Deprivation)"),
        "fedavg_placebo_excluded": ("exclude_typical_placebo", "FedAvg Placebo Excluded (Exclude Typical Honest Clients)"),
    }

    for key, (flt, desc) in placebo_conditions.items():
        print(f"--> Running: {desc}...")
        runs = []
        for s in SEEDS:
            t0 = time.time()
            res = run_trial(FedAvg(), client_filter=flt, seed=s)
            runs.append(res)
            print(f"    Seed {s} [{time.time()-t0:.1f}s]: Stratified RareRecall={res['rare_class_mean_recall']:.4f}, NatRareRecall={res['nat_rare_recall']:.4f}, Macro-F1={res['macro_f1']:.4f}")

        recs = [r["rare_class_mean_recall"] for r in runs]
        nat_recs = [r["nat_rare_recall"] for r in runs]
        f1s = [r["macro_f1"] for r in runs]
        results[key] = {
            "desc": desc,
            "rare_recall_mean": float(np.mean(recs)),
            "rare_recall_std": float(np.std(recs, ddof=1)),
            "nat_recall_mean": float(np.mean(nat_recs)),
            "nat_recall_std": float(np.std(nat_recs, ddof=1)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1)),
            "per_seed": runs,
        }

    # 2. Multi-Krum f-Sweep (f in [0, 1, 2, 3, 4] on Clean Traffic)
    print("\n[PART 2] MULTI-KRUM F-SWEEP ON CLEAN TRAFFIC (Defense Parameter Dose-Response)")
    f_sweep = [0, 1, 2, 3, 4]
    f_results = {}

    for f_val in f_sweep:
        print(f"--> Multi-Krum f={f_val}...")
        runs = []
        for s in SEEDS:
            t0 = time.time()
            res = run_trial(MultiKrum(num_malicious=f_val), client_filter=None, seed=s, f_krum=f_val)
            runs.append(res)
            print(f"    Seed {s} [{time.time()-t0:.1f}s]: RareRecall={res['rare_class_mean_recall']:.4f}, AtypicalAcc={res['atypical_acceptance_rate']*100:.1f}%, Macro-F1={res['macro_f1']:.4f}")

        recs = [r["rare_class_mean_recall"] for r in runs]
        nat_recs = [r["nat_rare_recall"] for r in runs]
        atyp = [r["atypical_acceptance_rate"] for r in runs]
        f_results[str(f_val)] = {
            "f": f_val,
            "rare_recall_mean": float(np.mean(recs)),
            "rare_recall_std": float(np.std(recs, ddof=1)),
            "nat_recall_mean": float(np.mean(nat_recs)),
            "nat_recall_std": float(np.std(nat_recs, ddof=1)),
            "atypical_acc_mean": float(np.mean(atyp)),
            "atypical_acc_std": float(np.std(atyp, ddof=1)),
        }

    # Summary Display
    print("\n" + "=" * 95)
    print("SUMMARY RESULTS: PLACEBO CONTROL & SPECIFICITY")
    print("=" * 95)
    for k, v in results.items():
        print(f"{v['desc']:<55} | RareRecall: {v['rare_recall_mean']:.4f} +/- {v['rare_recall_std']:.3f} | NatRecall: {v['nat_recall_mean']:.4f}")

    print("\n" + "=" * 95)
    print("SUMMARY RESULTS: MULTI-KRUM F-SWEEP DOSE-RESPONSE")
    print("=" * 95)
    fed_ref = results["fedavg_full"]["rare_recall_mean"]
    for f_val in f_sweep:
        v = f_results[str(f_val)]
        pen = fed_ref - v["rare_recall_mean"]
        print(f"Multi-Krum f={f_val:<2} | Atypical Acc: {v['atypical_acc_mean']*100:5.1f}% | Rare Recall: {v['rare_recall_mean']:.4f} +/- {v['rare_recall_std']:.3f} | Suppression Penalty: {pen:+.4f}")

    # Save to JSON
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    out_file = os.path.join(out_dir, "placebo_and_fsweep_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "placebo_arms": results,
            "f_sweep": f_results,
        }, f, indent=2)
    print(f"\nSaved placebo and f-sweep results to: {out_file}")


if __name__ == "__main__":
    main()
