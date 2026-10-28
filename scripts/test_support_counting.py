"""
Direct Support-Counting Test: k vs f on Real NF-ToN-IoT-V2 NetFlow Data.

Tests the fundamental theoretical proposition:
When rare-class samples are held by k clients in a federation with an aggregator
tolerating f Byzantine adversaries:
- If k <= f: The aggregator cannot distinguish honest atypical clients from adversaries,
  causing severe rare-class recall suppression.
- If k > f: Support counting exceeds f, consensus is established, and recall recovers.

We hold f=2 fixed and evaluate k in {1, 2, 3, 5} across 5 seeds.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.defense import FedAvg, MultiKrum
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator
from fl_ids.utils.metrics import compute_ids_metrics

SEEDS = [42, 43, 44, 45, 46]
NUM_ROUNDS = 5
SAMPLE_SIZE = 4000
F_FIXED = 2


def run_support_counting_trial(k_holders: int, defense_name: str, seed: int = 42):
    # 1. Load real NF-ToN-IoT-V2 data with guaranteed rare quota
    df, labels, meta = load_nf_ton_iot_v2(n_samples=SAMPLE_SIZE, min_rare_samples=100, random_state=seed)
    X = df.to_numpy(dtype=np.float32)
    rare_classes = meta["rare_classes"]
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]

    # 2. Extract disjoint root set
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=200, random_state=seed
    )

    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # 75/25 train/test split
    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 3. Controlled Support-Counting Partition:
    # Common classes (0, 2, 3, 4, 6, 8, 9) are partitioned among all 10 clients using Dirichlet(alpha=0.3).
    # Rare classes (1, 5, 7) are strictly distributed to exactly k designated holder clients.
    rng = np.random.RandomState(seed)
    num_clients = 10
    client_indices = {i: [] for i in range(num_clients)}

    # Designated holder clients for rare classes
    holders = list(range(k_holders))  # e.g. for k=1: client 0; for k=2: clients 0, 1; for k=3: clients 0, 1, 2

    for c in range(num_classes):
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)
        if c in rare_classes:
            # Distribute evenly among the k holder clients ONLY
            split_rare = np.array_split(idx_c, k_holders)
            for h_idx, h_cid in enumerate(holders):
                client_indices[h_cid].extend(split_rare[h_idx].tolist())
        else:
            # Common classes distributed via Dirichlet among all 10 clients
            props = rng.dirichlet(np.repeat(0.3, num_clients))
            counts = (props * len(idx_c)).astype(int)
            diff = len(idx_c) - counts.sum()
            for _ in range(abs(diff)):
                if diff > 0:
                    counts[rng.choice(num_clients)] += 1
                elif diff < 0 and counts.max() > 1:
                    counts[np.argmax(counts)] -= 1
            split_common = np.split(idx_c, np.cumsum(counts)[:-1])
            for cid in range(num_clients):
                client_indices[cid].extend(split_common[cid].tolist())

    clients = [
        FederatedClient(
            client_id=cid,
            X_train=X_train[np.array(client_indices[cid], dtype=np.int64)],
            y_train=y_train[np.array(client_indices[cid], dtype=np.int64)],
            batch_size=32,
            local_epochs=1,
            learning_rate=0.005,
        )
        for cid in range(num_clients)
    ]

    # Aggregator: Multi-Krum always drops f=2 clients (even on clean traffic)
    if defense_name == "multikrum":
        aggregator = MultiKrum(num_malicious=F_FIXED)
    else:
        aggregator = FedAvg()

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
        atypical_client_ids=holders,
        rare_class_ids=rare_classes,
    )

    history = sim.run(num_rounds=NUM_ROUNDS, verbose=False)
    final = history[-1]
    final["k_holders"] = k_holders
    final["holders"] = holders
    final["seed"] = seed
    return final


def main():
    print("=" * 85)
    print("SUPPORT-COUNTING PROPOSITION TEST: k vs f (f=2 fixed, k in {1, 2, 3, 5})")
    print("=" * 85)

    k_values = [1, 2, 3, 5]
    defenses = ["fedavg", "multikrum"]
    summary = {}

    for k in k_values:
        summary[str(k)] = {}
        print(f"\n---> Testing k = {k} Holder Clients (f = {F_FIXED} fixed)...")
        for d in defenses:
            recs = []
            atyp_accs = []
            f1s = []
            for s in SEEDS:
                t0 = time.time()
                res = run_support_counting_trial(k_holders=k, defense_name=d, seed=s)
                recs.append(res["rare_class_mean_recall"])
                atyp_accs.append(res["atypical_acceptance_rate"])
                f1s.append(res["macro_f1"])
                print(f"  {d:<10} | Seed {s} [{time.time()-t0:.1f}s]: RareRecall={res['rare_class_mean_recall']:.4f}, HolderAcc={res['atypical_acceptance_rate']*100:.1f}%, Macro-F1={res['macro_f1']:.4f}")

            rec_mean = float(np.mean(recs))
            rec_std = float(np.std(recs, ddof=1))
            acc_mean = float(np.mean(atyp_accs))
            summary[str(k)][d] = {
                "recall_mean": rec_mean,
                "recall_std": rec_std,
                "holder_acc_mean": acc_mean,
            }

    # Print Proposition Comparison Table
    print("\n" + "=" * 95)
    print("SUPPORT-COUNTING THEORETICAL THRESHOLD RESULTS (f=2 Byzantine Tolerance Fixed)")
    print("=" * 95)
    print(f"{'k (Holders)':<12} | {'Condition':<15} | {'FedAvg Recall':<18} | {'Multi-Krum Recall':<18} | {'Multi-Krum Holder Acc%':<22} | {'Suppression'}")
    print("-" * 95)

    for k in k_values:
        fa = summary[str(k)]["fedavg"]
        mk = summary[str(k)]["multikrum"]
        cond = "k <= f (Under-Supp)" if k <= F_FIXED else "k > f (Consensus)"
        penalty = fa["recall_mean"] - mk["recall_mean"]
        fa_str = f"{fa['recall_mean']:.4f} +/- {fa['recall_std']:.3f}"
        mk_str = f"{mk['recall_mean']:.4f} +/- {mk['recall_std']:.3f}"
        acc_str = f"{mk['holder_acc_mean']*100:.1f}%"
        print(f"k = {k:<8} | {cond:<15} | {fa_str:<18} | {mk_str:<18} | {acc_str:<22} | {penalty:+.4f}")

    print("=" * 95)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "support_counting_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved support-counting results to: {out_file}")


if __name__ == "__main__":
    main()
