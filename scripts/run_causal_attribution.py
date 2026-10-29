"""
Causal Attribution & Novelty Proof Benchmark on Real NF-ToN-IoT-V2 NetFlow Data.

Implements the 4 Pillars of Formal Causal Evaluation:
1. Powered Sample Size: Real streaming NetFlow flows with robust rare-class floor
   (dozens of test instances per seed, eliminating single-digit denominator noise).
2. Direct Causal Attribution via Do-Calculus Interventions:
   - FedAvg Natural vs. FedAvg Forced-Exclusion of atypical clients.
   - Multi-Krum Natural vs. Multi-Krum Forced-Inclusion of atypical clients.
   - FLTrust Natural vs. FLTrust Forced-Inclusion of atypical clients.
3. Dirichlet Non-IID Dose-Response Curve:
   - Alpha sweep: [0.05, 0.1, 0.3, 1.0, 5.0] showing monotonic penalty decay.
4. Rigorous Statistical & Baseline Reporting:
   - Clean reference = Clean FedAvg (10 clients).
   - Attacked reference = Benign-Only Oracle.
   - Clean-Model Baseline Target Rate (Null ASR).
   - Per-class recall numerators and denominators for all 10 classes.
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
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense import (
    FedAvg,
    TrimmedMean,
    MultiKrum,
    FLTrust,
    ClassAwareTrustDefense,
)
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator
from fl_ids.utils.metrics import compute_backdoor_asr, compute_ids_metrics

SEEDS = [42, 43, 44, 45, 46]
NUM_ROUNDS = 5
SAMPLE_SIZE = 5000  # 5,000 real flows: ~33 test flows per rare class per seed with 150 rare floor
DEFAULT_ALPHA = 0.3

DATA_CACHE = {}


def get_cached_dataset(seed: int, n_samples: int = SAMPLE_SIZE):
    if seed not in DATA_CACHE:
        df, labels, meta = load_nf_ton_iot_v2(n_samples=n_samples, min_rare_samples=150, random_state=seed)
        X = df.to_numpy(dtype=np.float32)
        DATA_CACHE[seed] = (X, labels, meta)
    return DATA_CACHE[seed]


class CausalInterventionAggregator:
    """
    Wraps an underlying aggregator to execute controlled causal interventions:
    - 'forced_exclusion': Forcibly purges atypical honest clients from the accepted set.
    - 'forced_inclusion': Forcibly includes atypical honest clients into the accepted set.
    - 'natural': Standard unperturbed aggregator execution.
    """

    def __init__(self, base_aggregator, intervention_type: str = "natural", target_clients: list = None):
        self.base_aggregator = base_aggregator
        self.intervention_type = intervention_type
        self.target_clients = target_clients if target_clients is not None else []
        self.name = f"{base_aggregator.name}_{intervention_type}"

    def aggregate(self, client_updates, global_weights, **kwargs):
        client_ids = list(client_updates.keys())
        
        if self.intervention_type == "forced_exclusion":
            # Strip out target atypical clients before aggregation
            filtered_updates = {cid: w for cid, w in client_updates.items() if cid not in self.target_clients}
            if not filtered_updates:
                filtered_updates = client_updates
            new_weights, meta = self.base_aggregator.aggregate(
                client_updates=filtered_updates,
                global_weights=global_weights,
                **kwargs
            )
            meta["accepted_clients"] = list(filtered_updates.keys())
            meta["rejected_clients"] = [cid for cid in client_ids if cid not in filtered_updates]
            return new_weights, meta

        # Run base aggregation
        new_weights, meta = self.base_aggregator.aggregate(
            client_updates=client_updates,
            global_weights=global_weights,
            **kwargs
        )

        if self.intervention_type == "forced_inclusion":
            # Forcibly restore target atypical clients into accepted set
            accepted = set(meta.get("accepted_clients", []))
            for tc in self.target_clients:
                if tc in client_updates:
                    accepted.add(tc)
            meta["accepted_clients"] = sorted(list(accepted))
            meta["rejected_clients"] = [cid for cid in client_ids if cid not in accepted]

            # Recompute simple average incorporating forced clients
            flat_global = np.concatenate([v.flatten() for v in global_weights.values() if np.issubdtype(v.dtype, np.floating)])
            acc_deltas = []
            for cid in meta["accepted_clients"]:
                flat_c = np.concatenate([client_updates[cid][k].flatten() for k in global_weights.keys() if np.issubdtype(client_updates[cid][k].dtype, np.floating)])
                acc_deltas.append(flat_c - flat_global)
            avg_delta = np.mean(acc_deltas, axis=0)

            # Reconstruct weights
            reconstructed = {}
            offset = 0
            for k, v in global_weights.items():
                if np.issubdtype(v.dtype, np.floating):
                    size = v.size
                    shape = v.shape
                    reconstructed[k] = (v + avg_delta[offset : offset + size].reshape(shape)).astype(np.float32)
                    offset += size
                else:
                    reconstructed[k] = v.copy()
            new_weights = reconstructed

        return new_weights, meta


def run_experiment_trial(
    aggregator_factory,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
    attack: bool = False,
    is_oracle: bool = False,
):
    # 1. Load real NF-ToN-IoT-V2 data with guaranteed rare floor (from cache)
    X, labels, meta = get_cached_dataset(seed)
    rare_classes = meta["rare_classes"]
    num_classes = meta["num_classes"]
    input_dim = X.shape[1]

    # 2. Extract disjoint server root set (stratified 20 samples per class = 200 flows)
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=200, random_state=seed
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
    num_clients = 8 if is_oracle else 10
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=alpha, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
    atypical_clients = p_meta["atypical_clients"]

    # 5. Attack Setup
    attack_obj = AdditiveBackdoorAttack(
        trigger_features=[0, 1, 2],
        trigger_magnitude=4.0,
        target_class=0,
        poison_rate=0.5,
        boost_factor=2.0,
    )
    X_trig_test = attack_obj.apply_trigger(X_test)
    typical_clients = [c for c in range(num_clients) if c not in atypical_clients]
    malicious_ids = typical_clients[:2] if (attack and not is_oracle) else []

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

    # 7. Server & Aggregator
    aggregator = aggregator_factory(atypical_clients, malicious_ids)
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

    # Evaluate exact per-class recall for all 10 classes
    server.global_model.eval()
    with torch.no_grad():
        preds = torch.argmax(server.global_model(torch.from_numpy(X_test).float()), dim=-1).cpu().numpy()
        trig_preds = torch.argmax(server.global_model(torch.from_numpy(X_trig_test).float()), dim=-1).cpu().numpy()

    # Clean model baseline target rate (even on clean runs)
    null_asr = compute_backdoor_asr(trig_preds, target_class=0, y_true=y_test)

    per_class_recalls = {}
    for c_idx in range(num_classes):
        c_name = meta["class_names"][c_idx]
        denom = int(np.sum(y_test == c_idx))
        num = int(np.sum((y_test == c_idx) & (preds == c_idx)))
        rec = float(num / denom) if denom > 0 else 0.0
        per_class_recalls[c_name] = {"num": num, "denom": denom, "recall": rec}

    final["per_class_recalls"] = per_class_recalls
    final["null_asr"] = null_asr
    final["seed"] = seed
    final["atypical_clients"] = atypical_clients
    final["n_test_samples"] = len(X_test)
    return final


def compute_paired_stats(clean_ref_recalls, condition_recalls):
    """Computes paired difference statistics: diff = clean_ref - condition."""
    diffs = [r - c for r, c in zip(clean_ref_recalls, condition_recalls)]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    se = std_diff / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    ci95 = float(2.776 * se)  # df=4 t-distribution
    # Paired t-statistic
    t_stat = mean_diff / se if se > 1e-8 else 0.0
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95": ci95,
        "t_stat": t_stat,
    }


def main():
    print("=" * 105)
    print("CAUSAL ATTRIBUTION & NOVELTY BENCHMARK SUITE (NF-ToN-IoT-V2)")
    print(f"Dataset: Real NF-ToN-IoT-V2 ({SAMPLE_SIZE} flows, 10 classes) across {len(SEEDS)} seeds")
    print("=" * 105)

    # Define experimental conditions for causal attribution
    conditions = {
        "fedavg_clean_ref": {
            "factory": lambda atyp, mal: FedAvg(),
            "attack": False,
            "is_oracle": False,
            "desc": "FedAvg Clean (True Clean Reference: 10 clients)",
        },
        "fedavg_forced_exclusion": {
            "factory": lambda atyp, mal: CausalInterventionAggregator(FedAvg(), "forced_exclusion", atyp),
            "attack": False,
            "is_oracle": False,
            "desc": "FedAvg + Causal Forced-Exclusion of Atypical Clients",
        },
        "multikrum_natural": {
            "factory": lambda atyp, mal: MultiKrum(num_malicious=len(mal)),
            "attack": False,
            "is_oracle": False,
            "desc": "Multi-Krum Natural (Distance Filtering)",
        },
        "multikrum_forced_inclusion": {
            "factory": lambda atyp, mal: CausalInterventionAggregator(MultiKrum(num_malicious=len(mal)), "forced_inclusion", atyp),
            "attack": False,
            "is_oracle": False,
            "desc": "Multi-Krum + Causal Forced-Inclusion of Atypical Clients",
        },
        "fltrust_natural": {
            "factory": lambda atyp, mal: FLTrust(),
            "attack": False,
            "is_oracle": False,
            "desc": "FLTrust Natural (Cosine Root Filtering)",
        },
        "fltrust_forced_inclusion": {
            "factory": lambda atyp, mal: CausalInterventionAggregator(FLTrust(), "forced_inclusion", atyp),
            "attack": False,
            "is_oracle": False,
            "desc": "FLTrust + Causal Forced-Inclusion of Atypical Clients",
        },
        "oracle_attacked_ref": {
            "factory": lambda atyp, mal: FedAvg(),
            "attack": False,
            "is_oracle": True,
            "desc": "Benign-Only Oracle (True Attacked Reference: 8 honest clients)",
        },
        "fedavg_under_attack": {
            "factory": lambda atyp, mal: FedAvg(),
            "attack": True,
            "is_oracle": False,
            "desc": "FedAvg Under Boosted Backdoor Attack",
        },
        "multikrum_under_attack": {
            "factory": lambda atyp, mal: MultiKrum(num_malicious=len(mal)),
            "attack": True,
            "is_oracle": False,
            "desc": "Multi-Krum Under Boosted Backdoor Attack",
        },
        "class_aware_trust_defense": {
            "factory": lambda atyp, mal: ClassAwareTrustDefense(
                client_clip_norm=3.0,
                server_clip_norm=3.0,
                min_trust_threshold=0.05,
                rare_class_boost=1.3,
                residual_threshold=3.0,
                coord_clip_val=0.35,
            ),
            "attack": True,
            "is_oracle": False,
            "desc": "ClassAwareTrust Defense Under Boosted Backdoor Attack",
        },
    }

    results = {}

    for cond_name, cfg in conditions.items():
        print(f"\n--> Running Condition: {cfg['desc']} ({cond_name})...")
        seed_runs = []
        for s in SEEDS:
            t0 = time.time()
            res = run_experiment_trial(
                aggregator_factory=cfg["factory"],
                seed=s,
                alpha=DEFAULT_ALPHA,
                attack=cfg["attack"],
                is_oracle=cfg["is_oracle"],
            )
            elapsed = time.time() - t0
            seed_runs.append(res)
            print(f"    Seed {s} [{elapsed:.1f}s]: Macro-F1={res['macro_f1']:.4f}, RareRecall={res['rare_class_mean_recall']:.4f}, AtypicalAcc={res['atypical_acceptance_rate']*100:.1f}%, ASR={res['backdoor_asr']:.4f}, NullASR={res['null_asr']:.4f}")

        recalls = [r["rare_class_mean_recall"] for r in seed_runs]
        f1s = [r["macro_f1"] for r in seed_runs]
        atyp = [r["atypical_acceptance_rate"] for r in seed_runs]
        asrs = [r["backdoor_asr"] for r in seed_runs]
        null_asrs = [r["null_asr"] for r in seed_runs]

        results[cond_name] = {
            "desc": cfg["desc"],
            "rare_recall_mean": float(np.mean(recalls)),
            "rare_recall_std": float(np.std(recalls, ddof=1)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1)),
            "atypical_acc_mean": float(np.mean(atyp)),
            "atypical_acc_std": float(np.std(atyp, ddof=1)),
            "asr_mean": float(np.mean(asrs)),
            "asr_std": float(np.std(asrs, ddof=1)),
            "null_asr_mean": float(np.mean(null_asrs)),
            "per_seed_runs": seed_runs,
        }

    # Summary Causal Table
    clean_ref_recalls = [r["rare_class_mean_recall"] for r in results["fedavg_clean_ref"]["per_seed_runs"]]
    
    print("\n" + "=" * 125)
    print(f"CAUSAL ATTRIBUTION RESULTS (Paired vs Clean FedAvg Reference, Rare Recall = {results['fedavg_clean_ref']['rare_recall_mean']:.4f})")
    print("=" * 125)
    print(f"{'Condition':<48} | {'Rare Recall':<16} | {'Atypical Acc%':<14} | {'Suppression Penalty (95% CI)':<30} | {'t-stat'}")
    print("-" * 125)

    for cond_name, res in results.items():
        cond_recalls = [r["rare_class_mean_recall"] for r in res["per_seed_runs"]]
        stats = compute_paired_stats(clean_ref_recalls, cond_recalls)
        diff_str = f"{stats['mean_diff']:+.4f} +/- {stats['ci95']:.4f}"
        rec_str = f"{res['rare_recall_mean']:.4f} +/- {res['rare_recall_std']:.3f}"
        atyp_str = f"{res['atypical_acc_mean']*100:.1f}%"
        print(f"{res['desc']:<48} | {rec_str:<16} | {atyp_str:<14} | {diff_str:<30} | {stats['t_stat']:+.2f}")

    # Section 2: Dirichlet Alpha Sweep (Dose-Response Evidence)
    print("\n" + "=" * 80)
    print("DIRICHLET ALPHA DOSE-RESPONSE SWEEP (Alpha in [0.05, 0.1, 0.3, 1.0, 5.0])")
    print("=" * 80)
    alpha_sweep_results = {}
    alphas = [0.05, 0.1, 0.3, 1.0, 5.0]

    for a in alphas:
        print(f"Evaluating Alpha = {a} across {len(SEEDS)} seeds...")
        fedavg_recs = []
        krum_recs = []
        for s in SEEDS:
            res_fa = run_experiment_trial(lambda atyp, mal: FedAvg(), seed=s, alpha=a, attack=False)
            res_mk = run_experiment_trial(lambda atyp, mal: MultiKrum(num_malicious=2), seed=s, alpha=a, attack=False)
            fedavg_recs.append(res_fa["rare_class_mean_recall"])
            krum_recs.append(res_mk["rare_class_mean_recall"])
        
        penalties = [fa - mk for fa, mk in zip(fedavg_recs, krum_recs)]
        alpha_sweep_results[str(a)] = {
            "alpha": a,
            "fedavg_recall": float(np.mean(fedavg_recs)),
            "krum_recall": float(np.mean(krum_recs)),
            "penalty_mean": float(np.mean(penalties)),
            "penalty_std": float(np.std(penalties, ddof=1)),
        }
        print(f"  Alpha {a:4.2f}: FedAvg={np.mean(fedavg_recs):.4f}, MultiKrum={np.mean(krum_recs):.4f} -> Penalty = {np.mean(penalties):+.4f} +/- {np.std(penalties, ddof=1):.4f}")

    # Save to json
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "causal_attribution_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({
            "causal_conditions": results,
            "alpha_sweep": alpha_sweep_results,
        }, f, indent=2)
    print(f"\nSaved complete causal attribution and alpha sweep benchmark to: {out_path}")


if __name__ == "__main__":
    main()
