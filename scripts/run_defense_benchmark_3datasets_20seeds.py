"""
Proposed Defense Benchmark Suite: 3 Datasets across 20 Fresh Seeds (3001-3020).

Evaluates the Proposed Defense (ClassAwareTrustDefense) with Q-Projection:
1. Backdoor ASR and utility evaluation on 3 datasets:
   - Dataset 1: NF-ToN-IoT-V2 Multiclass (10 classes, 41 features, ~56 rare test samples across classes)
   - Dataset 2: NF-ToN-IoT-V2 Binary (2 classes, 41 features, ~588 attack test samples)
   - Dataset 3: Synthetic-NetFlow-V2 (10 classes, 40 features)
2. Baselines: FedAvg, MultiKrum (f=2), TrimmedMean, FoolsGold, FLTrust, RepGuard.
3. 4-Arm Component Ablations:
   - w/o Subspace Projection
   - w/o Class-Aware Alignment
   - w/o Adaptive Clipping
   - w/o Root Norm
4. Statistics: Seed-level paired tests (Student's t-test, Wilcoxon signed-rank test, ddof=1 sample std).
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.synthetic import generate_synthetic_netflow, RARE_CLASSES
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack, LabelFlipAttack, AtypicalMimicryAttack
from fl_ids.defense import (
    FedAvg,
    TrimmedMean,
    MultiKrum,
    FoolsGold,
    FLTrust,
    RepGuard,
    ClassAwareTrustDefense,
)
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator

SEEDS = list(range(3001, 3021))  # 20 fresh seeds
NUM_ROUNDS = 3
N_SAMPLES = 3000
MIN_RARE_SAMPLES = 160  # Stratified floor ensuring rare class presence in train and test splits


def load_dataset_for_benchmark(dataset_name: str, seed: int):
    """Loads one of the 3 benchmark datasets."""
    if dataset_name == "NF-ToN-IoT-Multiclass":
        df, labels, meta = load_nf_ton_iot_v2(n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, binary=False, random_state=seed)
        rare_classes = meta.get("rare_classes", [7, 8, 9])
        num_classes = meta["num_classes"]
    elif dataset_name == "NF-ToN-IoT-Binary":
        df, labels, meta = load_nf_ton_iot_v2(n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, binary=True, random_state=seed)
        rare_classes = [1]  # Attack is rare/minority relative to standard flows
        num_classes = 2
    elif dataset_name == "Synthetic-NetFlow-Multiclass":
        df, labels, meta = generate_synthetic_netflow(n_samples=N_SAMPLES, n_features=40, random_state=seed)
        rare_classes = RARE_CLASSES
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = df.to_numpy(dtype=np.float32)
    return X, labels, rare_classes, num_classes


def run_single_trial(
    defense_name: str,
    attack_name: str,
    dataset_name: str,
    seed: int,
    ablation_kwargs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if ablation_kwargs is None:
        ablation_kwargs = {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Dataset Load
    X, labels, rare_classes, num_classes = load_dataset_for_benchmark(dataset_name, seed)
    input_dim = X.shape[1]

    # 2. Server Disjoint Root Set
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=150, random_state=seed
    )

    # 3. Leak-free Preprocessing
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # 25% test split
    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 4. Dirichlet Partitioner (10 clients)
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes if num_classes > 2 else None)
    atypical_clients = p_meta.get("atypical_clients", [])

    # 5. Attack Configuration
    attack_obj = None
    X_trig_test = None
    target_class = 7 if num_classes > 2 else 0
    malicious_ids = [0, 1] if attack_name not in ["none", "clean_control"] else []

    if attack_name == "backdoor":
        attack_obj = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=4.0,
            target_class=target_class,
            poison_rate=0.5,
            boost_factor=2.0,
        )
        X_trig_test = attack_obj.apply_trigger(X_test)
    elif attack_name == "clean_control":
        # Evaluates clean unpoisoned model on triggered test set to isolate false trigger baseline
        attack_obj = None
        eval_atk = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=4.0,
            target_class=target_class,
            poison_rate=0.5,
            boost_factor=2.0,
        )
        X_trig_test = eval_atk.apply_trigger(X_test)
    elif attack_name == "label_flip":
        attack_obj = LabelFlipAttack(source_class=1, target_class=0)
    elif attack_name == "mimicry":
        attack_obj = AtypicalMimicryAttack(
            atypical_class_id=rare_classes[0] if rare_classes else 1,
            trigger_features=[0, 1, 2],
            trigger_magnitude=4.0,
            boost_factor=2.0,
        )
        X_trig_test = attack_obj.apply_trigger(X_test)

    # 6. Defense Configuration
    if defense_name == "fedavg":
        aggregator = FedAvg()
    elif defense_name == "multikrum":
        aggregator = MultiKrum(num_malicious=2)
    elif defense_name == "trimmed_mean":
        aggregator = TrimmedMean(trim_fraction=0.2)
    elif defense_name == "foolsgold":
        aggregator = FoolsGold()
    elif defense_name == "fltrust":
        aggregator = FLTrust()
    elif defense_name == "repguard":
        aggregator = RepGuard()
    elif defense_name == "proposed":
        aggregator = ClassAwareTrustDefense(
            client_clip_norm=3.0,
            server_clip_norm=3.0,
            min_trust_threshold=0.10,
            rare_class_boost=1.35 if num_classes > 2 else 1.0,
            enable_class_aware=ablation_kwargs.get("enable_class_aware", True),
            enable_clipping=ablation_kwargs.get("enable_clipping", True),
            enable_root_norm=ablation_kwargs.get("enable_root_norm", True),
            enable_residual_check=ablation_kwargs.get("enable_residual_check", True),
            enable_coordinate_trim=True,
            trim_fraction=0.2,
            enable_subspace_projection=ablation_kwargs.get("enable_subspace_projection", True),
        )
    else:
        raise ValueError(f"Unknown defense: {defense_name}")

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

    # Calculate test samples for rare class
    rare_test_count = int(np.sum(np.isin(y_test, rare_classes)))

    res = {
        "seed": seed,
        "dataset": dataset_name,
        "defense": defense_name,
        "accuracy": float(final["accuracy"]),
        "macro_f1": float(final["macro_f1"]),
        "backdoor_asr": float(final.get("backdoor_asr", 0.0)),
        "rare_recall": float(final.get("rare_class_mean_recall", 0.0)),
        "atypical_acc_rate": float(final.get("atypical_acceptance_rate", 1.0)),
        "rare_test_samples": rare_test_count,
    }

    del sim, server, clients
    import gc
    gc.collect()
    return res


def compute_seed_level_comparison(proposed_vals: List[float], baseline_vals: List[float]) -> Dict[str, Any]:
    """Computes paired t-test, Wilcoxon signed-rank test, mean difference, and 95% CI."""
    p_arr = np.array(proposed_vals)
    b_arr = np.array(baseline_vals)
    diff = p_arr - b_arr
    mean_diff = float(np.mean(diff))
    n = len(diff)
    se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    # Paired t-test
    if np.all(diff == 0):
        t_stat, p_val_t = 0.0, 1.0
        wilc_stat, p_val_wilc = 0.0, 1.0
    else:
        t_stat, p_val_t = stats.ttest_rel(p_arr, b_arr)
        try:
            wilc_stat, p_val_wilc = stats.wilcoxon(p_arr, b_arr)
        except Exception:
            wilc_stat, p_val_wilc = 0.0, 1.0

    ci = (float(mean_diff - 1.96 * se), float(mean_diff + 1.96 * se))
    return {
        "mean_diff": mean_diff,
        "95_ci": ci,
        "t_stat": float(t_stat),
        "p_val_ttest": float(p_val_t),
        "p_val_wilcoxon": float(p_val_wilc),
    }


def main():
    print("=" * 95)
    print("PROPOSED DEFENSE BENCHMARK: 3 DATASETS across 20 FRESH SEEDS (3001-3020)")
    print("=" * 95)
    t0 = time.time()

    DATASETS = [
        "NF-ToN-IoT-Multiclass",
        "NF-ToN-IoT-Binary",
        "Synthetic-NetFlow-Multiclass",
    ]

    BASELINES = [
        "fedavg",
        "multikrum",
        "trimmed_mean",
        "foolsgold",
        "fltrust",
        "repguard",
    ]

    full_results = {
        "seeds": SEEDS,
        "datasets": {},
    }

    for d_name in DATASETS:
        print(f"\n=======================================================")
        print(f"EVALUATING DATASET: {d_name}")
        print(f"=======================================================")
        d_dict = {
            "backdoor_attack": {},
            "ablations": {},
            "seed_level_statistics": {},
        }

        # 1. Backdoor Attack: Proposed vs Baselines
        print(f"\n--- Backdoor Attack (20 Seeds) on {d_name} ---")
        
        # Clean Control (FedAvg on clean clients evaluated on triggered test set)
        clean_asrs = []
        for s in SEEDS:
            rc = run_single_trial("fedavg", "clean_control", d_name, s)
            clean_asrs.append(rc["backdoor_asr"])
        d_dict["clean_trigger_control"] = {
            "clean_asr_mean": float(np.mean(clean_asrs)),
            "clean_asr_std": float(np.std(clean_asrs, ddof=1)) if len(clean_asrs) > 1 else 0.0,
        }
        print(f"  Clean Control (Unpoisoned FedAvg on Triggered Test): ASR = {np.mean(clean_asrs):.4f}")

        # Proposed Defense
        prop_asrs, prop_accs, prop_recalls, prop_atypicals = [], [], [], []
        rare_test_samples_list = []
        for s in SEEDS:
            r = run_single_trial("proposed", "backdoor", d_name, s)
            prop_asrs.append(r["backdoor_asr"])
            prop_accs.append(r["accuracy"])
            prop_recalls.append(r["rare_recall"])
            prop_atypicals.append(r["atypical_acc_rate"])
            rare_test_samples_list.append(r["rare_test_samples"])

        asr_m = float(np.mean(prop_asrs))
        asr_s = float(np.std(prop_asrs, ddof=1))
        d_dict["backdoor_attack"]["proposed"] = {
            "asr_mean": asr_m,
            "asr_std": asr_s,
            "accuracy_mean": float(np.mean(prop_accs)),
            "accuracy_std": float(np.std(prop_accs, ddof=1)),
            "rare_recall_mean": float(np.mean(prop_recalls)),
            "rare_recall_std": float(np.std(prop_recalls, ddof=1)),
            "atypical_acc_rate_mean": float(np.mean(prop_atypicals)),
            "atypical_acc_rate_std": float(np.std(prop_atypicals, ddof=1)),
            "rare_test_samples_mean": float(np.mean(rare_test_samples_list)),
        }
        status_tag = "<= 0.20 Target Met" if asr_m <= 0.20 else "> 0.20 Threshold Exceeded"
        print(f"  Proposed Defense: ASR = {asr_m:.4f} +/- {asr_s:.4f} ({status_tag}) | Acc = {np.mean(prop_accs):.4f} | Rare Test N = {np.mean(rare_test_samples_list):.1f} | Atypical Acc Rate = {np.mean(prop_atypicals):.4f}")

        # Baseline Defenses
        for b_name in BASELINES:
            b_asrs, b_accs, b_recalls, b_atypicals = [], [], [], []
            for s in SEEDS:
                rb = run_single_trial(b_name, "backdoor", d_name, s)
                b_asrs.append(rb["backdoor_asr"])
                b_accs.append(rb["accuracy"])
                b_recalls.append(rb["rare_recall"])
                b_atypicals.append(rb["atypical_acc_rate"])

            d_dict["backdoor_attack"][b_name] = {
                "asr_mean": float(np.mean(b_asrs)),
                "asr_std": float(np.std(b_asrs, ddof=1)),
                "accuracy_mean": float(np.mean(b_accs)),
                "accuracy_std": float(np.std(b_accs, ddof=1)),
                "rare_recall_mean": float(np.mean(b_recalls)),
                "rare_recall_std": float(np.std(b_recalls, ddof=1)),
                "atypical_acc_rate_mean": float(np.mean(b_atypicals)),
                "atypical_acc_rate_std": float(np.std(b_atypicals, ddof=1)),
            }
            # Seed-level statistical comparison on ASR
            asr_comp = compute_seed_level_comparison(prop_asrs, b_asrs)
            d_dict["seed_level_statistics"][f"proposed_vs_{b_name}_asr"] = asr_comp
            print(f"  {b_name.upper():<12}: ASR = {np.mean(b_asrs):.4f} +/- {np.std(b_asrs, ddof=1):.4f} | Diff = {asr_comp['mean_diff']:+.4f} (p_wilc = {asr_comp['p_val_wilcoxon']:.4e})")

        # 2. Ablations of Proposed Defense (Backdoor Attack)
        print(f"\n--- Proposed Defense Ablations on {d_name} ---")
        ablations_to_test = {
            "w_o_subspace_proj": {"enable_subspace_projection": False},
            "w_o_class_aware": {"enable_class_aware": False},
            "w_o_clipping": {"enable_clipping": False},
            "w_o_root_norm": {"enable_root_norm": False},
        }
        for abl_name, abl_kw in ablations_to_test.items():
            a_asrs, a_accs, a_recalls, a_atypicals = [], [], [], []
            for s in SEEDS:
                ra = run_single_trial("proposed", "backdoor", d_name, s, ablation_kwargs=abl_kw)
                a_asrs.append(ra["backdoor_asr"])
                a_accs.append(ra["accuracy"])
                a_recalls.append(ra["rare_recall"])
                a_atypicals.append(ra["atypical_acc_rate"])

            d_dict["ablations"][abl_name] = {
                "asr_mean": float(np.mean(a_asrs)),
                "asr_std": float(np.std(a_asrs, ddof=1)),
                "accuracy_mean": float(np.mean(a_accs)),
                "accuracy_std": float(np.std(a_accs, ddof=1)),
                "rare_recall_mean": float(np.mean(a_recalls)),
                "rare_recall_std": float(np.std(a_recalls, ddof=1)),
                "atypical_acc_rate_mean": float(np.mean(a_atypicals)),
                "atypical_acc_rate_std": float(np.std(a_atypicals, ddof=1)),
            }
            print(f"  Ablation ({abl_name}): ASR = {np.mean(a_asrs):.4f} | Acc = {np.mean(a_accs):.4f} | Rare Recall = {np.mean(a_recalls):.4f}")

        full_results["datasets"][d_name] = d_dict

    total_time = time.time() - t0
    print(f"\n=======================================================")
    print(f"BENCHMARK COMPLETED IN {total_time:.1f}s.")
    print(f"=======================================================")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "proposed_defense_benchmark_3datasets_20seeds.json")
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved complete 3-dataset results to: {out_file}")


if __name__ == "__main__":
    main()
