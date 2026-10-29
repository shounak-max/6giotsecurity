"""
Unified CLI Experiment Runner for Federated Intrusion Detection (FL-IDS).

Supports running baselines and novel defenses on synthetic NetFlow or real flow datasets:
- Baselines: fedavg, trimmed_mean, multikrum, foolsgold, fltrust, repguard
- Proposed: class_aware_trust (ClassAwareTrustDefense)
- Attacks: none, sign_flip, label_flip, random_noise, backdoor, adaptive
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fl_ids.data.synthetic import generate_synthetic_netflow, RARE_CLASSES
from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.canary import NaiveBayesCanary
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import (
    SignFlipAttack,
    LabelFlipAttack,
    RandomNoiseAttack,
    AdditiveBackdoorAttack,
    AdaptiveAttacker,
)
from fl_ids.defense import (
    FedAvg,
    TrimmedMean,
    MultiKrum,
    FoolsGold,
    FLTrust,
    RepGuard,
    SketchCompressor,
    ClassAwareTrustDefense,
)
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator
from fl_ids.edge.profiler import EdgeProfiler
from fl_ids.models.mlp import EdgeIDSMLP


def get_defense(defense_name: str, num_malicious: int = 2):
    name = defense_name.lower()
    if name == "fedavg":
        return FedAvg()
    elif name == "trimmed_mean":
        return TrimmedMean(trim_fraction=0.2)
    elif name == "multikrum":
        return MultiKrum(num_malicious=num_malicious)
    elif name == "foolsgold":
        return FoolsGold()
    elif name == "fltrust":
        return FLTrust()
    elif name == "repguard":
        return RepGuard()
    elif name in ["class_aware_trust", "proposed"]:
        return ClassAwareTrustDefense(client_clip_norm=5.0, min_trust_threshold=0.05)
    else:
        raise ValueError(f"Unknown defense: {defense_name}")


def get_attack(attack_name: str, trigger_mag: float = 5.0, boost_factor: float = 2.0):
    name = attack_name.lower()
    if name == "none":
        return None
    elif name == "sign_flip":
        return SignFlipAttack(scale=1.5)
    elif name == "label_flip":
        return LabelFlipAttack(target_class=0)
    elif name == "random_noise":
        return RandomNoiseAttack(std=2.0)
    elif name == "backdoor":
        return AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_mag,
            target_class=0,
            poison_rate=0.5,
            boost_factor=boost_factor,
        )
    elif name == "adaptive":
        base = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_mag,
            target_class=0,
            boost_factor=boost_factor,
        )
        return AdaptiveAttacker(base_attack=base, max_norm_deviation=3.0)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def run(args):
    print("=" * 70)
    print("FL-IDS Experiment Configuration:")
    print(f"  Defense:       {args.defense}")
    print(f"  Attack:        {args.attack}")
    print(f"  Clients:       {args.clients} (Malicious: {args.malicious_clients})")
    print(f"  Dirichlet alpha: {args.alpha}")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Rounds:        {args.rounds}")
    print(f"  Compression:   {args.compression}")
    print("=" * 70)

    # 1. Dataset Generation or Loading Real Dataset
    if args.dataset.lower() in ["nf-ton-iot-v2", "nftoniotv2", "nf_ton_iot_v2", "real"]:
        print(f"[Data] Loading real NF-ToN-IoT-V2 dataset (samples={args.samples})...")
        df, labels, meta = load_nf_ton_iot_v2(
            parquet_path=args.dataset_path,
            n_samples=args.samples if args.samples > 0 else None,
            drop_identity_features=True,
            random_state=args.seed,
        )
        rare_class_ids = meta["rare_classes"]
        num_classes = meta["num_classes"]
    else:
        print(f"[Data] Generating synthetic NetFlow dataset (samples={args.samples})...")
        df, labels, meta = generate_synthetic_netflow(
            n_samples=args.samples,
            n_features=args.features,
            random_state=args.seed,
        )
        rare_class_ids = RARE_CLASSES
        num_classes = 10

    X = df.to_numpy(dtype=np.float32)
    actual_features = X.shape[1]
    print(f"[Data] Loaded {len(X)} samples with {actual_features} features, {num_classes} classes.")
    print(f"[Data] Rare classes: {rare_class_ids} ({meta.get('rare_class_names', [])})")

    # 2. Server Root Set Extraction
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=args.root_size, random_state=args.seed
    )

    # 3. Train-only scaling
    preproc = LeakageFreePreprocessor(random_state=args.seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # Train / Test split
    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 4. Leakage Canary Audit
    canary = NaiveBayesCanary()
    canary_result = canary.audit(X_train, y_train, X_test, y_test)
    print(f"[Canary] {canary_result['message']}")
    if canary_result["is_leakage_suspected"]:
        print("WARNING: Data leakage suspected by Canary! Halting experiment.")
        return

    # 5. Non-IID Partitioning
    partitioner = DirichletPartitioner(
        num_clients=args.clients,
        alpha=args.alpha,
        random_state=args.seed,
    )
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=RARE_CLASSES)
    atypical_clients = p_meta["atypical_clients"]
    print(f"[Data] Partitioned among {args.clients} clients. Atypical clients: {atypical_clients}")

    # 6. Attack & Trigger Setup
    attack_obj = get_attack(args.attack, trigger_mag=args.trigger_magnitude, boost_factor=args.boost_factor)
    X_triggered_test = None
    if isinstance(attack_obj, (AdditiveBackdoorAttack, AdaptiveAttacker)):
        base_backdoor = attack_obj if isinstance(attack_obj, AdditiveBackdoorAttack) else attack_obj.base_attack
        X_triggered_test = base_backdoor.apply_trigger(X_test)

    # 7. Sketch Compressor Setup
    compressor = None
    if args.compression != "none":
        compressor = SketchCompressor(method=args.compression, compression_ratio=0.05)

    # 8. Client Creation
    clients = []
    malicious_ids = list(range(args.malicious_clients))
    for cid in range(args.clients):
        c_idxs = client_splits[cid]
        c_atk = attack_obj if cid in malicious_ids else None
        client = FederatedClient(
            client_id=cid,
            X_train=X_train[c_idxs],
            y_train=y_train[c_idxs],
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            learning_rate=args.lr,
            attack=c_atk,
            compressor=compressor,
            device=args.device,
        )
        clients.append(client)

    # 9. Server & Defense Setup
    defense_obj = get_defense(args.defense, num_malicious=args.malicious_clients)
    server = FederatedServer(
        input_dim=actual_features,
        num_classes=num_classes,
        aggregator=defense_obj,
        X_root=X_root_scaled,
        y_root=y_root,
        device=args.device,
    )

    # 10. Run Simulation
    sim = FederatedSimulator(
        server=server,
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        X_triggered_test=X_triggered_test,
        atypical_client_ids=atypical_clients,
        rare_class_ids=rare_class_ids,
        backdoor_target_class=0,
    )

    history = sim.run(num_rounds=args.rounds, verbose=True)
    final_record = history[-1]

    # 11. Edge Hardware Profiling
    profiler = EdgeProfiler(server.global_model, device=args.device)
    params_info = profiler.count_parameters()
    latency_info = profiler.measure_inference_latency(input_dim=actual_features, batch_size=1)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT RESULTS: {args.defense.upper()} under {args.attack.upper()} (alpha={args.alpha})")
    print("=" * 70)
    print(f"  Macro-F1:                 {final_record['macro_f1']:.4f}")
    print(f"  Balanced Accuracy:        {final_record['balanced_accuracy']:.4f}")
    print(f"  Rare Class Mean Recall:   {final_record['rare_class_mean_recall']:.4f}")
    print(f"  Atypical Acceptance %:    {final_record['atypical_acceptance_rate']*100:.1f}%")
    if X_triggered_test is not None:
        print(f"  Backdoor ASR:             {final_record['backdoor_asr']:.4f}")
    print(f"  Inference Latency (med):  {latency_info['latency_median_ms']:.3f} ms")
    print(f"  Inference Latency (p95):  {latency_info['latency_p95_ms']:.3f} ms")
    print(f"  Model Parameters:         {params_info['total_params']:,}")
    print("=" * 70)

    # Output to JSON if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "config": vars(args),
                "final_metrics": final_record,
                "latency_metrics": latency_info,
                "history": history,
            }, f, indent=2)
        print(f"Saved experiment results to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL-IDS Experiment")
    parser.add_argument("--defense", type=str, default="class_aware_trust",
                        choices=["fedavg", "trimmed_mean", "multikrum", "foolsgold", "fltrust", "repguard", "class_aware_trust"])
    parser.add_argument("--attack", type=str, default="backdoor",
                        choices=["none", "sign_flip", "label_flip", "random_noise", "backdoor", "adaptive"])
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration parameter")
    parser.add_argument("--clients", type=int, default=10, help="Total clients")
    parser.add_argument("--malicious_clients", type=int, default=2, help="Number of malicious clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--root_size", type=int, default=100)
    parser.add_argument("--trigger_magnitude", type=float, default=5.0)
    parser.add_argument("--boost_factor", type=float, default=2.0)
    parser.add_argument("--compression", type=str, default="none", choices=["none", "topk", "sign"])
    parser.add_argument("--dataset", type=str, default="NF-ToN-IoT-V2",
                        choices=["NF-ToN-IoT-V2", "synthetic_netflow", "real"],
                        help="Dataset to evaluate on")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to parquet dataset file")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    run(args)
