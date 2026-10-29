"""
Post-Sweep Evaluation: Real Subspace-Constrained Adaptive Attacker & 20-Round Horizon.

1. Real Adaptive Attacker:
   The adversary estimates the server's clean subspace Q_hat using its own local per-class gradients.
   The poisoned update is explicitly projected onto span(Q_hat) before sending to the server:
       Delta_adaptive = Q_hat @ (Q_hat.T @ Delta_backdoor)
   This minimizes off-subspace energy rho, directly testing whether Hybrid with lambda > 0 is vulnerable.

2. 20-Round Longer Horizon:
   Evaluates FedAvg, v1, and Hybrid (lambda=0.10) over 20 communication rounds to determine
   whether the utility gap between pure projection and soft admission closes over extended training.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense import FedAvg, MultiKrum, ClassAwareTrustDefense, AdaptiveCATQDefense, HybridCATQDefense
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator
from fl_ids.models.mlp import EdgeIDSMLP

EVAL_SEEDS = [3001, 3002, 3003, 3004, 3005]  # Representative evaluation seeds
N_SAMPLES = 3000
MIN_RARE_SAMPLES = 160


def flatten_weights_dict(w: Dict[str, np.ndarray]) -> np.ndarray:
    flat = []
    for k, v in w.items():
        if np.issubdtype(v.dtype, np.floating):
            flat.append(v.flatten())
    return np.concatenate(flat)


def unflatten_weights_dict(flat: np.ndarray, template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    idx = 0
    for k, v in template.items():
        if np.issubdtype(v.dtype, np.floating):
            sz = v.size
            out[k] = flat[idx : idx + sz].reshape(v.shape).astype(v.dtype)
            idx += sz
        else:
            out[k] = v.copy()
    return out


class SubspaceEstimatedAdaptiveClient(FederatedClient):
    """
    Real Adaptive Attacker:
    1. Computes local class-conditional gradients on its own data to estimate Q_hat.
    2. Projects its poisoned update directly into span(Q_hat) to keep rho ~ 0.
    """

    def __init__(self, target_class: int = 7, trigger_magnitude: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        self.target_class = target_class
        self.trigger_magnitude = trigger_magnitude
        self.base_attack = AdditiveBackdoorAttack(
            trigger_features=[0, 1, 2],
            trigger_magnitude=trigger_magnitude,
            target_class=target_class,
            poison_rate=0.5,
            boost_factor=2.0,
        )

    def train(self, global_weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # 1. Estimate Q_hat from local per-class gradients
        model = EdgeIDSMLP(input_dim=self.X_train.shape[1], num_classes=10).to(self.device)
        model.set_weights_dict(global_weights)
        model.eval()

        t_x = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        t_y = torch.tensor(self.y_train, dtype=torch.long).to(self.device)

        # Overall gradient
        model.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss()
        l0 = loss_fn(model(t_x), t_y)
        l0.backward()
        g0_flat = np.concatenate([-p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        vectors = [g0_flat / (np.linalg.norm(g0_flat) + 1e-8)]

        # Class-conditional gradients
        classes = np.unique(self.y_train)
        for c in classes:
            c_mask = (t_y == c)
            if torch.sum(c_mask) == 0:
                continue
            model.zero_grad()
            lc = loss_fn(model(t_x[c_mask]), t_y[c_mask])
            lc.backward()
            gc_flat = np.concatenate([-p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
            vectors.append(gc_flat / (np.linalg.norm(gc_flat) + 1e-8))

        V = np.stack(vectors, axis=-1)
        Q_hat, _ = np.linalg.qr(V)

        # 2. Compute raw backdoor update via base training
        raw_weights, tele = super().train(global_weights)

        # 3. Project delta into span(Q_hat)
        g_flat = flatten_weights_dict(global_weights)
        raw_flat = flatten_weights_dict(raw_weights)
        raw_delta = raw_flat - g_flat

        # Adaptive projection: delta_adaptive = Q_hat @ (Q_hat.T @ raw_delta)
        delta_proj = Q_hat @ (Q_hat.T @ raw_delta)

        # Boost inside subspace
        delta_adapted = 2.0 * delta_proj

        adapted_flat = g_flat + delta_adapted
        adapted_weights = unflatten_weights_dict(adapted_flat, global_weights)

        return adapted_weights, tele


def run_adaptive_attack_experiment():
    print("=" * 80)
    print("EVALUATING REAL SUBSPACE-CONSTRAINED ADAPTIVE ATTACK")
    print("Attacker maximizes backdoor while constraining updates strictly inside span(Q_hat).")
    print("=" * 80)

    from scripts.run_hybrid_rigorous_sweep import load_calibrated_hyperparameters, instantiate_defense
    frozen_hyper, calib_thresh = load_calibrated_hyperparameters()

    defenses = ["fedavg", "multikrum_f2", "v1", "v2", "hybrid_lam0.05", "hybrid_lam0.10", "hybrid_lam0.25"]
    adaptive_results = {}

    for d_key in defenses:
        asr_list = []
        acc_list = []
        for seed in EVAL_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            df, labels, meta = load_nf_ton_iot_v2(n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, random_state=seed)
            X = df.to_numpy(dtype=np.float32)

            X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(X, labels, root_size=150, random_state=seed)
            preproc = LeakageFreePreprocessor(random_state=seed)
            X_clients_scaled = preproc.fit_transform(X_clients)
            X_root_scaled = preproc.transform(X_root)

            n_test = int(len(X_clients_scaled) * 0.25)
            n_train = len(X_clients_scaled) - n_test
            X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
            X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

            partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
            client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=[7, 8, 9])
            atypical_clients = p_meta.get("atypical_clients", [])

            eval_atk = AdditiveBackdoorAttack(trigger_features=[0, 1, 2], trigger_magnitude=4.0, target_class=7)
            X_trig_test = eval_atk.apply_trigger(X_test)

            aggregator = instantiate_defense(d_key, frozen_hyper, calib_thresh)

            clients = []
            for cid in range(10):
                if cid in [0, 1]:
                    client = SubspaceEstimatedAdaptiveClient(
                        client_id=cid,
                        X_train=X_train[client_splits[cid]],
                        y_train=y_train[client_splits[cid]],
                        target_class=7,
                        trigger_magnitude=4.0,
                    )
                else:
                    client = FederatedClient(client_id=cid, X_train=X_train[client_splits[cid]], y_train=y_train[client_splits[cid]], local_epochs=1)
                clients.append(client)

            server = FederatedServer(input_dim=X.shape[1], num_classes=10, aggregator=aggregator, X_root=X_root_scaled, y_root=y_root)
            sim = FederatedSimulator(server=server, clients=clients, X_test=X_test, y_test=y_test, X_triggered_test=X_trig_test, backdoor_target_class=7)
            history = sim.run(num_rounds=3, verbose=False)
            asr_list.append(history[-1].get("backdoor_asr", 0.0))
            acc_list.append(history[-1]["accuracy"])

        adaptive_results[d_key] = {
            "asr_mean": float(np.mean(asr_list)),
            "asr_std": float(np.std(asr_list, ddof=1)),
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list, ddof=1)),
        }
        print(f"Adaptive Attack | {d_key:15s} | ASR: {np.mean(asr_list):.4f} ± {np.std(asr_list, ddof=1):.4f} | Acc: {np.mean(acc_list)*100:5.2f}%")

    with open("results/adaptive_attacker_benchmark.json", "w") as f:
        json.dump(adaptive_results, f, indent=2)
    print("Adaptive attacker results saved to results/adaptive_attacker_benchmark.json.")


def run_20_rounds_horizon_experiment():
    print("\n" + "=" * 80)
    print("EVALUATING 20-ROUND LONGER HORIZON")
    print("Testing whether utility gap between v1, Hybrid (lambda=0.10), and FedAvg closes over 20 rounds.")
    print("=" * 80)

    from scripts.run_hybrid_rigorous_sweep import load_calibrated_hyperparameters, instantiate_defense
    frozen_hyper, calib_thresh = load_calibrated_hyperparameters()

    defenses = ["fedavg", "v1", "hybrid_lam0.10"]
    horizon_results = {d: {"rounds": [], "accuracy": [], "rare_recall": [], "asr": []} for d in defenses}

    # Evaluate across seed 3001
    seed = 3001
    torch.manual_seed(seed)
    np.random.seed(seed)
    df, labels, meta = load_nf_ton_iot_v2(n_samples=N_SAMPLES, min_rare_samples=MIN_RARE_SAMPLES, random_state=seed)
    X = df.to_numpy(dtype=np.float32)

    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(X, labels, root_size=150, random_state=seed)
    preproc = LeakageFreePreprocessor(random_state=seed)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    n_test = int(len(X_clients_scaled) * 0.25)
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=[7, 8, 9])

    atk = AdditiveBackdoorAttack(trigger_features=[0, 1, 2], trigger_magnitude=4.0, target_class=7, poison_rate=0.5, boost_factor=2.0)
    X_trig_test = atk.apply_trigger(X_test)

    for d_key in defenses:
        print(f"Running {d_key} for 20 rounds...")
        aggregator = instantiate_defense(d_key, frozen_hyper, calib_thresh)
        clients = []
        for cid in range(10):
            c_atk = atk if cid in [0, 1] else None
            client = FederatedClient(client_id=cid, X_train=X_train[client_splits[cid]], y_train=y_train[client_splits[cid]], local_epochs=1, attack=c_atk)
            clients.append(client)

        server = FederatedServer(input_dim=X.shape[1], num_classes=10, aggregator=aggregator, X_root=X_root_scaled, y_root=y_root)
        sim = FederatedSimulator(server=server, clients=clients, X_test=X_test, y_test=y_test, X_triggered_test=X_trig_test, backdoor_target_class=7, rare_class_ids=[7, 8, 9])
        history = sim.run(num_rounds=20, verbose=False)

        for rec in history:
            horizon_results[d_key]["rounds"].append(rec["round"])
            horizon_results[d_key]["accuracy"].append(float(rec["accuracy"]))
            horizon_results[d_key]["rare_recall"].append(float(rec.get("rare_class_mean_recall", 0.0)))
            horizon_results[d_key]["asr"].append(float(rec.get("backdoor_asr", 0.0)))

        print(f"  {d_key:15s} Round 20 | Acc = {history[-1]['accuracy']*100:.2f}%, Rare Recall = {history[-1]['rare_class_mean_recall']*100:.2f}%, ASR = {history[-1]['backdoor_asr']:.4f}")

    with open("results/20_rounds_horizon_benchmark.json", "w") as f:
        json.dump(horizon_results, f, indent=2)
    print("20-Round horizon results saved to results/20_rounds_horizon_benchmark.json.")


if __name__ == "__main__":
    run_adaptive_attack_experiment()
    run_20_rounds_horizon_experiment()
