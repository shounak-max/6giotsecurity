"""
Diagnostic script: Inspects client update representations, cosine similarities,
and evaluates clean-model baseline target rate.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.defense import FedAvg
from fl_ids.federation import FederatedClient, FederatedServer, FederatedSimulator
from fl_ids.utils.metrics import compute_backdoor_asr

seed = 42
df, labels, meta = load_nf_ton_iot_v2(n_samples=5000, random_state=seed)
X = df.to_numpy(dtype=np.float32)
rare_classes = meta["rare_classes"]
num_classes = meta["num_classes"]
input_dim = X.shape[1]

# Disjoint root
X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
    X, labels, root_size=200, random_state=seed
)
preproc = LeakageFreePreprocessor(random_state=seed)
X_clients_scaled = preproc.fit_transform(X_clients)
X_root_scaled = preproc.transform(X_root)

n_test = int(len(X_clients_scaled) * 0.25)
n_train = len(X_clients_scaled) - n_test
X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

# Attack definition
attack_obj = AdditiveBackdoorAttack(
    trigger_features=[0, 1, 2],
    trigger_magnitude=4.0,
    target_class=0,
    poison_rate=0.5,
    boost_factor=2.0,
)
X_trig_test = attack_obj.apply_trigger(X_test)

# Partition
partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=rare_classes)
atypical_clients = p_meta["atypical_clients"]
malicious_ids = [0, 1]

print(f"Atypical clients: {atypical_clients}")
print(f"Malicious clients: {malicious_ids}")

# 1. First, train clean FedAvg and measure baseline ASR on triggered inputs
clean_clients = [
    FederatedClient(cid, X_train[client_splits[cid]], y_train[client_splits[cid]], batch_size=32, local_epochs=1, learning_rate=0.005)
    for cid in range(10)
]
clean_server = FederatedServer(input_dim, num_classes, aggregator=FedAvg(), X_root=X_root_scaled, y_root=y_root)
clean_sim = FederatedSimulator(clean_server, clean_clients, X_test, y_test, X_triggered_test=X_trig_test, atypical_client_ids=atypical_clients, rare_class_ids=rare_classes)
clean_history = clean_sim.run(num_rounds=5, verbose=False)

clean_server.global_model.eval()
with torch.no_grad():
    trig_logits = clean_server.global_model(torch.tensor(X_trig_test, dtype=torch.float32))
    trig_preds = torch.argmax(trig_logits, dim=-1).numpy()
baseline_asr = compute_backdoor_asr(trig_preds, target_class=0, y_true=y_test)
print(f"\n--> Clean-Model Baseline Target Rate on Triggered Inputs: {baseline_asr:.4f}")

# 2. Inspect round 1 gradients under attack
server = FederatedServer(input_dim, num_classes, aggregator=FedAvg(), X_root=X_root_scaled, y_root=y_root)
root_grad, class_grads = server.compute_root_gradients()
g0 = np.concatenate([v.flatten() for v in root_grad.values()])
norm_g0 = np.linalg.norm(g0) + 1e-8

attacked_clients = [
    FederatedClient(cid, X_train[client_splits[cid]], y_train[client_splits[cid]], batch_size=32, local_epochs=1, learning_rate=0.005,
                    attack=attack_obj if cid in malicious_ids else None)
    for cid in range(10)
]

# Simulate 1 round of local training
updates = {}
for c in attacked_clients:
    w, _ = c.train(server.global_weights)
    delta = np.concatenate([(w[k] - server.global_weights[k]).flatten() for k in w])
    updates[c.client_id] = delta

print("\nRound 1 Update Diagnostics:")
for cid, delta in updates.items():
    norm_delta = np.linalg.norm(delta)
    cos_g0 = np.dot(delta, g0) / (norm_delta * norm_g0)
    
    # Class conditional max cos
    max_c_cos = -1.0
    best_c = -1
    for c_id, cg in class_grads.items():
        c_cos = np.dot(delta, cg) / (norm_delta * np.linalg.norm(cg))
        if c_cos > max_c_cos:
            max_c_cos = c_cos
            best_c = c_id
            
    is_mal = cid in malicious_ids
    is_atyp = cid in atypical_clients
    tag = "MALICIOUS" if is_mal else ("ATYPICAL" if is_atyp else "TYPICAL")
    print(f"Client {cid:2d} [{tag:<9}]: Norm={norm_delta:6.2f}, Cos(g0)={cos_g0:+.4f}, Best Class Cos={max_c_cos:+.4f} (class {best_c})")
