"""
Test Feature Saliency & Trigger Detection on Client Models.

Tests whether the server can use clean root data to compute input-feature attribution:
S_j(client_i) = mean_{x in X_root} | d Loss / d x_j |
Backdoor attackers who train on poisoned data have an extreme outlier concentration
on the trigger feature dimensions (features [0, 1, 2]).
Honest clients (even atypical ones) have smooth, natural feature attribution.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks import AdditiveBackdoorAttack
from fl_ids.federation import FederatedClient, FederatedServer
from fl_ids.models import EdgeIDSMLP

seed = 42
df, labels, meta = load_nf_ton_iot_v2(n_samples=4000, min_rare_samples=100, random_state=seed)
X = df.to_numpy(dtype=np.float32)
rare_classes = meta["rare_classes"]
num_classes = meta["num_classes"]
input_dim = X.shape[1]

# Extract disjoint root
X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
    X, labels, root_size=200, random_state=seed
)
preproc = LeakageFreePreprocessor(random_state=seed)
X_clients_scaled = preproc.fit_transform(X_clients)
X_root_scaled = preproc.transform(X_root)

# Partition
partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=seed)
client_splits, p_meta = partitioner.partition(y_clients, atypical_class_ids=rare_classes)
atypical_clients = p_meta["atypical_clients"]

# Attacker on clients 0, 1
attack_obj = AdditiveBackdoorAttack(
    trigger_features=[0, 1, 2],
    trigger_magnitude=4.0,
    target_class=0,
    poison_rate=0.5,
    boost_factor=2.0,
)
malicious_ids = [0, 1]

# Create server model
global_model = EdgeIDSMLP(input_dim=input_dim, num_classes=num_classes)
global_weights = global_model.get_weights_dict()

# Train 1 round for all clients
client_models = {}
for cid in range(10):
    c_atk = attack_obj if cid in malicious_ids else None
    c = FederatedClient(
        client_id=cid,
        X_train=X_clients_scaled[client_splits[cid]],
        y_train=y_clients[client_splits[cid]],
        batch_size=32,
        local_epochs=1,
        learning_rate=0.005,
        attack=c_atk,
    )
    w, _ = c.train(global_weights)
    client_models[cid] = w

# Server evaluates input feature attribution on clean root data
print("=== FEATURE ATTRIBUTION SALIENCY ANALYSIS ON CLEAN ROOT DATA ===")
criterion = nn.CrossEntropyLoss()
tensor_x_root = torch.tensor(X_root_scaled, dtype=torch.float32, requires_grad=True)
tensor_y_root = torch.tensor(y_root, dtype=torch.long)

saliency_scores = {}
for cid, weights in client_models.items():
    m = EdgeIDSMLP(input_dim=input_dim, num_classes=num_classes)
    m.set_weights_dict(weights)
    m.eval()
    
    x_input = tensor_x_root.clone().detach().requires_grad_(True)
    out = m(x_input)
    loss = criterion(out, tensor_y_root)
    loss.backward()
    
    # Feature-wise mean absolute gradient
    grad_norm = torch.mean(torch.abs(x_input.grad), dim=0).detach().numpy()
    
    # Trigger features [0, 1, 2] vs background features [3..40]
    trig_saliency = float(np.mean(grad_norm[0:3]))
    bg_saliency = float(np.mean(grad_norm[3:]))
    saliency_ratio = trig_saliency / (bg_saliency + 1e-8)
    
    # Kurtosis / max-to-mean concentration
    max_feat = int(np.argmax(grad_norm))
    max_ratio = float(np.max(grad_norm) / (np.mean(grad_norm) + 1e-8))
    
    is_mal = cid in malicious_ids
    is_atyp = cid in atypical_clients
    tag = "MALICIOUS" if is_mal else ("ATYPICAL" if is_atyp else "TYPICAL")
    print(f"Client {cid:2d} [{tag:<9}]: Trig Saliency={trig_saliency:.4f}, BG Saliency={bg_saliency:.4f}, Ratio={saliency_ratio:6.2f}, MaxFeat={max_feat:2d} (MaxRatio={max_ratio:5.2f})")
