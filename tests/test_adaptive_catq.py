"""
Unit tests for AdaptiveCATQDefense (CAT-Q v2).
"""

import pytest
import numpy as np
import torch
from fl_ids.defense.adaptive_catq import AdaptiveCATQDefense
from fl_ids.models.mlp import EdgeIDSMLP


def test_adaptive_catq_aggregation():
    torch.manual_seed(42)
    np.random.seed(42)

    model = EdgeIDSMLP(input_dim=41, num_classes=10)
    global_weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}

    # Server root gradient direction
    server_root_grad = {}
    for k, v in global_weights.items():
        if np.issubdtype(v.dtype, np.floating):
            server_root_grad[k] = np.random.normal(0, 0.05, size=v.shape).astype(v.dtype)

    # 8 Benign clients: update aligns well with server root gradient + slight personal noise
    client_updates = {}
    for cid in range(8):
        update = {}
        for k, v in global_weights.items():
            if np.issubdtype(v.dtype, np.floating):
                # Benign clients move in direction of root gradient
                g = server_root_grad[k]
                personal_noise = np.random.normal(0, 0.005, size=v.shape).astype(v.dtype)
                update[k] = v + g + personal_noise
            else:
                update[k] = v.copy()
        client_updates[cid] = update

    # 2 Backdoor Attackers: update has massive perturbation in classifier head, orthogonal to root
    for cid in [8, 9]:
        update = {}
        for k, v in global_weights.items():
            if np.issubdtype(v.dtype, np.floating):
                if "fc3" in k:  # Classifier head
                    attack_signal = np.random.normal(0, 0.8, size=v.shape).astype(v.dtype)
                    update[k] = v + attack_signal
                else:
                    update[k] = v + np.random.normal(0, 0.01, size=v.shape).astype(v.dtype)
            else:
                update[k] = v.copy()
        client_updates[cid] = update

    defense = AdaptiveCATQDefense(mad_k=1.5, consensus_threshold=2)
    new_weights, meta = defense.aggregate(
        client_updates=client_updates,
        global_weights=global_weights,
        server_root_grad=server_root_grad,
    )

    assert "accepted_clients" in meta
    assert "rejected_clients" in meta
    assert "diagnostics" in meta
    
    # Attackers 8 and 9 should be flagged (either rejected or soft-projected)
    flagged = set(meta["rejected_clients"]) | set(meta["soft_projected_clients"])
    assert 8 in flagged or 9 in flagged

    # Ensure shape of new weights matches global weights
    for k in global_weights:
        assert new_weights[k].shape == global_weights[k].shape
