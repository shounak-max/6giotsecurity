"""
Unit tests for HybridCATQDefense and endpoint verification:
1. lambda = 0 must reproduce pure subspace projection (v1 behaviour).
2. lambda = 1 with detector disabled must reproduce standard averaging (FedAvg behaviour).
3. Robust LOO Coordinate-Median resistance against 2 colluding attackers.
"""

import pytest
import numpy as np
import torch
from fl_ids.defense.hybrid_catq import HybridCATQDefense
from fl_ids.defense.fedavg import FedAvg
from fl_ids.defense.novel_defense import ClassAwareTrustDefense
from fl_ids.models.mlp import EdgeIDSMLP


def test_endpoint_lambda_1_matches_fedavg():
    """At lambda=1 with detector off and no clipping, HybridCATQ matches FedAvg exactly."""
    model = EdgeIDSMLP(input_dim=41, num_classes=10)
    global_weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}

    np.random.seed(42)
    client_updates = {}
    for cid in range(5):
        update = {}
        for k, v in global_weights.items():
            if np.issubdtype(v.dtype, np.floating):
                update[k] = v + np.random.normal(0, 0.05, size=v.shape).astype(v.dtype)
            else:
                update[k] = v.copy()
        client_updates[cid] = update

    server_root_grad = {k: np.random.normal(0, 0.01, size=v.shape).astype(v.dtype) for k, v in global_weights.items() if np.issubdtype(v.dtype, np.floating)}

    # FedAvg
    fedavg = FedAvg()
    fedavg_weights, _ = fedavg.aggregate(client_updates, global_weights)

    # Hybrid with lambda=1.0, detector disabled, clipping unbounded
    hybrid = HybridCATQDefense(
        lambda_val=1.0,
        enable_detector=False,
        client_clip_norm=1e9,
        server_clip_norm=1e9,
        enable_root_norm=False,
    )
    hybrid_weights, _ = hybrid.aggregate(
        client_updates=client_updates,
        global_weights=global_weights,
        server_root_grad=server_root_grad,
    )

    for k in global_weights:
        np.testing.assert_allclose(hybrid_weights[k], fedavg_weights[k], rtol=1e-5, atol=1e-5)


def test_endpoint_lambda_0_pure_subspace_projection():
    """At lambda=0, all updates are projected onto Q, with orthogonal component = 0."""
    model = EdgeIDSMLP(input_dim=41, num_classes=10)
    global_weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}

    np.random.seed(42)
    client_updates = {}
    for cid in range(5):
        update = {}
        for k, v in global_weights.items():
            if np.issubdtype(v.dtype, np.floating):
                update[k] = v + np.random.normal(0, 0.05, size=v.shape).astype(v.dtype)
            else:
                update[k] = v.copy()
        client_updates[cid] = update

    server_root_grad = {k: np.random.normal(0, 0.01, size=v.shape).astype(v.dtype) for k, v in global_weights.items() if np.issubdtype(v.dtype, np.floating)}

    hybrid_0 = HybridCATQDefense(
        lambda_val=0.0,
        enable_detector=False,
        client_clip_norm=1e9,
        server_clip_norm=1e9,
        enable_root_norm=False,
    )
    h0_weights, _ = hybrid_0.aggregate(
        client_updates=client_updates,
        global_weights=global_weights,
        server_root_grad=server_root_grad,
    )

    # Compute Q
    from fl_ids.defense.hybrid_catq import _flatten_weights
    g0 = _flatten_weights(server_root_grad)
    q0 = g0 / np.linalg.norm(g0)
    # The aggregated delta must lie entirely in span(q0)
    agg_delta = _flatten_weights(h0_weights) - _flatten_weights(global_weights)
    proj = q0 * np.dot(agg_delta, q0)
    np.testing.assert_allclose(agg_delta, proj, rtol=1e-5, atol=1e-5)


def test_robust_loo_median_collusion_resistance():
    """When 2 attackers collude with identical malicious updates, LOO median isolates honest centroid."""
    model = EdgeIDSMLP(input_dim=41, num_classes=10)
    global_weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}

    np.random.seed(42)
    # 8 honest clients around +0.01
    client_updates = {}
    for cid in range(8):
        client_updates[cid] = {
            k: (v + np.random.normal(0.01, 0.001, size=v.shape).astype(v.dtype) if np.issubdtype(v.dtype, np.floating) else v.copy())
            for k, v in global_weights.items()
        }
    # 2 colluding attackers at -0.50
    for cid in [8, 9]:
        client_updates[cid] = {
            k: (v - 0.50 if np.issubdtype(v.dtype, np.floating) else v.copy())
            for k, v in global_weights.items()
        }

    server_root_grad = {k: np.ones_like(v) * 0.01 for k, v in global_weights.items() if np.issubdtype(v.dtype, np.floating)}

    defense = HybridCATQDefense(lambda_val=0.1, mad_k=1.5, drop_flagged=True)
    _, meta = defense.aggregate(client_updates, global_weights, server_root_grad=server_root_grad)

    # Both colluding attackers (8 and 9) must be flagged and rejected
    assert 8 in meta["rejected_clients"]
    assert 9 in meta["rejected_clients"]
