"""
Unit tests for robust aggregators and the novel ClassAwareTrustDefense.
"""

import numpy as np
import pytest

from fl_ids.defense.fedavg import FedAvg
from fl_ids.defense.coordinate_filters import TrimmedMean, CoordinateMedian
from fl_ids.defense.distance_filters import MultiKrum, FoolsGold
from fl_ids.defense.trust_filters import FLTrust, RepGuard
from fl_ids.defense.novel_defense import ClassAwareTrustDefense
from fl_ids.defense.compression import SketchCompressor


def _make_dummy_updates(n_clients=5, n_dim=10):
    global_weights = {"fc.weight": np.zeros(n_dim, dtype=np.float32)}
    client_updates = {}
    for i in range(n_clients):
        # Benign clients have small positive updates
        client_updates[i] = {"fc.weight": np.ones(n_dim, dtype=np.float32) * (1.0 + i * 0.1)}
    return global_weights, client_updates


def test_fedavg():
    glob, updates = _make_dummy_updates(n_clients=3, n_dim=4)
    # [1.0, 1.1, 1.2] -> mean 1.1
    agg = FedAvg()
    res, meta = agg.aggregate(updates, glob)
    assert np.allclose(res["fc.weight"], np.ones(4) * 1.1)
    assert len(meta["accepted_clients"]) == 3


def test_trimmed_mean():
    glob, updates = _make_dummy_updates(n_clients=5, n_dim=4)
    # Add a huge malicious outlier at client 4
    updates[4]["fc.weight"] = np.ones(4, dtype=np.float32) * 1000.0

    agg = TrimmedMean(trim_fraction=0.2)  # Trims 1 from each side
    res, meta = agg.aggregate(updates, glob)

    # Outlier 1000.0 should be completely trimmed
    assert res["fc.weight"][0] < 5.0


def test_multi_krum_rejects_outlier():
    glob, updates = _make_dummy_updates(n_clients=5, n_dim=4)
    # Malicious client 4
    updates[4]["fc.weight"] = np.ones(4, dtype=np.float32) * -50.0

    agg = MultiKrum(num_malicious=1, num_to_aggregate=4)
    res, meta = agg.aggregate(updates, glob)

    assert 4 not in meta["accepted_clients"]
    assert 4 in meta["rejected_clients"]


def test_fltrust():
    glob, updates = _make_dummy_updates(n_clients=4, n_dim=4)
    # Malicious client 3 with opposite direction
    updates[3]["fc.weight"] = np.ones(4, dtype=np.float32) * -10.0

    root_grad = {"fc.weight": np.ones(4, dtype=np.float32) * 1.0}

    agg = FLTrust()
    res, meta = agg.aggregate(updates, glob, server_root_grad=root_grad)

    assert meta["trust_scores"][3] == 0.0
    assert 3 in meta["rejected_clients"]


def test_sketch_compressor():
    vec = np.arange(100, dtype=np.float32)

    # Top-k
    comp_topk = SketchCompressor(method="topk", compression_ratio=0.1)
    vals, meta = comp_topk.compress_vector(vec)
    assert len(vals) == 10
    decomp = comp_topk.decompress_vector(vals, meta)
    assert len(decomp) == 100
    assert np.all(decomp[90:] == np.arange(90, 100, dtype=np.float32))

    # Sign sketch
    comp_sign = SketchCompressor(method="sign")
    vals_sign, meta_sign = comp_sign.compress_vector(vec - 50.0)
    assert meta_sign["compression_ratio"] == 1.0 / 32.0


def test_class_aware_trust_preserves_atypical_client():
    glob = {"fc.weight": np.zeros(10, dtype=np.float32)}
    updates = {}

    # Majority benign clients update along coordinate 0..4
    for i in range(4):
        u = np.zeros(10, dtype=np.float32)
        u[:5] = 1.0
        updates[i] = {"fc.weight": u}

    # Atypical client 4 carries a rare attack class (e.g. Worms), updating coordinate 5..9
    atypical_u = np.zeros(10, dtype=np.float32)
    atypical_u[5:] = 1.0
    updates[4] = {"fc.weight": atypical_u}

    # Malicious backdoor client 5 has boosted random perturbation
    mal_u = np.ones(10, dtype=np.float32) * -50.0
    updates[5] = {"fc.weight": mal_u}

    # Overall server root gradient mostly captures benign majority:
    root_grad = {"fc.weight": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}

    # Class-conditional gradient anchor for rare class 7:
    class_conditional_grads = {
        0: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }

    defense = ClassAwareTrustDefense(client_clip_norm=5.0, min_trust_threshold=0.05)
    res, meta = defense.aggregate(
        updates,
        glob,
        server_root_grad=root_grad,
        class_conditional_grads=class_conditional_grads,
    )

    # The atypical client 4 MUST be accepted because its update aligns with class 7 anchor!
    assert 4 in meta["accepted_clients"]
    # The malicious client 5 must be rejected
    assert 5 in meta["rejected_clients"]
