"""
Unit tests for Byzantine, Backdoor, and Adaptive attack generators.
"""

import numpy as np
import pytest

from fl_ids.attacks.byzantine import SignFlipAttack, LabelFlipAttack, RandomNoiseAttack
from fl_ids.attacks.backdoor import AdditiveBackdoorAttack
from fl_ids.attacks.adaptive import AdaptiveAttacker


def test_sign_flip_attack():
    attack = SignFlipAttack(scale=1.5)
    global_w = {"layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    local_w = {"layer1": np.array([1.2, 2.4, 2.8], dtype=np.float32)}

    # delta = [0.2, 0.4, -0.2]
    # poisoned = global_w - scale * delta = [1.0 - 0.3, 2.0 - 0.6, 3.0 - (-0.3)] = [0.7, 1.4, 3.3]
    poisoned = attack.poison_update(local_w, global_w, client_id=1)
    expected = np.array([0.7, 1.4, 3.3], dtype=np.float32)
    assert np.allclose(poisoned["layer1"], expected)


def test_label_flip_attack():
    attack = LabelFlipAttack(source_class=2, target_class=0)
    X = np.ones((10, 4), dtype=np.float32)
    y = np.array([0, 1, 2, 2, 3, 0, 2, 1, 4, 2])

    X_p, y_p = attack.poison_data(X, y, client_id=1)
    assert np.all(y_p[y == 2] == 0)
    assert y_p[1] == 1  # Unaffected class


def test_additive_backdoor_attack():
    attack = AdditiveBackdoorAttack(
        trigger_features=[0, 2],
        trigger_magnitude=10.0,
        target_class=0,
        poison_rate=0.5,
        boost_factor=2.0,
    )

    X = np.zeros((10, 5), dtype=np.float32)
    y = np.ones(10, dtype=int) * 3  # All class 3

    X_p, y_p = attack.poison_data(X, y, client_id=1)
    # Check that half of the samples were poisoned with target class 0
    poisoned_idx = np.where(y_p == 0)[0]
    assert len(poisoned_idx) == 5
    # Trigger features should be 10.0
    assert np.all(X_p[poisoned_idx, 0] == 10.0)
    assert np.all(X_p[poisoned_idx, 2] == 10.0)
    assert np.all(X_p[poisoned_idx, 1] == 0.0)  # Untouched feature

    # Test boost factor in model update
    global_w = {"w": np.zeros(4, dtype=np.float32)}
    local_w = {"w": np.ones(4, dtype=np.float32)}
    boosted = attack.poison_update(local_w, global_w, client_id=1)
    assert np.allclose(boosted["w"], np.ones(4, dtype=np.float32) * 2.0)


def test_adaptive_attacker_norm_bounding():
    base_atk = AdditiveBackdoorAttack(boost_factor=100.0)
    adaptive = AdaptiveAttacker(base_attack=base_atk, max_norm_deviation=2.0)

    global_w = {"w": np.zeros(10, dtype=np.float32)}
    local_w = {"w": np.ones(10, dtype=np.float32)}

    poisoned = adaptive.poison_update(local_w, global_w, client_id=1)

    # Check deviation from honest update
    honest_delta = local_w["w"] - global_w["w"]
    poison_delta = poisoned["w"] - global_w["w"]
    pert_norm = np.linalg.norm(poison_delta - honest_delta)

    # Norm deviation must be capped at max_norm_deviation (2.0)
    assert pert_norm <= 2.0001
