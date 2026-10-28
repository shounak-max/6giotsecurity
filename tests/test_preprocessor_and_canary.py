"""
Unit tests for LeakageFreePreprocessor and NaiveBayesCanary.
"""

import numpy as np
import pandas as pd
import pytest

from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.canary import NaiveBayesCanary
from fl_ids.data.synthetic import generate_synthetic_netflow


def test_canonicalize_placeholders():
    df = pd.DataFrame({
        "feat_a": [1.0, np.inf, 3.0, -np.inf],
        "feat_b": ["?", "normal", "NaN", "Infinity"],
        "feat_c": [np.nan, 2.0, 3.0, 4.0],
    })
    cleaned = LeakageFreePreprocessor.canonicalize_placeholders(df)
    assert not cleaned.isin([np.inf, -np.inf, "?", "NaN", "Infinity"]).any().any()
    assert cleaned.isna().sum().sum() == 0


def test_deduplicate():
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 1.0, 3.0],
        "feat_b": [5.0, 6.0, 5.0, 7.0],
    })
    dedup = LeakageFreePreprocessor.deduplicate(df)
    assert len(dedup) == 3


def test_train_only_scaling():
    rng = np.random.RandomState(42)
    train_data = rng.normal(loc=100.0, scale=10.0, size=(200, 5))
    test_data = rng.normal(loc=200.0, scale=10.0, size=(50, 5))

    preproc = LeakageFreePreprocessor(scaler_type="standard", clip_outliers=False)
    X_train_scaled = preproc.fit_transform(train_data)

    # Train mean should be close to 0 and std close to 1
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-1)
    assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-1)

    # Test transformed with TRAIN statistics
    X_test_scaled = preproc.transform(test_data)
    # Since test data had mean 200 vs train 100 with scale 10, test mean should be around (200 - 100) / 10 = 10
    assert np.allclose(X_test_scaled.mean(axis=0), 10.0, atol=1.0)


def test_disjoint_root_set_extraction():
    df, labels, _ = generate_synthetic_netflow(n_samples=500, random_state=42)
    X = df.to_numpy()
    root_size = 50

    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=root_size, random_state=42
    )

    assert len(X_root) == root_size
    assert len(X_clients) == 500 - root_size
    assert len(X_root) + len(X_clients) == len(X)
    # Ensure no overlap
    root_bytes = {row.tobytes() for row in X_root}
    client_bytes = {row.tobytes() for row in X_clients}
    assert len(root_bytes.intersection(client_bytes)) == 0


def test_naive_bayes_canary_detects_leakage():
    canary = NaiveBayesCanary(leakage_threshold_balanced_acc=0.95)

    # 1. Clean synthetic data: weak Naive Bayes should have moderate/realistic performance (canary passes)
    df, labels, _ = generate_synthetic_netflow(n_samples=1000, random_state=42)
    X = df.to_numpy()
    X_train, y_train = X[:800], labels[:800]
    X_test, y_test = X[800:], labels[800:]

    clean_audit = canary.audit(X_train, y_train, X_test, y_test)
    assert clean_audit["is_leakage_suspected"] is False
    assert "Canary Passed" in clean_audit["message"]

    # 2. Injected target leakage: copy label directly into a feature column
    X_train_leaked = X_train.copy()
    X_test_leaked = X_test.copy()
    X_train_leaked[:, 0] = y_train * 1000.0
    X_test_leaked[:, 0] = y_test * 1000.0

    leaked_audit = canary.audit(X_train_leaked, y_train, X_test_leaked, y_test)
    assert leaked_audit["is_leakage_suspected"] is True
    assert "LEAKAGE ALERT" in leaked_audit["message"]
