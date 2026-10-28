"""Unit tests for real NF-ToN-IoT-V2 NetFlow dataset loader and leakage verification."""

import pytest
import numpy as np
import pandas as pd

from fl_ids.data.nftoniot import load_nf_ton_iot_v2
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.canary import NaiveBayesCanary


def test_real_dataset_loading_structure():
    n_target = 1000
    df, labels, meta = load_nf_ton_iot_v2(n_samples=n_target, random_state=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_target
    assert len(labels) == n_target
    assert df.shape[1] == 41  # 41 NetFlow features, no IP addresses
    assert "IPV4_SRC_ADDR" not in df.columns
    assert "IPV4_DST_ADDR" not in df.columns
    assert "Attack" not in df.columns
    assert "Label" not in df.columns

    # Verify Benign is class 0
    assert meta["class_names"][0] == "Benign"
    assert meta["num_classes"] == 10

    # Verify rare classes identified
    assert len(meta["rare_classes"]) > 0
    for r_id in meta["rare_classes"]:
        name = meta["class_names"][r_id]
        assert name in ["backdoor", "mitm", "ransomware"]


def test_real_dataset_binary_mode():
    df, labels, meta = load_nf_ton_iot_v2(n_samples=500, binary=True, random_state=42)
    assert meta["num_classes"] == 2
    assert meta["class_names"] == {0: "Benign", 1: "Attack"}
    assert set(np.unique(labels)).issubset({0, 1})


def test_real_dataset_leakage_canary_integrity():
    df, labels, _ = load_nf_ton_iot_v2(n_samples=1500, random_state=42)
    X = df.to_numpy(dtype=np.float32)

    # Disjoint server root extraction
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=100, random_state=42
    )

    # Train-only scaling
    preproc = LeakageFreePreprocessor(random_state=42)
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # Train/test split
    n_test = 300
    n_train = len(X_clients_scaled) - n_test
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # Canary audit
    canary = NaiveBayesCanary()
    res = canary.audit(X_train, y_train, X_test, y_test)
    assert not res["is_leakage_suspected"], f"Canary suspected leakage: {res['message']}"
