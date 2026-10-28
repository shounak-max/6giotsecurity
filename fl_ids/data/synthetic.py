"""
Synthetic NetFlow v2 IDS Dataset Generator.

Simulates NetFlow / CSE-CIC-IDS2018 / NF-UNSW-NB15 tabular feature distributions,
including realistic flow stats and minority attack classes:
- Class 0: Benign (majority ~70%)
- Class 1: DDoS
- Class 2: DoS
- Class 3: PortScan
- Class 4: Bot
- Class 5: Infiltration
- Class 6: BruteForce
- Class 7: Worms (extremely rare minority class)
- Class 8: SQL Injection (minority class)
- Class 9: Fingerprinting / Uploading (minority class)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "Benign",
    1: "DDoS",
    2: "DoS",
    3: "PortScan",
    4: "Bot",
    5: "Infiltration",
    6: "BruteForce",
    7: "Worms",  # Rare class (<0.1%)
    8: "SQL_Injection",  # Rare class (~0.3%)
    9: "Fingerprinting",  # Rare class (~0.4%)
}

RARE_CLASSES = [7, 8, 9]


def generate_synthetic_netflow(
    n_samples: int = 5000,
    n_features: int = 40,
    inject_placeholders: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, any]]:
    """
    Generates a realistic synthetic tabular NetFlow dataset.

    Args:
        n_samples: Total number of network flow records to generate.
        n_features: Number of flow statistical features (default: 40).
        inject_placeholders: Whether to inject raw placeholders (NaN, Inf, strings) for testing preprocessor.
        random_state: Random seed.

    Returns:
        df_features: DataFrame containing flow feature columns.
        labels: 1D numpy array of integer class labels (0..9).
        metadata: Dictionary with class distribution info and rare class indices.
    """
    rng = np.random.RandomState(random_state)

    # Class probability distribution mimicking real-world heavy skew & extreme rare classes
    probs = np.array([
        0.65,    # 0: Benign
        0.12,    # 1: DDoS
        0.08,    # 2: DoS
        0.06,    # 3: PortScan
        0.04,    # 4: Bot
        0.025,   # 5: Infiltration
        0.015,   # 6: BruteForce
        0.002,   # 7: Worms (rare)
        0.003,   # 8: SQL Injection (rare)
        0.005,   # 9: Fingerprinting (rare)
    ])
    probs = probs / probs.sum()

    # Assign labels
    labels = rng.choice(10, size=n_samples, p=probs)

    # Ensure at least a few samples exist for each rare class
    for rare_c in RARE_CLASSES:
        if np.sum(labels == rare_c) < 5:
            # Force at least 5 samples
            replace_indices = rng.choice(np.where(labels == 0)[0], size=5, replace=False)
            labels[replace_indices] = rare_c

    # Generate feature base matrix
    features = np.zeros((n_samples, n_features), dtype=np.float32)

    # Base benign traffic: log-normal and standard normal flow statistics
    for j in range(n_features):
        if j % 4 == 0:  # Byte counts / Packet sizes (heavy-tailed)
            features[:, j] = rng.lognormal(mean=4.0, sigma=1.5, size=n_samples)
        elif j % 4 == 1:  # Inter-arrival times / Durations (exponential)
            features[:, j] = rng.exponential(scale=2.0, size=n_samples)
        elif j % 4 == 2:  # Packet counts / Flags (Poisson-like)
            features[:, j] = rng.poisson(lam=8.0, size=n_samples).astype(np.float32)
        else:  # Flow rates / Ratios (Gaussian)
            features[:, j] = rng.normal(loc=10.0, scale=3.0, size=n_samples)

    # Inject class-specific signatures into distinctive feature subsets
    for c in range(1, 10):
        c_mask = (labels == c)
        if not np.any(c_mask):
            continue

        sig_features = [(c * 3 + offset) % n_features for offset in range(4)]
        if c == 1:  # DDoS: huge packet bursts
            features[c_mask, sig_features[0]] *= 15.0
            features[c_mask, sig_features[1]] += 80.0
        elif c == 2:  # DoS: long duration, high packet rate
            features[c_mask, sig_features[0]] += 50.0
        elif c == 3:  # PortScan: many connections, low bytes
            features[c_mask, sig_features[2]] *= 8.0
            features[c_mask, sig_features[0]] = np.clip(features[c_mask, sig_features[0]] * 0.1, 0, None)
        elif c in RARE_CLASSES:  # Worms / SQL Injection: distinct high-order payload/flag anomaly
            features[c_mask, sig_features[0]] += 25.0
            features[c_mask, sig_features[1]] *= 4.0
            features[c_mask, sig_features[2]] += 12.0

    feature_cols = [f"flow_feat_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_cols)

    if inject_placeholders:
        # Inject occasional Infs and NaNs in a few random rows
        bad_idx = rng.choice(n_samples, size=max(5, n_samples // 100), replace=False)
        df.iloc[bad_idx[0], 0] = np.inf
        df.iloc[bad_idx[1], 1] = -np.inf
        df.iloc[bad_idx[2], 2] = np.nan

    classes, counts = np.unique(labels, return_counts=True)
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "class_counts": {CLASS_NAMES[int(c)]: int(cnt) for c, cnt in zip(classes, counts)},
        "rare_classes": RARE_CLASSES,
        "rare_class_names": [CLASS_NAMES[c] for c in RARE_CLASSES],
    }

    return df, labels, metadata
