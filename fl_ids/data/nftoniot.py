"""
NF-ToN-IoT-V2 Real NetFlow Intrusion Detection Dataset Loader.

Loads the real NF-ToN-IoT-V2 NetFlow v2 benchmark dataset with:
1. Memory-efficient streaming extraction from Parquet (no full 13.1M row RAM blowup).
2. Exact class preservation for all 10 classes, including rare classes:
   - ransomware (~0.025%)
   - mitm (~0.06%)
   - backdoor (~0.12%)
3. Canonicalization of placeholders (Infs, NaNs, missing values).
4. Deduplication before any federated splitting.
5. Removal of identity-leaking features (IPv4 addresses).
6. Guaranteed class 0 = Benign.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fl_ids.data.preprocessor import LeakageFreePreprocessor

DEFAULT_DATASET_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "nftoniotv2", "NF-ToN-IoT-V2.parquet"),
    os.path.expanduser(r"~/.cache/kagglehub/datasets/dhoogla/nftoniotv2/versions/2/NF-ToN-IoT-V2.parquet"),
]

IDENTITY_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
LABEL_COLUMNS = ["Label", "Attack"]

# Real dataset class distribution proportions (precomputed from 13.1M flows)
KNOWN_CLASS_PROPORTIONS = {
    "Benign": 0.2741,
    "scanning": 0.2285,
    "xss": 0.1865,
    "ddos": 0.1329,
    "password": 0.0756,
    "injection": 0.0503,
    "dos": 0.0498,
    "backdoor": 0.0012,
    "mitm": 0.0006,
    "ransomware": 0.0003,
}


def get_default_parquet_path() -> str:
    """Finds the first existing NF-ToN-IoT-V2.parquet path."""
    for p in DEFAULT_DATASET_PATHS:
        if os.path.exists(p):
            return p
    return DEFAULT_DATASET_PATHS[0]


def load_nf_ton_iot_v2(
    parquet_path: Optional[str] = None,
    n_samples: Optional[int] = 5000,
    binary: bool = False,
    drop_identity_features: bool = True,
    rare_threshold_ratio: float = 0.02,
    min_rare_samples: int = 150,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, any]]:
    """
    Memory-efficient loader for real NF-ToN-IoT-V2 dataset.

    Args:
        parquet_path: Path to NF-ToN-IoT-V2.parquet.
        n_samples: Target number of samples to load (stratified). Default: 5000.
        binary: If True, binary classification (0=Benign, 1=Attack).
        drop_identity_features: If True, drops IP address columns if present.
        rare_threshold_ratio: Ratio below which attack classes are treated as rare minority.
        min_rare_samples: Guaranteed minimum sample quota for rare attack classes.
        random_state: Random seed for sampling.

    Returns:
        df_features: DataFrame of numerical flow features.
        labels: 1D numpy array of integer labels (0 = Benign).
        metadata: Dict with class mappings, counts, rare classes, and feature names.
    """
    if parquet_path is None:
        parquet_path = get_default_parquet_path()

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"NF-ToN-IoT-V2 parquet file not found at: {parquet_path}")

    # Disk cache check to avoid repeatedly streaming 13M rows
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "nftoniotv2", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"sample_{n_samples}_{min_rare_samples}_{random_state}_{'bin' if binary else 'multi'}.parquet")
    if os.path.exists(cache_file):
        cached_df = pd.read_parquet(cache_file)
        labels = cached_df["_label"].to_numpy(dtype=np.int64)
        if binary and len(np.unique(labels)) > 2:
            labels = (labels > 0).astype(np.int64)
        df_features = cached_df.drop(columns=["_label"])
        # Reconstruct metadata
        unique_labels = np.unique(labels)
        if binary:
            cached_class_names = {0: "Benign", 1: "Attack"}
            rare_classes = [1]
        else:
            ordered_classes = ["Benign"] + [a for a in sorted(list(KNOWN_CLASS_PROPORTIONS.keys())) if a != "Benign"]
            cached_class_names = {idx: cls_name for idx, cls_name in enumerate(ordered_classes)}
            KNOWN_RARE = {"backdoor", "mitm", "ransomware"}
            rare_classes = [int(c) for c in unique_labels if int(c) != 0 and (cached_class_names.get(int(c), "").lower() in KNOWN_RARE or float(np.mean(labels == c)) < rare_threshold_ratio)]
        return df_features, labels, {
            "num_classes": len(unique_labels),
            "class_names": cached_class_names,
            "rare_classes": rare_classes,
            "class_proportions": {c: float(np.mean(labels == c)) for c in unique_labels},
            "feature_names": list(df_features.columns),
        }

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    rng = np.random.RandomState(random_state)

    # Determine columns to load
    all_col_names = pf.schema.names
    cols_to_load = [c for c in all_col_names if c not in IDENTITY_COLUMNS] if drop_identity_features else all_col_names

    # If n_samples is specified and less than total_rows, stream and extract stratified samples
    if n_samples is not None and n_samples < total_rows:
        # Determine quota per class
        class_quotas = {}
        allocated = 0
        effective_min_rare = min(min_rare_samples, max(15, int(n_samples * 0.03)))
        for cls_name, frac in KNOWN_CLASS_PROPORTIONS.items():
            q = int(round(n_samples * frac))
            # Guarantee at least min_rare_samples for rare classes so FL clients and test evaluation have robust representation
            if frac < rare_threshold_ratio:
                q = max(effective_min_rare, q)
            class_quotas[cls_name] = q
            allocated += q

        # Adjust Benign quota to match target n_samples exactly
        diff = allocated - n_samples
        if diff > 0 and class_quotas["Benign"] > diff + 50:
            class_quotas["Benign"] -= diff
        elif diff < 0:
            class_quotas["Benign"] += abs(diff)

        collected_dfs = {cls_name: [] for cls_name in class_quotas}
        counts_collected = {cls_name: 0 for cls_name in class_quotas}

        # Stream batches until quotas are satisfied (batch_size=10000 prevents Windows commit spikes)
        batch_size = 10000
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols_to_load):
            batch_df = batch.to_pandas()
            for cls_name, quota in class_quotas.items():
                needed = quota - counts_collected[cls_name]
                if needed <= 0:
                    continue
                match_mask = (batch_df["Attack"] == cls_name)
                matched_rows = batch_df[match_mask]
                if len(matched_rows) > 0:
                    take = matched_rows.iloc[:needed]
                    collected_dfs[cls_name].append(take)
                    counts_collected[cls_name] += len(take)

            # Check if all quotas satisfied
            if all(counts_collected[c] >= class_quotas[c] for c in class_quotas):
                break

        # Concatenate collected parts
        all_parts = []
        for cls_name, parts in collected_dfs.items():
            if parts:
                all_parts.append(pd.concat(parts, ignore_index=True))
        df = pd.concat(all_parts, ignore_index=True)
    else:
        # Full dataset load (warning: requires sufficient RAM)
        df = pf.read(columns=cols_to_load).to_pandas()

    # 1. Canonicalize placeholders
    df = LeakageFreePreprocessor.canonicalize_placeholders(df)

    # 2. Deduplicate before any splitting
    df = LeakageFreePreprocessor.deduplicate(df)

    # 3. Label encoding
    if binary:
        labels = df["Label"].to_numpy(dtype=np.int64)
        class_names = {0: "Benign", 1: "Attack"}
    else:
        # Multiclass: ensure Benign is index 0
        raw_attacks = df["Attack"].astype(str)
        unique_attacks = sorted(list(raw_attacks.unique()))
        ordered_classes = ["Benign"] + [a for a in unique_attacks if a != "Benign"]
        str_to_id = {cls_name: idx for idx, cls_name in enumerate(ordered_classes)}
        class_names = {idx: cls_name for cls_name, idx in str_to_id.items()}
        labels = raw_attacks.map(str_to_id).to_numpy(dtype=np.int64)

    # Feature matrix: drop label columns
    feat_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    df_features = df[feat_cols].astype(np.float32).copy()

    # Shuffle rows randomly using seed
    indices = np.arange(len(df_features))
    rng.shuffle(indices)
    if n_samples is not None and len(indices) > n_samples:
        indices = indices[:n_samples]
    df_features = df_features.iloc[indices].reset_index(drop=True)
    labels = labels[indices]

    # Calculate statistics & rare classes
    classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_counts = {class_names.get(int(c), str(c)): int(cnt) for c, cnt in zip(classes, counts)}

    KNOWN_RARE_NAMES = {"backdoor", "mitm", "ransomware"}
    rare_classes = [
        int(c) for c, cnt in zip(classes, counts)
        if int(c) != 0 and (
            class_names.get(int(c), str(c)).lower() in KNOWN_RARE_NAMES
            or (cnt / total_samples) < rare_threshold_ratio
        )
    ]
    rare_class_names = [class_names.get(c, str(c)) for c in rare_classes]

    metadata = {
        "dataset_name": "NF-ToN-IoT-V2",
        "n_samples": total_samples,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "num_classes": len(classes),
        "class_names": class_names,
        "class_counts": class_counts,
        "rare_classes": rare_classes,
        "rare_class_names": rare_class_names,
    }

    # Save to disk cache
    try:
        df_to_save = df_features.copy()
        df_to_save["_label"] = labels
        df_to_save.to_parquet(cache_file)
    except Exception:
        pass

    return df_features, labels, metadata
