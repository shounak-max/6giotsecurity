"""
Leakage-Free Preprocessor for Network Intrusion Detection Datasets.

Enforces:
1. Canonicalization of placeholders (NaN, Inf, -Inf, string representations of nulls).
2. Deduplication before splitting.
3. Strict fit-on-train only (test and server root splits are transformed with train statistics).
4. Stratified / disjoint extraction of the server root set.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


class LeakageFreePreprocessor:
    """Preprocesses tabular network intrusion detection flows without data leakage."""

    def __init__(
        self,
        scaler_type: str = "standard",
        clip_outliers: bool = True,
        clip_quantile: float = 0.999,
        random_state: int = 42,
    ):
        """
        Args:
            scaler_type: 'standard' (StandardScaler) or 'robust' (RobustScaler).
            clip_outliers: Whether to clip extreme positive outliers based on train quantile.
            clip_quantile: Upper quantile for clipping if clip_outliers is True.
            random_state: Random seed for reproducibility.
        """
        self.scaler_type = scaler_type
        self.clip_outliers = clip_outliers
        self.clip_quantile = clip_quantile
        self.random_state = random_state

        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.clip_thresholds: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None
        self.fitted: bool = False

    @staticmethod
    def canonicalize_placeholders(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces infinity, -infinity, and placeholder string representations with NaN,
        then replaces NaNs with 0.0 or column median.
        """
        df = df.copy()
        # Replace string representations of missing values
        placeholder_strings = ["?", "Infinity", "inf", "-Infinity", "-inf", "NaN", "nan", "null", "None"]
        df.replace(placeholder_strings, np.nan, inplace=True)

        # Replace numpy Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill NaNs with 0.0
        df.fillna(0.0, inplace=True)
        return df

    @staticmethod
    def deduplicate(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """Removes duplicate rows before any dataset splitting."""
        before = len(df)
        df_dedup = df.drop_duplicates(subset=subset).reset_index(drop=True)
        after = len(df_dedup)
        return df_dedup

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray]) -> "LeakageFreePreprocessor":
        """
        Fits scaler strictly on train split data only.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_arr = X_train.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_train, dtype=np.float32)

        # Canonicalize if array has any inf or nan
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        if self.clip_outliers:
            # Compute upper quantile per feature strictly on train
            self.clip_thresholds = np.quantile(X_arr, self.clip_quantile, axis=0)
            X_arr = np.minimum(X_arr, self.clip_thresholds)

        self.scaler.fit(X_arr)
        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transforms data using pre-fitted train statistics.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        if self.clip_outliers and self.clip_thresholds is not None:
            X_arr = np.minimum(X_arr, self.clip_thresholds)

        return self.scaler.transform(X_arr).astype(np.float32)

    def fit_transform(self, X_train: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fits strictly on train and transforms train."""
        return self.fit(X_train).transform(X_train)

    @staticmethod
    def extract_disjoint_root_set(
        X: np.ndarray,
        y: np.ndarray,
        root_size: int = 100,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts a disjoint, clean server root set for trust-based defense filtering (FLTrust, RepGuard, etc.).
        Returns:
            X_clients, y_clients, X_root, y_root
        """
        rng = np.random.RandomState(random_state)
        n_samples = len(X)
        if root_size >= n_samples:
            raise ValueError(f"root_size ({root_size}) must be strictly less than n_samples ({n_samples})")

        # Stratified sampling if possible
        classes, class_counts = np.unique(y, return_counts=True)
        root_indices = []

        samples_per_class = max(1, root_size // len(classes))
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            selected = rng.choice(
                cls_indices,
                size=min(len(cls_indices), samples_per_class),
                replace=False,
            )
            root_indices.extend(selected.tolist())

        # If we need more samples to reach root_size, randomly pick from remaining
        remaining_indices = list(set(range(n_samples)) - set(root_indices))
        needed = root_size - len(root_indices)
        if needed > 0 and len(remaining_indices) >= needed:
            extra = rng.choice(remaining_indices, size=needed, replace=False)
            root_indices.extend(extra.tolist())

        root_indices = np.array(root_indices)
        client_mask = np.ones(n_samples, dtype=bool)
        client_mask[root_indices] = False

        X_root, y_root = X[root_indices], y[root_indices]
        X_clients, y_clients = X[client_mask], y[client_mask]

        return X_clients, y_clients, X_root, y_root
