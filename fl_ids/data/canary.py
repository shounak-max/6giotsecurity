"""
Naive Bayes Leakage Canary.

Audits dataset splits and preprocessing for target leakage, serialisation artifacts,
and illicit train-test correlation. As identified in literature audits (Edge-IIoTset, 2608.15761),
leaking serialisation artifacts or global encodings inflate weak classifiers to 1.0000.
"""

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, f1_score


class NaiveBayesCanary:
    """
    Leakage detection canary using a baseline Gaussian Naive Bayes classifier.
    If a weak, non-linear-free Naive Bayes model scores abnormally high on multi-class
    IDS datasets (above threshold, e.g. balanced accuracy > 0.98 on complex datasets),
    an alert is raised.
    """

    def __init__(
        self,
        leakage_threshold_balanced_acc: float = 0.98,
        leakage_threshold_f1: float = 0.98,
    ):
        self.leakage_threshold_balanced_acc = leakage_threshold_balanced_acc
        self.leakage_threshold_f1 = leakage_threshold_f1
        self.model = GaussianNB()

    def audit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Fits a Gaussian Naive Bayes model on X_train and evaluates on X_test.
        Returns metrics and a boolean flag indicating suspected leakage.
        """
        # Ensure clean inputs
        X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.fit(X_train_clean, y_train)
        y_pred = self.model.predict(X_test_clean)

        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        is_suspicious = (
            bal_acc >= self.leakage_threshold_balanced_acc
            or macro_f1 >= self.leakage_threshold_f1
        )

        msg = (
            "LEAKAGE ALERT: Weak Naive Bayes canary achieved suspiciously high performance "
            f"(Balanced Acc: {bal_acc:.4f}, Macro-F1: {macro_f1:.4f}). Check for serialisation artifacts or split overlap!"
            if is_suspicious
            else f"Canary Passed: Balanced Acc: {bal_acc:.4f}, Macro-F1: {macro_f1:.4f} (within realistic bounds)."
        )

        return {
            "canary_balanced_acc": bal_acc,
            "canary_macro_f1": macro_f1,
            "is_leakage_suspected": is_suspicious,
            "message": msg,
        }
