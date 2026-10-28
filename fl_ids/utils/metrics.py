"""
Evaluation Metrics for Federated Intrusion Detection.

Calculates:
- Utility: Macro-F1, Balanced Accuracy, Standard Accuracy, FPR
- Rare-Class Recall: Per-class recall specifically tracked for minority classes (Worms, SQL Injection, Fingerprinting)
- Robustness: Backdoor Attack Success Rate (ASR), Atypical Client Acceptance Rate
"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_ids_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rare_class_ids: Optional[List[int]] = None,
    benign_class_id: int = 0,
) -> Dict[str, float]:
    """
    Computes comprehensive IDS utility and per-class minority recall metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class recall
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # False Positive Rate (Benign misclassified as any attack)
    benign_mask = (y_true == benign_class_id)
    if np.any(benign_mask):
        fp = np.sum((y_pred != benign_class_id) & benign_mask)
        tn = np.sum((y_pred == benign_class_id) & benign_mask)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "fpr": fpr,
    }

    # Record rare class recall
    if rare_class_ids is not None:
        rare_recalls = []
        for rc in rare_class_ids:
            if rc < len(recalls):
                rc_recall = float(recalls[rc])
                metrics[f"recall_class_{rc}"] = rc_recall
                rare_recalls.append(rc_recall)
        if len(rare_recalls) > 0:
            metrics["rare_class_mean_recall"] = float(np.mean(rare_recalls))

    return metrics


def compute_backdoor_asr(
    y_pred_triggered: np.ndarray,
    target_class: int = 0,
    y_true: Optional[np.ndarray] = None,
) -> float:
    """
    Attack Success Rate (ASR): Percentage of triggered non-target test samples
    that are successfully tricked into being classified as the target class.
    """
    if len(y_pred_triggered) == 0:
        return 0.0
    if y_true is not None:
        non_target_mask = (y_true != target_class)
        if np.sum(non_target_mask) == 0:
            return 0.0
        success = np.sum((y_pred_triggered == target_class) & non_target_mask)
        return float(success / np.sum(non_target_mask))
    success = np.sum(y_pred_triggered == target_class)
    return float(success / len(y_pred_triggered))


def compute_atypical_acceptance_rate(
    accepted_client_ids: List[int],
    atypical_client_ids: List[int],
) -> float:
    """
    Computes the percentage of atypical / rare-class carrying clients
    that were successfully included in the robust aggregate.
    """
    if not atypical_client_ids:
        return 1.0
    accepted_atypical = set(accepted_client_ids).intersection(set(atypical_client_ids))
    return float(len(accepted_atypical) / len(atypical_client_ids))
