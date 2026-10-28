"""
Federated Server representation.
Manages global model state, server-side root gradient evaluation,
robust aggregation, and global test set evaluation (Macro-F1, ASR).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_ids.models.mlp import EdgeIDSMLP
from fl_ids.defense.base import BaseAggregator
from fl_ids.utils.metrics import compute_ids_metrics, compute_backdoor_asr


class FederatedServer:
    """Central aggregating server for federated intrusion detection."""

    def __init__(
        self,
        input_dim: int = 40,
        num_classes: int = 10,
        aggregator: Optional[BaseAggregator] = None,
        X_root: Optional[np.ndarray] = None,
        y_root: Optional[np.ndarray] = None,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.aggregator = aggregator
        self.X_root = X_root
        self.y_root = y_root
        self.device = torch.device(device)

        self.global_model = EdgeIDSMLP(input_dim=input_dim, num_classes=num_classes).to(self.device)
        self.global_weights = self.global_model.get_weights_dict()

    def compute_root_gradients(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[int, np.ndarray]]]:
        """
        Computes clean reference gradient and class-conditional sub-gradients on the disjoint server root set.
        """
        if self.X_root is None or self.y_root is None or len(self.X_root) == 0:
            return None, None

        self.global_model.eval()
        tensor_x = torch.tensor(self.X_root, dtype=torch.float32).to(self.device)
        tensor_y = torch.tensor(self.y_root, dtype=torch.long).to(self.device)

        # 1. Overall root gradient
        self.global_model.zero_grad()
        logits = self.global_model(tensor_x)
        loss = nn.CrossEntropyLoss()(logits, tensor_y)
        loss.backward()

        root_grad = {}
        for name, param in self.global_model.named_parameters():
            if param.grad is not None:
                # Delta direction is negative gradient step: Delta = -lr * grad
                root_grad[name] = -param.grad.detach().cpu().numpy().copy()
            else:
                root_grad[name] = np.zeros_like(param.detach().cpu().numpy())

        # 2. Class-conditional sub-gradients
        class_conditional_grads = {}
        classes = np.unique(self.y_root)
        for c in classes:
            c_mask = (tensor_y == c)
            if torch.sum(c_mask) == 0:
                continue
            self.global_model.zero_grad()
            c_logits = self.global_model(tensor_x[c_mask])
            c_loss = nn.CrossEntropyLoss()(c_logits, tensor_y[c_mask])
            c_loss.backward()

            c_grads_flat = []
            for name, param in self.global_model.named_parameters():
                if param.grad is not None:
                    c_grads_flat.append(-param.grad.detach().cpu().numpy().flatten())
            class_conditional_grads[int(c)] = np.concatenate(c_grads_flat)

        return root_grad, class_conditional_grads

    def aggregate_round(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        client_sample_counts: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """Runs the configured robust aggregator on client updates."""
        root_grad, class_grads = self.compute_root_gradients()

        # Check if aggregator supports class-conditional gradients
        kwargs = {
            "client_updates": client_updates,
            "global_weights": self.global_weights,
            "client_sample_counts": client_sample_counts,
            "server_root_grad": root_grad,
            "server_root_data": (self.X_root, self.y_root) if self.X_root is not None else None,
        }
        if hasattr(self.aggregator, "aggregate"):
            import inspect
            sig = inspect.signature(self.aggregator.aggregate)
            if "class_conditional_grads" in sig.parameters:
                kwargs["class_conditional_grads"] = class_grads

            new_weights, meta = self.aggregator.aggregate(**kwargs)
        else:
            new_weights = self.global_weights
            meta = {}

        self.global_weights = new_weights
        self.global_model.set_weights_dict(new_weights)
        return meta

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rare_class_ids: Optional[List[int]] = None,
        X_triggered: Optional[np.ndarray] = None,
        target_class: int = 0,
    ) -> Dict[str, float]:
        """Evaluates global model on clean test data and triggered backdoor test data."""
        self.global_model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_test).float().to(self.device)
            logits = self.global_model(tensor_x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            del tensor_x, logits

        metrics = compute_ids_metrics(y_true=y_test, y_pred=preds, rare_class_ids=rare_class_ids)

        if X_triggered is not None:
            with torch.no_grad():
                trig_x = torch.from_numpy(X_triggered).float().to(self.device)
                trig_logits = self.global_model(trig_x)
                trig_preds = torch.argmax(trig_logits, dim=-1).cpu().numpy()
                del trig_x, trig_logits
            metrics["backdoor_asr"] = compute_backdoor_asr(trig_preds, target_class=target_class, y_true=y_test)

        return metrics
