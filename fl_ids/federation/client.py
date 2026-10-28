"""
Federated Client Representation.
Manages local dataset, local training, optional poisoning, and sketch compression.
"""

from typing import Dict, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_ids.models.mlp import EdgeIDSMLP
from fl_ids.attacks.base import BaseAttack
from fl_ids.defense.compression import SketchCompressor


class FederatedClient:
    """Simulates a 6G edge device / client node in the FL-IDS system."""

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32,
        local_epochs: int = 2,
        learning_rate: float = 0.005,
        attack: Optional[BaseAttack] = None,
        compressor: Optional[SketchCompressor] = None,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.attack = attack
        self.compressor = compressor
        self.device = torch.device(device)

        # Poison data if under data-poisoning attack (e.g. Backdoor, LabelFlip)
        if self.attack is not None:
            self.X_train, self.y_train = self.attack.poison_data(X_train, y_train, self.client_id)
        else:
            self.X_train, self.y_train = X_train, y_train

        self.num_samples = len(self.X_train)

    def train(
        self,
        global_weights: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        """
        Executes local SGD training on client's partition.
        Returns:
            updated_weights: Dictionary of numpy weights (poisoned/compressed if configured).
            telemetry: Dict of local training metrics.
        """
        # Instantiate model and load current global weights
        input_dim = self.X_train.shape[1]
        num_classes = 10
        model = EdgeIDSMLP(input_dim=input_dim, num_classes=num_classes).to(self.device)
        model.set_weights_dict(global_weights)
        model.train()

        # Build local DataLoader
        tensor_x = torch.tensor(self.X_train, dtype=torch.float32)
        tensor_y = torch.tensor(self.y_train, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        drop_last = (len(dataset) > self.batch_size and len(dataset) % self.batch_size == 1)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, max(1, len(dataset))), shuffle=True, drop_last=drop_last)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        total_loss = 0.0
        n_batches = 0
        for _ in range(self.local_epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        local_weights = model.get_weights_dict()

        # Apply model-poisoning attack if configured
        if self.attack is not None:
            final_weights = self.attack.poison_update(local_weights, global_weights, self.client_id)
        else:
            final_weights = local_weights

        telemetry = {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "avg_loss": float(total_loss / max(1, n_batches)),
            "is_malicious": self.attack is not None,
        }

        # Apply sketch compression if configured
        if self.compressor is not None:
            # Flatten delta and compress
            flat_delta = []
            for k in global_weights.keys():
                if np.issubdtype(global_weights[k].dtype, np.floating):
                    flat_delta.append((final_weights[k] - global_weights[k]).flatten())
            full_delta = np.concatenate(flat_delta)
            comp_vals, comp_meta = self.compressor.compress_vector(full_delta)
            telemetry["compression"] = comp_meta

        return final_weights, telemetry
