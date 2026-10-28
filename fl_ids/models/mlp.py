"""
Edge-Deployable Tabular MLP for Network Flow Intrusion Detection.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class EdgeIDSMLP(nn.Module):
    """
    Lightweight, gateway-deployable Multi-Layer Perceptron for flow classification.
    Suitable for edge devices (e.g., Raspberry Pi 4 / 6G edge gateway).
    """

    def __init__(
        self,
        input_dim: int = 40,
        num_classes: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate layer representation for class-aware representation analysis."""
        return self.feature_extractor(x)

    def get_weights_dict(self) -> Dict[str, np.ndarray]:
        """Returns model state dict as a dictionary of numpy arrays."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}

    def set_weights_dict(self, weights: Dict[str, np.ndarray]):
        """Loads weights from a dictionary of numpy arrays."""
        state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}
        self.load_state_dict(state_dict)

    def get_flattened_params(self) -> np.ndarray:
        """Returns all trainable model parameters as a 1D numpy array."""
        params = []
        for p in self.parameters():
            params.append(p.detach().cpu().numpy().flatten())
        return np.concatenate(params)
