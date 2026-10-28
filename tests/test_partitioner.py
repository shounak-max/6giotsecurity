"""
Unit tests for DirichletPartitioner and non-IID partitioning.
"""

import numpy as np
import pytest

from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.data.synthetic import generate_synthetic_netflow, RARE_CLASSES


def test_dirichlet_partitioner_sample_conservation():
    df, labels, _ = generate_synthetic_netflow(n_samples=1000, random_state=42)
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.3, random_state=42)

    client_dict, meta = partitioner.partition(labels, atypical_class_ids=RARE_CLASSES)

    assert len(client_dict) == 10
    total_allocated = sum(len(idxs) for idxs in client_dict.values())
    assert total_allocated == 1000
    assert meta["total_samples_allocated"] == 1000


def test_dirichlet_alpha_heterogeneity_sweep():
    df, labels, _ = generate_synthetic_netflow(n_samples=1200, random_state=42)

    # Sweep alpha in {0.1, 0.3, 1.0} as specified in requirements
    for alpha in [0.1, 0.3, 1.0]:
        partitioner = DirichletPartitioner(num_clients=5, alpha=alpha, random_state=42)
        client_dict, meta = partitioner.partition(labels, atypical_class_ids=RARE_CLASSES)
        assert len(client_dict) == 5
        assert meta["alpha"] == alpha
        assert "atypical_clients" in meta


def test_atypical_client_tracking():
    df, labels, _ = generate_synthetic_netflow(n_samples=1000, random_state=42)
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.1, random_state=42)

    client_dict, meta = partitioner.partition(labels, atypical_class_ids=RARE_CLASSES)

    atypical = meta["atypical_clients"]
    assert isinstance(atypical, list)
    # Check that any atypical client indeed has rare classes
    matrix = np.array(meta["client_class_matrix"])
    for cid in atypical:
        has_rare = any(matrix[cid, rc] > 0 for rc in RARE_CLASSES)
        assert has_rare
