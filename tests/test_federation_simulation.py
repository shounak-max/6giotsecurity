"""
Integration test for FederatedSimulator end-to-end multi-round execution.
"""

import numpy as np
import pytest

from fl_ids.data.synthetic import generate_synthetic_netflow, RARE_CLASSES
from fl_ids.data.preprocessor import LeakageFreePreprocessor
from fl_ids.data.partitioner import DirichletPartitioner
from fl_ids.attacks.backdoor import AdditiveBackdoorAttack
from fl_ids.defense.novel_defense import ClassAwareTrustDefense
from fl_ids.defense.fedavg import FedAvg
from fl_ids.federation.client import FederatedClient
from fl_ids.federation.server import FederatedServer
from fl_ids.federation.simulator import FederatedSimulator


def test_federated_simulation_pipeline():
    # 1. Generate synthetic dataset
    df, labels, meta = generate_synthetic_netflow(n_samples=1000, n_features=20, random_state=42)
    X = df.to_numpy()

    # 2. Extract disjoint root set
    X_clients, y_clients, X_root, y_root = LeakageFreePreprocessor.extract_disjoint_root_set(
        X, labels, root_size=100, random_state=42
    )

    # 3. Train-only scaling
    preproc = LeakageFreePreprocessor()
    X_clients_scaled = preproc.fit_transform(X_clients)
    X_root_scaled = preproc.transform(X_root)

    # Split clients into train and test
    n_train = 700
    X_train, y_train = X_clients_scaled[:n_train], y_clients[:n_train]
    X_test, y_test = X_clients_scaled[n_train:], y_clients[n_train:]

    # 4. Partition train data among 5 clients with Dirichlet alpha=0.3
    partitioner = DirichletPartitioner(num_clients=5, alpha=0.3, random_state=42)
    client_splits, p_meta = partitioner.partition(y_train, atypical_class_ids=RARE_CLASSES)

    # 5. Configure backdoor attack on client 0
    backdoor_attack = AdditiveBackdoorAttack(
        trigger_features=[0, 1],
        trigger_magnitude=5.0,
        target_class=0,
        poison_rate=0.4,
        boost_factor=1.5,
    )
    X_triggered_test = backdoor_attack.apply_trigger(X_test)

    # 6. Initialize clients
    clients = []
    for cid in range(5):
        c_idxs = client_splits[cid]
        c_atk = backdoor_attack if cid == 0 else None
        client = FederatedClient(
            client_id=cid,
            X_train=X_train[c_idxs],
            y_train=y_train[c_idxs],
            batch_size=32,
            local_epochs=1,
            learning_rate=0.01,
            attack=c_atk,
        )
        clients.append(client)

    # 7. Initialize server with ClassAwareTrustDefense
    defense = ClassAwareTrustDefense(client_clip_norm=5.0, min_trust_threshold=0.05)
    server = FederatedServer(
        input_dim=20,
        num_classes=10,
        aggregator=defense,
        X_root=X_root_scaled,
        y_root=y_root,
    )

    # 8. Run simulator for 2 rounds
    simulator = FederatedSimulator(
        server=server,
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        X_triggered_test=X_triggered_test,
        atypical_client_ids=p_meta["atypical_clients"],
        rare_class_ids=RARE_CLASSES,
        backdoor_target_class=0,
    )

    history = simulator.run(num_rounds=2, verbose=False)

    assert len(history) == 2
    assert "macro_f1" in history[0]
    assert "backdoor_asr" in history[0]
    assert "atypical_acceptance_rate" in history[0]
    assert history[1]["macro_f1"] >= 0.0
