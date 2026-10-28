"""
Federated Simulation Orchestrator.
Runs multi-round federated experiments with telemetry and metrics tracking.
"""

from typing import Any, Dict, List, Optional
import time
import numpy as np

from fl_ids.federation.client import FederatedClient
from fl_ids.federation.server import FederatedServer
from fl_ids.utils.metrics import compute_atypical_acceptance_rate


class FederatedSimulator:
    """Orchestrates rounds of federated training, aggregation, and evaluation."""

    def __init__(
        self,
        server: FederatedServer,
        clients: List[FederatedClient],
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_triggered_test: Optional[np.ndarray] = None,
        atypical_client_ids: Optional[List[int]] = None,
        rare_class_ids: Optional[List[int]] = None,
        backdoor_target_class: int = 0,
    ):
        self.server = server
        self.clients = clients
        self.X_test = X_test
        self.y_test = y_test
        self.X_triggered_test = X_triggered_test
        self.atypical_client_ids = atypical_client_ids if atypical_client_ids is not None else []
        self.rare_class_ids = rare_class_ids
        self.backdoor_target_class = backdoor_target_class

        self.history: List[Dict[str, Any]] = []

    def run(self, num_rounds: int = 5, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Executes num_rounds of federated training.
        """
        if verbose:
            print(f"Starting Federated Simulation: {num_rounds} rounds, {len(self.clients)} clients.")

        for r in range(1, num_rounds + 1):
            round_start = time.time()

            # 1. Local client training
            client_updates = {}
            client_sample_counts = {}
            local_telemetry = []

            for client in self.clients:
                weights, tel = client.train(self.server.global_weights)
                client_updates[client.client_id] = weights
                client_sample_counts[client.client_id] = client.num_samples
                local_telemetry.append(tel)

            # 2. Server aggregation
            agg_meta = self.server.aggregate_round(
                client_updates=client_updates,
                client_sample_counts=client_sample_counts,
            )

            # 3. Global model evaluation
            test_metrics = self.server.evaluate(
                X_test=self.X_test,
                y_test=self.y_test,
                rare_class_ids=self.rare_class_ids,
                X_triggered=self.X_triggered_test,
                target_class=self.backdoor_target_class,
            )

            # 4. Atypical client acceptance telemetry
            accepted_clients = agg_meta.get("accepted_clients", list(client_updates.keys()))
            atypical_acc_rate = compute_atypical_acceptance_rate(
                accepted_client_ids=accepted_clients,
                atypical_client_ids=self.atypical_client_ids,
            )

            round_duration = time.time() - round_start

            round_record = {
                "round": r,
                "duration_sec": float(round_duration),
                "macro_f1": test_metrics.get("macro_f1", 0.0),
                "balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
                "accuracy": test_metrics.get("accuracy", 0.0),
                "fpr": test_metrics.get("fpr", 0.0),
                "backdoor_asr": test_metrics.get("backdoor_asr", 0.0),
                "atypical_acceptance_rate": atypical_acc_rate,
                "rare_class_mean_recall": test_metrics.get("rare_class_mean_recall", 0.0),
                "num_accepted": len(accepted_clients),
                "num_rejected": len(agg_meta.get("rejected_clients", [])),
            }

            # Add individual rare class recall if present
            for k, v in test_metrics.items():
                if k.startswith("recall_class_"):
                    round_record[k] = v

            self.history.append(round_record)

            if verbose:
                asr_str = f", Backdoor ASR: {round_record['backdoor_asr']:.4f}" if self.X_triggered_test is not None else ""
                print(
                    f"Round {r:02d}/{num_rounds:02d} [{round_duration:.2f}s] - "
                    f"Macro-F1: {round_record['macro_f1']:.4f}, "
                    f"Bal Acc: {round_record['balanced_accuracy']:.4f}, "
                    f"Rare Recall: {round_record['rare_class_mean_recall']:.4f}, "
                    f"Atypical Acc%: {atypical_acc_rate*100:.1f}%"
                    f"{asr_str}"
                )

        return self.history
