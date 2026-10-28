"""
Edge Cost and Hardware Profiler for 6G Gateway Deployments.

Measures:
- Inference latency (median and p95 ms)
- Model parameter count and footprint in memory (MB)
- Simulated uplink transmission latency across link speeds (e.g. 50 kbps LoRaWAN, 1 Mbps IoT, 100 Mbps 6G Edge)
- Uplink data volume per round under sketch compression
"""

from typing import Dict, List, Optional
import time
import numpy as np
import torch
import torch.nn as nn


class EdgeProfiler:
    """Profiles latency, memory, and communication overhead for edge IDS."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def count_parameters(self) -> Dict[str, int]:
        """Counts total and trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}

    def measure_inference_latency(
        self,
        input_dim: int = 40,
        batch_size: int = 1,
        warmup_runs: int = 50,
        test_runs: int = 200,
    ) -> Dict[str, float]:
        """
        Measures single-inference and batch inference latency on edge hardware.
        Returns median and p95 latency in milliseconds.
        """
        self.model.eval()
        dummy_input = torch.randn(batch_size, input_dim, dtype=torch.float32).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(test_runs):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)  # ms

        latencies = np.array(latencies)
        return {
            "batch_size": batch_size,
            "latency_median_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_std_ms": float(np.std(latencies)),
        }

    @staticmethod
    def estimate_uplink_transmission_time(
        payload_bytes: int,
        link_speed_kbps: float = 1000.0,
    ) -> float:
        """
        Estimates transmission time in seconds over a constrained uplink channel.
        Example link speeds:
        - 50 kbps: LoRaWAN / constrained edge sensor
        - 1000 kbps (1 Mbps): standard NB-IoT / LTE-M
        - 50000 kbps (50 Mbps): 5G/6G sub-carrier edge channel
        """
        bits = payload_bytes * 8.0
        kb = bits / 1000.0
        return float(kb / link_speed_kbps)

    @staticmethod
    def compute_bandwidth_savings(
        full_precision_bytes: int,
        compressed_bytes: int,
    ) -> Dict[str, float]:
        """Calculates compression factor and bandwidth reduction ratio."""
        factor = float(full_precision_bytes / max(1, compressed_bytes))
        reduction_pct = float((1.0 - (compressed_bytes / full_precision_bytes)) * 100.0)
        return {
            "full_precision_kb": full_precision_bytes / 1024.0,
            "compressed_kb": compressed_bytes / 1024.0,
            "compression_factor_x": factor,
            "bandwidth_reduction_pct": reduction_pct,
        }
