"""
Multi-Method Benchmark Matrix Runner.

Evaluates baselines vs ClassAwareTrust under:
- Clean baseline
- Boosted Backdoor attack
- Defense-aware Adaptive attack
Generates a markdown table comparing Utility, Backdoor ASR, Rare Class Recall, and Atypical Acceptance %.
"""

import sys
import subprocess
import json
import tempfile
import os

METHODS = ["fedavg", "multikrum", "fltrust", "class_aware_trust"]
ATTACKS = [
    ("clean", "none"),
    ("backdoor", "backdoor"),
    ("adaptive", "adaptive"),
]


def run_benchmark_matrix():
    python_exe = sys.executable
    results = []

    print("Running Federated IDS Benchmark Matrix...")
    print("-" * 80)

    for method in METHODS:
        for attack_label, attack_type in ATTACKS:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                python_exe,
                "scripts/run_experiment.py",
                "--defense", method,
                "--attack", attack_type,
                "--clients", "10",
                "--malicious_clients", "2" if attack_type != "none" else "0",
                "--rounds", "3",
                "--samples", "2000",
                "--features", "30",
                "--alpha", "0.3",
                "--output", tmp_path,
            ]

            print(f"-> Evaluating [{method}] under [{attack_label}] attack...")
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"Error running {method} with {attack_label}:", ret.stderr)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                continue

            with open(tmp_path, "r") as f:
                data = json.load(f)

            os.remove(tmp_path)

            fm = data["final_metrics"]
            lat = data["latency_metrics"]

            results.append({
                "Defense": method,
                "Attack": attack_label,
                "Macro-F1": fm["macro_f1"],
                "Bal Acc": fm["balanced_accuracy"],
                "Rare Recall": fm["rare_class_mean_recall"],
                "Atypical Acc %": fm["atypical_acceptance_rate"] * 100.0,
                "Backdoor ASR": fm.get("backdoor_asr", 0.0) if attack_type != "none" else 0.0,
                "Latency (ms)": lat["latency_median_ms"],
            })

    # Print Markdown Table
    print("\n" + "=" * 95)
    print("BENCHMARK MATRIX RESULTS")
    print("=" * 95)
    header = "| Defense | Threat Model | Macro-F1 | Bal Acc | Rare Recall | Atypical Acc % | Backdoor ASR | Latency (ms) |"
    sep = "|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in results:
        asr_str = f"{r['Backdoor ASR']:.4f}" if r["Attack"] != "clean" else "N/A"
        print(
            f"| {r['Defense']} | {r['Attack']} | {r['Macro-F1']:.4f} | {r['Bal Acc']:.4f} | "
            f"{r['Rare Recall']:.4f} | {r['Atypical Acc %']:.1f}% | {asr_str} | {r['Latency (ms)']:.3f} |"
        )
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark_matrix()
