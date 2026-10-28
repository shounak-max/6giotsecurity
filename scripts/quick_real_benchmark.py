import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scripts.run_rigorous_matrix import run_single

def main():
    defenses = ['fedavg', 'multikrum', 'trimmed_mean', 'fltrust', 'class_aware_trust']
    attacks = ['none', 'backdoor']
    results = {}

    for d in defenses:
        for a in attacks:
            key = f"{d}_{a}"
            print(f"Running {key} on NF-ToN-IoT-V2...")
            res = run_single(d, a, seed=42, dataset="NF-ToN-IoT-V2")
            results[key] = {
                "macro_f1": round(float(res["macro_f1"]), 4),
                "bal_acc": round(float(res["balanced_accuracy"]), 4),
                "rare_recall": round(float(res["rare_class_mean_recall"]), 4),
                "atypical_acc": round(float(res["atypical_acceptance_rate"]), 4),
                "asr": round(float(res.get("backdoor_asr", 0.0)), 4),
                "n_test": res.get("n_test_samples", 0),
                "n_triggered": res.get("n_triggered_samples", 0),
            }
            print(f"  {key} -> F1: {results[key]['macro_f1']}, BalAcc: {results[key]['bal_acc']}, RareRecall: {results[key]['rare_recall']}, AtypicalAcc: {results[key]['atypical_acc']}, ASR: {results[key]['asr']}")

    with open("results/real_dataset_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/real_dataset_comparison.json")

if __name__ == "__main__":
    main()
