import json

with open("results/causal_attribution_benchmark.json") as f:
    data = json.load(f)

print("=== CAUSAL BENCHMARK EXTRACTED METRICS ===")
for k, v in data["causal_conditions"].items():
    print(f"{k:<30} | RareRecall: {v['rare_recall_mean']:.4f} +/- {v['rare_recall_std']:.4f} | AtypicalAcc: {v['atypical_acc_mean']*100:5.1f}% | MacroF1: {v['macro_f1_mean']:.4f} | ASR: {v['asr_mean']:.4f} | NullASR: {v['null_asr_mean']:.4f}")

print("\n=== ALPHA SWEEP ===")
for a, v in data["alpha_sweep"].items():
    print(f"Alpha {a:<4} | FedAvg: {v['fedavg_recall']:.4f} | MultiKrum: {v['krum_recall']:.4f} | Penalty: {v['penalty_mean']:+.4f} +/- {v['penalty_std']:.4f}")

# Mediation Ratio Calculation
# Natural vs Forced-Include on Multi-Krum and FLTrust
# Mediation = (Natural - Forced-Include) / (Natural - FedAvg_ref)
fed_ref = data["causal_conditions"]["fedavg_clean_ref"]["rare_recall_mean"]
mk_nat = data["causal_conditions"]["multikrum_natural"]["rare_recall_mean"]
mk_inc = data["causal_conditions"]["multikrum_forced_inclusion"]["rare_recall_mean"]

fl_nat = data["causal_conditions"]["fltrust_natural"]["rare_recall_mean"]
fl_inc = data["causal_conditions"]["fltrust_forced_inclusion"]["rare_recall_mean"]

print("\n=== MEDIATION ATTRIBUTION RATIOS ===")
print(f"FedAvg Clean Reference Recall: {fed_ref:.4f}")
print(f"Multi-Krum Natural Recall:     {mk_nat:.4f} (Suppression Penalty = {fed_ref - mk_nat:+.4f})")
print(f"Multi-Krum Forced-Include:     {mk_inc:.4f}")
print(f"FLTrust Natural Recall:        {fl_nat:.4f} (Atypical Acceptance = {data['causal_conditions']['fltrust_natural']['atypical_acc_mean']*100:.1f}%)")
print(f"FLTrust Forced-Include Recall: {fl_inc:.4f} (Atypical Acceptance = 100.0%)")
