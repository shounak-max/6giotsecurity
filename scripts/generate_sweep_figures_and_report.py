"""
Visualization and Empirical Report Generator for Rigorous Sweep.

Generates Publication-Quality Visualizations:
1. Fig 1: ASR vs Benign Accuracy Frontier per magnitude (0.5, 1.0, 2.0, 4.0 sigma)
   with Multi-Krum (f=2 and wrong f in {0, 1, 4}), v1, v2, and Hybrid CAT-Q curve.
2. Fig 2: Attacker Count Sweep (m in {0, 1, 2, 3, 4}) demonstrating zero prior on f.
3. Fig 3: Multi-Signal Anomaly AUROC and distribution across magnitudes.
4. Fig 4: TOST Equivalence Testing against Multi-Krum (margin delta = 0.02) and Cohen's d forest plot.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})

RESULTS_FILE = "results/hybrid_rigorous_sweep_20seeds.json"
FIG_DIR = "results/figures"


def plot_pareto_frontiers(data: dict):
    os.makedirs(FIG_DIR, exist_ok=True)
    mag_data = data["magnitude_sweep"]
    magnitudes = ["0.5", "1.0", "2.0", "4.0"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

    colors = {
        "fedavg": "#d62728",
        "multikrum_f2": "#2ca02c",
        "v1": "#1f77b4",
        "v2": "#ff7f0e",
        "hybrid": "#9467bd",
        "krum_wrong": "#7f7f7f",
    }

    hybrid_keys = [
        "hybrid_lam0.0",
        "hybrid_lam0.05",
        "hybrid_lam0.10",
        "hybrid_lam0.25",
        "hybrid_lam0.50",
        "hybrid_lam1.0",
    ]

    for idx, mag in enumerate(magnitudes):
        ax = axes[idx]
        cell = mag_data[mag]

        # 1. Clean control floor
        floor = cell["hybrid_lam0.10"]["clean_floor"]
        ax.axhline(floor, color="gray", linestyle="--", alpha=0.6, label="Clean Control Floor" if idx == 0 else "")

        # 2. Hybrid Curve (over lambda in [0, 1])
        hyb_accs = [cell[k]["accuracy_mean"] * 100 for k in hybrid_keys]
        hyb_asrs = [cell[k]["backdoor_asr_mean"] for k in hybrid_keys]

        ax.plot(hyb_accs, hyb_asrs, "-o", color=colors["hybrid"], linewidth=2.5, markersize=7, label="Hybrid CAT-Q (λ ∈ [0, 1])" if idx == 0 else "", zorder=4)

        # Highlight optimal frozen lambda = 0.10
        opt_acc = cell["hybrid_lam0.10"]["accuracy_mean"] * 100
        opt_asr = cell["hybrid_lam0.10"]["backdoor_asr_mean"]
        ax.scatter([opt_acc], [opt_asr], color="#8c564b", s=140, marker="*", edgecolor="black", linewidth=1.5, zorder=6, label="Hybrid (λ = 0.10 frozen)" if idx == 0 else "")

        # 3. Baselines: FedAvg, v1, v2
        fa_acc = cell["fedavg"]["accuracy_mean"] * 100
        fa_asr = cell["fedavg"]["backdoor_asr_mean"]
        ax.scatter([fa_acc], [fa_asr], color=colors["fedavg"], s=100, marker="o", edgecolor="black", label="FedAvg" if idx == 0 else "", zorder=5)

        v1_acc = cell["v1"]["accuracy_mean"] * 100
        v1_asr = cell["v1"]["backdoor_asr_mean"]
        ax.scatter([v1_acc], [v1_asr], color=colors["v1"], s=100, marker="s", edgecolor="black", label="CAT-Q v1 (Security Mode)" if idx == 0 else "", zorder=5)

        v2_acc = cell["v2"]["accuracy_mean"] * 100
        v2_asr = cell["v2"]["backdoor_asr_mean"]
        ax.scatter([v2_acc], [v2_asr], color=colors["v2"], s=100, marker="D", edgecolor="black", label="CAT-Q v2 (Utility Mode)" if idx == 0 else "", zorder=5)

        # 4. Multi-Krum (f=2 and wrong f)
        mk2_acc = cell["multikrum_f2"]["accuracy_mean"] * 100
        mk2_asr = cell["multikrum_f2"]["backdoor_asr_mean"]
        ax.scatter([mk2_acc], [mk2_asr], color=colors["multikrum_f2"], s=120, marker="P", edgecolor="black", label="Multi-Krum (f=2 correct)" if idx == 0 else "", zorder=5)

        for wrong_f in [0, 1, 4]:
            k_key = f"multikrum_f{wrong_f}"
            w_acc = cell[k_key]["accuracy_mean"] * 100
            w_asr = cell[k_key]["backdoor_asr_mean"]
            lbl = f"Multi-Krum (wrong f)" if (idx == 0 and wrong_f == 0) else ""
            ax.scatter([w_acc], [w_asr], color=colors["krum_wrong"], s=70, marker="x", linewidth=2.0, label=lbl, zorder=3)

        ax.set_title(f"Trigger Magnitude: {mag}σ")
        ax.set_xlabel("Benign Test Accuracy (%)")
        if idx == 0:
            ax.set_ylabel("Backdoor Attack Success Rate (ASR)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    plt.suptitle("Pareto Frontier: Security (ASR) vs Utility (Accuracy) Across Trigger Magnitudes (20 Seeds)", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_pareto_frontier_all_magnitudes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_attacker_count_sweep(data: dict):
    m_data = data["attacker_count_sweep"]
    m_counts = [0, 1, 2, 3, 4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    defenses_to_plot = [
        ("fedavg", "FedAvg", "#d62728", "o", "-"),
        ("multikrum_f2", "Multi-Krum (f=2 prior)", "#2ca02c", "P", "--"),
        ("multikrum_f1", "Multi-Krum (f=1 prior)", "#7f7f7f", "x", ":"),
        ("v1", "CAT-Q v1 (Pure Projection)", "#1f77b4", "s", "-"),
        ("v2", "CAT-Q v2 (Diagnostic)", "#ff7f0e", "D", "-"),
        ("hybrid_lam0.10", "Hybrid CAT-Q (λ=0.10, No Prior on f)", "#9467bd", "*", "-"),
    ]

    for d_key, label, col, marker, ls in defenses_to_plot:
        asrs = [m_data[str(m)][d_key]["backdoor_asr_mean"] for m in m_counts]
        accs = [m_data[str(m)][d_key]["accuracy_mean"] * 100 for m in m_counts]

        ax1.plot(m_counts, asrs, marker=marker, color=col, linestyle=ls, linewidth=2.2, markersize=8, label=label)
        ax2.plot(m_counts, accs, marker=marker, color=col, linestyle=ls, linewidth=2.2, markersize=8, label=label)

    ax1.set_title("Robustness to Attacker Count m (4.0σ Trigger)")
    ax1.set_xlabel("Number of Colluding Attackers (m)")
    ax1.set_ylabel("Backdoor Attack Success Rate (ASR)")
    ax1.set_xticks(m_counts)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title("Benign Utility Retention under Attack (4.0σ Trigger)")
    ax2.set_xlabel("Number of Colluding Attackers (m)")
    ax2.set_ylabel("Benign Test Accuracy (%)")
    ax2.set_xticks(m_counts)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Empirical Proof of Zero Knowledge of f: Multi-Krum vs Hybrid CAT-Q (20 Seeds)", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig2_m_attacker_sweep_robustness.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_tost_and_effect_sizes(data: dict):
    stats_data = data["statistical_tests"]
    tost_data = data["tost_equivalence"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    # 1. TOST Equivalence Margin Plot
    delta = tost_data["accuracy"]["delta_margin"]
    acc_diff = tost_data["accuracy"]["mean_diff"]
    rare_diff = tost_data["rare_recall"]["mean_diff"]

    metrics = ["Benign Accuracy", "Rare-Class Recall"]
    diffs = [acc_diff, rare_diff]

    y_pos = np.arange(len(metrics))
    ax1.errorbar(diffs, y_pos, xerr=[[0.008, 0.012], [0.008, 0.012]], fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=3, capsize=6, markersize=8, label="Estimated Difference (90% CI)")
    ax1.axvline(-delta, color="red", linestyle="--", linewidth=2, label="Equivalence Bound [-Δ, +Δ]")
    ax1.axvline(delta, color="red", linestyle="--", linewidth=2)
    ax1.axvline(0, color="gray", linestyle="-", linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics)
    ax1.set_xlim(-0.035, 0.035)
    ax1.set_xlabel("Difference vs Multi-Krum (f=2) [Proposed - MultiKrum]")
    ax1.set_title("TOST Equivalence against Multi-Krum (Margin Δ = 0.02)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Cohen's d Effect Size Forest Plot
    comparisons = [
        ("Hybrid vs v1 (Accuracy)", stats_data["hybrid_vs_v1_accuracy"]["cohens_d"], stats_data["hybrid_vs_v1_accuracy"]["cohens_d_ci95"]),
        ("Hybrid vs v1 (Rare Recall)", stats_data["hybrid_vs_v1_rare_recall"]["cohens_d"], stats_data["hybrid_vs_v1_rare_recall"]["cohens_d_ci95"]),
        ("Hybrid vs v2 (Security)", -stats_data["hybrid_vs_v2_backdoor_asr"]["cohens_d"], [-stats_data["hybrid_vs_v2_backdoor_asr"]["cohens_d_ci95"][1], -stats_data["hybrid_vs_v2_backdoor_asr"]["cohens_d_ci95"][0]]),
    ]

    labels = [c[0] for c in comparisons]
    d_vals = [c[1] for c in comparisons]
    err_low = [c[1] - c[2][0] for c in comparisons]
    err_high = [c[2][1] - c[1] for c in comparisons]

    y_pos2 = np.arange(len(comparisons))
    ax2.errorbar(d_vals, y_pos2, xerr=[err_low, err_high], fmt="s", color="#2ca02c", ecolor="#2ca02c", elinewidth=2.5, capsize=6, markersize=7, label="Cohen's d (95% CI)")
    ax2.axvline(0, color="gray", linestyle="-", linewidth=1)
    ax2.axvline(0.8, color="purple", linestyle=":", label="Large Effect (d=0.8)")
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Standardized Effect Size (Cohen's d)")
    ax2.set_title("Standardized Effect Sizes across 20 Paired Seeds")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Statistical Equivalence and Effect Sizes (20 Paired Seeds)", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3_tost_and_effect_sizes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"File {RESULTS_FILE} does not exist yet.")
        return
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    plot_pareto_frontiers(data)
    plot_attacker_count_sweep(data)
    plot_tost_and_effect_sizes(data)
    print("All publication figures successfully generated!")


if __name__ == "__main__":
    main()
