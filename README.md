# Federated Intrusion Detection System for 6G Edge Networks (FL-IDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](tests/)

A Byzantine-resilient Federated Intrusion Detection System designed for high-throughput 6G edge gateways. This repository implements the **Class-Aware Trust Defense (`ClassAwareTrustDefense`)**, which combines **Verified Clean Root Subspace Projection ($Q$-Projection)** with class-conditional trust alignment.

On multiclass 6G NetFlow traffic, the mechanism suppresses additive backdoor attacks to near-zero ($\text{ASR} \le 0.004$), providing robust protection without requiring prior knowledge of the attacker fraction $f$. This README documents the empirical performance, the exact results across 20 independent random seeds, and the inherent trade-offs between trigger neutralization and benign utility.

---

## 🚀 Architectural Principles & Mechanism

1. **Verified Clean Root Subspace Projection ($Q$-Projection)**:
   - A clean server root set $D_r$ ($n=150$) is used to compute the unconditioned root gradient $g_0$ and class-conditional anchor gradients $\{g_c\}$.
   - We construct an orthonormal basis $Q \in \mathbb{R}^{d \times k}$ ($k \le C+1$) via thin QR decomposition on $M = [g_0, g_1, \dots, g_C]$.
   - Client parameter updates $\Delta w_i$ are projected onto $Q$: $\widetilde{\Delta w}_i = Q (Q^T \Delta w_i)$. Any additive trigger perturbations orthogonal to clean flow dynamics satisfy $Q^T \delta_{\text{trigger}} = 0$, suppressing backdoor dimensions.
   - **Advantage over Distance Filters**: Unlike Multi-Krum, which requires an exact prior bound on the number of compromised edge nodes $f$, $Q$-projection is client-count agnostic.

2. **Class-Aware Trust Alignment & Adaptive Clipping**:
   - Directional trust alignment is evaluated class-by-class against clean server anchors, rather than computing a monolithic parameter-wide cosine similarity.
   - Multi-tier gradient norm clipping ($\|g\|_2 \le 3.0$) and coordinate-wise trimmed outlier bounding prevent high-norm Byzantine poisoning.

3. **Strict Leakage-Free Preprocessing**:
   - Implements `LeakageFreePreprocessor`: all feature standardizations ($\mu, \sigma$) are computed strictly on local training folds, preventing data leakage across client-server boundaries.

4. **Seed-Level Rigorous Statistical Protocol**:
   - Evaluated across 20 fresh random seeds (`3001`–`3020`).
   - Uses paired Student's $t$-test (`scipy.stats.ttest_rel`), Wilcoxon signed-rank test (`scipy.stats.wilcoxon`), and sample standard deviations ($\text{ddof}=1$).

---

## 📊 Benchmark Overview

The benchmark evaluates the defense across **3 Datasets**, **20 Fresh Random Seeds** (`3001`–`3020`), **6 State-of-the-Art Baselines**, and **4 Component Ablations**:

### Evaluated Datasets
1. **NF-ToN-IoT-Multiclass**: Real NetFlow v2 traffic, 10 classes, 41 features. Contains an average of 56.3 test samples across the 3 minority classes (~18.8 samples/rare class).
2. **NF-ToN-IoT-Binary**: Real NetFlow v2 traffic, 2 classes (Benign vs. Attack), 41 features (mean rare test $N = 588.5$).
3. **Synthetic-NetFlow-Multiclass**: 40 synthetic NetFlow features simulating high-velocity 6G edge flows under Dirichlet non-IID class imbalance ($\alpha=0.3$).

### Evaluated Aggregators & Baselines
- **Standard**: `FedAvg` (McMahan et al.)
- **Distance-Based**: `MultiKrum` (Blanchard et al., parameterized with true $f=2$)
- **Coordinate-Wise**: `TrimmedMean` (Yin et al., $\beta=0.2$)
- **Diversity / Sybil-Resistant**: `FoolsGold` (Fung et al.)
- **Trust-Based**: `FLTrust` (Cao et al.)
- **Reputation-Based**: `RepGuard` (Xu et al.)
- **Proposed**: `ClassAwareTrustDefense` (Ours with Clean Root Subspace Projection)

---

## 📈 Empirical Results (20 Seeds $\times$ 3 Datasets)

All values report empirical **mean $\pm$ sample standard deviation** across 20 independent seeds (`3001`–`3020`). Statistical tests are two-sided seed-level paired comparisons against the Proposed defense.

### Table 1: Main Defense Benchmark (Backdoor ASR & Benign Utility)

| Dataset | Aggregator / Defense | Backdoor ASR $\downarrow$ | Test Accuracy $\uparrow$ | Rare Recall $\uparrow$ | Paired Diff ($\Delta_{\text{ASR}}$) | Wilcoxon $p$-val |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NF-ToN-IoT-Multiclass** | **Proposed (Ours)** | **0.0037 $\pm$ 0.0090** | 49.29% | 7.74% | — | — |
| (10 classes, 41 features) | `FedAvg` | 0.6807 $\pm$ 0.4342 | 58.23% | 15.40% | -0.6770 | $3.80 \times 10^{-4}$ |
| (Rare test $N \approx 56.3$ total) | `MultiKrum` ($f=2$) | 0.0065 $\pm$ 0.0195 | 54.26% | 7.24% | -0.0028 | $0.4631$ |
| | `TrimmedMean` | 0.5784 $\pm$ 0.3361 | 59.16% | 10.31% | -0.5748 | $1.91 \times 10^{-6}$ |
| | `FoolsGold` | 0.8346 $\pm$ 0.3279 | 56.93% | 22.01% | -0.8309 | $9.18 \times 10^{-5}$ |
| | `FLTrust` | 0.3417 $\pm$ 0.3255 | 55.77% | 23.21% | -0.3380 | $3.42 \times 10^{-4}$ |
| | `RepGuard` | 0.4454 $\pm$ 0.3920 | 58.21% | 14.93% | -0.4417 | $2.33 \times 10^{-4}$ |
| **NF-ToN-IoT-Binary** | **Proposed (Ours)** | **0.3208 $\pm$ 0.4228** | 69.10% | 75.63% | — | — |
| (2 classes, 41 features) | `FedAvg` | 0.5155 $\pm$ 0.4632 | 88.17% | 99.23% | -0.1948 | $0.4074$ |
| (Rare test $N = 588.5$) | `MultiKrum` ($f=2$) | 0.5081 $\pm$ 0.3769 | 88.01% | 92.09% | -0.1873 | $0.0707$ |
| | `TrimmedMean` | 0.8059 $\pm$ 0.3194 | 88.43% | 93.39% | -0.4852 | $2.23 \times 10^{-3}$ |
| | `FoolsGold` | 0.9226 $\pm$ 0.1360 | 90.93% | 96.11% | -0.6018 | $2.10 \times 10^{-4}$ |
| | `FLTrust` | 0.6261 $\pm$ 0.4591 | 66.50% | 65.86% | -0.3053 | $8.25 \times 10^{-3}$ |
| | `RepGuard` | 0.6965 $\pm$ 0.3593 | 87.42% | 92.37% | -0.3758 | $1.26 \times 10^{-2}$ |
| **Synthetic-NetFlow** | **Proposed (Ours)** | **0.0000 $\pm$ 0.0000** | 68.09% | 0.00% | — | — |
| (10 classes, 40 features) | `FedAvg` | 0.2914 $\pm$ 0.4240 | 79.98% | 0.00% | -0.2914 | $1.80 \times 10^{-2}$ |
| | `MultiKrum` ($f=2$) | 0.0060 $\pm$ 0.0257 | 75.24% | 0.00% | -0.0060 | $0.1797$ |
| | `TrimmedMean` | 0.1124 $\pm$ 0.2665 | 80.65% | 0.00% | -0.1124 | $1.80 \times 10^{-2}$ |
| | `FoolsGold` | 0.6253 $\pm$ 0.4326 | 80.67% | 0.00% | -0.6253 | $4.36 \times 10^{-4}$ |
| | `FLTrust` | 0.0007 $\pm$ 0.0031 | 67.16% | 0.00% | -0.0007 | $0.3173$ |
| | `RepGuard` | 0.1254 $\pm$ 0.2423 | 80.53% | 0.00% | -0.1254 | $3.35 \times 10^{-3}$ |

---

### Table 2: 4-Arm Component Ablations of Proposed Defense

| Dataset | Ablation Variant | Backdoor ASR $\downarrow$ | Test Accuracy $\uparrow$ | Rare Recall $\uparrow$ | Causal Insight |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **NF-ToN-IoT-Multiclass** | **Full Proposed Defense** | **0.0037 $\pm$ 0.0090** | **49.29%** | **7.74%** | Full subspace projection + class-aware alignment |
| | w/o Subspace Projection | 0.3872 $\pm$ 0.2913 | 57.68% | 15.68% | Backdoor ASR surges **105$\times$**; confirms projection neutralizes triggers |
| | w/o Class-Aware Alignment | 0.0211 $\pm$ 0.0364 | 40.34% | 41.51% | Overall accuracy collapses to 40.34% without class anchors |
| | w/o Adaptive Clipping | 0.0039 $\pm$ 0.0094 | 49.40% | 7.91% | Minor effect when subspace projection is active |
| | w/o Root Norm | 0.0019 $\pm$ 0.0057 | 51.54% | 7.37% | Minor calibration adjustment |
| **NF-ToN-IoT-Binary** | **Full Proposed Defense** | **0.3208 $\pm$ 0.4228** | **69.10%** | **75.63%** | Subspace rank $\le 2$ constrains separation |
| | w/o Subspace Projection | 0.7323 $\pm$ 0.2991 | 87.96% | 90.53% | ASR more than doubles (0.32 $\to$ 0.73) |
| | w/o Class-Aware Alignment | 0.7369 $\pm$ 0.4267 | 39.76% | 32.65% | Severe collapse in binary utility |
| | w/o Root Norm | 0.1643 $\pm$ 0.3211 | 85.62% | 88.35% | Root norm scaling is over-restrictive in binary setting |
| **Synthetic-NetFlow** | **Full Proposed Defense** | **0.0000 $\pm$ 0.0000** | **68.09%** | **0.00%** | Complete trigger eradication |
| | w/o Subspace Projection | 0.0595 $\pm$ 0.1827 | 84.71% | 0.00% | Backdoor activates without root projection |
| | w/o Class-Aware Alignment | 0.0062 $\pm$ 0.0187 | 37.65% | 14.17% | Accuracy collapses to 37.65% |

---

## 🔍 Critical Discussion & Empirical Insights

### 1. Multiclass Trigger Neutralization vs. Binary Limitations
- **Multiclass**: On `NF-ToN-IoT-Multiclass` and `Synthetic-NetFlow`, $Q$-projection suppresses ASR to $\mathbf{\le 0.004}$, outperforming `FedAvg` ($0.681$), `FoolsGold` ($0.835$), `TrimmedMean` ($0.578$), and `FLTrust` ($0.342$).
- **Binary**: On `NF-ToN-IoT-Binary`, ASR is **$0.321 \pm 0.423$**, which fails the $\le 0.20$ criterion and does not achieve statistical significance over FedAvg ($p=0.41$). In a binary setting, the server basis $Q$ has rank $\le 2$, causing substantial subspace overlap between benign attack-detection gradients and trigger perturbations.

### 2. The Benign Utility Tax
- Constraining client updates to an orthonormal basis of at most $C+1 = 11$ server directions imposes an inherent utility cost:
  - Multiclass accuracy is **$49.29\%$** compared to $58.23\%$ for FedAvg and $54.26\%$ for MultiKrum.
  - Rare-class recall is **$7.74\%$** compared to $15.40\%$ for FedAvg.
- This represents a fundamental trade-off: strong backdoor suppression via geometric projection restricts parameter degrees of freedom for edge models.

### 3. Comparison with Multi-Krum
- `MultiKrum` achieves similar multiclass ASR ($0.0065 \pm 0.0195$) with higher benign accuracy ($54.26\%$) when provided with the **exact ground-truth attacker count ($f=2$)**.
- However, Multi-Krum is vulnerable to misparameterization of $f$. $Q$-projection operates **without prior knowledge of attacker count $f$**, providing robustness in dynamic edge deployments where the number of compromised nodes is unknown.

### 4. Clean Control Baseline (Trigger Neutralization Proof)
- Evaluating an unattacked model (`FedAvg` trained on clean clients) against triggered test flows yields a **Clean Trigger Activation Rate of 0.0000 (0.0%)**.
- This confirms that the baseline ASR of $68.07\%$ in `FedAvg` is an active backdoor effect, and the $0.0037$ ASR achieved by the Proposed Defense reflects true trigger neutralization.

---

## 🛠️ Repository Structure

```text
winner/
├── fl_ids/
│   ├── attacks/               # Poisoning and evasion attack modules
│   │   ├── backdoor.py        # Additive & boosted NetFlow backdoor attacks
│   │   ├── labelflip.py       # Source-target label flip attack
│   │   ├── mimicry.py         # Atypical minority class mimicry attack
│   │   └── signflip.py        # Byzantine sign-inversion attack
│   ├── data/                  # Data loaders, partitioners, and preprocessors
│   │   ├── canary.py          # Data leakage verification canary
│   │   ├── nftoniot.py        # Real-world NF-ToN-IoT-V2 dataset loader
│   │   ├── partitioner.py     # Dirichlet non-IID partitioner (alpha=0.3)
│   │   ├── preprocessor.py    # Strict fit-on-train leak-free preprocessor
│   │   └── synthetic.py       # Controlled synthetic NetFlow generator
│   ├── defense/               # Federated aggregation defenses
│   │   ├── coordinate_filters.py # TrimmedMean and CoordinateMedian
│   │   ├── distance_filters.py   # MultiKrum and FoolsGold
│   │   ├── novel_defense.py      # ClassAwareTrustDefense (Proposed)
│   │   └── trust_filters.py      # FLTrust and RepGuard
│   ├── federation/            # FL simulation engine
│   │   ├── client.py          # FederatedClient with local training
│   │   ├── server.py          # FederatedServer with root evaluation
│   │   └── simulator.py       # Orchestrator with round-by-round metrics
│   └── models/                # Deep learning models
│       └── mlp.py             # High-throughput Edge IDS MLP
├── scripts/
│   ├── run_defense_benchmark_3datasets_20seeds.py  # 20-seed 3-dataset benchmark
│   ├── run_causal_attribution.py                   # Causal attribution interventions
│   └── test_feature_saliency.py                    # Attribution detector test
├── results/                   # Benchmark output JSONs
├── tests/                     # 22 automated unit and integration tests
├── requirements.txt           # Dependencies
└── pyproject.toml             # Build configuration
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shounak-max/6giotsecurity.git
   cd 6giotsecurity
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running Tests & Benchmarks

### 1. Verify Integrity with PyTest
```bash
pytest tests/ -v
```
*(All 22 unit and integration tests pass)*

### 2. Run the 20-Seed, 3-Dataset Benchmark
```bash
python scripts/run_defense_benchmark_3datasets_20seeds.py
```
Outputs are dynamically computed across 20 fresh seeds (`3001`–`3020`) and written to `results/proposed_defense_benchmark_3datasets_20seeds.json`.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
