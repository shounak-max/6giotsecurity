# Federated Intrusion Detection for 6G Edge: Problem Statement, Requirements, Roadmap and Targets

## 1. Problem statement

How can federated intrusion detection (FL-IDS) for 6G edge, IoT and vehicular networks jointly optimize privacy, poisoning (Byzantine and backdoor) robustness, non-IID learning, minority-attack recall, communication cost, and edge latency and energy, under an evaluation protocol that is free of leakage and oracle assumptions?

## 2. Requirements

A solution must satisfy all of the following at once.

| Axis | Requirement |
|---|---|
| Utility | High macro-F1 and balanced accuracy under strong non-IID partitions (Dirichlet α down to 0.1) |
| Poisoning robustness | Resists sign-flip, label-flip and backdoor attacks at 20–30% malicious clients, including defense-aware adaptive attackers |
| Trigger-agnostic defense | No knowledge of the attacker's trigger at the server |
| Minority recall | Does not exclude or penalise clients that hold rare attack classes |
| Privacy | Reports a formally accounted (ε, δ), or uses secure aggregation. Privacy claims are kept separate from robustness claims |
| Communication | Per-round uplink well below full-precision FedAvg, with the robust filter still functional |
| Edge cost | Low latency, memory and energy per inference on a gateway-class device, plus uplink time on a constrained link |
| Evaluation integrity | Train-only scaling, disjoint server root set, deduplication before split, canonicalised placeholders, multi-seed paired tests, weak-model leakage canary |

## 3. Literature findings

Sources: RepGuard leakage-free evaluation (arXiv 2610.14767), EdgeDetect (2604.14663), PVBRF-ID (2610.12704), Edge-IIoTset audit (2608.15761). Other papers in the pool (BRFID 2609.28599, FBID 2608.04073, FedTransKD-IDS 2608.06447, SoK 2607.10914, among others) were seen as abstracts only. The Consensus search quota was exhausted, so retrieval used alphaXiv.

### 3.1 Reported metrics

| Paper | Setup | Key numbers |
|---|---|---|
| RepGuard | CSE-CIC-IDS2018, 10 clients, Dirichlet α=0.3, 2 Byzantine, harsh regime, n=6 seeds | Macro-F1 0.683 (FLTrust 0.626, FedAvg 0.296). Backdoor ASR 0.93–0.97 for all methods under strong triggers |
| RepGuard | Heterogeneity sweep, trigger 1.0 | α=0.1: ASR 0.48 vs FLTrust 0.86. f=4 of 10: ASR 0.64 vs FLTrust 1.00 |
| RepGuard | NF-UNSW-NB15-v2 | Best macro-F1 0.212, balanced accuracy 0.281, ASR 0.400. Rarest classes (Worms: 5 samples) unlearnable |
| RepGuard | DP-SGD, CSE-CIC-IDS2018 | ε from 16.5 to 0.32 costs macro-F1 0.654 to 0.623. ASR stays at 0.73 or higher |
| RepGuard | Oracle ablation | ASR 0.49 trigger-agnostic vs 0.12 with trigger knowledge. Removing root anchoring raises ASR 0.49 to 0.71 |
| PVBRF-ID | Edge-IIoTset 15-class, 10 sites, Dirichlet 0.5, 10 seeds | Clean macro-F1 0.708 (FedAvg 0.706). Sign-flip 0.696 vs FedAvg 0.091 |
| PVBRF-ID | Adaptive attacker, 30% of sites | 0.641 with secure clipping (FedAvg 0.046, multi-Krum 0.693, trimmed mean 0.597) |
| PVBRF-ID | Backdoor | ASR 1.0 unfiltered. Clip + noise + aggregate clip at z=1: ASR 0.150, macro-F1 0.680, but ε=41 |
| PVBRF-ID | Cost and fairness | 19.4× FedAvg upload (1.72 MiB per round). Multi-Krum accepts atypical sites 40.7% vs 93.5% for typical. Fingerprinting recall about 0.10, SQL injection 0.49, Uploading 0.43 |
| EdgeDetect | CIC-IDS2017, 50 clients | Accuracy 98.0%, macro-F1 97.9%. 450 MB to 14 MB per round (32×). α=0.1: 94.2%, or 95.1% with FedProx |
| EdgeDetect | Edge (claimed) | Raspberry Pi 4: 0.8 ms, 12 mJ per inference, 4.2 MB memory |
| Edge-IIoTset audit | Binary, 5-fold × 3 repeats | Under the distributed recipe five of six models score exactly 1.0000. Corrected: best macro-F1 0.9503 (hist. gradient boosting). Naive Bayes falls 0.3005 |
| Edge-IIoTset audit | Leave-one-device-out, AgriEdge | Random forest 0.9988 to 0.5083 balanced accuracy across the perception/actuation layer. Non-IID costs at most 0.0037 macro-F1. 20-round LoRaWAN run costs 4.6 hours of uplink |

### 3.2 Cautions on comparability

- EdgeDetect mixes a Random Forest model with gradient compression and Paillier encryption, reports robustness at both 5% and 20% poisoning with different numbers, and uses random splits on CIC-IDS2017. Treat its numbers as claims to reproduce.
- Edge-IIoTset results above 99% that used the distributed preprocessing recipe are likely inflated by a label-leaking serialisation artifact. Label, ordinal and frequency encodings leak identically.
- Trust-based defenses are often evaluated with root sets overlapping client data, single seeds, unaccounted "DP" and filters that know the attacker's trigger.
- A best-effort reimplementation of P4P did not transfer in the RepGuard study, which illustrates how hard published FL-IDS defense results are to reproduce without released code.

## 4. Research gaps

1. Trigger-agnostic backdoor defense under strong triggers. ASR is about 0.95 on CSE-CIC-IDS2018 for every method; the best single-dataset result is 0.40–0.48.
2. No paper jointly evaluates poisoning, non-IID data, privacy, compression and minority recall under one protocol. PVBRF-ID has robustness and privacy at 19× cost. EdgeDetect has compression and privacy without verified robustness.
3. Distance and norm filters (multi-Krum, tight norm bounds) systematically penalise atypical clients, which are the clients that carry rare attack classes.
4. Privacy budgets are rarely reported honestly. Where they are, they are weak (ε of 17–113 for PVBRF-ID noise runs).
5. Rare-class performance is poor (macro-F1 0.17–0.21 on NF-UNSW-NB15-v2).
6. On constrained links, bandwidth may matter more than statistical heterogeneity.
7. Cross-dataset and cross-device transfer is largely unmeasured in FL-IDS papers.
8. No standard 6G-specific FL-IDS benchmark exists. Existing evaluations use general IoT and NetFlow datasets.

## 5. Evaluation protocol

**Datasets:** CSE-CIC-IDS2018, NF-ToN-IoT, NF-UNSW-NB15-v2, and corrected Edge-IIoTset or AgriEdge.

**Partitioning:** Dirichlet α in {0.1, 0.3, 1}, plus label-skew partitions and per-device or per-farm clients where attribution exists.

**Threat model:** sign-flip, label-flip, targeted flip, additive-trigger backdoor with a magnitude sweep, boosted backdoor, defense-aware adaptive attacker, and triggerless or dynamic backdoors as a stretch case. Malicious fraction 20% and 30%.

**Integrity rules:** train-only standardisation, server root set disjoint from clients, canonicalise placeholders and deduplicate before splitting, random row sampling rather than file prefixes, and a naive Bayes leakage canary on every dataset.

**Statistics:** at least 6 seeds, mean ± std, paired t-test and Wilcoxon, effect sizes and 95% confidence intervals. Report non-significant results.

**Metrics:**
- Utility: macro-F1, balanced accuracy, accuracy, per-class recall for rare classes, false positive rate.
- Robustness: backdoor ASR (trigger-agnostic), macro-F1 under attack, atypical-client acceptance rate.
- Privacy: accounted (ε, δ), plus an update-inference attack success measure (class-mix correlation).
- Efficiency: MB per client per round, total MB to reach a target accuracy, rounds to target, server overhead in seconds.
- Edge: inference latency (median and p95), memory, energy, and uplink time on a constrained link.

**Baselines:** FedAvg, FedProx, Krum or multi-Krum, Trimmed-Mean, FoolsGold, FLTrust, RepGuard, PVBRF-ID. EdgeDetect as an unverified baseline.

## 6. Roadmap

### Phase 0: Benchmark protocol
- Implement the leakage-free pipeline and the integrity rules in section 5.
- Build all dataset partitions and attack generators as single-command configs with fixed seeds.
- Define the reporting template so every result carries utility, robustness, privacy and efficiency columns.

### Phase 1: Reproduce baselines
- Implement the baselines in section 5. The released protocol and code from RepGuard and PVBRF-ID give a starting point.
- Validate against published numbers within tolerance and document where reproduction fails.

### Phase 2: Method
- **Class-aware trust scoring.** Score clients on class-conditional representations so that rare-class sites are not filtered out. Track the acceptance rate of atypical clients as a metric.
- **Compression compatible with robust aggregation.** Top-k or sign sketches that the trust filter can still score, avoiding the overhead of hiding updates entirely.
- **Two-level clipping and root anchoring.** Root anchoring was the dominant lever in RepGuard, and PVBRF-ID shows clipping outperforms noise alone against boosted backdoors.
- **Honest privacy.** Secure aggregation or DP with a reported ε, evaluated separately from robustness.
- **Latency-aware participation.** Partial client selection and asynchronous rounds for 6G edge settings.

### Phase 3: Stress tests
- Defense-aware adaptive attackers, trigger-magnitude sweeps, triggerless or dynamic backdoors.
- Leave-one-dataset-out and leave-one-device-out transfer.
- Ablate each component of the method (trust scoring, compression, clipping, root anchoring, privacy layer).

### Phase 4: Edge cost and write-up
- Measure latency, memory and energy on gateway-class hardware, and uplink time on a constrained link.
- Release code, configs and the evaluation protocol.
- Report negative and non-significant results alongside headline numbers.

## 7. Targets

These are proposed targets derived from the gaps above, not published numbers.

| Metric | Current best found | Target |
|---|---|---|
| Backdoor ASR, strong trigger, trigger-agnostic | About 0.95 (CSE-CIC-IDS2018); 0.40 best single-dataset case | At most 0.20 on all three NetFlow datasets, with macro-F1 loss at most 0.03 vs clean FedAvg |
| Macro-F1, harsh non-IID poisoning, CSE-CIC-IDS2018 | 0.683 (RepGuard) | At least 0.72 with std at most 0.02 |
| NF-UNSW-NB15-v2 macro-F1 | 0.212 (RepGuard+) | At least 0.30 |
| Adaptive attacker at 30%, Edge-IIoTset 15-class | 0.641 (PVBRF-ID); 0.693 (multi-Krum) | At least 0.69 with atypical-site acceptance at least 85% |
| Secure + robust communication overhead | 19.4× FedAvg (PVBRF-ID) | At most 5× |
| Compression with robust aggregation intact | 32× with no verified robustness (EdgeDetect) | At least 16× with accuracy within 1 pp of uncompressed |
| Privacy budget | ε 0.3–16 with ASR unchanged; ε 17–113 for noise-based backdoor mitigation | ε at most 8 with macro-F1 loss at most 0.02 |
| Rare-class recall | Fingerprinting about 0.10, SQL injection 0.49, Uploading 0.43 | Improve each over FedAvg without lowering clean macro-F1 |
| Cross-device transfer | 0.5083 balanced accuracy across perception/actuation layer (random forest) | Report on every benchmark and improve over the strongest baseline |
| Edge inference | 0.8 ms, 12 mJ, 4.2 MB (EdgeDetect, claimed, Random Forest on Pi 4) | Match or beat after independent measurement |

The primary claim to aim for is the first row: a trigger-agnostic defense that holds under strong triggers and an honest protocol. Matching published 98–99% accuracy figures is not a meaningful goal, since many come from leaky or random-split setups.

## 8. Risks and open questions

- Trust-based defenses assume a clean disjoint root set and a known bound on the number of attackers. Evaluate sensitivity to both.
- A defense-aware attacker can be tuned against any specific scoring rule. Include attackers designed against the proposed method, not only against baselines.
- Compression may degrade the signal that trust scoring depends on. Test this interaction directly.
- Results may not transfer between datasets. NF-UNSW-NB15-v2 behaved differently from CSE-CIC-IDS2018 and NF-ToN-IoT in the RepGuard study.
- Non-collusion assumptions in two-server secure designs (PVBRF-ID) are assumptions, not results.
- This review used a limited set of papers. Re-run the literature search on Consensus for broader coverage once the quota resets.

## 9. Sources

- RepGuard leakage-free evaluation of trust-based FL-IDS defenses: arXiv 2610.14767
- EdgeDetect: arXiv 2604.14663
- PVBRF-ID: arXiv 2610.12704
- Edge-IIoTset serialisation artifact and AgriEdge benchmark: arXiv 2608.15761
- Pool of related FL-IDS papers seen as abstracts: arXiv 2609.28599, 2609.03420, 2608.06447, 2608.04073, 2609.27202, 2605.05644, 2605.05896, 2610.19808, 2607.10914, 2607.01305
- Problem-statement source document: Consensus-generated report on cybersecurity and 6G research problem statements, section on federated IDS and the 6G edge overlap