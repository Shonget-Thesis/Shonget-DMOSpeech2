# Objective 1 Analysis & Audit Report: Characterization of Teacher Handover Timing

**Document Status:** Development-Set Problem Characterization & Decision-Rule Audit (N=100 Speakers / 800 Runs)  
**Governing Thesis Artifact:** Chapter 3 Research Objective 1  

---

## Test Description & Protocol Specification

- **Research Objective:** Objective 1 — Characterize the empirical effect of teacher handover timing on quality, failure probability, and compute on the Development Split.
- **Dataset & Split Scope:** 100 unique, speaker-disjoint Common Voice speakers drawn from the Seed-TTS English evaluation dataset (`development_manifest.json`).
- **Acoustic Regimes:** Dual-condition evaluation for every speaker: `clean` and `noisy15` (Additive White Gaussian Noise injected at 15.0 +/- 0.1 dB relative to prompt power before feature extraction and RMS normalization).
- **Candidate Handover Grid:** Teacher steps 4, 5, 6, and 8, corresponding to actual swayed Euler integration times $t \in [0.0761, 0.1181, 0.1685, 0.2929]$.
- **Trajectory Probe (Zero Additional NFE):** Normalized endpoint drift evaluated on generated frames only with zero additional neural-network forward passes:
  $$d_i = \frac{\| \hat{x}_1(t_i) - \hat{x}_1(t_{i-1}) \|_2}{\| \hat{x}_1(t_i) \|_2 + \epsilon}$$
- **Student Consistency Completion:** Fixed restart level $j=1$ (times $\tau \in [0.0761, 0.2929, 0.6173]$, exactly 3 student forward passes).
- **Cost Accounting (Primary Unit = Forward NFE):**
  - Step 4: 5 teacher steps ($2\times$ CFG) + 3 student steps = **13 NFE** (38.1% savings vs reference)
  - Step 5: 6 teacher steps ($2\times$ CFG) + 3 student steps = **15 NFE** (28.6% savings vs reference)
  - Step 6: 7 teacher steps ($2\times$ CFG) + 3 student steps = **17 NFE** (19.0% savings vs reference)
  - Step 8 (Reference): 9 teacher steps ($2\times$ CFG) + 3 student steps = **21 NFE** (0.0% savings)
- **Chapter 3 Definition 3.1 Failure Criterion (Reference-Aware Degradation):**
  $$\text{fail}(x, k) = \text{WER}(x, k) > \max(0.125, \text{WER}(x, \text{ref}) + 0.01) \lor \text{SIM}(x, k) < \text{SIM}(x, \text{ref}) - 0.02$$
  Evaluated strictly against the matched Step-8 reference synthesized with identical input and student noise seeds.
- **Evaluation Metrics:** Whisper large-v3 for Word Error Rate (WER), WavLM-large x-vector cosine similarity against ground-truth target audio (SIM), and SpeechMOS UTMOS for naturalness.

---

## Hardware Environment & Execution Limitations

- **Host Processor (CPU):** AMD Ryzen 7 5700U with Radeon Graphics (8 physical cores, 16 logical threads, base clock 1.8 GHz, boost up to 4.3 GHz).
- **GPU Accelerator Availability:** Integrated AMD Radeon Graphics without native NVIDIA CUDA execution support (`torch.cuda.is_available() == False`).
- **Execution Target:** Host CPU execution using evaluation mode (`torch.inference_mode()`), batch size 1, full 32-bit floating point precision (`torch.float32` / FP32), and isolated local `torch.Generator` instances for bitwise reproducibility.
- **Computational Limitations & Precise Latency Accounting:**
  Total sequential execution time across all 800 evaluations on host CPU was **716.20 minutes (~11.94 hours)**. The exact time accounting breaks down as:
  - **3.581 minutes (214.86 seconds)** per utterance-condition case (200 cases across 100 speakers × 2 conditions).
  - **7.162 minutes** per speaker (including both clean and noisy15 conditions).
  - **53.715 seconds** per candidate evaluation (800 candidate evaluations).
  - **Bare model-generation runtime** accounted for a mean of **9.08 seconds** (median **8.75 seconds**) per candidate evaluation (121.02 minutes cumulative across 800 evaluations). The remainder (~44.64 seconds per candidate evaluation) represents evaluation-pipeline overhead (Whisper large-v3 transcription, WavLM-large x-vector extraction and target cosine comparison, UTMOS scoring, tensor operations, and file I/O).
- **Data Isolation:** Calibration and held-out test splits were strictly isolated and never accessed by this runner.

---

## 1. Candidate Failure Rates (Definition 3.1)

> [!IMPORTANT]
> **Reference-Relative Interpretation:** Failure denotes unacceptable *additional degradation* relative to the matched Step 8 reference. Passing the failure rule does not by itself establish high absolute synthesis quality, especially when the reference is degraded by an adverse prompt condition. Step 8 failure is 0% by definition; confidence intervals are not applicable.

### 1.1 Condition-Specific Failure Rates (Exact Clopper-Pearson 95% Confidence Intervals)

| Condition | Teacher Step | Actual Time | Total NFE | Failures / Total | Failure Rate (%) | Exact 95% Clopper-Pearson CI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `clean` | Step 4 | 0.0761 | 13 | 2 / 100 | 2.00% | [0.24%, 7.04%] |
| `clean` | Step 5 | 0.1181 | 15 | 3 / 100 | 3.00% | [0.62%, 8.52%] |
| `clean` | Step 6 | 0.1685 | 17 | 4 / 100 | 4.00% | [1.10%, 9.93%] |
| `clean` | Step 8 | 0.2929 | 21 | 0 / 100 | 0.00% | 0% by definition (CI not applicable) |
| `noisy15` | Step 4 | 0.0761 | 13 | 63 / 100 | 63.00% | [52.76%, 72.44%] |
| `noisy15` | Step 5 | 0.1181 | 15 | 49 / 100 | 49.00% | [38.86%, 59.20%] |
| `noisy15` | Step 6 | 0.1685 | 17 | 53 / 100 | 53.00% | [42.76%, 63.06%] |
| `noisy15` | Step 8 | 0.2929 | 21 | 0 / 100 | 0.00% | 0% by definition (CI not applicable) |

### 1.2 Combined Descriptive Results (Cluster Bootstrap by Utterance, 1,000 resamples)

| Teacher Step | Total NFE | Failures / Total | Cluster Mean Rate (%) | Cluster Bootstrap 95% CI |
|:---:|:---:|:---:|:---:|:---:|
| Step 4 | 13 | 65 / 200 | 32.50% | [27.50%, 37.50%] |
| Step 5 | 15 | 52 / 200 | 26.00% | [21.00%, 31.50%] |
| Step 6 | 17 | 57 / 200 | 28.50% | [23.50%, 33.50%] |
| Step 8 | 21 | 0 / 200 | 0.00% | 0% by definition (CI not applicable) |

---

## 2. Failure Mode Decomposition & Distributional Statistics

To explain discrepancies between mean WER and failure rate (such as extreme Whisper repetition loops in noisy audio), we report failure breakdown by modality along with Median WER and Interquartile Range (IQR):

| Condition | Step | Total Failures | WER-Only Fail | SIM-Only Fail | Both Failed | Median WER (%) | WER IQR (%) | Paired Mean $\Delta$WER (%) | Paired Median $\Delta$WER (%) | Paired Median $\Delta$SIM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `clean` | Step 4 | **2** | 1 | 1 | 0 | 0.00% | 0.00% | 0.36% | 0.00% | -0.0001 |
| `clean` | Step 5 | **3** | 0 | 3 | 0 | 0.00% | 0.00% | 0.07% | 0.00% | +0.0008 |
| `clean` | Step 6 | **4** | 0 | 4 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | -0.0003 |
| `clean` | Step 8 | **0** | 0 | 0 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | +0.0000 |
| `noisy15` | Step 4 | **63** | 21 | 27 | 15 | 17.42% | 43.25% | 9.12% | 0.00% | -0.0100 |
| `noisy15` | Step 5 | **49** | 17 | 17 | 15 | 11.81% | 33.93% | 52.17% | 0.00% | +0.0003 |
| `noisy15` | Step 6 | **53** | 23 | 20 | 10 | 10.00% | 34.09% | 45.95% | 0.00% | +0.0041 |
| `noisy15` | Step 8 | **0** | 0 | 0 | 0 | 10.00% | 22.22% | 0.00% | 0.00% | +0.0000 |

> [!NOTE]
> **Explanation of Noisy WER Outliers & Modality Contributions:**
> 1. For noisy prompts, Mean WER at Step 5 (70.40%) and Step 6 (64.17%) was heavily distorted by extreme ASR repetition loops: one utterance (`common_voice_en_31959919-common_voice_en_31959922`) exhibited WER = 2,690.9% at Step 5 and 2,600.0% at Step 6, while two additional separate utterances experienced repetition loops at Step 5 (WER = 2,220.0%) and Step 6 (WER = 1,730.8%). The **Median WER** is strictly monotonic: Step 4 (17.42%) -> Step 5 (11.81%) -> Step 6 (10.00%) = Step 8 (10.00%).
> 2. Both intelligibility and speaker-similarity degradation contributed substantially to noisy-condition failures. At Step 4, SIM was involved in 42 failures (27 SIM-only, 15 both) and WER was involved in 36 failures (21 WER-only, 15 both).

---

## 3. Oracle Step Distribution by Condition

The **oracle step** is defined as the cheapest non-failing candidate ($\min \{k \in \{4, 5, 6, 8\} \mid \text{fail}(x, k) = 0\}$).

| Condition | Oracle Step 4 (NFE 13) | Oracle Step 5 (NFE 15) | Oracle Step 6 (NFE 17) | Oracle Step 8 (NFE 21) |
|---|:---:|:---:|:---:|:---:|
| `clean` | 98.0% (98) | 2.0% (2) | 0.0% (0) | 0.0% (0) |
| `noisy15` | 37.0% (37) | 23.0% (23) | 8.0% (8) | 32.0% (32) |

---

## 4. Signal Correlation Audit: Condition Confounding and Drift Diagnosis

### 4.1 Condition-Specific vs. Pooled Spearman Rank Correlations with Oracle Step

> [!NOTE]
> Pooled Spearman correlations treat paired clean and noisy observations as independent and are reported descriptively. Condition-specific analyses contain exactly one observation per speaker and are statistically independent.

| Feature | Pooled $\rho$ (N=200, Descriptive) | Pooled $p$-value | Clean-Only $\rho$ (N=100) | Clean $p$-value | Noisy-Only $\rho$ (N=100) | Noisy $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `d_4` | -0.6139 | 4.22e-22 | -0.0124 | 9.03e-01 | -0.2257 | 2.40e-02 |
| `d_5` | -0.6018 | 4.33e-21 | -0.1559 | 1.21e-01 | -0.1590 | 1.14e-01 |
| `d_6` | -0.5457 | 6.52e-17 | -0.1435 | 1.54e-01 | -0.1372 | 1.73e-01 |
| `prompt_snr` | -0.5800 | 2.27e-19 | 0.0644 | 5.24e-01 | -0.1190 | 2.38e-01 |

### 4.2 Diagnostic Resolution of Condition Confounding on Drift

1. **Condition Confounding Across Regimes:**
   - In `clean` audio, normalized drift is large (mean $d_4 = 0.595$, median $0.605$) and oracle step is uniformly low (98% at Step 4).
   - In `noisy15` audio, normalized drift exhibited substantially lower values (mean $d_4 = 0.179$, median $0.175$). The present experiment did not isolate the cause of this reduction; it may reflect changes in predicted velocity, endpoint magnitude, normalization behavior, or their interaction. At the same time, noisy audio requires higher oracle steps (32% at Step 8).
   - Pooling clean and noisy samples artificially pairs high drift with low oracle steps, producing an unadjusted negative correlation ($\rho = -0.6139$).
2. **Within-Condition Predictive Power:**
   - Within clean speech, variance is minimal because 98% of inputs succeed at Step 4.
   - Within noisy speech, passing samples have slightly *higher* drift ($0.186 \pm 0.030$) than failing samples ($0.175 \pm 0.037$).
   - Consequently, the simple univariate rule $d_i \le \lambda$ alone cannot differentiate passing from failing noisy utterances.

---

## 5. Comparative Evaluation of Candidate Schedulers (Development Split)

To test whether trajectory drift provides additional risk-compute value beyond prompt SNR alone, we benchmark 10 candidate policies across all 100 development speakers (200 condition-pairs):

| Scheduler Policy | Decision Rule | Mean NFE | NFE Savings (%) | Robust Failure Rate (%) | Clean Fail Rate (%) | Noisy15 Fail Rate (%) | Step Distribution {4, 5, 6, 8} |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Step 4** | Always Step 4 | 13.00 | **38.1%** | **64.0%** | 2.0% | 63.0% | {200, 0, 0, 0} |
| **Fixed Step 5** | Always Step 5 | 15.00 | **28.6%** | **51.0%** | 3.0% | 49.0% | {0, 200, 0, 0} |
| **Fixed Step 6** | Always Step 6 | 17.00 | **19.0%** | **56.0%** | 4.0% | 53.0% | {0, 0, 200, 0} |
| **Fixed Step 8** | Always Step 8 (Reference) | 21.00 | **0.0%** | **0.0%** | 0.0% | 0.0% | {0, 0, 0, 200} |
| **SNR-Only (theta=20.0 dB)** | If $\text{SNR} \ge 20\text{ dB} \to 4$, else $8$ | 17.00 | **19.0%** | **2.0%** | 2.0% | 0.0% | {100, 0, 0, 100} |
| **Drift-Only (lambda=0.60)** | If $d_4 \le 0.60 \to 4$, else $8$ | 15.04 | **28.4%** | **63.0%** | 1.0% | 63.0% | {149, 0, 0, 51} |
| **Binary SNR + d_4 diagnostic policy (lambda=0.60)** | If $\text{SNR} \ge 20\text{ dB}$ AND $d_4 \le 0.60 \to 4$, else $8$ | 19.04 | **9.3%** | **1.0%** | 1.0% | 0.0% | {49, 0, 0, 151} |
| **Sequential Policy (lambda=0.60, theta=20.0 dB)** | Sequential $4\to 5\to 6$ if $d_i \le 0.60$ and $\text{SNR} \ge 20$, else $8$ | 17.61 | **16.1%** | **1.0%** | 1.0% | 0.0% | {49, 41, 10, 100} |
| **Sequential Policy (lambda=0.70, theta=20.0 dB)** | Sequential $4\to 5\to 6$ if $d_i \le 0.70$ and $\text{SNR} \ge 20$, else $8$ | 17.14 | **18.4%** | **1.0%** | 1.0% | 0.0% | {87, 12, 1, 100} |
| **Oracle Policy** | Cheapest non-failing candidate per utterance | 14.69 | **30.0%** | **0.0%** | 0.0% | 0.0% | {135, 25, 8, 32} |

### Key Takeaways from the Policy Comparison:
- **Drift-Only Fails Disastrously under Noise (63.0% Robust Failure Rate):** Because noisy prompts collapse $d_4$ to $\sim 0.18$ (which is $\le 0.60$), a drift-only rule triggers early exit on 100% of noisy inputs, resulting in severe degradation.
- **SNR-Only Baseline Performance:** On the development split, the SNR-only policy achieved 19.0% mean NFE reduction with a 2.0% robust failure rate. Independent calibration and held-out testing are required before making a reliability claim.
- **Binary SNR + $d_4$ Diagnostic Filter:** Adding the $d_4$ filter in the binary diagnostic policy reduced clean development failures from two cases to one while reducing mean NFE savings from 19.0% to 9.3%. A one-case difference is insufficient to establish a statistically reliable improvement.
- **Complete Sequential Scheduler Recovers Compute Efficiency:** Evaluating the complete sequential $\{4, 5, 6, 8\}$ policy restores substantial compute savings (**16.1% savings at $\lambda=0.60$** and **18.4% savings at $\lambda=0.70$**) with a **1.0% robust failure rate**. This occurs because boundary clean cases that do not pass $d_4 \le \lambda$ can safely exit at Step 5 (15 NFE) or Step 6 (17 NFE) instead of immediately defaulting to Step 8 (21 NFE).

---

## 6. Preregistration & Policy Freezing Before Task 5 Calibration

To maintain rigorous Learn-Then-Test guarantees and prevent model-family selection from leaking into calibration data, we freeze the following specifications using development-split insights only:

1. **Primary Policy Family:** Sequential Input-Adaptive Scheduler with prompt-SNR gate and normalized trajectory drift probe.
2. **Candidate Decision Sequence:** Evaluated in order $4 \to 5 \to 6 \to 8$. At candidate $i \in \{4, 5, 6\}$, the scheduler selects step $i$ if:
   $$\text{Prompt SNR} \ge \theta \; (20.0\text{ dB}) \quad \text{AND} \quad d_i \le \lambda$$
   If no candidate qualifies, the policy falls back to Reference Step 8.
3. **Preregistered Threshold Grid:** $\Lambda = [0.20, 0.25, 0.30, 0.35, \dots, 1.20]$ tested strictly in conservative-to-aggressive order.
4. **Statistical Target:** Primary Learn-Then-Test guarantee controlling robust speaker-level failure risk at level $\alpha = 0.05$ with confidence $1 - \delta = 0.90$ ($P(R(\pi_{\hat{\lambda}}) > \alpha) \le \delta$).
5. **Independent Calibration Unit:** Paired robust Bernoulli loss $\text{robust\_fail}_i(\lambda) = \max(\text{fail}_{i,\text{clean}}, \text{fail}_{i,\text{noisy15}})$ over $n = 200$ independent calibration speakers.
6. **Prespecified Comparators for Task 6:** Fixed steps {4, 5, 6, 8}, SNR-Only ($\theta = 20.0\text{ dB}$), and Drift-Only ($\lambda = 0.60$) are retained as prespecified baseline comparators on the held-out test split, but will not be selected post-hoc during calibration.

---

## 7. Quality & Computational Cost Trade-Off Summary

| Condition | Candidate Step | NFE | NFE Savings (%) | Mean WER (%) | Median WER (%) | Mean SIM | Median SIM | Mean UTMOS | Failure Rate (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `clean` | Step 4 | 13 | 38.1% | 2.17% | 0.00% | 0.9316 | 0.9459 | 3.8364 | 2.0% |
| `clean` | Step 5 | 15 | 28.6% | 1.88% | 0.00% | 0.9310 | 0.9403 | 3.8592 | 3.0% |
| `clean` | Step 6 | 17 | 19.0% | 1.81% | 0.00% | 0.9301 | 0.9386 | 3.8594 | 4.0% |
| `clean` | Step 8 | 21 | 0.0% | 1.81% | 0.00% | 0.9311 | 0.9402 | 3.8489 | 0.0% |
| `noisy15` | Step 4 | 13 | 38.1% | 27.35% | 17.42% | 0.6803 | 0.6804 | 1.2512 | 63.0% |
| `noisy15` | Step 5 | 15 | 28.6% | 70.40% | 11.81% | 0.6914 | 0.7011 | 1.2498 | 49.0% |
| `noisy15` | Step 6 | 17 | 19.0% | 64.17% | 10.00% | 0.6950 | 0.7035 | 1.2485 | 53.0% |
| `noisy15` | Step 8 | 21 | 0.0% | 18.23% | 10.00% | 0.6964 | 0.6942 | 1.2480 | 0.0% |