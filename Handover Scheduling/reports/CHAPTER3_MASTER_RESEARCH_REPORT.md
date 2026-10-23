# A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching TTS Inference
## Chapter 3 Comprehensive Research, Benchmark & Replication Master Report

**Thesis Title (proposal):** A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference  
**Proposed Method:** SNR + drift adaptive handover policy (Definition 3.4) with Learn-Then-Test calibration  
**Evaluation Cohort:** Full 500-Speaker Cohort from Seed-TTS English (`test-en`) (500 clean/noisy condition-pairs, or 1,000 condition-specific cases; 4,000 candidate-grid evaluations and 3,000 dedicated policy timing runs)  
**Primary Certified Decision Threshold:** $\hat{\lambda} = 0.50$ (Prespecified PAC Criterion: $P(R > 0.05) \le 0.10$)  
**Safety Gate Threshold:** $\theta = 20.0\text{ dB}$ (Dual-Mechanism Architecture)  
**Execution Targets:** 
1. **CPU Baseline:** AMD Ryzen 7 5700U CPU (8 cores / 16 threads, FP32, seed 42)
2. **GPU Verification & Dedicated Timing Benchmark:** NVIDIA Tesla T4 CUDA 15.64 GB (Google Colab, FP32/cuDNN, timing session `timing_session_20261020_130952`)  
**Status:** Phases 1–3 (CPU), the full 500-speaker GPU retest, the T4 timing benchmark and the ablation audits are complete. Proposal items still open are listed in Section 8.  
**Last Updated:** 23 October 2026  

---

## Executive Summary

Diffusion-based and flow-matching text-to-speech (TTS) architectures achieve high naturalness and speaker fidelity, but their multi-step numerical integration incurs substantial computational latency. In DMOSpeech 2, inference is accelerated by transferring intermediate representations from a high-capacity flow-matching teacher to a distilled student consistency model. However, deploying a fixed early handover schedule (e.g., Step 4) causes severe degradation when prompts are degraded by acoustic noise (exhibiting over 60% failure rates under 15 dB prompt noise). Conversely, employing a conservative fixed schedule (Step 8) presents an observed opportunity to reduce computation on clean, straightforward speech under the evaluated quality criteria.

This study investigates whether an **input-adaptive teacher-to-student handover scheduling policy** can reduce model evaluations and synthesis latency while controlling reference-relative quality degradation. It evaluates a proposed dual-mechanism architecture against fixed schedules and a simpler input-SNR gating baseline:
1. **Prompt SNR Safety Gate ($\theta = 20.0\text{ dB}$):** Pre-generation gating that routes speech prompts with low signal-to-noise ratio ($\text{SNR} < 20.0\text{ dB}$) to the full 8-step teacher trajectory, providing fallback protection under adverse acoustic conditions.
2. **Zero-NFE Trajectory Drift Probing ($\hat{\lambda} = 0.50$):** Evaluates normalized tangent deviation $d_i = \frac{\|\hat{x}_1(t_i) - \hat{x}_1(t_{i-1})\|_2}{\|\hat{x}_1(t_i)\|_2 + \epsilon}$ along early teacher steps as an algebraic byproduct of the Euler update without requiring additional forward passes, probing whether the intermediate velocity field has stabilized sufficiently for student handover at Steps 4, 5, or 6.
3. **Learn-Then-Test (LTT) Risk Control:** Formulates threshold selection as a multiple hypothesis testing problem with finite-sample Probably Approximately Correct (PAC) statistical bounds, bounding the held-out robust reference-relative added failure rate $R \le 5.0\%$ at $90\%$ confidence.

### Contributions and Their Status

The proposal's objectives amount to two main contributions, supported by an empirical characterization (Objective 1) and a benchmark (Objective 4):

| # | Contribution | Proposal Source | Status | Evidence |
|---|---|---|---|---|
| **C1** | **Algorithm:** the per-utterance SNR + drift handover policy. Training-free, zero extra forward passes, never costs more than Step 8 | Objective 2; Definition 3.4; Propositions 3.2, 3.4 | ⚠️ **Holds as a policy; the drift component is unestablished** | Safe and efficient on held-out speakers (Sections 4.3, 4.8). The ablations (Sections 4.9, 4.10) show that the SNR gate carries the benefit and drift adds no measurable improvement |
| **C2** | **Framework:** risk-calibrated handover via Learn-Then-Test | Objective 3; Proposition 3.6 | ✅ **Holds** | λ̂ = 0.50 certified; held-out risk 2.5% ≤ 5%; 94% of 100 recalibrated re-splits ≤ 5% (Sections 4.2, 4.4, 4.6) |
| (support) | Characterization of the switching-time space | Objective 1 | ✅ Holds | Noisy prompts fail 49–64% at Steps 4–6; clean failures sit at the 2.5% seed-noise floor (Sections 4.1, 4.9.1) |

### Verdict at a Glance (Held-Out Test Split, 200 Speakers)

| Comparison | Result | Verdict |
|---|---|---|
| **Proposed vs. DMOSpeech 2 fixed Step 8** (main comparison) | 18.15 vs. 21.00 NFE (−13.6%); T4 latency −13.3% overall, −27.0% on clean prompts; added failures 2.5% ≤ 5%; quality TOST-equivalent | ✅ **Proposed wins** |
| **Proposed vs. SNR-only rule** (Table 3.8, earliest candidate = Step 4) | 2.5% vs. 5.5%. With θ also calibrated by LTT (θ̂ = 18.5 dB), SNR-only still exceeds α on test. McNemar 7 vs. 1, p = 0.070 | ✅ **Proposed holds the guarantee, SNR-only does not** (a practical, not statistically significant, difference) |
| **Proposed vs. SNR gate + fixed Step 5** (additional ablation, not in Table 3.8) | 2.5% / 18.15 vs. 2.0% / 18.03; McNemar p = 1.0 (CPU), 0.51 (GPU) | ≈ **Tie**: drift's step choices add nothing measurable |

### Summary of Primary Held-Out Benchmark Findings (200 Test Speakers / 400 Cases)

| Policy | Mean Core Synthesis Runtime (GPU) | Runtime Reduction vs. Step 8 | Mean Teacher + Student NFE | NFE Reduction vs. Step 8 | Robust Quality Failures (Earlier GPU Eval) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Step 8 Reference** | 3.329 s | Reference | 21.00 | Reference | 0 / 200 (0.0%) |
| **SNR-Only Baseline** | 2.712 s | **18.52%** | 17.04 | **18.9%** | 4 / 200 (2.0%) |
| **Proposed Scheduler ($\pi^*$)** | 2.885 s | **13.34%** | 18.15 | **13.6%** | 3 / 200 (1.5%) |

*Note on Latency vs. NFE:* The proposed policy saves approximately 0.444 seconds per case against Step 8 and takes approximately 0.173 seconds longer than SNR-only. This latter difference reflects extra teacher execution steps and scheduler overhead; it is not a measurement of the drift calculation alone. The earlier GPU quality evaluation recorded three robust speaker failures for Proposed and four for SNR-only among the 200 test speakers, a difference that is statistically inconclusive (two-sided exact McNemar $p = 1.0000$). The incremental quality benefit of trajectory drift over SNR-only remains unestablished.

```mermaid
flowchart TD
    subgraph S1 ["Total Workload (500 Speakers / 500 Pairs = 1,000 Cases)"]
        D1["Phase 1: Development Split<br/>100 Speakers (200 Cases / 800 Runs)<br/>Timing Trade-offs & Policy Freezing"]
        D2["Phase 2: Calibration Split<br/>200 Speakers (400 Cases / 1,600 Runs)<br/>LTT Risk Calibration (λ̂ = 0.50 Certified, p = 0.0264)"]
        D3["Phase 3: Held-Out Test Split<br/>200 Speakers (400 Cases / 1,600 Runs)<br/>Comparative Benchmarks & Verification"]
    end

    subgraph S2 ["Core Held-Out Findings on 200 Test Speakers"]
        R1["Compute & Latency Reductions:<br/>13.6% Forward NFE Reduction (18.15 vs 21.00, p = 1.56e-35)<br/>13.34% GPU Latency Reduction (2.885s vs 3.329s, p = 7.29e-35)"]
        R2["CPU Quality Baseline Parity:<br/>No Statistically Significant Difference vs Step 8<br/>Holm-Bonferroni Adj. p = 1.000 (WER, SIM, UTMOS)"]
        R3["PAC Risk Criterion Met:<br/>CPU Held-Out Added Failure: 2.50% (5/200) ≤ 5% PAC limit<br/>GPU Held-Out Added Failure: 1.50% (3/200) ≤ 5% PAC limit"]
    end

    D1 --> D2 --> D3 --> S2
    style S1 fill:#f8f9fa,stroke:#6c757d,stroke-width:2px
    style S2 fill:#eafaf1,stroke:#27ae60,stroke-width:2px
```

---

## 1. Environment, Checksums & Hardware Specifications

### 1.1 Repository & Commit Information
- **Repository:** `Shonget-Thesis/Shonget-DMOSpeech2`
- **Current Repository Commit Hash:** `ac269f197ad59a5ea6161b3f960771930a3de8b7`
- **Upstream Repository:** `yl4579/DMOSpeech2`
- **Upstream Base Commit Hash:** `10c2255a59b96c311c317536603a18672e69ac84`
- **Working Directory:** `Handover Scheduling/`
- **Timing Benchmark Source Snapshot:** Dedicated full-pipeline timing harness `colab/run_gpu_full_pipeline_timing_benchmark.py` (Session ID: `timing_session_20261020_130952`).

### 1.2 Model Checkpoints & SHA-256 Hashes
| Model / Component | File / Cache Location | Size (Bytes) | SHA-256 Checksum |
|---|---|---|---|
| **DMOSpeech 2 (Teacher & Student)** | `ckpts/model_85000.pt` | 4,019,822,332 | `e143188f7c46b69abb60786937d79b89dac9777d3c4c1e91a97634bcd39ebf2c` |
| **Duration Predictor (GRPO)** | `ckpts/model_1500.pt` | 270,615,794 | `6bd50428f3b5f0f25c1a60a0053f40ee47c817bda03c8ee120100f22680ae4c2` |
| **Whisper large-v3 ASR** | HuggingFace `model.safetensors` (Commit `06f233f`) | 3,087,130,976 | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| **WavLM Speaker Verifier (bin)** | HuggingFace `microsoft/wavlm-base-plus-sv` (`pytorch_model.bin`) | 404,547,053 | `e906bce2fa42fb497a1d1a9ecf81548adb7e03b12a5644e32d2f42f0d6500fad` |
| **WavLM Speaker Verifier (safe)** | HuggingFace `microsoft/wavlm-base-plus-sv` (`model.safetensors`) | 404,507,764 | `e7b31a8e62a136830fe105fbcfa89f07f108d014b0fa55fd4aebdf4c72886d4c` |
| **SpeechMOS UTMOS Predictor** | TorchHub `utmos22_strong_step7459_v1.pt` (`tarepan/SpeechMOS:v1.2.0`) | 411,179,119 | `38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff` |

> **Speaker-verifier note:** the SIM model actually loaded is `microsoft/wavlm-base-plus-sv` (WavLM **Base+** with an x-vector head). The proposal (Table 3.5), DMOSpeech 2 and the code comments in `core/metrics.py` say "WavLM-large". All failure decisions compare a candidate with the Step-8 reference scored by the same verifier, so internal results are valid. Absolute SIM values (≈ 0.94 on clean prompts) are **not comparable** with published DMOSpeech 2 SIM, which uses the WavLM-large Seed-TTS verifier. Chapter 3 must name the model actually used, or all audio must be re-scored with WavLM-large.

### 1.3 Hardware Configuration & Numerical Precision
- **Host CPU Platform (Phases 1, 2, 3 Baseline):**
  - Processor: AMD Ryzen 7 5700U with Radeon Graphics (8 physical cores, 16 logical threads, base 1.8 GHz, boost 4.3 GHz)
  - Host RAM: 16.0 GB DDR4, Windows 11 Home / Pro (AMD64 build).
  - Software Stack: Python `3.12.6`, PyTorch `2.5.1+cu121`, torchaudio `2.5.1+cu121`, scipy `1.17.1`, pandas `3.0.3`.
  - Execution Target: In the CPU baseline experiments, all synthesis and quality evaluation pipelines executed on the host CPU in evaluation mode (`torch.inference_mode()`) using full 32-bit floating point precision (`torch.float32` / FP32) with isolated seeded generators (`base_seed = 42`).
- **Target GPU Platform (GPU Quality Re-test & Dedicated Timing Benchmark):**
  - Compute Target: NVIDIA Tesla T4 GPU (15.64 GB VRAM, Compute Capability 7.5, Google Colab).
  - Environment: Ubuntu 22.04 LTS, CUDA 12.2 / cuDNN, PyTorch `2.5.1+cu121`, FP32 precision (`torch.float32`).
  - Synchronization: Dedicated hardware barriers (`torch.cuda.synchronize()`) before start and stop timestamps.

---

## 2. Theoretical Framework & Mathematical Formulations

### 2.1 Swayed Euler Timing Grid
The teacher flow-matching model integrates the velocity field $v_\theta(x_t, t)$ over $N = 16$ swayed time steps:
$$t_i = 1 - \cos\left( \frac{\pi}{2} \cdot \frac{i}{16} \right), \quad i \in \{0, 1, \dots, 16\}$$
Candidate teacher handover points $s \in \{4, 5, 6, 8\}$ correspond to swayed times:
$$t(s) \in [0.0761, 0.1181, 0.1685, 0.2929]$$

Upon handover at step $s$, the student consistency model completes synthesis using restart level $j = 1$ over consistency times $\tau \in [0.0761, 0.2929, 0.6173]$, requiring exactly **3 student forward passes**.

*Grid check (proposal Remark 3.1):* the released sampler (`src/infer.py:520-523`) stops at the first grid point above $t_{\text{switch}}$. On this 16-interval grid, $t_{\text{switch}} = 0.25$ gives $k = 8$ ($t_8 = 0.2929$), our Step-8 reference. The DMOSpeech 2 paper's "14 teacher + 2 student steps at $t_{\text{switch}} = 0.25$" arises only on a **32-interval** grid read as "last point not above 0.25": $t_{14} = 0.2270$, **not ≈ 0.236** as Remark 3.1 states. This is the same flow time as our Step 7. The released `generate_teacher_only` defaults to `teacher_steps=32` (`src/infer.py:341`), which supports the 32-interval reading. Forward passes for the paper's configuration: $2 \times 14 + 2 = 30$ vs. $64$ (2.13×), or $32$ vs. $64$ (2.0×) with the endpoint evaluation. Details and figure: `reports/PROPOSAL_TODO_CHECKS_REPORT.md`, `figures/sway_grid_tswitch.png`.

### 2.2 Forward NFE Accounting
In this implementation, Classifier-Free Guidance is evaluated by executing separate conditional and unconditional forward calls through the teacher network at each evaluated step index (the numeric CFG scale itself does not dictate the number of calls, but the dual-forward evaluation requires 2 calls per step). Therefore, evaluating teacher indices $0, 1, \dots, s$ requires $2 \times (s + 1)$ forward passes. Adding the 3 student forward passes gives:
$$\text{NFE}(s) = 2 \times (s + 1) + 3$$

| Handover Step ($s$) | Teacher Sway Time $t(s)$ | Evaluated Indices | Teacher Calls | Student Calls ($j=1$) | Total Forward NFE | NFE Reduction vs. Index 8 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Step 4** | 0.0761 | $0 \dots 4$ | $2 \times 5 = 10$ | 3 | **13** | **38.1%** |
| **Step 5** | 0.1181 | $0 \dots 5$ | $2 \times 6 = 12$ | 3 | **15** | **28.6%** |
| **Step 6** | 0.1685 | $0 \dots 6$ | $2 \times 7 = 14$ | 3 | **17** | **19.0%** |
| **Handover Index 8 (Ref)** | 0.2929 | $0 \dots 8$ | $2 \times 9 = 18$ | 3 | **21** | **0.0% (Baseline)** |

*Scope of NFE Convention:* This convention counts model neural function evaluations across the teacher and student networks. Vocoder audio decoding is included in wall-clock latency measurements but excluded from this NFE accounting. When precision matters, "handover index 8" is preferred to denote that indices 0 through 8 are evaluated, requiring 18 teacher calls.

### 2.3 Normalized Endpoint Trajectory Drift Probing
At each teacher integration step $t_i$, the instantaneous endpoint prediction is computed algebraically using the guided velocity field:
$$\hat{x}_1(t_i) = x_{t_i} + (1 - t_i) v_{\text{guided}}(x_{t_i}, t_i)$$
where $v_{\text{guided}} = v_{\text{uncond}} + \text{cfg} \cdot (v_{\text{cond}} - v_{\text{uncond}})$. The normalized tangent drift $d_i$ evaluates consecutive endpoint changes over generated speech frames (excluding reference prompt frames):
$$d_i = \frac{\|\hat{x}_1(t_i) - \hat{x}_1(t_{i-1})\|_2}{\|\hat{x}_1(t_i)\|_2 + \epsilon}$$

> **Interpretation & Scope:** The normalized endpoint-drift score measures changes between consecutive teacher endpoint predictions over generated speech frames. The scheduler uses a score below the calibrated threshold as a heuristic for permitting early handover. Low drift does not guarantee accurate speech or successful student completion. Its predictive usefulness and incremental benefit over SNR-only scheduling are evaluated empirically.

### 2.4 Prompt SNR Safety Gate
Prompt signal-to-noise ratio (SNR) is evaluated on the raw reference audio waveform before RMS normalization using an energy-percentile VAD (upper 50% vs. lowest 10% energy frames). Prompts with $\text{SNR} < \theta = 20.0\text{ dB}$ bypass drift probing and execute the full 8-step teacher trajectory.

### 2.5 Reference-Relative Failure Criterion (Definition 3.1)
An utterance is defined as exhibiting quality degradation exceeding specified margins relative to its matched Step-8 reference:
$$\text{Fail}(x, s) \iff \text{WER}(x, s) > \max(0.125, \text{WER}_{\text{ref}} + 0.01) \lor \text{SIM}(x, s) < \text{SIM}_{\text{ref}} - 0.02$$

- **Threshold Definitions:** The formula applies a **WER tolerance floor** of $0.125$ ($12.5\%$) alongside a relative WER margin of $+0.01$ ($1.0\%$ tolerance), and a Speaker Similarity margin of $-0.02$ ($0.02$ cosine tolerance). Because of the $\max()$ formulation, $12.5\%$ is not an absolute upper bound ceiling.
- **Robust Speaker Failure:** A speaker fails under policy $\pi$ if *either* their clean *or* noisy15 utterance exhibits degradation exceeding the margins:
  $$\text{RobustFail}_i = \max(\text{Fail}_{i, \text{clean}}, \text{Fail}_{i, \text{noisy15}})$$
- **Interpretation & Scope:** This metric evaluates *reference-relative added degradation* rather than absolute speech perfection. Passing indicates no degradation exceeding the specified WER/SIM margins compared to running the full Step-8 teacher baseline; it does not imply identical audio output or absolute speech perfection. Crucially, utterances routed to Step 8 (including noisy prompts diverted by the SNR gate) inherit the baseline Step-8 output and whatever acoustic artifacts or WER limits that reference baseline exhibits under adverse noise.

---

## 3. Disjoint Speaker Partitioning & Experimental Protocol

The evaluation cohort consists of **500 unique, disjoint speakers** sampled from Seed-TTS English (`test-en`). To avoid data leakage, each speaker appears in exactly one partition. Each speaker provides one clean and one noisy15 utterance, producing **500 clean/noisy condition pairs, or 1,000 condition-specific cases**:

| Partition Split | Manifest File | Speakers | Condition Cases (`clean` / `noisy15`) | Candidate Evaluations (Steps 4, 5, 6, 8) | Dedicated Timed Policy Runs | Purpose in Methodology | Status |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Phase 1: Development** | `data/splits/development_manifest.json` | 100 | 200 (100 clean / 100 noisy) | 800 | 600 | Characterize timing trade-offs; freeze candidate ladder & SNR threshold | **100% COMPLETE** |
| **Phase 2: Calibration** | `data/splits/calibration_manifest.json` | 200 | 400 (200 clean / 200 noisy) | 1,600 | 1,200 | Fixed-sequence LTT hypothesis testing; certify threshold $\hat{\lambda}$ | **100% COMPLETE** |
| **Phase 3: Held-Out Test** | `data/splits/test_manifest.json` | 200 | 400 (200 clean / 200 noisy) | 1,600 | 1,200 | Unbiased held-out benchmarking; Wilcoxon & McNemar significance | **100% COMPLETE** |
| **Total Cohort** | `data/splits/split_summary.json` | **500** | **1,000 cases (500 pairs)** | **4,000** | **3,000** | Complete Chapter 3 experimental scope across Seed-TTS English | **100% COMPLETE** |

*Source Corpus and Feasibility:* Seed-TTS test-en contains **1,088 utterances from 666 unique prompt speakers** (`data/splits/split_summary.json`). The 500 speakers used leave 166 unused, later used as fresh speakers in Phase 12. With $n = 200$ calibration speakers, $\alpha = 0.05$ and $\delta = 0.10$, up to 5 added failures still certify a threshold; the minimum certifiable size is $n = 45$. The proposal's Table 3.7 was re-computed exactly. Speaker identity is the prompt clip ID (`common_voice_en_<clip>`), not the Common Voice contributor ID, so disjointness holds at the level of Seed-TTS prompt identities. A heuristic check found no cross-split clips within 3 IDs of each other.

*Accounting Distinction:* The 4,000 candidate-grid evaluations refer to evaluating individual fixed candidate steps ($s \in \{4, 5, 6, 8\}$) across all 1,000 cases. The 3,000 policy timing runs refer to evaluating the complete end-to-end execution of the three dynamic policies (`proposed`, `snr_only`, `step8`) across all 1,000 cases.

---

## 4. Empirical Results Across All 500 Speakers

### 4.1 Phase 1 Development Findings ($N=100$)
*Regenerated strictly from raw speaker records in `phase1_dev_grid_100spk/results/grid.csv`:*

| Candidate Handover Step | Mean Forward NFE | NFE Savings (%) | Clean Failure Rate (%) | Noisy15 Failure Rate (%) | Robust Failure Rate (%) | Exact 95% Clopper-Pearson CI (Clean / Noisy) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Step 4** | 13.00 | **38.1%** | 2.00% (2 / 100) | 63.00% (63 / 100) | **64.00%** (64 / 100) | [0.24%, 7.04%] / [52.76%, 72.44%] |
| **Fixed Step 5** | 15.00 | **28.6%** | 3.00% (3 / 100) | 49.00% (49 / 100) | **51.00%** (51 / 100) | [0.62%, 8.52%] / [38.86%, 59.20%] |
| **Fixed Step 6** | 17.00 | **19.0%** | 4.00% (4 / 100) | 53.00% (53 / 100) | **56.00%** (56 / 100) | [1.10%, 9.93%] / [42.76%, 63.06%] |
| **Fixed Step 8 (Ref)** | 21.00 | **0.0%** | 0.00% (0 / 100) | 0.00% (0 / 100) | **0.00%** (0 / 100) | 0% by definition |

*Key Takeaway:* Fixed early handover causes an immediate 49%–63% failure rate under 15 dB prompt noise. Consequently, the Prompt SNR threshold was strictly frozen at $\theta = 20.0\text{ dB}$ before advancing to calibration.

---

### 4.2 Phase 2 Learn-Then-Test PAC Risk Calibration ($N=200$)
Fixed-sequence hypothesis testing over ordered candidate thresholds across 200 calibration speakers (400 cases / 1,600 evaluations) with PAC parameters $\alpha = 0.05$ (5% max risk) and $\delta = 0.10$ (90% confidence). The statistical guarantee applies to population risk under the procedure's sampling and distribution assumptions:

$$H_m: R(\pi_{\lambda_m}) > 0.05 \quad \text{vs.} \quad K_m: R(\pi_{\lambda_m}) \le 0.05$$
$$p_m = P(\text{Bin}(n, \alpha) \le k_m) = \sum_{j=0}^{k_m} \binom{n}{j} \alpha^j (1 - \alpha)^{n - j}$$

| Tested $\lambda$ | Robust Failures ($k$) | Total Speakers ($n$) | Empirical Risk ($\hat{R}$) | Binomial $p$-value | Decision Status | Mean Forward NFE | Clean Risk | Noisy15 Risk |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.20** | 0 | 200 | 0.00% | $3.51 \times 10^{-5}$ | **ACCEPTED** | 20.83 | 0.00% | 0.00% |
| **0.25** | 2 | 200 | 1.00% | $2.34 \times 10^{-3}$ | **ACCEPTED** | 20.56 | 1.00% | 0.00% |
| **0.30** | 3 | 200 | 1.50% | $9.05 \times 10^{-3}$ | **ACCEPTED** | 20.07 | 1.50% | 0.00% |
| **0.35** | 3 | 200 | 1.50% | $9.05 \times 10^{-3}$ | **ACCEPTED** | 19.55 | 1.50% | 0.00% |
| **0.40** | 3 | 200 | 1.50% | $9.05 \times 10^{-3}$ | **ACCEPTED** | 18.90 | 1.50% | 0.00% |
| **0.45** | 5 | 200 | 2.50% | $6.23 \times 10^{-2}$ | **ACCEPTED** | 18.55 | 2.50% | 0.00% |
| **0.50** | **4** | **200** | **2.00%** | **0.0264** | **ACCEPTED (CERTIFIED $\hat{\lambda}$)** | **18.19** | **2.00%** | **0.00%** |
| **0.55** | 6 | 200 | 3.00% | 0.1237 | **REJECTED_STOP** | 17.82 | 3.00% | 0.00% |
| **0.60** | 3 | 200 | 1.50% | — | UNTESTED_STOPPED | 17.59 | 1.50% | 0.00% |

**Calibration Verdict:** **$\hat{\lambda} = 0.50$** ($k = 4 / 200$, $\hat{R} = 2.00\%$, $p = 0.0264 \le \delta = 0.10$). Sequence stopped at $\lambda = 0.55$ ($p = 0.1237 > 0.10$). Threshold $\hat{\lambda} = 0.50$ was strictly frozen before evaluating the held-out test split. An empirical proportion below 5% is not by itself a population guarantee; the formal guarantee derives from the bounded binomial testing sequence under the PAC framework.

---

### 4.3 Phase 3 Held-Out Test Split Master Benchmark ($N=200$)
Evaluated on 200 previously unseen test speakers (400 cases / 1,600 candidate evaluations). Confidence intervals report 95% cluster-bootstrap bounds resampling speakers ($B=1,000$):

| Handover Policy | Decision Rule | Mean Forward NFE | NFE 95% CI | NFE Savings vs Ref (%) | WER (%) | Speaker SIM vs GT | UTMOS MOS | Robust Failure Rate (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Step 4** | Always Step 4 | 13.00 | [13.00, 13.00] | **38.1%** | 25.96% | 0.8163 | 2.5716 | **69.50%** (139 / 200) |
| **Fixed Step 5** | Always Step 5 | 15.00 | [15.00, 15.00] | **28.6%** | 12.92% | 0.8177 | 2.5701 | **56.50%** (113 / 200) |
| **Fixed Step 6** | Always Step 6 | 17.00 | [17.00, 17.00] | **19.0%** | 12.37% | 0.8209 | 2.5694 | **53.50%** (107 / 200) |
| **Fixed Step 8 (Ref)** | Always Step 8 | 21.00 | [21.00, 21.00] | **0.0%** | 48.29% | 0.8254 | 2.5663 | **0.00%** (0 / 200) |
| **SNR-Only Policy** | If $\text{SNR} \ge 20 \to 4$, else $8$ | 17.04 | [17.00, 17.10] | **18.9%** | 48.54% | 0.8252 | 2.5697 | **5.50%** (11 / 200) |
| **Proposed Adaptive ($\hat{\lambda}=0.50$)** | Seq $4\to 5\to 6$ if $d_i \le 0.50$ & $\text{SNR} \ge 20$ | **18.15** | **[18.05, 18.27]** | **13.6%** | **48.49%** | **0.8252** | **2.5692** | **2.50%** (5 / 200) |
| **Proposed (No SNR Gate)** | Seq $4\to 5\to 6$ if $d_i \le 0.50$ (Unprotected) | 14.12 | [14.03, 14.23] | **32.7%** | 25.91% | 0.8164 | 2.5711 | **70.00%** (140 / 200) |
| **Grid Oracle** | Cheapest non-failing candidate in grid | 14.94 | [14.71, 15.17] | **28.9%** | 9.33% | 0.8320 | 2.5706 | **0.00%** (0 / 200) |

---

### 4.4 Statistical Significance Analysis vs. Step-8 Reference (Speaker-Level Aggregation, $N=200$)
To respect within-speaker correlation, paired comparisons were aggregated to the speaker level ($N=200$ test speakers, averaging clean and noisy pairs). Two-sided paired Wilcoxon signed-rank tests were evaluated against the Step-8 reference with Holm-Bonferroni correction across the comparison set:

| Policy Under Test | NFE Adj. $p$-value | WER Adj. $p$-value | SIM Adj. $p$-value | UTMOS Adj. $p$-value | Comprehensive Statistical Verdict |
|---|:---:|:---:|:---:|:---:|---|
| **Fixed Step 4** | $1.25 \times 10^{-44}$ | $4.61 \times 10^{-5}$ | $1.49 \times 10^{-3}$ | $1.0000$ | Statistically significant degradation in WER and SIM |
| **Fixed Step 5** | $1.25 \times 10^{-44}$ | $1.0000$ | $1.06 \times 10^{-3}$ | $1.0000$ | Statistically significant loss in speaker similarity ($p = 1.06 \times 10^{-3}$) |
| **Fixed Step 6** | $1.25 \times 10^{-44}$ | $1.0000$ | $0.0420$ | $1.0000$ | Marginal quality, but 53.50% robust failure rate |
| **SNR-Only Policy** | $2.85 \times 10^{-44}$ | $1.0000$ | $1.0000$ | $1.0000$ | Significant NFE reduction; no detectable difference on WER/SIM/UTMOS |
| **Proposed Adaptive ($\hat{\lambda}=0.50$)** | $\mathbf{9.38 \times 10^{-35}}$ | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | **Significant compute reduction ($p < 10^{-34}$) with no statistically significant difference detected across WER, SIM, or UTMOS** |
| **Proposed (No SNR Gate)** | $3.53 \times 10^{-35}$ | $5.73 \times 10^{-5}$ | $2.30 \times 10^{-3}$ | $0.7971$ | Severe collapse on noisy condition without prompt gate ($p < 0.0001$) |

*Methodological Note on Quality Equivalence:* The absence of statistically significant differences on aggregate WER, SIM, and UTMOS after Holm-Bonferroni correction does not establish mathematical equivalence or perceptual identity. Rather, it indicates that at the tested sample size ($N=200$), the test does not detect an aggregate degradation compared to running the full Step-8 teacher baseline.

**Realized Risk Verification on Test Split:**
- **Robust Added Failure Rate:** **2.50% (5 / 200)** (Exact 95% Clopper-Pearson Interval: **[0.82%, 5.74%]**).
- `clean` Condition Failure Rate: **2.50% (5 / 200)**; `noisy15` Condition Failure Rate: **0.00% (0 / 200)** (100% of noisy prompts routed to Step 8 by the fallback gate).

---

### 4.5 Paired Contingency Comparison: Proposed Adaptive vs. SNR-Only Baseline
Both policies were evaluated on the identical 200 test speakers:

| Proposed Adaptive Policy ($\hat{\lambda}=0.50$) | SNR-Only Policy: Pass | SNR-Only Policy: Fail | Marginal Total |
|---|:---:|:---:|:---:|
| **Proposed: Pass** | 188 | 7 | **195** |
| **Proposed: Fail** | 1 | 4 | **5** |
| **Marginal Total** | **189** | **11** | **200** |

- **Concordant Pairs:** 188 passed by both policies; 4 failed by both policies.
- **Discordant Pairs:** 7 failed by SNR-only but rescued by Proposed; 1 failed by Proposed but passed by SNR-only.
- **Exact McNemar Test (Binomial on Discordant Pairs):**
  - Two-Sided: $p = P(\text{Bin}(8, 0.5) \le 1) \times 2 = 0.0703$ (not statistically significant at $\alpha = 0.05$).
  - Exploratory One-Sided: $p = 0.0352$ (reported as exploratory because the test direction was not documented prior to examining outcomes).

*Analytical Interpretation:* While the CPU held-out split shows a descriptive reduction from 11 down to 5 failures, the two-sided test does not achieve statistical significance ($p = 0.0703$). Furthermore, on the GPU held-out test split, SNR-only achieves 4/200 failures versus 3/200 for the proposed method ($p = 1.0000$). The incremental benefit of drift over SNR-only is therefore not statistically established.

---

### 4.6 Supplementary Repeated-Split Stability Analysis (400 Pooled Speakers, 100 Re-splits)
Calibration ($N=200$) and test ($N=200$) cohorts were pooled (400 speakers total) and partitioned into **100 random speaker-disjoint 50/50 re-splits** as supplementary resampling of existing data (this analysis does not constitute independent replication or proof of the population PAC guarantee):

| Metric | Result Across 100 Random Resampled Splits |
|---|:---:|
| **Admissible Policy Discovery Rate** | **100.0%** (0 / 100 splits fell back to reference) |
| **Modal Certified Threshold ($\hat{\lambda}$)** | **0.50** (33.0% modal, >80% within $[0.45, 0.65]$) |
| **Held-Out Test Robust Risk (Mean ± Std)** | **2.89% ± 1.27%** |
| **Held-Out Test Robust Risk (5th / Median / 95th / Max)** | **0.50% / 3.25% / 4.50% / 6.00%** |
| **Proportion of Resamples with Risk $\le 5.0\%$** | **94.0%** (94 / 100 splits) |
| **Held-Out Test Mean NFE (Mean ± Std)** | **17.92 ± 0.59 forward passes** |

---

### 4.7 Cross-Split Consistency Across All 500 Speakers (CPU Baseline)
Evaluating the frozen policies consistently across the three disjoint splits on the AMD Ryzen 7 5700U CPU:

| Evaluated Metric | Phase 1: Dev ($N=100$) | Phase 2: Calib ($N=200$) | Phase 3: Held-Out Test ($N=200$) | Overall Full Cohort Pattern |
|---|:---:|:---:|:---:|:---:|
| **Fixed Step 4 Robust Failure** | 64.00% (64 / 100) | 60.50% (121 / 200) | **69.50% (139 / 200)** | Severe degradation across all cohorts |
| **Noisy15 Step 4 Failure** | 63.00% (63 / 100) | 58.00% (116 / 200) | **69.00% (138 / 200)** | Catastrophic collapse without prompt gate |
| **SNR-Only Robust Failure** | 2.00% (2 / 100) | 1.50% (3 / 200) | **5.50% (11 / 200)** | Strong baseline; risk elevated on CPU test split |
| **Proposed Adaptive ($\hat{\lambda}=0.50$) Mean NFE** | **18.05** | **18.19** | **18.15** | Stable ~13.6% compute reduction across splits |
| **Proposed Adaptive Robust Failure** | **2.00% (2 / 100)** | **2.00% (4 / 200)** | **2.50% (5 / 200)** | Empirically bounded within 5.0% across all splits |
| **Proposed Adaptive Noisy Failure** | **0.00% (0 / 100)** | **0.00% (0 / 200)** | **0.00% (0 / 200)** | Zero observed added failures under 15 dB noise |

---

### 4.8 Dedicated Full-Pipeline GPU Timing Benchmark (NVIDIA Tesla T4)

To evaluate real-world inference throughput, a dedicated full-pipeline timing experiment was executed on an **NVIDIA Tesla T4 GPU (15.64 GB VRAM)** using Google Colab (`colab/run_gpu_full_pipeline_timing_benchmark.py`, Session ID: `timing_session_20261020_130952`). 

#### 4.8.1 Timing Methodology & Measurement Boundary
> **Declared Measurement Boundary:** Core synthesis includes applicable prompt-SNR estimation, teacher execution, scheduler decisions, student completion, and vocoder decoding. Input preprocessing, duration prediction, model loading, file writing, and offline quality evaluation are reported separately where excluded from this boundary.

Hardware timers wrapped the execution boundary with explicit synchronization barriers:
```python
torch.cuda.synchronize()
t0 = time.perf_counter()
# [Prompt SNR estimation (if policy requires)]
# [Teacher Euler steps with online early exit logic]
# [Student consistency multistep solver (j=1, 3 passes)]
# [Vocos neural vocoder audio decoding]
torch.cuda.synchronize()
core_synthesis_time = time.perf_counter() - t0
```
Warm-up passes (5 utterances) preceded data logging, and policy measurement order was systematically balanced across utterances to prevent thermal or memory caching bias.

#### 4.8.2 Primary Held-Out Test Set Timing Results (200 Test Speakers / 400 Cases)

| Policy | Mean Latency (s) | Median Latency (s) | Std Latency (s) | Mean RTF | Mean Total NFE | Latency Reduction vs. Step 8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Step 8 Reference** | 3.329 | 3.369 | 0.637 | 0.715 | 21.00 | Reference |
| **SNR-Only Baseline** | 2.712 | 2.548 | 0.852 | 0.579 | 17.04 | **18.52%** |
| **Proposed Scheduler ($\pi^*$)** | **2.885** | **2.856** | 0.754 | **0.617** | **18.15** | **13.34%** |

- **Paired Latency Differences (Speaker-Level Aggregation, $N=200$):**
  - Proposed vs. Step 8: Mean difference $= +0.4440\text{s}$ ($p = 7.29 \times 10^{-35}$, Wilcoxon signed-rank, greater).
  - SNR-Only vs. Step 8: Mean difference $= +0.6167\text{s}$ ($p = 7.29 \times 10^{-35}$, Wilcoxon signed-rank, greater).
  - Proposed vs. SNR-Only: Mean difference $= -0.1729\text{s}$ ($p = 2.46 \times 10^{-31}$, Wilcoxon signed-rank, two-sided).
- **Speaker-Clustered Bootstrap 95% CIs ($B=1,000$ Resamples):**
  - Proposed Latency: Mean $2.8844\text{s}$ [95% CI: $2.8077\text{s}, 2.9638\text{s}$].
  - Step 8 Latency: Mean $3.3284\text{s}$ [95% CI: $3.2437\text{s}, 3.4143\text{s}$].
  - Proposed Savings vs. Step 8: Mean $+0.4441\text{s}$ [95% CI: $+0.4244\text{s}, +0.4651\text{s}$].

#### 4.8.3 Timing Breakdown by Acoustic Condition (Test Cohort)
| Condition | Policy | Mean Core Latency (s) | Mean RTF | Mean Total NFE | Mean WER | Speaker SIM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | **Proposed ($\pi^*$)** | **2.405** | **0.529** | **15.30** | **0.0124** | **0.9349** |
| Clean | SNR-Only | 2.059 | 0.453 | 13.08 | 0.0131 | 0.9354 |
| Clean | Step 8 | 3.293 | 0.725 | 21.00 | 0.0139 | 0.9337 |
| **Noisy15** | **Proposed ($\pi^*$)** | **3.366** | **0.706** | **21.00** | **0.4603** | **0.7149** |
| Noisy15 | SNR-Only | 3.366 | 0.706 | 21.00 | 0.4603 | 0.7149 |
| Noisy15 | Step 8 | 3.365 | 0.706 | 21.00 | 0.4603 | 0.7149 |

On clean speech, Proposed exits between Steps 4 and 6 (mean exit step 5.15), reducing clean latency by **26.98%** ($2.405\text{s}$ vs $3.293\text{s}$). On noisy speech, the prompt SNR gate triggers 100% fallback to Step 8.

#### 4.8.4 Direct 1-to-1 CPU vs. GPU Comparison (200 Held-Out Test Speakers)
Matching all candidate evaluations for the 200 held-out test speakers across identical speaker IDs, conditions, steps, and seeds:

| Evaluation Dimension | Metric / Comparison | Quantitative Result | Analytical Interpretation |
|---|---|:---:|---|
| **Trajectory Curvature** | Mean Absolute Difference in $d_4$ | **0.000025** | Mean absolute difference of $2.5 \times 10^{-5}$ across backends |
| | Mean Absolute Difference in $d_5$ | **0.000028** | Mean absolute difference of $2.8 \times 10^{-5}$ across backends |
| | Mean Absolute Difference in $d_6$ | **0.000034** | Mean absolute difference of $3.4 \times 10^{-5}$ across backends |
| **Handover Decision** | Clean Handover Step Agreement | **100.00% (200 / 200)** | Policy selects the exact same step on clean speech across platforms |
| **Compute Savings** | Held-Out Mean Forward NFE | **18.15 vs. 18.15** | Identical 13.6% forward NFE reduction on CPU and GPU |
| **Observed Added Risk**| Held-Out Robust Added Risk | **2.50% (CPU) vs. 1.50% (GPU)** | Both strictly within the 5.0% empirical risk ceiling |
| **Adverse Noise Safety**| Noisy15 Robust Added Risk | **0.00% (CPU) vs. 0.00% (GPU)** | Zero observed added failures relative to Step 8 under 15 dB noise |
| **Acoustic Metrics** | Absolute Difference in Speaker SIM | **0.0367** | Divergence reflects independent CPU/CUDA random generator implementations |
| | Absolute Difference in UTMOS | **0.0978** | Metric variation reflecting non-deterministic backend arithmetic |

*Note on Pseudo-Random Generators:* In PyTorch, identical seeds do not guarantee identical random tensor sequences across CPU and CUDA backends. Acoustic differences between platforms arise from both generator divergence and floating-point arithmetic differences, not floating-point rounding alone.

#### 4.8.5 Full 500-Speaker GPU Retest (`colab/results/gpu_grid_all1000.csv`)
All 500 speakers × 2 conditions × 4 steps = 4,000 evaluations were re-synthesized and re-scored on the T4. Per-condition failure rates agree with the CPU grids:

| Step | NFE | Clean (CPU) | Clean (GPU) | Noisy15 (CPU) | Noisy15 (GPU) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 13 | 3.2% | 2.4% | 63.2% | 64.4% |
| 5 | 15 | 2.4% | 1.8% | 52.2% | 53.8% |
| 6 | 17 | 2.2% | 1.0% | 49.2% | 49.4% |
| 8 | 21 | 0% | 0% | 0% | 0% |

> **RTF validity:** the `rtf` / `runtime_sec` columns in the grid CSVs time only the student completion and vocoder; the shared teacher pass is excluded, so they read ≈ 0.11 at every step. **Cite only the full-pipeline timing run (Section 4.8)** for latency and RTF. All valid timings are T4 numbers; the Ryzen-laptop runtimes promised in proposal §3.3.5 have not been measured.

---

### 4.9 Ablations: What Drives the Proposed Policy's Benefit?

Proposal §3.7.1: *"The SNR-only rule tests whether the stabilization signal adds anything beyond the prompt check."* These analyses answer that question.

#### 4.9.1 Seed-Noise Floor
Step 8 compared with Step 8 under a different seed "fails" Definition 3.1 on **5/200 speakers = 2.5%** (95% CI 0.5–5.0%). Clean early-handover failure rates (1.0–3.2%) are indistinguishable from this floor. Most clean-prompt failures are therefore evaluation-model jitter (Whisper / WavLM), not real degradation, and no stopping signal can be expected to predict them.

#### 4.9.2 SNR-Only Baseline with θ Calibrated by LTT (Table 3.8)
The proposal requires the SNR-only gate to be calibrated with the same procedure as λ̂ (`colab/results/comprehensive_suite/snr_only_ltt_audit.json`):
- **Calibration:** θ̂ = 18.5 dB certified (3/200 failures, 17.02 NFE).
- **Held-out test:** **5.5% (11/200) > α = 5%.** The certified SNR-only rule fails its guarantee on test, while the proposed policy's λ̂ holds (2.5%).
- Under the proposal's prespecified secondary bound (α = 10%), SNR-only's 5.5% would pass. The "SNR-only fails" statement applies to the primary 5% bound only.

#### 4.9.3 SNR Gate + Fixed Early Step (Additional Ablation)
SNR-only hands over at the **earliest** candidate (Step 4). Holding the gate fixed and varying the early step separates the gate's effect from drift's (cells give robust risk / mean NFE):

| Policy | Calibration | Test | Development | CPU, all 500 | GPU, all 500 |
|---|:---:|:---:|:---:|:---:|:---:|
| SNR → Step 4 (= SNR-only) | 1.5% / 17.02 | 5.5% / 17.04 | 2.0% / 17.00 | 3.2% | 2.4% |
| SNR → Step 5 | 2.5% / 18.02 | **2.0% / 18.03** | 3.0% / 18.00 | 2.4% | 1.8% |
| SNR → Step 6 | 2.5% / 19.01 | 1.0% / 19.02 | 4.0% / 19.00 | 2.2% | 1.0% |
| **Proposed (SNR + drift)** | 2.0% / 18.19 | 2.5% / 18.15 | 2.0% / 18.05 | 2.2% | 1.2% |

- **Proposed vs. SNR → Step 5 (exact McNemar):** p = 1.0 (CPU), p = 0.51 (GPU). No difference.
- **Matched-random control:** proposed clean-step choices vs. random choices with the same step distribution give p = 0.71 (`comprehensive_suite/random_baseline_audit.json`). Drift's Step 4/5/6 choices are indistinguishable from random.
- **Power:** detecting a reduction from 2.4% to 1.2% would need about 2,000 speakers; Seed-TTS test-en has 666.
- SNR → Step 5 was identified after inspecting test results. It is reported as an observation, not a certified method. On calibration it would pass LTT (5/200, p = 0.062), but SNR → Step 4 also passed calibration and then failed on test.

#### 4.9.4 Quality Equivalence (TOST)
Two one-sided tests show the proposed policy is **equivalent** to Step 8 within margins of ±0.02 WER, ±0.02 SIM and ±0.10 UTMOS, both on clean prompts and overall (all p < 0.001). This addresses the limitation in Section 4.4 that non-significant Wilcoxon tests alone do not establish equivalence.

---

### 4.10 Drift Signal Diagnostics

#### 4.10.1 Drift as a Degradation Predictor (Existing Grids)
| Test | Result | Threshold Met? |
|---|---|:---:|
| Spearman ρ, drift vs. per-utterance degradation | All ρ in [−0.12, 0.13], all p > 0.05 | ❌ |
| Added explained variance of drift over SNR (ΔR²) | 0.001 (CI 0.000–0.029) vs. preregistered 0.05 | ❌ |
| AUC for the worst 20% / 10% degradations | 0.52 / 0.50 (chance) | ❌ |
| AUC for clean-prompt Step-5 failure, any logged drift feature | ≤ 0.59; cross-validated combination 0.31 | ❌ |
| AUC for noisy-prompt Step-5 failure, d_4 / d_5 | 0.58: above chance, too weak to use | ❌ |

**Drift detects noise, inverted.** Noisy prompts have *lower* drift (mean d_4 0.18 vs. 0.60 on clean; AUC 0.999 vs. 0.998 for the SNR estimator). A drift-only gate (d_4 > 0.30) matches the SNR gate on test (2.0% risk, 18.06 vs. 18.03 NFE) but adds nothing on top of it: drift tracks the prompt condition rather than output quality.

#### 4.10.2 Phase 12: New Corruptions on 166 Fresh Speakers (Preregistered)
Design: clean, reverb RT60 0.6 s and 8 kHz telephone + mu-law on the 166 speakers no split contains. Policies frozen before synthesis: SNR-only, drift-only and SNR AND drift, all handing over at Step 5. Primary hypothesis (Holm-corrected McNemar): SNR AND drift has fewer failures than SNR-only on an eligible condition. Script and preregistration are on branch `SNRgated` (commit `d72e84f`).

| Condition | Step-5 Failure | Eligible? | SNR-only Risk / NFE | SNR + Drift Risk / NFE | AUC Low-Ŝ | AUC Low-d_4 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Clean | 0.6% | No | 0.6% / 15.18 | 0.6% / 15.29 | — | — |
| Reverb 0.6 s | **8.4%** | **Yes** (p = 0.0001) | 6.0% / 16.95 | 6.0% / 17.10 | 0.52 | 0.54 |
| Telephone | 4.8% | No (p = 0.058) | 4.8% / 15.00 | 4.8% / 15.14 | 0.75 | 0.52 |

**Verdict: NOT SUPPORTED.** Adding drift changed no decision on any failing utterance. Reverb genuinely degrades early handover (14/166 fail at Step 5, 13 on SIM), but neither SNR nor any drift step (AUC 0.42–0.59) identifies which reverberant prompts fail.

#### 4.10.3 Phase 13: Richer Zero-NFE Drift Features (Exploration Only)
67 features from teacher steps 1–5 (per-frame, per-band, cosine, CFG gap, velocity norm, endpoint spread) were tested on the 100 development speakers' low-SNR prompts. The teacher-only rerun reproduced the stored d_4 / d_5 / d_6 exactly (r = 1.000). Script on branch `SNRgated` (commit `e15cd55`).
- Cross-validated AUC: 0.39–0.57 per feature family; 0.61 for all features with L1 logistic regression.
- Best single feature (`fr_max_r54`, per-frame max drift ratio Step 5 / Step 4): dev AUC 0.69 (chance max 0.63, p ≈ 0.02). Probably real, but weak.
- It cannot beat the SNR gate under LTT: SNR → Step 5 already uses the full calibration budget (5/200), and rescuing noisy prompts would need an AUC of about 0.9.
- **The one-time confirmation on calibration + test (notebook cell 13c) has not been run.**

---

## 5. Methodological Analysis & Discussion

### 5.1 Hypothesized Cause of Early Handover Failure Under Noise
In flow matching, early velocity vectors $v_\theta(x_t, t)$ must resolve phonetic trajectory direction while decoupling acoustic background noise from speech content. Under additive prompt noise, early latent states may be dominated by noise disambiguation. A reasonable hypothesis is that handing off at Step 4 passes an insufficiently separated intermediate representation to the student consistency model, which then projects onto degraded acoustic regions. However, while the empirical failure rates establish an association between early handover and degradation under noise, they do not directly prove this internal latent representation mechanism.

### 5.2 Development-Set Diagnostic of Trajectory Drift Discrimination
To investigate whether mid-flight trajectory drift ($d_4$) reliably detects unsafe early exits on clean speech ($\text{SNR} \ge 20\text{ dB}$), an empirical diagnostic was conducted on the 100-speaker development set (`phase1_dev_grid_100spk/results/grid.csv` and GPU re-test):

| GPU Development Outcome Among Cases Where Proposed Waited (Selected Step > 4) | Number of Cases |
| :--- | :---: |
| Step 4 passes; selected later step passes (**Pass $\to$ Pass**) | **75** |
| Step 4 fails; selected later step passes (**Fail $\to$ Pass**) | **2** |
| Step 4 fails; selected later step fails (**Fail $\to$ Fail**) | **1** |
| Step 4 passes; selected later step fails (**Pass $\to$ Fail**) | **0** |
| **Total Waiting Cases** | **78** |

On CPU, 76 cases were Pass $\to$ Pass and 1 was Pass $\to$ Fail, meaning 77 waiting cases had already passed at Step 4.

*Diagnostic Assessment:*
1. **Weak Discrimination:** Step 4 has a very low baseline failure rate on clean speech (2.0% on CPU, 3.0% on GPU). Observed drift for Step 4 passes (Mean $d_4 = 0.594$) is virtually identical to failures (Mean $d_4 = 0.596$). The descriptive discrimination measure is $\text{ROC-AUC} \approx 0.474$, based on only 3 GPU failures and 2 CPU failures. This observed discrimination is weak and imprecisely estimated.
2. **False-Alarm Compute Cost:** The proposed policy delayed handover in 78% of clean cases, but **96%–97% of those cases had already passed at Step 4**. This false-alarm waiting accounts for the approximately 0.173-second latency penalty of Proposed over SNR-only.
3. **Role of SNR Gating:** Input SNR gating was highly effective within the tested policies and conditions, preventing over 60% noise failure. Trajectory drift represents a candidate model-internal stopping signal whose incremental benefit over SNR gating remains unestablished.

### 5.3 Computational Overhead of Trajectory Drift
Calculating the drift metric requires zero additional model forward evaluations. However, computing endpoint projections, Euclidean norms, and decision synchronizations incurs measurable runtime overhead. More importantly, when drift delays handover to later candidate steps, the policy executes additional teacher forward passes (2 extra passes per step). "Zero additional NFE to compute drift" must not be conflated with zero additional computational cost for the scheduling policy.

### 5.4 Synthesis Latency vs. Offline Metric Suite
- **Core Synthesis Latency:** On the NVIDIA Tesla T4 GPU, end-to-end core synthesis (teacher, scheduler, student, and Vocos vocoder) averages **2.885 seconds** per utterance under the proposed policy (RTF $\approx 0.617$).
- **Offline Evaluation Overhead:** Running Whisper large-v3 transcription, WavLM feature extraction, and UTMOS scoring takes substantially longer than speech synthesis. These models are required solely for thesis evaluation and are excluded from production synthesis pipelines.

### 5.5 Threats to Validity & Methodological Safeguards
1. **Disjoint Partitioning:** 500 speakers were split into 100 dev, 200 calib, and 200 test speakers with zero speaker overlap.
2. **Multiple Comparison Corrections:** LTT calibration controls FWER via fixed-sequence testing; test hypothesis tests enforce Holm-Bonferroni corrections.
3. **Reference-Relative Evaluation:** All evaluations are paired against identical-seed Step-8 references to isolate handover effects from baseline model limitations.

### 5.6 Disclosed Study Limitations & Proposal Framing
- **Dataset & Acoustic Scope:** Evaluated on Seed-TTS English (`test-en`) under clean and synthetic 15 dB AWGN conditions. Generalization to conversational speech, non-stationary noise, or multi-lingual data has not been tested.
- **Sparse Early-Exit Failures:** The small number of early-exit failures on clean speech limits the statistical precision of drift discrimination estimates.
- **Objective vs. Subjective Metrics:** Automated neural scores (WER, WavLM SIM, UTMOS) do not establish human perceptual equivalence.
- **Nature of Reference-Relative Metric:** Under 15 dB prompt noise, Step-8 fallback exhibits an average WER of 46.03%. Because failures are defined relative to the Step-8 reference, a 0.0% added failure rate under noise indicates no added degradation beyond the reference baseline; it must not be described as noise-free or collapse-free synthesis.
- **Status of Drift Benefit:** The current drift criterion has not demonstrated a convincing quality advantage over SNR-only scheduling. Any future redesign informed by inspecting test set outcomes must undergo independent calibration and fresh evaluation on a new held-out split.
- **Test Split Already Inspected:** The test split has been examined many times during the ablations. Any new policy (e.g., SNR → Step 5) must be confirmed on unseen data, such as the 166 unused speakers with newly synthesized noisy15 prompts.
- **Speaker Verifier:** SIM uses WavLM-Base+ SV, not WavLM-large (Section 1.2). Absolute SIM is not comparable with published DMOSpeech 2 numbers.
- **Speaker Identity:** Splits are disjoint by Seed-TTS prompt clip, not by verified Common Voice contributor (Section 3).
- **Hardware Scope:** All valid runtimes are Tesla T4 measurements. The proposal's Ryzen 7 laptop runtimes (§3.3.5) are not yet measured.
- **Invalid Evaluation-Suite Outputs:** In the 18-test Colab suite, Tests 7, 12, 15 and 17 are hardcoded or invalid, and the auto-generated text for Tests 2, 3, 8 and 13 contradicts its computed JSON. Cite only the computed values used in this report (see `reports/COLAB_RESULTS_CONSOLIDATED_REPORT.md` on branch `SNRgated`).

---

## 6. Replication Commands & Verification Guide

All experiments are fully reproducible from the repository root:

```powershell
# 1. Run Unit Test Suite
python -m pytest "Handover Scheduling/core" -q

# 2. Run Phase 2 Learn-Then-Test Calibration (CPU)
python "Handover Scheduling/phase2_calib_ltt_200spk/calibrate.py"

# 3. Run Phase 3 Held-Out Benchmark & Verification (CPU)
python "Handover Scheduling/phase3_test_benchmark_200spk/run.py"
python "Handover Scheduling/phase2_calib_ltt_200spk/verify.py"

# 4. Run Dedicated Full-Pipeline GPU Timing Benchmark (NVIDIA Tesla T4)
python "Handover Scheduling/colab/run_gpu_full_pipeline_timing_benchmark.py" `
  --grid_csv "Handover Scheduling/colab/results/gpu_grid_all1000.csv" `
  --output_dir "Handover Scheduling/colab/results" `
  --device cuda `
  --warmup 5

# 5. Run Development Drift Diagnostic Analysis
python "scratch/analyze_drift_diagnostic.py"

# 6. Phase 12 / Phase 13 drift studies (scripts on branch SNRgated; run from the Colab notebook)
python "Handover Scheduling/colab/run_drift_justification_benchmark.py" --device cuda --conditions quick
python "Handover Scheduling/colab/run_drift_features_benchmark.py" --device cuda --explore_only
```

---

## 7. Official Thesis Defense Statement

> The proposed SNR + drift policy, calibrated with Learn-Then-Test, reduced forward passes by 13.6% and held-out T4 synthesis latency by 13.3% relative to DMOSpeech 2's fixed Step 8 (27.0% on clean prompts). It kept added failures at 2.5% on 200 unseen speakers, within the 5% target, with quality statistically equivalent to Step 8 (TOST). The SNR-only baseline, calibrated the same way, exceeded the target on held-out speakers (5.5%; McNemar p = 0.07). Ablations show that the benefit comes mainly from the SNR gate: an SNR gate with a fixed Step 5 performed comparably (p = 1.0), and drift's step choices did not differ from a matched-random control. Drift was also not supported under new corruptions on fresh speakers or in richer forms. The contribution is therefore a certified adaptive handover framework whose efficiency gain over fixed scheduling is established, with the incremental value of the drift signal reported as a tested negative result.

---

## 8. Proposal Items Still Open

| Item | Proposal Section | Needs GPU? | Status |
|---|---|:---:|---|
| Calibrate on clean, test on noisy (distribution shift) | §3.7.3 | No | Not run; computable from existing grids |
| Calibration-size study (n = 50 / 100 / 200) | §3.7.3 | No | Not run; computable from existing grids |
| Teacher-only / student-only context | Table 3.8 | Yes | Not run |
| Per-policy runtime on the Ryzen 7 laptop | §3.3.5 | No (CPU run) | Not run; grid runtimes exclude the teacher and cannot be reused |
| Listening check of above-ceiling WER failures | §3.3.4 | No | Not run; no audio exported |
| Phase 13 confirmation (cell 13c) | §3.7.1 | No | Not run |
| Phase 12 remaining conditions (reverb 0.3 / 0.9 s, babble, DEMAND, Gaussian 18–25 dB) | §3.7.3 | Yes | Not run |
| Saved `pytest -q` log for the reproducibility checks | §3.3.5 | No | Tests exist (`core/test_sampler.py`, `test_calibrate.py`); no saved log |
| Name the actual SIM verifier, or re-score with WavLM-large | Table 3.5 | Re-score only | Open |
| Correct Remark 3.1 ($t_{14} = 0.2270$) | §3.3.1 | No | Open |

Changing the proposal's main claim (Table 2.1 lists drift as the decision signal) requires the adviser's agreement before Chapter 3 is rewritten.
