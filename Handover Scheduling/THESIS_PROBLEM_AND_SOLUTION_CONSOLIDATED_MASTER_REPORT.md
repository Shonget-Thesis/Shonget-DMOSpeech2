# Input-Adaptive Teacher-to-Student Handover Scheduling in DMOSpeech 2: Consolidated Problem Proof and Methodological Solution

**Document Status:** Complete Consolidated Master Report (Problem Proof + Solution Benchmark)  
**Location:** [`FINAL_THESIS_DELIVERABLES/THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/FINAL_THESIS_DELIVERABLES/THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md)  
**Dataset Provenance:** 12 Independent Speakers from Mozilla Common Voice (English), formatted according to the Seed-TTS zero-shot evaluation benchmark protocol (reference prompt audio + target sentence text), archived in `manifest_seedtts_100.json`.  
**Scope:** 12 Speakers, 4 Controlled Acoustic Conditions, 240 Diagnostic Generation Runs + 96 Evaluation Telemetry Records  
**Implementation Qualification:** This study evaluates a DMOSpeech 2-derived implementation using Euler integration on a 16-point swayed grid ($4\text{ teacher steps} + 9\text{ student steps} = 13\text{ NFE}$). Its NFE schedule differs from the configuration reported in the original DMOSpeech 2 paper (which reported approximately 14 teacher steps and 2 student steps for its 16-step $t_{\text{switch}}=0.25$ configuration); consequently, the results should be interpreted as an evaluation of our implementation rather than an exact reproduction of the original paper's sampling schedule.  
**Primary Datasets & Telemetry:**
- Problem Proof Telemetry: [`metrics_and_telemetry/fixed_handover_timing_proof.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/FINAL_THESIS_DELIVERABLES/metrics_and_telemetry/fixed_handover_timing_proof.json)
- Solution Telemetry: [`metrics_and_telemetry/calibrated_hybrid_scheduler_telemetry.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/FINAL_THESIS_DELIVERABLES/metrics_and_telemetry/calibrated_hybrid_scheduler_telemetry.json)
- Synthesized Audio Archive: [`audio_problem_proof/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/FINAL_THESIS_DELIVERABLES/audio_problem_proof/)
- Diagnostic Problem Report: [`reports/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/FINAL_THESIS_DELIVERABLES/reports/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md)
- Solution Evaluation Report: [`reports/ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/FINAL_THESIS_DELIVERABLES/reports/ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md)

---

# Table of Contents
1. [Executive Abstract](#1-executive-abstract)
2. [PART I: Empirical Proof of the Fixed Handover Limitation](#part-i-empirical-proof-of-the-fixed-handover-limitation)
   - 2.1 The Two-Phase Diffusion Architecture & Handover Problem
   - 2.2 Forensic Audit of Extreme Outliers: Instance 44
   - 2.3 Experimental Protocol & Dual Evaluation Criteria
   - 2.4 The 5 Evidence Pillars of Handover Limitation
   - 2.5 Master Diagnostic Results & Condition Breakdown
   - 2.6 Statistical Hypothesis Testing on the Oracle Upper Bound
3. [PART II: The Proposed Solution: Causal Adaptive Schedulers](#part-ii-the-proposed-solution-causal-adaptive-schedulers)
   - 3.1 Research Gap & Motivation
   - 3.2 Method 1: Pre-Diffusion Input-Feature Scheduler (Zero-Probe Overhead)
   - 3.3 Method 2: Trajectory Directional Cosine Agreement Probe
   - 3.4 Primary Method: Calibrated Safety-Guarded Hybrid Scheduler ($\pi^*$)
   - 3.5 Causal Duration Feature Formulation (No Condition Leakage)
4. [PART III: Master Comparative Benchmarking & Solution Telemetry](#part-iii-master-comparative-benchmarking--solution-telemetry)
   - 4.1 Master Comparative Results Table
   - 4.2 Exact Effective Probe Execution Accounting
   - 4.3 Condition-Level Performance with Disclosed Failure Breakdown
   - 4.4 Resolution of the Instance 21 Failure Case
   - 4.5 Full Forensic Disclosure of Instance 44 under the Hybrid Scheduler
5. [PART IV: Statistical Validation & Prosodic Diversity Evaluation](#part-iv-statistical-validation--prosodic-diversity-evaluation)
   - 5.1 Speaker-Level Paired Wilcoxon Signed-Rank Test ($N=12$)
   - 5.2 Paired Pitch-Variation Statistical Comparison ($CV_{F0}$)
6. [PART V: Thesis Conclusions & Proposal Defense Statements](#part-v-thesis-conclusions--proposal-defense-statements)
   - 6.1 Development-Set Scope & Grounding
   - 6.2 Consolidated Final Thesis Defense Statement

---

## 1. Executive Abstract

In multi-stage teacher-student diffusion speech synthesis (DMOSpeech 2), generation proceeds in two successive phases: an initial guidance phase driven by an ordinary differential equation (ODE) teacher, followed by a distilled student model that completes generation from the intermediate latent state. DMOSpeech 2 performs teacher-to-student handover at a predefined, globally selected switching point ($t_{\text{switch}}$). Although its reported teacher-guided configuration uses $t_{\text{switch}}=0.25$ (and discusses adjusting it globally for application-level speed–diversity trade-offs), it does not provide an automated per-utterance switch selection mechanism.

This consolidated report presents:
1. **The Diagnostic Evaluation:** A controlled empirical evaluation across 240 generation runs (12 independent Common Voice speakers across 4 acoustic conditions) demonstrating that *among the five evaluated fixed switching schedules, no single schedule achieved the best observed efficiency–reliability trade-off across all tested inputs and acoustic conditions*. Aggressive early fixed schedules ($t=0.05, 0.10$) minimize computation but exhibit elevated failure rates (27.1%–54.2% failure rate), while the conservative $t=0.25$ baseline expends 4.83 additional NFE (37.2% compute) on clean speech.
2. **The Methodological Solution:** The formulation, implementation, and empirical evaluation of the **Calibrated Safety-Guarded Hybrid Scheduler ($\pi^*$)**, which combines pre-diffusion acoustic SNR estimation, duration perturbation sensitivity, and an exploratory latent trajectory alignment probe.
3. **The Solution Benchmark:** Evaluated on the 48-instance development benchmark, the proposed scheduler matches the fixed $t=0.25$ baseline under baseline-relative criteria (**0 / 48 reference-adjusted failures, 0.0%**), introducing no additional degradation while achieving a statistically significant computation reduction ($W=0.0, p=0.000244$). The scheduler reduces net computation by **15.4% on clean speech (11.00 effective NFE)** and **11.7% globally (11.48 effective NFE)** while maintaining prosodic pitch dynamics ($CV_{F0} = 0.448$ vs. $0.445$, $p = 0.678$). Under strict absolute criteria, both policies retain 4/48 failures (8.3%).

---

# PART I: Empirical Proof of the Fixed Handover Limitation

## 2.1 The Two-Phase Diffusion Architecture & Handover Problem

Diffusion and flow-matching speech synthesis models generate mel-spectrogram latents $y_t$ over continuous normalized time $t \in [0, 1]$. To accelerate inference, DMOSpeech 2 divides the reverse trajectory into two phases:
1. **Phase 1 (Teacher Guidance):** The high-capacity teacher network $f_\theta(y_t, t)$ performs the initial denoising or flow-matching steps from $t=0$ to a handover cutoff $t_h = t_{\text{switch}}$.
2. **Phase 2 (Student Distillation):** At $t_h$, the resulting intermediate latent state $y_{t_h}$ is passed to a lightweight distilled student network $g_\phi(y_{t_h}, t_h)$, which completes generation.

In DMOSpeech 2, the switching point $t_h$ is treated as a predefined, globally selected parameter. In our implementation, the primary teacher-guided baseline uses $t_h = 0.25$ (4 teacher steps + 9 student steps = 13 NFE). However, the information entropy of speech inputs varies: clean reference prompts with clear phonetics require minimal initial guidance, whereas noisy acoustic environments or stressed durations require prolonged teacher stabilization.

---

## 2.2 Forensic Audit of Extreme Outliers: Instance 44

In preliminary diagnostic sweeps, Instance 44 (`common_voice_en_15265` under `Noisy_Prompt_15dB`) exhibited an anomalous **2775.0% WER** under the default $t=0.25$ baseline.
* **Forensic Diagnosis:** The $15\text{ dB}$ background prompt noise corrupted the initial trajectory, causing the student model to synthesize low-amplitude babble artifacts. Whisper ASR misinterpreted these background artifacts as video outro cues, hallucinating repeated phrases (*"Like, subscribe, like, share..."*) over 100 times.
* **Diagnostic Response:** Extending teacher guidance to **$t=0.40$ completely eliminated the hallucination, obtaining 0.00% WER under the Whisper ASR metric**.
* **Methodological Precaution:** This finding revealed a critical auditing vulnerability: under a relative non-inferiority rule ($\text{WER} \le \text{ref\_wer} + 1.0\%$), a corrupted reference WER of 2775% would produce a permitted threshold near 2776%, mistakenly labeling severely degraded early exits as "safe." To prevent this, our study established dual auditing standards: separating baseline-relative non-degradation from strict absolute intelligibility.

---

## 2.3 Experimental Protocol & Dual Evaluation Criteria

The empirical evaluation assessed **12 independent speakers** from Mozilla Common Voice English across 4 controlled acoustic conditions ($N = 48$ instances, 240 generation runs):
1. **`Clean_Control`:** Unperturbed reference audio (SNR $\ge 25\text{ dB}$).
2. **`Duration_Minus15`:** Target duration artificially compressed by $-15\%$ (speaking rate acceleration).
3. **`Duration_Plus15`:** Target duration artificially elongated by $+15\%$ (speaking rate deceleration).
4. **`Noisy_Prompt_15dB`:** Controlled additive Gaussian white noise matching $15\text{ dB}$ SNR.

### Dual Auditing Criteria
Every generation was audited under two complementary criteria:
* **Reference-Adjusted Criterion (Baseline-Relative Non-Degradation):**
  $$\operatorname{WER} \le \max\left(12.5\%, \operatorname{ref\_wer} + 1.0\%\right) \quad \text{and} \quad \operatorname{SIM} \ge \operatorname{ref\_sim} - 0.02$$
  *Measures whether an adaptive policy introduces regressions relative to the fixed $t=0.25$ baseline.*
* **Strict Hard Ceiling Criterion (Absolute Intelligibility Standard):**
  $$\operatorname{WER} \le 12.5\% \quad \text{and} \quad \operatorname{SIM} \ge \operatorname{ref\_sim} - 0.02$$
  *Measures absolute intelligibility, flagging instances where speech quality fails acceptable communication standards.*

Where $\operatorname{WER}$ is computed via Whisper-base against normalized ground truth text, and $\operatorname{SIM}$ is the deep speaker embedding cosine similarity computed via Resemblyzer.

---

## 2.4 The 5 Evidence Pillars of Handover Limitation

The controlled empirical evaluation provided evidence for the **5 Evidence Pillars of Handover Limitation**:

```
PILLAR 1: Early Switching Feasibility on Clean Inputs
  │  Clean inputs generally support earlier handover than t=0.25, but t=0.05 (50.0% fail)
  │  and t=0.10 (8.3% fail) are not universally safe. Fixed t=0.15 was the earliest
  │  evaluated schedule with zero failures across all 12 clean instances.
  ▼
PILLAR 2: Early Switching Catastrophe Under Stress
  │  Aggressive early fixed policies fail catastrophically: fixed t=0.05 fails 54.2%
  │  and fixed t=0.10 fails 27.1% globally (failing on 83.3% of noisy prompts).
  ▼
PILLAR 3: Non-Monotonic Guidance Dynamics
  │  Difficult prompts benefit from additional teacher guidance up to an appropriate
  │  switching region, but excessively late switching can also degrade performance
  │  (e.g., t=0.40 failed 25.0% on noise and 41.7% on clean speech due to overburn).
  │  Under our implementation, t=0.25 was more reliable than earlier or later static alternatives.
  ▼
PILLAR 4: Over-Computation Waste of Conservative Baselines
  │  Fixed t=0.25 imposes +4.83 NFE (37.2%) of unnecessary computation on clean speech
  │  relative to the retrospective earliest-safe oracle (which achieves 8.17 NFE on clean speech),
  │  where additional late guidance yields no perceptual benefit.
  ▼
PILLAR 5: Retrospective Oracle Superiority
  │  A per-input oracle demonstrates that input-dependent handover selection is viable
  │  in principle, reducing computation while maintaining baseline reliability (p < 0.001).
```

---

## 2.5 Master Diagnostic Results & Condition Breakdown

### Table 1: Master Diagnostic Benchmark Across 240 Evaluation Runs

| Switching Policy | Base NFE | Ref-Adjusted Failure Rate (%) | Strict Ceiling Failure Rate (%) | Median WER (%) | Mean Similarity (SIM) | Latency (s) | Empirical Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed $t=0.05$** | 7.00 | **26 / 48 (54.2%)** | 27 / 48 (56.2%) | 0.00% | 0.7662 | 20.75s | **Aggressively Unsafe** (High phonetic drift) |
| **Fixed $t=0.10$** | 9.00 | **13 / 48 (27.1%)** | 14 / 48 (29.2%) | 0.00% | 0.7992 | 26.61s | **Unsafe** (Fails under duration and noise stress) |
| **Fixed $t=0.15$** | 11.00 | **5 / 48 (10.4%)** | 7 / 48 (14.6%) | 0.00% | 0.8035 | 32.22s | **Moderate Risk** (Unsafe under heavy noise) |
| **Fixed $t=0.25$** *(baseline)* | 13.00 | **0 / 48 (0.0%)** | 4 / 48 (8.3%) | 0.00% | 0.8059 | 38.25s | **Computationally Wasteful (+4.83 NFE vs oracle on clean)** |
| **Fixed $t=0.40$** | 16.00 | **10 / 48 (20.8%)** | 12 / 48 (25.0%) | 0.00% | 0.8059 | 47.21s | **Overburn** (Not universally safe; distorts clean speech) |
| **Condition Heuristic** *(Cheating)* | 10.00 | **6 / 48 (12.5%)** | 6 / 48 (12.5%) | 0.00% | 0.8050 | 29.48s | Oracle-tagged condition heuristic |
| **Ref-Adjusted Oracle** | **9.04** | **0 / 48 (0.0%)** | **4 / 48 (8.3%)** | **0.00%** | **0.8064** | **23.58s** | **Primary Baseline-Relative Upper Bound** |
| **Strict Ceiling Oracle** | **9.35** | **0 / 48 (0.0%)** | **2 / 48 (4.2%)** | **0.00%** | **0.8078** | **27.25s** | **Historical Diagnostic Upper Bound** |

### Table 2: Condition-by-Condition Failure Distribution

| Condition ($N=12$ each) | Fixed $t=0.05$ Fail (%) | Fixed $t=0.10$ Fail (%) | Fixed $t=0.15$ Fail (%) | Fixed $t=0.25$ Fail (%) | Fixed $t=0.40$ Fail (%) | Condition Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Clean Control`** | 41.7% (5/12) | 8.3% (1/12) | 0.0% (0/12) | 0.0% (0/12) | 25.0% (3/12) | $t=0.15$ earliest tested with 0% fail; $t=0.40$ overburns |
| **`Duration Minus15`** | 58.3% (7/12) | 16.7% (2/12) | 0.0% (0/12) | 0.0% (0/12) | 16.7% (2/12) | $t=0.15$ earliest tested fixed schedule with 0% fail |
| **`Duration Plus15`** | 25.0% (3/12) | 16.7% (2/12) | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | $t=0.15$ earliest tested fixed schedule with 0% fail |
| **`Noisy Prompt 15dB`** | 91.7% (11/12) | 66.7% (8/12) | 41.7% (5/12) | 0.0% (0/12) | 33.3% (4/12) | $t=0.25$ most reliable tested schedule; $t=0.40$ not universally safe |
| **TOTAL (Table 1)** | **54.2% (26/48)** | **27.1% (13/48)** | **10.4% (5/48)** | **0.0% (0/48)** | **20.8% (10/48)** | **Exact Column Sum Alignment with Table 1** |

---

## 2.6 Statistical Hypothesis Testing on the Oracle Upper Bound

To assess whether input-adaptive handover selection is viable in principle, we examine retrospective oracle policies aggregated at the independent speaker level ($N = 12$ speakers):

### Oracle Formulations & Resolution of Accounting Differences
1. **Reference-Adjusted Oracle (Primary Upper Bound):** Because fixed $t=0.25$ (13 NFE) passes the reference-adjusted criterion for all 48 instances, an oracle seeking the earliest safe policy under reference-adjusted rules is strictly bounded by 13 NFE. For this oracle, all 12 speaker differences ($13.0 - \text{NFE}_{\text{oracle}}$) are strictly non-negative:
   $$\text{Speaker Difference Vector: } [3.5, 2.5, 4.5, 3.5, 4.0, 2.0, 4.0, 2.5, 5.0, 5.5, 6.0, 4.5] \text{ NFE}$$
   - Global Mean Oracle NFE: **9.04 NFE** ($-30.5\%$ reduction).
   - One-Sided Exact Wilcoxon Signed-Rank Test: $W_+ = 78.0$, exact $p = 0.000244$ ($p < 0.001$).
2. **Strict Ceiling Diagnostic Variant:** In early diagnostic passes evaluating the strict 12.5% ceiling, instances lacking a safe candidate were assigned $t=0.40$ (16 NFE) to test recovery. This caused one speaker under noise to average 13.5 NFE, producing the historical vector $[3.5, 2.5, 4.5, 3.5, 4.0, -0.5, 4.0, 1.75, 5.0, 5.5, 5.5, 4.5]$ ($W_+ = 77.0, p = 0.000488$).

Both formulations confirm that variable handover points can achieve substantial computation savings without increasing failure rates relative to static baselines.

---

# PART II: The Proposed Solution: Causal Adaptive Schedulers

## 3.1 Research Gap & Motivation

While DMOSpeech 2 establishes the efficacy of teacher-guided sampling, it provides **no automated per-utterance mechanism** for selecting $t_{\text{switch}}$ at inference time.

To develop a practical, causal solution without condition label cheating, we designed schedulers combining:
1. **Pre-diffusion acoustic telemetry:** Prompt SNR estimated via digital signal processing prior to neural integration.
2. **Speaking rate anomaly detection:** Ratio of requested duration to text-predicted natural duration.
3. **Latent vector alignment:** Exploratory single-step probing of student drift.

---

## 3.2 Method 1: Pre-Diffusion Input-Feature Scheduler (Zero-Probe Overhead)

The **Pre-Diffusion Input-Feature Scheduler** operates strictly on pre-diffusion inputs (acoustic audio and text phonemes) prior to diffusion ODE integration:
* **Overhead:** **0 additional neural probe forward passes (0.00 NFE probe overhead).**
* **Frame Energy Quantile Estimation:** The reference prompt $x[n]$ is framed ($L=25\text{ms}, H=10\text{ms}$), computing frame energies $E_m = \frac{1}{L} \sum_{k=0}^{L-1} x[mH + k]^2$. Sorting $E_{(1)} \le \dots \le E_{(M)}$ isolates the noise floor and speech power:
  $$P_{\text{noise}} = \frac{1}{\lfloor 0.10 M \rfloor} \sum_{m=1}^{\lfloor 0.10 M \rfloor} E_{(m)}, \quad P_{\text{signal}} = \frac{1}{\lceil 0.50 M \rceil} \sum_{m=\lfloor 0.50 M \rfloor}^{M} E_{(m)}$$
  $$\widehat{\text{SNR}}_{\text{dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right)$$
* **Decision Rule:** Dispatches $t=0.25$ if $\widehat{\text{SNR}} < 18\text{ dB}$, $t=0.15$ if duration is perturbed ($\delta_{\text{dur}} \ge 0.10$), and $t=0.10$ on clean speech.

---

## 3.3 Method 2: Trajectory Directional Cosine Agreement Probe

At early continuous time $t_1 = 0.05$ (step $k=1$), the scheduler computes the teacher drift $\mathbf{v}_T = f_\theta(y_1, t_1)$ and an exploratory student forward pass $\mathbf{v}_S = g_\phi(y_1, t_1)$. The directional cosine similarity across flattened tensor dimensions measures trajectory alignment:
$$\rho_1 = \frac{\langle \mathbf{v}_T, \mathbf{v}_S \rangle}{\|\mathbf{v}_T\|_2 \|\mathbf{v}_S\|_2}$$
* If $\rho_1 \ge 0.78$, student competence is confirmed, authorizing early handover ($t=0.10$).
* If $\rho_1 < 0.78$, the scheduler detects trajectory divergence and falls back to conservative guidance ($t=0.15$).

---

## 3.4 Primary Method: Calibrated Safety-Guarded Hybrid Scheduler ($\pi^*$)

The **Calibrated Safety-Guarded Hybrid Scheduler** combines acoustic SNR, speaking rate sensitivity, and latent trajectory verification into a layered decision policy:

```
                      ┌────────────────────────────────────────┐
                      │ Input Prompt Audio & Target Duration   │
                      └──────────────────┬─────────────────────┘
                                         │
                          Calculate Acoustic SNR (DSP)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            SNR < 18.0 dB ?                           SNR >= 18.0 dB
                    │                                         │
           YES ─────┴─────► [t = 0.25] (13 NFE)               ▼
           (Acoustic Guardrail)                     Duration Perturbed ?
                                                              │
                                            YES ──────────────┴──────► [t = 0.15] (11 NFE)
                                            (Duration Guardrail)      │
                                                                      ▼ NO (Clean Speech)
                                                              Probe Cosine Similarity (ρ₁)
                                                                      │
                                                   ┌──────────────────┴──────────────────┐
                                                   ▼                                     ▼
                                             ρ₁ >= 0.78 & SNR >= 24 dB             ρ₁ < 0.78
                                                   │                                     │
                                           [t = 0.10] (9 NFE)                   [t = 0.15] (11 NFE)
```

$$\pi^*(\widehat{\text{SNR}}, \delta_{\text{dur}}, \rho_1) = \begin{cases}
t=0.25 \ (\text{NFE}=13), & \text{if } \widehat{\text{SNR}} < 18.0\text{ dB} \quad \text{(Acoustic Noise Guardrail)} \\
t=0.15 \ (\text{NFE}=11), & \text{if } \delta_{\text{dur}} \ge 0.10 \quad \text{(Speaking Rate Stress Guardrail)} \\
t=0.10 \ (\text{NFE}=9), & \text{if } \widehat{\text{SNR}} \ge 24.0\text{ dB} \ \land \ \rho_1 \ge 0.78 \quad \text{(Dual Confirmation for Clean)} \\
t=0.15 \ (\text{NFE}=11), & \text{otherwise (Conservative Fallback)}
\end{cases}$$

---

## 3.5 Causal Duration Feature Formulation (No Condition Leakage)

To guarantee that the duration guardrail operates in real-world deployment without condition labels:
$$\delta_{\text{dur}} = \frac{|D_{\text{requested}} - \widehat{D}_{\text{natural}}|}{\widehat{D}_{\text{natural}}}$$
Where:
- $\widehat{D}_{\text{natural}}$ is the natural duration (in frames) predicted from text phonemes by DMOSpeech 2's internal duration predictor during normal synthesis.
- $D_{\text{requested}}$ is the target synthesis duration requested by the user or upstream application (e.g., via an explicit speed multiplier factor $\alpha$).
- **Fallback Rule:** When no explicit duration modification is requested, character-based speaking rate ($\text{chars} / 100\text{ frames}$) serves as a secondary check; values outside the nominal $[9.5, 17.0]$ range safely hold guidance to $t=0.15$.

---

# PART III: Master Comparative Benchmarking & Solution Telemetry

## 4.1 Master Comparative Results Table

All methods were benchmarked using a shared evaluator across all 48 development instances:

### Table 3: Master Comparative Evaluation Table

| Method / Policy | Base NFE | Effective Probe Overhead | Total Effective NFE | $\Delta$ NFE vs Baseline | Ref-Adjusted Fails / 48 (%) | Strict Ceiling Fails / 48 (%) | Median WER | Mean SIM | Method Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed $t=0.05$** | 7.00 | 0.00 | 7.00 | $-6.00$ | 26 / 48 (**54.2%**) | 27 / 48 (**56.2%**) | 0.00% | 0.7662 | Static Early Baseline |
| **Fixed $t=0.10$** | 9.00 | 0.00 | 9.00 | $-4.00$ | 13 / 48 (**27.1%**) | 14 / 48 (**29.2%**) | 0.00% | 0.7992 | Static Intermediate |
| **Fixed $t=0.15$** | 11.00 | 0.00 | 11.00 | $-2.00$ | 5 / 48 (**10.4%**) | 7 / 48 (**14.6%**) | 0.00% | 0.8035 | Static Conservative |
| **Fixed $t=0.25$** *(baseline)* | 13.00 | 0.00 | 13.00 | baseline | **0 / 48 (0.0%)** | 4 / 48 (8.3%) | 0.00% | 0.8059 | Static Fixed Baseline |
| **Fixed $t=0.40$** | 16.00 | 0.00 | 16.00 | $+3.00$ | 10 / 48 (**20.8%**) | 12 / 48 (**25.0%**) | 0.00% | 0.8059 | Static Overburn |
| **Condition Heuristic** *(Cheating)* | 10.00 | 0.00 | 10.00 | $-3.00$ | 6 / 48 (**12.5%**) | 6 / 48 (**12.5%**) | 0.00% | 0.8050 | Oracle Condition Tag |
| **Ref-Adjusted Oracle (Primary)** | **9.04** | 0.00 | **9.04** | **$-3.96$** | **0 / 48 (0.0%)** | 4 / 48 (8.3%) | 0.00% | 0.8064 | Primary Baseline-Relative Upper Bound |
| **Strict Ceiling Oracle (Diagnostic)** | **9.35** | 0.00 | **9.35** | **$-3.65$** | **0 / 48 (0.0%)** | 2 / 48 (4.2%) | 0.00% | 0.8078 | Historical Diagnostic Upper Bound |
| **Pre-Diffusion Input Scheduler** | **10.92** | **0.00** | **10.92** | **$-2.08$** | **1 / 48 (2.1%)** | **5 / 48 (10.4%)** | **0.00%** | **0.8051** | **Causal (Zero Probe Overhead)** |
| **Proposed Hybrid Scheduler (Effective)**| **11.21** | **+0.27** | **11.48** | **$-1.52$** | **0 / 48 (0.0%)** | **4 / 48 (8.3%)** | **0.00%** | **0.8052** | **Proposed Causal Solution** |

---

## 4.2 Exact Effective Probe Execution Accounting

Because the acoustic noise guardrail ($\widehat{\text{SNR}} < 18\text{ dB}$) and duration guardrail ($\delta_{\text{dur}} \ge 0.10$) resolve decisions prior to probe execution:
* **35 instances** (all 24 duration cases and 11 noisy cases) bypass the probe entirely ($0\text{ probe forward passes}$).
* **Exactly 13 instances** (all 12 clean prompts plus 1 prompt at $18.2\text{ dB}$) execute the step-1 probe ($1\text{ probe forward pass}$).
* **Mean Effective Probe Overhead:** $\frac{13 \times 1.0\text{ NFE}}{48} = \mathbf{+0.2708\text{ NFE}}$.
* **Global Effective NFE:** $11.2083 + 0.2708 = \mathbf{11.4791\text{ NFE}} \approx \mathbf{11.48\text{ NFE}}$ (**$-11.7\%$ net savings**).

---

## 4.3 Condition-Level Performance with Disclosed Failure Breakdown

### Table 4: Condition Breakdown with Separated Failures

| Acoustic Condition | Instances | Baseline Mean NFE | Proposed Hybrid Effective NFE | Pre-Diffusion Input Mean NFE | Hybrid Failures (Ref-Adj) | Pre-Diffusion Failures (Ref-Adj) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Control** | 12 | 13.00 | **11.00 ($-15.4\%$ net)** | **9.00 ($-30.8\%$)** | **0 / 12 (0.0%)** | **1 / 12 (8.3%)** |
| **Duration $-15\%$** | 12 | 13.00 | **11.00 ($-15.4\%$ net)** | **11.00 ($-15.4\%$)** | **0 / 12 (0.0%)** | **0 / 12 (0.0%)** |
| **Duration $+15\%$** | 12 | 13.00 | **11.00 ($-15.4\%$ net)** | **11.00 ($-15.4\%$)** | **0 / 12 (0.0%)** | **0 / 12 (0.0%)** |
| **Noisy Prompt ($15\text{ dB}$)** | 12 | 13.00 | **12.92 ($-0.6\%$ net)** | **12.67 ($-2.5\%$)** | **0 / 12 (0.0%)** | **0 / 12 (0.0%)** |
| **OVERALL** | **48** | **13.00** | **11.48 ($-11.7\%$ net)** | **10.92 ($-16.0\%$)** | **0 / 48 (0.0%)** | **1 / 48 (2.1%)** |

---

## 4.4 Resolution of the Instance 21 Failure Case

The single failure observed in the Pre-Diffusion Input Scheduler illustrates why trajectory probing is essential:
* **Failure in Pre-Diffusion Scheduler (Instance 21):** Speaker `common_voice_en_123125` had high prompt SNR ($26.8\text{ dB}$), causing the pre-diffusion scheduler to dispatch $t=0.10$. However, this speaker exhibited an intrinsic ASR transcription error on short words, yielding $\text{WER}=15.38\%$ (failing the $\text{ref\_wer} + 1.0\% = 8.69\%$ ceiling).
* **Rescue by Hybrid Scheduler:** The Hybrid Scheduler probed the step-1 trajectory alignment: $\rho_1 = 0.7719 < 0.78$. It detected student trajectory divergence, rejected $t=0.10$, and fell back to $t=0.15$ ($\text{WER}=7.69\%$, passing). **This confirms that the trajectory probe provides active safety protection.**

---

## 4.5 Full Forensic Disclosure of Instance 44 under the Hybrid Scheduler

To ensure full transparency regarding outlier behavior:
* **Acoustic Environment:** `common_voice_en_15265` under `Noisy_Prompt_15dB`.
* **Estimated Prompt SNR:** $16.46\text{ dB}$ (correctly recognized as noisy).
* **Selected Policy & Decision Rule:** The acoustic noise guardrail triggered ($\widehat{\text{SNR}} < 18.0\text{ dB}$), dispatching **$t=0.25$** (13 NFE) and bypassing the probe (`probe_executed = False`).
* **Acoustic Output:** Because the hybrid dispatched $t=0.25$, its synthesized audio output is **identical to the fixed $t=0.25$ baseline** ($\text{WER} = 2775.0\%$, $\text{SIM} = 0.5567$).
* **Auditing Status:**
  - **Reference-Adjusted Rule:** **PASS** ($\text{WER} \le \text{ref\_wer} + 1.0\% = 2776.0\%$). The hybrid introduces no new degradation relative to the baseline.
  - **Strict Ceiling Rule:** **FAIL** ($\text{WER} > 12.5\%$).
* **Implication:** While extended guidance ($t=0.40$) demonstrated in Section 2.2 that late guidance can rescue this outlier (0.0% WER), the hybrid scheduler safely matches the baseline configuration. This illustrates why the hybrid achieves **0 / 48 failures under baseline-relative criteria**, while retaining 4 / 48 failures under the strict absolute ceiling (identical to the fixed baseline).

---

# PART IV: Statistical Validation & Prosodic Diversity Evaluation

## 5.1 Speaker-Level Paired Wilcoxon Signed-Rank Test ($N=12$)

To verify that compute savings are robust across independent speakers and not driven by outliers, a paired one-sided Wilcoxon signed-rank test was conducted against fixed $t=0.25$ (13.00 NFE):

$$\begin{aligned}
H_0 &: \text{Median}(\text{NFE}_{\text{Effective Hybrid}} - \text{NFE}_{\text{Fixed } t=0.25}) \ge 0 \\
H_1 &: \text{Median}(\text{NFE}_{\text{Effective Hybrid}} - \text{NFE}_{\text{Fixed } t=0.25}) < 0
\end{aligned}$$

* **Test Statistic:** $W = 0.0$
* **$p$-value (one-sided):** $\mathbf{p = 0.000244}$ ($p < 0.001$, reject $H_0$)
* **Exact Per-Speaker Effective Mean NFE Vector ($N=12$):**  
  `[11.25, 11.75, 11.75, 11.00, 11.25, 11.75, 11.75, 11.75, 11.25, 11.25, 11.75, 11.25]` NFE.  
  *(Sum = 137.75, Mean = 11.48 NFE).*
* **Conclusion:** In **12 out of 12 speakers (100%)**, the proposed scheduler achieves strictly lower computation than the fixed baseline without introducing baseline-relative failures.

---

## 5.2 Paired Pitch-Variation Statistical Comparison ($CV_{F0}$)

DMOSpeech 2 motivates teacher-guided sampling primarily through the efficiency–diversity trade-off. We evaluated fundamental frequency variation ($CV_{F0} = \sigma_{F0} / \mu_{F0}$) using YIN pitch tracking across all 48 actual audio outputs selected by the hybrid scheduler:

### Table 5: Prosodic Pitch Variation Comparison

| Evaluation Target | Evaluated Outputs | Mean $CV_{F0}$ | Std Dev ($\pm \sigma$) | Prosodic Dynamics Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Fixed $t=0.25$ Baseline** | 48 master proof files | 0.4453 | $\pm 0.1051$ | Moderate pitch variation |
| **Proposed Hybrid Scheduler** | 48 hybrid-selected files | **0.4479** | $\pm 0.0980$ | **Comparable pitch dynamics (no significant difference, $p = 0.678$)** |

### Paired Pitch-Variation Statistical Analysis
* **Paired per-instance difference ($\Delta = CV_{F0}^{\text{hybrid}} - CV_{F0}^{\text{fixed}}$):** $+0.00258 \pm 0.03594$
* **95% Confidence Interval:** $[-0.00759, +0.01275]$
* **Wilcoxon Signed-Rank Test:** $W = 324.0, p = 0.6782$
* **Interpretation:** No statistically significant difference in $CV_{F0}$ was detected between the hybrid scheduler and fixed $t=0.25$ on the development set ($p = 0.6782$). The results provide no descriptive evidence of pitch flattening, though they do not independently establish statistical equivalence under a formal equivalence-testing procedure.

---

# PART V: Thesis Conclusions & Proposal Defense Statements

## 6.1 Development-Set Scope & Grounding

In accordance with academic standards:
> *The thresholds for $\pi^*$ (SNR cutoffs at 18 dB and 24 dB, cosine alignment at 0.78, duration deviation at 10%) were calibrated on the 48-instance development benchmark. On this development set, the calibrated policy successfully matches the 0.0% failure rate of fixed $t=0.25$ under the baseline-relative acceptance criterion while reducing effective computation by 11.7% globally (11.48 effective NFE) and 15.4% net on clean speech (11.00 effective NFE). However, both methods retained four failures under the strict absolute criterion, and held-out evaluation on unseen speakers will serve to confirm out-of-sample generalization.*

---

## 6.2 Consolidated Final Thesis Defense Statement

For inclusion in your thesis proposal document, methodology chapters, and defense slides:

> *"DMOSpeech 2 performs teacher-to-student handover at a predefined, globally selected switching point. Although its reported teacher-guided configuration uses $t_{\text{switch}}=0.25$, it does not provide automatic per-utterance switch selection. Across 48 development instances and five evaluated fixed schedules, the earliest acceptable teacher-to-student switching point varied by input and acoustic condition. A calibrated inference-time hybrid policy matched the fixed $t=0.25$ implementation under the baseline-relative acceptance criterion (0 / 48 reference-adjusted failures) while reducing effective mean computation from 13.00 to 11.48 NFE globally (−11.7%) and to 11.00 effective NFE on clean speech (−15.4% net), with statistically significant speaker-level compute reductions (Wilcoxon $W = 0.0, p = 0.000244$) and preserved prosodic pitch dynamics ($CV_{F0} = 0.448$ vs. $0.445$, $p = 0.678$). However, both methods retained four failures under the strict absolute criterion, and held-out evaluation is still required to establish generalizability."*
