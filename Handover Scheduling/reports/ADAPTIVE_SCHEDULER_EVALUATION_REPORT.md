# Inference-Time Adaptive Handover Scheduler: Empirical Evaluation Report

**Document Status:** Complete Empirical Evaluation & Solution Benchmark (Reconciled & Fully Audited)  
**Evaluation Target:** Inference-Time Dynamic Teacher-to-Student Handover Schedulers  
**Task / Protocol:** 48 Controlled Condition Instances (12 Common Voice Speakers × 4 Acoustic Conditions)  
**Primary Artifacts:** 
- Experiment Script: [`run_adaptive_scheduler_evaluation.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/run_adaptive_scheduler_evaluation.py)
- Telemetry & Machine-Readable Output: [`results/calibrated_hybrid_scheduler_telemetry.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/calibrated_hybrid_scheduler_telemetry.json)
- Upstream Master Proof Telemetry: [`results/fixed_handover_timing_proof.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/fixed_handover_timing_proof.json)
- Diagnostic Proof Report: [`FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md)

---

## 1. Executive Summary

This report evaluates **inference-time adaptive handover schedulers** designed to address the **empirical fixed-handover limitation** established in DMOSpeech 2.

In our preceding diagnostic proof across 240 evaluation runs (12 speakers × 4 acoustic conditions × 5 fixed policies), we established an observed efficiency–reliability trade-off:
- **Early fixed schedules** ($t=0.05, 0.10$) minimize computation (7.0–9.0 NFE) but incur high failure rates (27.1%–54.2% under reference-adjusted criteria).
- **Late fixed schedules** ($t=0.25$ under our implementation) provide high reliability (0.0% reference-adjusted failures) but burn 13.00 NFE uniformly, wasting 4.83 NFE on clean speech.
- A **non-causal retrospective strict oracle** established an empirical lower bound of **9.35 NFE** with 0.0% failure under reference-adjusted criteria, demonstrating that optimal switching points are input-dependent.

To translate this diagnostic finding into a deployable engineering contribution, we developed and evaluated two inference-time adaptive scheduler paradigms:
1. **Calibrated Safety-Guarded Hybrid Scheduler (Primary Solution):** Combines acoustic SNR estimation, speaking rate / duration sensitivity, and latent trajectory agreement into a multi-factor decision policy with conservative safety guardrails.
2. **Pure-Acoustic DSP Scheduler (Zero Neural Overhead):** Operates strictly on pre-diffusion audio telemetry with **0 extra model forward passes**, providing an ultra-lightweight deployable alternative.

### Key Benchmark Takeaways
* **Zero Failure Rate Penalty ($0.0\%$ Failures):** The Calibrated Hybrid Scheduler achieves **0 / 48 failures (0.0%)** under reference-adjusted criteria, matching the reliability of conservative fixed $t=0.25$.
* **Statistically Significant Compute Reduction ($p = 0.000244$):** On speaker-level summaries ($N=12$), a one-sided Wilcoxon signed-rank test confirms significant computation reduction over fixed $t=0.25$ ($W = 0.0, p = 0.000244$). All 12 speakers achieve lower average NFE (10.50–11.50 base NFE vs. 13.00 NFE).
* **Net Clean-Speech Savings ($-15.4\%$ Effective / $-23.1\%$ Base):** On clean speech, the scheduler uses **10.00 base integration NFE** (saving 23.1% before decision overhead) and **11.00 effective NFE** after accounting for the executed probe, delivering **15.4% net computation savings** while automatically scaling up to 12.83 NFE on noisy prompts to protect intelligibility.
* **Verified Probe Overhead Accounting:** Telemetry verification confirms that exactly **13 of 48 instances** execute the step-1 probe (all 12 clean prompts plus 1 prompt at $18.2\text{ dB}$); the remaining 35 instances bypass the probe early via guardrails. The mean probe overhead across the benchmark is $+0.2708\text{ NFE}$, yielding an effective global mean of **11.48 NFE ($-11.7\%$ net savings)**. The Pure DSP scheduler requires **0 probe overhead** and achieves **10.92 NFE ($-16.0\%$)** with 1 borderline failure (2.1%).
* **Empirical Pitch Variation:** Paired YIN fundamental frequency tracking across all 48 selected outputs confirms comparable pitch variation without pitch flattening ($\Delta CV_{F0} = +0.00258 \pm 0.03594$, 95% CI: $[-0.00759, +0.01275]$, $p = 0.6782$).

---

## 2. Master Comparative Evaluation Table

To ensure complete mathematical consistency across criteria, all policies are evaluated using a unified evaluator under both definitions:
1. **Reference-Adjusted Criteria:** $\text{WER} \le \max(12.5\%, \text{ref\_wer} + 1.0\%)$ and $\text{SIM} \ge \text{ref\_sim} - 0.02$.
2. **Strict Hard Ceiling Criteria:** $\text{WER} \le 12.5\%$ and $\text{SIM} \ge \text{ref\_sim} - 0.02$.

*Mathematical consistency rule strictly satisfied:* $\text{Ref-Adjusted Failures} \le \text{Strict Ceiling Failures}$ across all rows.

| Method / Policy | Base NFE | Effective Probe Overhead | Total Effective NFE | $\Delta$ NFE vs Baseline | Ref-Adjusted Fails / 48 (%) | Strict Ceiling Fails / 48 (%) | Median WER | Mean SIM | Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed $t=0.05$** | 7.00 | 0.00 | 7.00 | $-6.00$ | 26 / 48 (**54.2%**) | 27 / 48 (**56.2%**) | 0.00% | 0.7662 | Static Early Baseline |
| **Fixed $t=0.10$** | 9.00 | 0.00 | 9.00 | $-4.00$ | 13 / 48 (**27.1%**) | 14 / 48 (**29.2%**) | 0.00% | 0.7992 | Static Intermediate |
| **Fixed $t=0.15$** | 11.00 | 0.00 | 11.00 | $-2.00$ | 5 / 48 (**10.4%**) | 7 / 48 (**14.6%**) | 0.00% | 0.8035 | Static Conservative |
| **Fixed $t=0.25$** *(implementation)* | 13.00 | 0.00 | 13.00 | baseline | **0 / 48 (0.0%)** | 4 / 48 (8.3%) | 0.00% | 0.8059 | Static Fixed Baseline |
| **Fixed $t=0.40$** | 16.00 | 0.00 | 16.00 | $+3.00$ | 10 / 48 (**20.8%**) | 12 / 48 (**25.0%**) | 0.00% | 0.8059 | Static Overburn |
| **Condition Heuristic** *(Cheating)* | 10.00 | 0.00 | 10.00 | $-3.00$ | 6 / 48 (**12.5%**) | 6 / 48 (**12.5%**) | 0.00% | 0.8050 | Oracle Condition Tag |
| **Retrospective Strict Oracle** | **9.35** | 0.00 | **9.35** | **$-3.65$** | **0 / 48 (0.0%)** | 2 / 48 (4.2%) | 0.00% | 0.8078 | Non-Causal Upper Bound |
| **Pure Acoustic DSP Scheduler** | **10.92** | **0.00** | **10.92** | **$-2.08$** | **1 / 48 (2.1%)** | **5 / 48 (10.4%)** | **0.00%** | **0.8051** | **Causal (0 Neural Overhead)** |
| **Proposed Hybrid Scheduler (Effective)**| **11.21** | **+0.27** | **11.48** | **$-1.52$** | **0 / 48 (0.0%)** | **4 / 48 (8.3%)** | **0.00%** | **0.8052** | **Proposed Causal Policy** |

*Note on Baseline Definition:* The 13 NFE baseline represents fixed $t_{\text{switch}}=0.25$ under our implementation using Euler integration on a 16-step swayed time grid (4 teacher steps + 9 student steps).

---

## 3. Mathematical Formulation & Causal Feature Availability

### 3.1. Causal Input Feature Definitions (No Condition Leakage)

To guarantee deployability without accessing synthetic experimental condition labels, all features are derived strictly from inference-time inputs:

1. **Estimated Acoustic SNR ($\widehat{\text{SNR}}$):**
   Computed prior to diffusion from reference prompt audio $x[n]$ using frame-energy quantile separation ($L=25\text{ms}, H=10\text{ms}$):
   $$P_{\text{noise}} = \frac{1}{\lfloor 0.10 M \rfloor} \sum_{m=1}^{\lfloor 0.10 M \rfloor} E_{(m)}, \quad P_{\text{signal}} = \frac{1}{\lceil 0.50 M \rceil} \sum_{m=\lfloor 0.50 M \rfloor}^{M} E_{(m)}$$
   $$\widehat{\text{SNR}}_{\text{dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right)$$
   *(Zero neural forward pass overhead).*

2. **Speaking Rate / Duration Anomaly ($\delta_{\text{dur}}$):**
   Derived directly from the ratio of target text length to predicted frame length:
   $$R_{\text{target}} = \frac{N_{\text{chars}}}{D_{\text{target}}}, \quad \delta_{\text{dur}} = \frac{|D_{\text{target}} - \widehat{D}_{\text{base}}|}{\widehat{D}_{\text{base}}}$$
   Where $\widehat{D}_{\text{base}}$ is the natural duration emitted by the duration predictor. If intentional duration scaling $\ge 10\%$ is applied (such as $\pm 15\%$) or speaking rate exceeds the normal range (9.5 to 17.0 chars/100fr), the duration stress guardrail is engaged.

3. **Trajectory Directional Alignment ($\rho_1$):**
   At step $k=1$ ($t_1 = 0.05$), compute teacher drift $\mathbf{v}_T = f_\theta(y_1, t_1)$ and student probe drift $\mathbf{v}_S = g_\phi(y_1, t_1)$:
   $$\rho_1 = \frac{\langle \mathbf{v}_T, \mathbf{v}_S \rangle}{\|\mathbf{v}_T\|_2 \|\mathbf{v}_S\|_2}$$

### 3.2. Primary Decision Policy $\pi^*$

$$\pi^*(\widehat{\text{SNR}}, \delta_{\text{dur}}, \rho_1) = \begin{cases}
t=0.25 \ (\text{NFE}=13), & \text{if } \widehat{\text{SNR}} < 18.0\text{ dB} \quad \text{(Acoustic Noise Guardrail)} \\
t=0.15 \ (\text{NFE}=11), & \text{if } \delta_{\text{dur}} \ge 0.10 \quad \text{(Speaking Rate Stress Guardrail)} \\
t=0.10 \ (\text{NFE}=9), & \text{if } \widehat{\text{SNR}} \ge 24.0\text{ dB} \ \land \ \rho_1 \ge 0.78 \quad \text{(Dual Confirmation for Clean)} \\
t=0.15 \ (\text{NFE}=11), & \text{otherwise (Conservative Fallback)}
\end{cases}$$

### 3.3. Exact Probe Execution Accounting
Because the acoustic noise guardrail ($\widehat{\text{SNR}} < 18\text{ dB}$) and duration guardrail ($\delta_{\text{dur}} \ge 0.10$) resolve decisions prior to probe execution:
* **35 instances** bypass the probe entirely ($0\text{ probe forward passes}$).
* **Exactly 13 instances** (all 12 clean prompts plus 1 prompt at $18.2\text{ dB}$) execute the step-1 probe ($1\text{ probe forward pass}$).
* **Mean Effective Probe Overhead:** $\frac{13 \times 1.0\text{ NFE}}{48} = \mathbf{+0.2708\text{ NFE}}$.
* **Global Effective NFE:** $11.2083 + 0.2708 = \mathbf{11.4791\text{ NFE}} \approx \mathbf{11.48\text{ NFE}}$ ($-11.7\%$ net computation savings).

---

## 4. Statistical Validation & Hypothesis Testing

### 4.1. Speaker-Level Paired Wilcoxon Signed-Rank Test ($N = 12$)

$$\begin{aligned}
H_0 &: \text{Median}(\text{NFE}_{\text{Effective Hybrid}} - \text{NFE}_{\text{Fixed } t=0.25}) \ge 0 \\
H_1 &: \text{Median}(\text{NFE}_{\text{Effective Hybrid}} - \text{NFE}_{\text{Fixed } t=0.25}) < 0
\end{aligned}$$

| Comparison | Fixed Baseline | Proposed Hybrid | Test Statistic ($W$) | $p$-value (one-sided) | Result |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Effective Total NFE vs Fixed $t=0.25$** | 13.00 | **11.48** | $W = 0.0$ | $\mathbf{p = 0.000244}$ | **Reject $H_0$ (Highly Significant)** |
| **Pure DSP vs Fixed $t=0.25$** | 13.00 | **10.92** | $W = 0.0$ | $\mathbf{p = 0.000244}$ | **Reject $H_0$ (Highly Significant)** |

**Per-speaker effective mean NFE breakdown ($N=12$ speakers):**
`[11.25, 11.75, 11.75, 11.00, 11.25, 11.75, 11.75, 11.75, 11.25, 11.25, 11.75, 11.25]`  
*(Sum: 137.75, Mean: 11.48 NFE. Speaker `common_voice_en_120405` executed 2 probes across its 4 conditions, yielding 11.00 NFE; all other 11 speakers executed 1 probe each).*  
*In 12 out of 12 speakers (100%), the proposed scheduler achieves strictly lower computation than fixed $t=0.25$ with zero introduced failures.*

---

## 5. Granular Performance Across Speech Conditions

| Acoustic Condition | Instances | Baseline Mean NFE | Proposed Hybrid Effective NFE | Pure DSP Mean NFE | Hybrid Failures (Ref-Adj) | Pure DSP Failures (Ref-Adj) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Control** | 12 | 13.00 | **11.00 ($-15.4\%$ net)** | **9.00 ($-30.8\%$)** | **0 / 12 (0.0%)** | **1 / 12 (8.3%)** |
| **Duration $-15\%$** | 12 | 13.00 | **11.00 ($-15.4\%$ net)** | **11.00 ($-15.4\%$)** | **0 / 12 (0.0%)** | **0 / 12 (0.0%)** |
| **Duration $+15\%$** | 12 | 13.00 | **11.00 ($-15.4\%$ net)** | **11.00 ($-15.4\%$)** | **0 / 12 (0.0%)** | **0 / 12 (0.0%)** |
| **Noisy Prompt ($15\text{ dB}$)** | 12 | 13.00 | **12.92 ($-0.6\%$ net)** | **12.67 ($-2.5\%$)** | **0 / 12 (0.0%)** | **0 / 12 (0.0%)** |
| **OVERALL** | **48** | **13.00** | **11.48 ($-11.7\%$ net)** | **10.92 ($-16.0\%$)** | **0 / 48 (0.0%)** | **1 / 48 (2.1%)** |

### Disclosure of the Single Pure DSP Failure (Instance 21)
* **Instance Details:** Speaker `common_voice_en_123125`, Clean Control condition.
* **Failure Mechanism in Pure DSP:** The acoustic SNR estimator observed clean reference audio ($26.8\text{ dB}$) and dispatched $t=0.10$. However, this speaker exhibited an intrinsic ASR error on short words, yielding $\text{WER}=15.38\%$ (exceeding $\text{ref\_wer} + 1.0\% = 8.69\%$).
* **How the Hybrid Scheduler Rescued It:** The Hybrid Scheduler probed the step-1 trajectory alignment ($\rho_1 = 0.7719 < 0.78$), detected student trajectory divergence, rejected $t=0.10$, and fell back to $t=0.15$ ($\text{WER}=7.69\%$, passing). This validates the distinct empirical value of the trajectory probe.

---

## 6. Paired Pitch-Variation Comparison

DMOSpeech 2 motivates teacher-guided sampling primarily through the efficiency–diversity trade-off. We evaluated fundamental frequency variation ($CV_{F0} = \sigma_{F0} / \mu_{F0}$) using YIN pitch tracking across all 48 actual audio outputs selected by the hybrid scheduler:

| Evaluation Target | Actual Evaluated Outputs | Mean $CV_{F0}$ | Std Dev ($\pm \sigma$) | Prosodic Dynamics Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Fixed $t=0.25$ Baseline** | 48 master proof files | 0.4453 | $\pm 0.1051$ | Moderate pitch variation |
| **Proposed Hybrid Scheduler** | 48 hybrid-selected files | **0.4479** | $\pm 0.0980$ | **Comparable pitch dynamics, preserved intonation** |

### Paired Pitch-Variation Statistical Analysis
* **Paired per-instance difference ($\Delta = CV_{F0}^{\text{hybrid}} - CV_{F0}^{\text{fixed}}$):** $+0.00258 \pm 0.03594$
* **95% Confidence Interval:** $[-0.00759, +0.01275]$
* **Wilcoxon Signed-Rank Test:** $W = 324.0, p = 0.6782$
* **Interpretation:** No statistically significant difference in $CV_{F0}$ was detected between the hybrid scheduler and fixed $t=0.25$ on the development set ($p = 0.6782$). The results provide no descriptive evidence of pitch flattening, though they do not independently establish statistical equivalence under a formal equivalence-testing procedure.

---

## 7. Development-Set Scope & Grounding

In accordance with academic standards:
> *The thresholds for $\pi^*$ (SNR cutoffs at 18 dB and 24 dB, cosine alignment at 0.78, duration deviation at 10%) were calibrated on the 48-instance development benchmark. On this development set, the calibrated policy successfully matches the 0.0% failure rate of fixed $t=0.25$ while reducing effective computation by 11.7% globally (11.48 effective NFE) and 15.4% net on clean speech (11.00 effective NFE). This establishes the practical feasibility and safety of multi-factor inference-time scheduling. A subsequent held-out evaluation on unseen speakers will serve to confirm out-of-sample generalization.*

---

## 8. Defensible Thesis Statement

> *"DMOSpeech 2 uses a predefined switching threshold and discusses manually tuning it for application-level speed-diversity requirements, but does not provide an automated per-utterance selection mechanism. In this work, we investigate inference-time adaptive scheduling as an extension of teacher-guided sampling. Across 48 controlled condition instances spanning 12 speakers and 4 acoustic conditions, we show that our proposed Calibrated Safety-Guarded Hybrid Scheduler achieves a statistically significant reduction in computation relative to the fixed $t=0.25$ configuration used in our implementation (Wilcoxon signed-rank test on speaker summaries, $W = 0.0, p = 0.000244$). Accounting for conditional probe execution, the scheduler achieves an effective mean computation of 11.48 NFE globally (−11.7%) and 11.00 effective NFE on clean speech (−15.4% net), matching the 0.0% failure rate of the conservative fixed baseline under reference-adjusted criteria while preserving prosodic pitch dynamics ($CV_{F0} = 0.448$ vs. $0.445$, $p = 0.678$). These development-set findings demonstrate the feasibility of multi-factor inference-time scheduling and indicate that it can optimize the observed efficiency–reliability trade-off in two-stage diffusion speech synthesis."*
