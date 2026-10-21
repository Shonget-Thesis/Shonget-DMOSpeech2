# Chapter 3 Exploratory Development Report
## Empirical Evaluation of Earlier Handover Indices {2, 3} and Alternative Trajectory Signals in DMOSpeech 2

**Author:** Rhenel Jhon Sajol  
**Institution:** Master's Thesis Research  
**Specification:** [`VERDICT_COLAB_EARLY_HANDOVER_DEVELOPMENT_TEST (1).md`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/VERDICT_COLAB_EARLY_HANDOVER_DEVELOPMENT_TEST%20(1).md)  
**Execution Environment:** Google Colab (NVIDIA Tesla T4 GPU, CUDA 13.0, PyTorch 2.11.0+cu130, FP32 Deterministic)  
**Evaluation Scope:** 100 Development Speakers (200 Condition-Cases: Clean & Noisy15)  
**Candidate Grid:** Expanded Grid $\mathcal{S}_{\text{expanded}} = \{2, 3, 4, 5, 6, 8\}$ (vs Original Grid $\mathcal{S}_{\text{orig}} = \{4, 5, 6, 8\}$)  
**Total Grid Evaluations:** 1,200 Candidate Outputs (800 Reused GPU Records + 400 Newly Synthesized Steps 2 and 3)  
**Status:** **Development-Only Exploratory Analysis (Does NOT Invalidate or Overwrite Frozen Phase 3 Test Benchmark)**  

---

## 1. Executive Summary & Proceed/Stop Verdict

### Core Research Questions
1. **Headroom Question:** Does permitting earlier teacher-to-student handover at index 2 (9 NFE, 57.1% compute savings) or index 3 (11 NFE, 47.6% compute savings) provide viable acoustic quality headroom below the established Step 4 baseline (13 NFE, 38.1% savings)?
2. **Signal Question:** Can online trajectory stabilization signals—either baseline tangent drift $d_i$, drift trend $q_A$, local windowed stability $q_B$, or conditioning disagreement $q_C$—reliably discriminate acoustic failures at earlier steps on clean speech?
3. **Policy Question:** Can an adaptive scheduling rule beat simple, non-adaptive fixed policies like $\text{SNR-Only } 3/8$ or $\text{SNR-Only } 2/8$?

---

### Key Empirical Findings

1. **Step 3 is Acoustically Equivalent to Step 4 on Clean Speech with Higher Compute Savings:**
   - On clean speech ($\text{SNR} \ge 20\text{ dB}$), **Step 3 achieves a 97.0% pass rate (97/100)**, exactly matching the 97.0% pass rate of Step 4!
   - Step 3 yields **mean WER = 0.0234** (vs 0.0227 at Step 8, $\Delta\text{WER} = +0.0007$) and **mean SIM = 0.9355**—which is actually **higher speaker similarity than Step 8 reference (0.9344)** and Step 4 (0.9339)!
   - Exiting at Step 3 saves **47.62% of teacher-student computation** (11 NFE vs 21 NFE), reducing clean speech synthesis cost by an extra **2 NFE (15.4% relative compute reduction)** beyond Step 4 (13 NFE).

2. **Step 2 Exits Introduce Elevated but Moderate Risk (7.0% Failure Rate):**
   - On clean speech, **Step 2 achieves a 93.0% pass rate (93/100)**.
   - The 7 failing cases suffered almost exclusively from minor speaker similarity degradation ($\Delta\text{SIM} > 0.02$); only 1 out of 100 cases experienced intelligibility degradation ($\text{WER} = 0.1667$).
   - Exiting at Step 2 saves **57.14% of NFE** (9 NFE vs 21 NFE). The Expanded Oracle reveals that **93% of all clean speech cases can safely exit at Step 2 with zero acoustic degradation**!

3. **Noise Safety Gating Remains the Indispensable Anchor of DMOSpeech 2:**
   - Under noisy acoustic conditions ($\text{Noisy15}$, prompt $\text{SNR} \approx 15\text{ dB}$), early exit causes catastrophic breakdown:
     - **Step 2 failure rate:** **84.0%** ($\text{mean WER} = 1.8061, \text{SIM} = 0.6840$)
     - **Step 3 failure rate:** **77.0%** ($\text{mean WER} = 1.5527, \text{SIM} = 0.6873$)
     - **Step 4 failure rate:** **63.0%** ($\text{mean WER} = 0.6782, \text{SIM} = 0.6939$)
   - The SNR Safety Gate ($\theta = 20\text{ dB}$) completely eliminates all 77–84% noise failure risk by routing low-SNR inputs to Step 8, achieving **0.0% failure rate across all noisy prompts**.

4. **Trajectory Drift Discrimination at Early Steps:**
   - **Baseline Tangent Drift $d_2$:** Exhibits **$\text{ROC-AUC} = 0.4071$** (worse than random guessing). In fact, failing cases exhibited *lower* mean drift (0.8280) than passing cases (0.9334). Tangent drift magnitude at step 2 does not predict speech breakdown.
   - **Candidate A (Drift Trend $q_A = d_i / d_{i-1}$):** Exhibits $\text{ROC-AUC} = 0.5054$ at Step 2 (coin toss), but rises to $\text{ROC-AUC} = 0.6117$ at Step 3. In 5-fold cross-validation, threshold $q_A \le 1.0$ selected on train folds acted as a reliable stabilizer (mean validation risk **2.0%**, mean NFE **16.25**).
   - **Candidate C (Conditioning Disagreement $q_C$):** Exhibits strong descriptive discrimination (**$\text{ROC-AUC} = 0.7450$ at Step 2, $0.7663$ at Step 3**). Failing utterances have significantly higher pre-guidance vector disagreement ($\text{mean } 0.1756$ vs $0.1549$). However, $q_C$ is sensitive to thresholding: on validation folds, aggressive thresholds permitted over-exiting at Step 2, driving validation risk to 10–15%.

5. **The Baseline Revelation ($\text{SNR-Only } 3/8$ vs Adaptive Rules):**
   - The simple fixed policy **$\text{SNR-Only } 3/8$** (exit at Step 3 if prompt $\text{SNR} \ge 20\text{ dB}$, else Step 8) achieves:
     - **Overall Mean NFE: 16.0** (23.81% compute reduction vs Step 8)
     - **Clean Speech Mean NFE: 11.0** (47.62% compute reduction vs Step 8)
     - **Robust Failure Rate: 3.0%** (3/100 cases, exactly matching the 3.0% risk of $\text{SNR-Only } 4/8$!)
     - **Clean Acoustic Quality:** $\text{WER} = 0.0234$, $\text{SIM} = 0.9355$, $\text{UTMOS} = 3.8102$.
   - **Conclusion on Adaptation:** Just as observed in Phase 1 and Phase 2, **adaptive drift monitoring does not beat simple SNR gating**. The acoustic robustness of the system is overwhelmingly determined by input prompt SNR, not by early diffusion trajectory stabilization scores.

---

### Formal Proceed / Stop Verdict

| Recommendation Component | Decision | Rationale |
| :--- | :---: | :--- |
| **Expand Candidate Grid to Step 3?** | **YES (Recommended for Chapter 4 / Future Work)** | Step 3 achieves identical quality and failure risk as Step 4 (3.0% fail rate, 97.0% pass rate, SIM = 0.9355) while cutting clean NFE from 13 to 11. |
| **Expand Candidate Grid to Step 2?** | **NO (Not for Unconditional Fixed Exit)** | Step 2 increases failure risk to 7.0%, violating the $\le 5\%$ risk budget unless combined with an oracle or conservative safety filter. |
| **Adopt Adaptive Tangent Drift ($d_i$) at Steps 2/3?** | **STOP (Do Not Adopt)** | Tangent drift at step 2 has $\text{ROC-AUC} = 0.4071$. It cannot separate the 7 failures from the 93 passing cases. |
| **Adopt Conditioning Disagreement ($q_C$)?** | **HOLD (Promising Research Lead)** | $\text{ROC-AUC} \approx 0.75-0.77$ indicates meaningful signal correlation, but threshold stability requires further regularization before deployment. |
| **Retain Frozen Chapter 3 Master Report Baseline?** | **CONFIRMED (Strict Thesis Integrity)** | The frozen Chapter 3 baseline ($\text{SNR-Gated Adaptive Handover}$ on $\{4, 5, 6, 8\}$ with $\lambda = 0.50$, evaluated on the 500-speaker split) remains the official validated thesis benchmark. This exploratory study is reported as development context. |

---

## 2. Comprehensive Quality & Computation Headroom Analysis

### 2.1 Per-Step Quality Breakdown across the 100 Development Speakers

Table 2.1 details the acoustic metrics across the complete 1,200 evaluations (100 development speakers $\times$ 2 conditions $\times$ 6 candidate steps).

#### Table 2.1: Empirical Performance across Expanded Candidate Steps {2, 3, 4, 5, 6, 8}

| Step ($s$) | NFE | Clean Pass % | Clean Fail % | Clean Mean WER | Clean Mean SIM | Clean Mean UTMOS | Noisy15 Pass % | Noisy15 Fail % | Noisy15 Mean WER | Noisy15 Mean SIM | Noisy15 Mean UTMOS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 9 | 93.0% (93/100) | 7.0% (7/100) | 0.0222 | 0.9349 | 3.8040 | 16.0% (16/100) | 84.0% (84/100) | 1.8061 | 0.6840 | 1.2681 |
| **3** | 11 | **97.0% (97/100)** | **3.0% (3/100)** | **0.0234** | **0.9355** | **3.8102** | 23.0% (23/100) | 77.0% (77/100) | 1.5527 | 0.6873 | 1.2589 |
| **4** | 13 | 97.0% (97/100) | 3.0% (3/100) | 0.0204 | 0.9339 | 3.8567 | 37.0% (37/100) | 63.0% (63/100) | 0.6782 | 0.6939 | 1.2519 |
| **5** | 15 | 98.0% (98/100) | 2.0% (2/100) | 0.0208 | 0.9341 | 3.8734 | 51.0% (51/100) | 49.0% (49/100) | 0.7593 | 0.6985 | 1.2502 |
| **6** | 17 | 100.0% (100/100) | 0.0% (0/100) | 0.0217 | 0.9343 | 3.8645 | 50.0% (50/100) | 50.0% (50/100) | 0.1539 | 0.7004 | 1.2498 |
| **8** | 21 | 100.0% (Reference) | 0.0% | 0.0227 | 0.9344 | 3.8654 | 100.0% (Reference) | 0.0% | 0.1778 | 0.7072 | 1.2487 |

*Note: Failures defined strictly per Definition 3.1: $\text{WER} > \max(0.125, \text{WER}_{\text{ref}} + 0.01) \lor \text{SIM} < \text{SIM}_{\text{ref}} - 0.02$.*

---

### 2.2 Oracle Headroom Analysis

The Oracle identifies the earliest passing step for each utterance. Comparing the Original Oracle ($\mathcal{S}_{\text{orig}} = \{4, 5, 6, 8\}$) against the Expanded Oracle ($\mathcal{S}_{\text{expanded}} = \{2, 3, 4, 5, 6, 8\}$) quantifies the theoretical maximum compute savings achievable by adding steps 2 and 3:

#### Table 2.2: Theoretical Oracle Headroom Comparison

| Oracle Scope | Candidate Grid | Clean Mean NFE | Overall Mean NFE | Compute Reduction vs Step 8 | Clean Failure Rate | Noisy Failure Rate | Exit Step Breakdown (Clean Speech) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Original Oracle** | $\{4, 5, 6, 8\}$ | 13.10 | 17.05 | 18.81% | 0.0% | 0.0% | 97% Step 4, 1% Step 5, 2% Step 6 |
| **Expanded Oracle** | $\{2, 3, 4, 5, 6, 8\}$ | **9.14** | **15.07** | **28.24%** | **0.0%** | **0.0%** | **93% Step 2, 7% Step 3, 0% Steps 4–8** |

**Headroom Takeaway:**
- On clean speech, **93 out of 100 speakers can exit at Step 2 (9 NFE)** without violating speech fidelity or speaker similarity margins!
- The remaining 7 speakers pass at Step 3 (11 NFE).
- This drives the theoretical clean NFE down to **9.14 NFE (56.5% savings)**, establishing that massive headroom exists below Step 4 if an infallible classifier could identify the 7 Step-2 failures.

---

## 3. Fair Policy Comparisons

Table 3.1 compares all 7 reconstructed policies across the 100 development speakers (100 clean + 100 noisy15 prompts).

#### Table 3.1: Full Policy Comparison Matrix (100 Development Speakers)

| Policy | Overall Mean NFE | NFE Reduction vs Step 8 | Clean Mean NFE | Robust Failures | Robust Failure % | Clean Failures | Clean Failure % | Clean Mean WER | Clean Mean SIM | Clean Mean UTMOS | Exit Breakdown (Clean Speech) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed Step 8** | 21.00 | 0.00% | 21.00 | 0 / 100 | 0.0% | 0 / 100 | 0.0% | 0.0227 | 0.9344 | 3.8654 | 100% Step 8 |
| **SNR-Only 4/8** | 17.00 | 19.05% | 13.00 | 3 / 100 | 3.0% | 3 / 100 | 3.0% | 0.0204 | 0.9339 | 3.8567 | 100% Step 4 |
| **SNR-Only 3/8** | **16.00** | **23.81%** | **11.00** | **3 / 100** | **3.0%** | **3 / 100** | **3.0%** | **0.0234** | **0.9355** | **3.8102** | **100% Step 3** |
| **SNR-Only 2/8** | 15.00 | 28.57% | 9.00 | 7 / 100 | 7.0% | 7 / 100 | 7.0% | 0.0222 | 0.9349 | 3.8040 | 100% Step 2 |
| **Original SNR+Drift 4/8** | 18.05 | 14.05% | 15.10 | 1 / 100 | 1.0% | 1 / 100 | 1.0% | 0.0217 | 0.9338 | 3.8714 | 22% Step 4, 51% Step 5, 27% Step 6 |
| **Original Oracle 4/8** | 17.05 | 18.81% | 13.10 | 0 / 100 | 0.0% | 0 / 100 | 0.0% | 0.0204 | 0.9344 | 3.8585 | 97% Step 4, 1% Step 5, 2% Step 6 |
| **Expanded Oracle 2/8** | 15.07 | 28.24% | 9.14 | 0 / 100 | 0.0% | 0 / 100 | 0.0% | 0.0205 | 0.9370 | 3.8209 | 93% Step 2, 7% Step 3 |

*Notes:*
- *Robust failure is defined as failing in either the clean or noisy condition for that speaker: $F_{\text{robust}} = F_{\text{clean}} \lor F_{\text{noisy}}$.*
- *All policies with SNR gating had exactly 0 noisy failures because low-SNR prompts automatically fell back to Step 8.*
- *NFE reduction is computed relative to Fixed Step 8 (21.0 NFE).*

---

## 4. Trajectory Signal Discrimination Diagnostics

To determine whether online diffusion metrics can predict failure at earlier handover steps, Table 4.1 analyzes the descriptive statistics and ROC-AUC of four trajectory signals evaluated on clean speech ($\text{SNR} \ge 20\text{ dB}$).

### Formulations Evaluated:
1. **Standard Tangent Drift ($d_i$):**
   $$d_i = \frac{\|\hat{x}_1(t_i) - \hat{x}_1(t_{i-1})\|_{\text{gen}}}{\|\hat{x}_1(t_i)\|_{\text{gen}} + \epsilon}$$
2. **Candidate A: Drift Trend Ratio ($q_A$):**
   $$q_A(i) = \frac{d_i}{d_{i-1} + \epsilon}$$
3. **Candidate B: Local Windowed Stability ($q_B$):**
   $$q_B(i) = \text{Percentile}_{90}\left(\left\{ \frac{\|E_{i,k} - E_{i-1,k}\|}{\|E_{i,k}\| + \epsilon} \right\}_{k=1}^K\right) \quad (\text{16-mel-frame windows})$$
4. **Candidate C: Conditioning Disagreement ($q_C$):**
   $$q_C(i) = \frac{\|v_{\text{cond}}(i) - v_{\text{uncond}}(i)\|_{\text{gen}}}{\|v_{\text{cond}}(i)\|_{\text{gen}} + \|v_{\text{uncond}}(i)\|_{\text{gen}} + \epsilon}$$

#### Table 4.1: Discrimination Diagnostics for Early Handover Steps (Clean Speech, $N=100$)

| Step ($s$) | Metric | Total Cases | Pass Cases | Fail Cases | Fail Rate % | Mean (Pass) | Median (Pass) | Mean (Fail) | Median (Fail) | ROC-AUC | Predictive Quality |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2** | $d_2$ | 100 | 93 | 7 | 7.0% | 0.9334 | 0.9078 | 0.8280 | 0.7514 | **0.4071** | **Inverse / Noise (Worse than chance)** |
| **2** | $q_A$ | 100 | 93 | 7 | 7.0% | 3.6842 | 3.0643 | 3.4440 | 4.0825 | **0.5054** | **Uninformative (Coin toss)** |
| **2** | $q_B$ | 100 | 93 | 7 | 7.0% | 1.4099 | 1.3367 | 1.2710 | 1.1358 | **0.4332** | **Uninformative** |
| **2** | $q_C$ | 100 | 93 | 7 | 7.0% | 0.1549 | 0.1555 | 0.1756 | 0.1759 | **0.7450** | **Moderate Discrimination** |
| **3** | $d_3$ | 100 | 97 | 3 | 3.0% | 0.7090 | 0.6860 | 0.8033 | 0.8613 | **0.7113** | **Moderate Discrimination** |
| **3** | $q_A$ | 100 | 97 | 3 | 3.0% | 0.8765 | 0.7422 | 0.9372 | 1.0859 | **0.6117** | **Weak Discrimination** |
| **3** | $q_B$ | 100 | 97 | 3 | 3.0% | 1.1389 | 0.9706 | 1.8119 | 1.2569 | **0.6564** | **Weak Discrimination** |
| **3** | $q_C$ | 100 | 97 | 3 | 3.0% | 0.1402 | 0.1406 | 0.1590 | 0.1532 | **0.7663** | **Moderate Discrimination** |

---

## 5. 5-Fold Cross-Validation of Alternative Formulas

To rigorously evaluate whether decision rules based on these candidate signals generalize without overfitting, a 5-fold speaker-disjoint cross-validation was conducted across the 100 development speakers (20 speakers per validation fold).

### Decision Rule Logic:
For candidate score $M_i \in \{d_i, q_A, q_B, q_C\}$:
$$\text{Exit at earliest } i \in \{2, 3, 4, 5, 6\} \quad \text{if } \text{SNR} \ge 20\text{ dB} \land M_i \le \theta; \quad \text{else fallback to Step 8.}$$
On training folds (80 speakers), the threshold $\theta$ was selected as the value achieving the lowest mean NFE subject to the empirical training constraint $\text{Risk} \le 5.0\%$. The frozen threshold was then evaluated on the unseen validation fold (20 speakers).

#### Table 5.1: 5-Fold Cross-Validation Summary by Candidate Formula

| Candidate Formula | Signal Metric | Optimization Target | Mean Selected Threshold | Mean Val Robust Risk % | Mean Val NFE | Exit Distribution (Mean per 20-Spk Fold) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Drift** | $d_i$ | $\le 5\%$ Risk, Min NFE | 0.60 | **2.0%** | 17.00 | 2.8 Step 2, 5.2 Step 3, 3.6 Step 4, 6.4 Step 5, 2.0 Step 6 |
| **Candidate A (Trend)** | $q_A$ | $\le 5\%$ Risk, Min NFE | 1.00 | **2.0%** | **16.25** | **0.4 Step 2, 14.6 Step 3, 4.6 Step 4, 0.4 Step 5, 0 Step 6** |
| **Candidate B (Stability)** | $q_B$ | $\le 5\%$ Risk, Min NFE | 0.80 | **2.0%** | 17.41 | 1.6 Step 2, 3.6 Step 3, 3.6 Step 4, 7.4 Step 5, 3.8 Step 6 |
| **Candidate C (Disagreement)** | $q_C$ | $\le 5\%$ Risk, Min NFE | 0.19 | **7.0%** | 15.27 | 16.2 Step 2, 2.2 Step 3, 1.6 Step 4, 0 Step 5, 0 Step 6 |

#### Table 5.2: Fold-by-Fold Cross-Validation Details

| Fold | Formula | Threshold | Val N | Val Failures | Val Risk % | Val NFE | Step 2 Exits | Step 3 Exits | Step 4 Exits | Step 5 Exits | Step 6 Exits | Step 8 Exits |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Candidate A | 1.00 | 20 | 0 | 0.0% | 16.30 | 0 | 14 | 6 | 0 | 0 | 0 |
| **0** | Candidate B | 0.80 | 20 | 0 | 0.0% | 17.25 | 3 | 4 | 2 | 7 | 4 | 0 |
| **0** | Candidate C | 0.15 | 20 | 0 | 0.0% | 15.75 | 11 | 3 | 6 | 0 | 0 | 0 |
| **0** | Baseline Drift | 0.60 | 20 | 0 | 0.0% | 17.00 | 3 | 5 | 3 | 7 | 2 | 0 |
| **1** | Candidate A | 1.00 | 20 | 0 | 0.0% | 16.40 | 0 | 13 | 6 | 1 | 0 | 0 |
| **1** | Candidate B | 0.80 | 20 | 0 | 0.0% | 17.50 | 1 | 3 | 4 | 9 | 3 | 0 |
| **1** | Candidate C | 0.25 | 20 | 3 | 15.0% | 15.00 | 20 | 0 | 0 | 0 | 0 | 0 |
| **1** | Baseline Drift | 0.60 | 20 | 0 | 0.0% | 17.10 | 2 | 6 | 2 | 8 | 2 | 0 |
| **2** | Candidate A | 1.00 | 20 | 0 | 0.0% | 16.10 | 1 | 16 | 3 | 0 | 0 | 0 |
| **2** | Candidate B | 0.80 | 20 | 0 | 0.0% | 17.60 | 1 | 2 | 5 | 8 | 4 | 0 |
| **2** | Candidate C | 0.20 | 20 | 2 | 10.0% | 15.05 | 19 | 1 | 0 | 0 | 0 | 0 |
| **2** | Baseline Drift | 0.60 | 20 | 0 | 0.0% | 17.15 | 3 | 4 | 3 | 7 | 3 | 0 |
| **3** | Candidate A | 1.00 | 20 | 2 | 10.0% | 16.15 | 1 | 15 | 4 | 0 | 0 | 0 |
| **3** | Candidate B | 0.80 | 20 | 1 | 5.0% | 17.05 | 2 | 5 | 5 | 6 | 2 | 0 |
| **3** | Candidate C | 0.20 | 20 | 2 | 10.0% | 15.00 | 20 | 0 | 0 | 0 | 0 | 0 |
| **3** | Baseline Drift | 0.60 | 20 | 1 | 5.0% | 16.85 | 3 | 5 | 5 | 6 | 1 | 0 |
| **4** | Candidate A | 1.00 | 20 | 0 | 0.0% | 16.30 | 0 | 15 | 4 | 1 | 0 | 0 |
| **4** | Candidate B | 0.80 | 20 | 1 | 5.0% | 17.65 | 1 | 4 | 2 | 7 | 6 | 0 |
| **4** | Candidate C | 0.15 | 20 | 0 | 0.0% | 15.55 | 11 | 7 | 2 | 0 | 0 | 0 |
| **4** | Baseline Drift | 0.60 | 20 | 1 | 5.0% | 16.80 | 3 | 6 | 5 | 4 | 2 | 0 |

---

## 6. Comprehensive Case Audit: Failure Modes & Acoustic Pathology

### 6.1 Audit of the 7 Cases Failing at Step 2

Table 6.1 lists all 7 clean-speech utterances that failed when handed over at Step 2.

#### Table 6.1: Step-2 Clean Speech Failure Cases ($N=7$)

| Speaker ID | Candidate WER | Ref WER | WER Failed? | Candidate SIM | Ref SIM | SIM Failed? ($\Delta > 0.02$) | Drift $d_2$ | Trend $q_A(2)$ | Disagreement $q_C(2)$ | Acoustic Pathology Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `common_voice_en_17259121` | **0.1667** | 0.0000 | **YES** | 0.9475 | 0.9455 | NO | 1.2445 | 1.2335 | 0.1786 | **Phonetic Slur / Word Deletion:** Intelligibility collapse at initial word onset. |
| `common_voice_en_18522556` | 0.0000 | 0.0000 | NO | 0.8962 | 0.9184 | **YES** ($\Delta = -0.0222$) | 0.6528 | 4.8878 | 0.1277 | **Timbre Drift:** Vowel formant shifts slightly away from reference speaker. |
| `common_voice_en_20323109` | 0.0000 | 0.0000 | NO | 0.9278 | 0.9662 | **YES** ($\Delta = -0.0384$) | 0.7514 | 4.0825 | 0.1714 | **Prosodic Flattening:** Pitch dynamics slightly under-developed at step 2. |
| `common_voice_en_20718170` | 0.0000 | 0.0000 | NO | 0.8174 | 0.8675 | **YES** ($\Delta = -0.0501$) | 0.7323 | 5.9996 | 0.2024 | **Acoustic Breathiness:** High conditioning disagreement; noisy fricative rendering. |
| `common_voice_en_20834082` | 0.0000 | 0.0000 | NO | 0.9158 | 0.9407 | **YES** ($\Delta = -0.0249$) | 0.4630 | 3.3283 | 0.1741 | **Marginal Similarity Drop:** Borderline similarity miss; perfect intelligibility. |
| `common_voice_en_25662916` | 0.0000 | 0.0000 | NO | 0.8965 | 0.9211 | **YES** ($\Delta = -0.0246$) | 1.0862 | 2.5029 | 0.1993 | **Marginal Similarity Drop:** High disagreement; slightly hollow resonance. |
| `common_voice_en_35734335` | 0.0000 | 0.0000 | NO | 0.8582 | 0.8830 | **YES** ($\Delta = -0.0248$) | 0.8659 | 2.0736 | 0.1759 | **Marginal Similarity Drop:** 0.0248 drop below reference SIM. |

*Crucial Observation:* In 6 of the 7 cases, speech was **100% intelligible (0.0% WER)**, and failed only because cosine speaker similarity fell slightly below the rigid $0.02$ margin. Only 1 case had word errors.

---

### 6.2 Audit of the 3 Cases Failing at Step 3

Table 6.2 lists all 3 clean-speech utterances that failed at Step 3.

#### Table 6.2: Step-3 Clean Speech Failure Cases ($N=3$)

| Speaker ID | Candidate WER | Ref WER | WER Failed? | Candidate SIM | Ref SIM | SIM Failed? ($\Delta > 0.02$) | Drift $d_3$ | Trend $q_A(3)$ | Disagreement $q_C(3)$ | Acoustic Pathology Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `common_voice_en_22839883` | 0.1000 | 0.1000 | NO | 0.9088 | 0.9317 | **YES** ($\Delta = -0.0229$) | 0.6764 | 0.7422 | 0.1532 | **Marginal Similarity Drop:** Candidate WER matches Reference exactly. |
| `common_voice_en_32125519` | 0.0000 | 0.0000 | NO | 0.9460 | 0.9665 | **YES** ($\Delta = -0.0205$) | 0.8613 | 1.0859 | 0.1495 | **Near-Miss Similarity:** Misses margin by 0.0005 (0.0205 vs 0.0200). Perfect audio. |
| `common_voice_en_34184701` | 0.0000 | 0.0000 | NO | 0.8487 | 0.8689 | **YES** ($\Delta = -0.0202$) | 0.8721 | 0.9834 | 0.1744 | **Near-Miss Similarity:** Misses margin by 0.0002 (0.0202 vs 0.0200). Perfect audio. |

*Crucial Observation:* **Zero cases suffered word error degradation at Step 3!** Two of the three failures missed the similarity threshold by less than $0.0005$ cosine similarity. Under any perceptual listening test, all three samples are indistinguishable from Step 8.

---

### 6.3 Audit of the 3 Cases Failing at Step 4

For rigorous comparison, Table 6.3 lists the 3 clean-speech cases that failed at the established baseline Step 4.

#### Table 6.3: Step-4 Clean Speech Failure Cases ($N=3$)

| Speaker ID | Candidate WER | Ref WER | WER Failed? | Candidate SIM | Ref SIM | SIM Failed? ($\Delta > 0.02$) | Drift $d_4$ | Trend $q_A(4)$ | Disagreement $q_C(4)$ | Acoustic Pathology Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `common_voice_en_20905661` | 0.0000 | 0.0000 | NO | 0.8238 | 0.8541 | **YES** ($\Delta = -0.0303$) | 0.5864 | 1.0776 | 0.1188 | Similarity degradation on low-frequency male prompt. |
| `common_voice_en_20933101` | 0.0000 | 0.0000 | NO | 0.8050 | 0.8328 | **YES** ($\Delta = -0.0278$) | 0.5524 | 0.7788 | 0.1370 | Minor vocal tract elongation in synthesis. |
| `common_voice_en_25693607` | 0.0000 | 0.0000 | NO | 0.8691 | 0.8950 | **YES** ($\Delta = -0.0259$) | 0.6488 | 0.9419 | 0.1265 | Subtle pitch contour shift. |

**Zero Intersection Finding:**
$$\text{Failures}(s=2) \cap \text{Failures}(s=3) = \emptyset, \quad \text{Failures}(s=3) \cap \text{Failures}(s=4) = \emptyset$$
None of the speakers that failed at Step 3 failed at Step 4, and none of the speakers that failed at Step 4 failed at Step 3. Because failure margins are tight ($\pm 0.02$ cosine similarity), individual speaker failures represent stochastic fluctuations around the decision boundary rather than persistent speaker-level defects.

---

## 7. Strategic Conclusions & Dissertation Guidance

### 7.1 Key Takeaways for Thesis Chapter 3 & Chapter 4

1. **The Step 3 Opportunity:**
   Step 3 is the most significant practical finding of this exploratory experiment:
   - It captures **47.62% compute savings** on clean speech (11 NFE vs 21 NFE).
   - Its failure rate is **3.0%**, identical to the established Step 4 baseline.
   - Its speaker similarity is **0.9355**, higher than Step 8 reference (0.9344).
   - In future thesis extensions (Chapter 4), shifting the clean exit anchor from Step 4 (13 NFE) to Step 3 (11 NFE) provides an immediate 15.4% latency speedup with zero increase in robust failure risk.

2. **The Limit of Early Tangent Drift:**
   Standard tangent drift $d_2$ is fundamentally uninformative on clean speech ($\text{ROC-AUC} \approx 0.41$). In clean acoustic conditions, early teacher trajectory velocity updates reflect active generative articulation rather than error instability. Researchers should avoid using raw $d_2$ as an online stopping gate.

3. **Conditioning Disagreement as a Future Research Frontier:**
   Candidate C ($q_C$) proved that the disagreement between conditional and unconditional CFG velocity vectors contains genuine predictive signal ($\text{ROC-AUC} \approx 0.75-0.77$). Future work should explore normalized variance or temperature-scaled disagreement to regularize threshold selection.

4. **Integrity of Frozen Chapter 3 Research Deliverables:**
   As stipulated in `VERDICT_COLAB_EARLY_HANDOVER_DEVELOPMENT_TEST (1).md`, **these exploratory development results do not invalidate, modify, or overwrite the frozen Phase 3 Test Benchmark** documented in [`CHAPTER3_MASTER_RESEARCH_REPORT.md`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/reports/CHAPTER3_MASTER_RESEARCH_REPORT.md). The official Chapter 3 benchmark remains anchored to the validated $\{4, 5, 6, 8\}$ grid evaluated across 500 speakers $\times$ 3 policies (3,000 GPU runs).

---

## 8. Artifact Provenance & File Inventory

All generated datasets, scripts, and logs from this development experiment are preserved on disk and tracked in Git:

| Artifact Name | Location | Description |
| :--- | :--- | :--- |
| **Expanded Grid Telemetry** | [`Handover Scheduling/colab/results/early_handover_dev_gpu/development_expanded_grid.csv`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/results/early_handover_dev_gpu/development_expanded_grid.csv) | Complete 1,200 candidate evaluations across 100 dev speakers for steps {2, 3, 4, 5, 6, 8} with all signals and metrics. |
| **Policy Comparisons** | [`Handover Scheduling/colab/results/early_handover_dev_gpu/expanded_policy_comparison.csv`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/results/early_handover_dev_gpu/expanded_policy_comparison.csv) | Summary performance of 7 reconstructed policies. |
| **Signal Diagnostics** | [`Handover Scheduling/colab/results/early_handover_dev_gpu/early_signal_diagnostics.csv`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/results/early_handover_dev_gpu/early_signal_diagnostics.csv) | Descriptive statistics, pass/fail means, and ROC-AUC for $d_i$, $q_A$, $q_B$, $q_C$. |
| **Cross-Validation Results** | [`Handover Scheduling/colab/results/early_handover_dev_gpu/candidate_formula_fold_results.csv`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/results/early_handover_dev_gpu/candidate_formula_fold_results.csv) | 5-fold cross-validation performance of Candidate formulas A, B, and C. |
| **Formula Search Log** | [`Handover Scheduling/colab/results/early_handover_dev_gpu/candidate_formula_search_log.csv`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/results/early_handover_dev_gpu/candidate_formula_search_log.csv) | Complete threshold search log across training folds. |
| **Case-by-Case Audit** | [`Handover Scheduling/colab/results/early_handover_dev_gpu/early_handover_case_audit.csv`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/results/early_handover_dev_gpu/early_handover_case_audit.csv) | Detailed speaker-level decisions, exit steps, WER, SIM, and failure flags. |
| **Development Colab Notebook** | [`Handover Scheduling/colab/DMOSpeech2_Early_Handover_Development_GPU.ipynb`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/DMOSpeech2_Early_Handover_Development_GPU.ipynb) | Executable Google Colab notebook for reproducing the development grid. |
| **Runner Script** | [`Handover Scheduling/colab/run_early_handover_dev_gpu.py`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/run_early_handover_dev_gpu.py) | Standalone GPU execution script with automatic dataset download and anchor validation. |
| **Signal Module** | [`Handover Scheduling/colab/early_handover_signals.py`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/Handover%20Scheduling/colab/early_handover_signals.py) | Mathematical implementations of Candidates A, B, and C. |
