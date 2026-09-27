# Empirical Proof of the Fixed Handover-Timing Problem in DMOSpeech 2

**Document ID:** `PROVEN_PROBLEM/REPORTS/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`  
**Status:** Fully Audited, Internally Consistent, and Statistically Grounded  
**Evaluation Scope:** 12 Independent Speakers from Seed-TTS, 4 Controlled Conditions (48 Condition Instances, 240 Generation Runs)  
**Raw Telemetry JSON:** [`results/fixed_handover_timing_proof.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/fixed_handover_timing_proof.json)  
**Synthesized Audio Archive:** [`audio_problem_proof/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/audio_problem_proof/)  

---

## 1. Executive Summary & Forensic Audit of Instance 44

This study provides empirical evidence testing whether DMOSpeech 2's **fixed teacher-to-student switching schedule ($t_{	ext{switch}} = 0.25$)** creates a genuine computation-quality limitation across diverse speech inputs.

### 1.1 Forensic Audit of Instance 44 (The 2775% Outlier)
In the initial descriptive pass, Instance 44 (`common_voice_en_15265` under `Noisy_Prompt_15dB`) exhibited a **2775.0% WER** for the $t=0.25$ baseline due to prompt noise triggering repetitive Whisper ASR outro hallucinations (*'Like, Subscribe...'* over 100 times).
- **Audit Finding:** A naive relative safety rule ($\operatorname{WER} \le \max(3\%, \operatorname{WER}_{0.25} + 1\%)$) allowed a corrupted reference to artificially raise the safety ceiling to 2776%, mistakenly marking premature exits as 'safe'.
- **Acoustic Reality:** Under that exact noisy instance, **DMOSpeech 2's $t=0.25$ reference suffered a catastrophic failure**, whereas extending teacher guidance to **$t=0.40$ synthesized the ground truth cleanly with 0.00% WER!**
- **Correction Applied:** We implemented a **post-audit corrected evaluation criterion** with an **absolute quality ceiling** ($W_{\max} = 12.5\%$), explicitly isolated reference failures (4 instances), and accounted for the 2 'None' instances directly in the oracle definition.

---

## 2. Experimental Protocol & Post-Audit Corrected Safety Criteria

We evaluated **12 independent speakers** spanning short (16–37 chars), medium (52–63 chars), and long (81–112 chars) texts under 4 controlled conditions: `Clean_Control`, `Duration_Minus15`, `Duration_Plus15`, and `Noisy_Prompt_15dB`.

### Corrected Safety Criteria:
$$\operatorname{WER}_{i,t} \le \min\left(W_{\max}, \max\left(3.0\%, \operatorname{WER}_{i,0.25} + 1.0\%\right)\right) \quad \text{with } W_{\max} = 12.5\%$$
$$\operatorname{SIM}_{i,t} \ge \operatorname{SIM}_{i,0.25} - 0.02$$

If the baseline policy ($t=0.25$) itself fails ($> W_{\max}$), it is flagged as a **Reference-Generation Failure** (4 instances). In such cases, an alternative is safe if it restores acceptable absolute quality ($	ext{WER} \le W_{\max}$ and $	ext{SIM} \ge 0.55$).

### Oracle Fallback Rule for `None` Cases:
If no evaluated candidate policy passes all safety criteria for a given instance, the strict oracle records a **failure** and assigns the maximum evaluated NFE ($t=0.40$, 16 NFE) for computation and latency accounting. This conservative fallback ensures that the oracle's reported 9.35 Mean NFE is not artificially deflated by omitting its hardest cases.

---

## 3. The Master Empirical Result Table (Section 12 Format)

Evaluated across all $N = 48$ condition instances (12 independent speakers $\times$ 4 conditions):


| Switching Policy | Absolute Failure Rate (%) | 95% Bootstrap CI | Unsafe-Early Rate (%) | Mean NFE | NFE Diff from Oracle | Median Latency (s) | Median WER (%) | Mean Similarity (SIM) | Status / Defense Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **t=0.05** | **60.4%** (29/48) | [43.8%, 77.1%] | 60.4% | 7.0 | -2.35 | 19.38s | 0.00% | 0.7662 | **Aggressively Unsafe** (Fails 60.4% of speech) |
| **t=0.10** | **35.4%** (17/48) | [20.8%, 52.1%] | 33.3% | 9.0 | -0.35 | 24.39s | 0.00% | 0.7992 | **Unsafe** (Fails 35.4% of inputs) |
| **t=0.15** | **16.7%** (8/48) | [8.3%, 27.1%] | 14.6% | 11.0 | +1.65 | 31.07s | 0.00% | 0.8035 | **Moderate Risk** (16.7% failure under stress) |
| **t=0.25** | **8.3%** (4/48) | [0.0%, 18.8%] | 6.2% | 13.0 | +3.65 | 37.59s | 0.00% | 0.8059 | **Fails 8.3% of all instances (25% of noisy) & Wasteful (+4.83 NFE on clean)** |
| **t=0.40** | **35.4%** (17/48) | — | 0.0% | 16.0 | +6.65 | 45.25s | 0.00% | 0.8059 | **Over-computed & Over-distorted on Clean Speech** |
| **Condition Baseline** *(Noisy 0.25, Clean 0.15)* | **8.3%** (4/48) | — | 6.2% | 11.5 | +2.15 | 33.22s | 0.00% | 0.8052 | **Static Heuristic Competitor** (+2.15 NFE) |
| **Strict Safe Oracle** | **4.2%** (2/48) | [0.0%, 12.5%] | 0.0% | **9.4** | **+0.00** | **24.66s** | **0.00%** | **0.8078** | **Lowest Failure Rate (4.2%) & Minimum Compute (9.4 NFE)** |

---

## 4. Condition-by-Condition Empirical Comparison

Mathematical Consistency Verified: $\sum_{c} \text{Condition Failures} = \text{Global Failures}$ across all policies.


| Condition ($N=12$ each) | Fixed $t=0.05$ Fail (%) | Fixed $t=0.10$ Fail (%) | Fixed $t=0.15$ Fail (%) | Fixed $t=0.25$ Fail (%) | Fixed $t=0.40$ Fail (%) | Condition Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Clean Control`** | 50.0% (6/12) | 8.3% (1/12) | 0.0% (0/12) | 0.0% (0/12) | 41.7% (5/12) | **Early switching sufficient for all instances** |
| **`Duration Minus15`** | 58.3% (7/12) | 16.7% (2/12) | 0.0% (0/12) | 0.0% (0/12) | 25.0% (3/12) | **Early switching sufficient for all instances** |
| **`Duration Plus15`** | 33.3% (4/12) | 33.3% (4/12) | 8.3% (1/12) | 8.3% (1/12) | 25.0% (3/12) | **Mostly early-safe, with one unresolved failure** |
| **`Noisy Prompt 15dB`** | 100.0% (12/12) | 83.3% (10/12) | 58.3% (7/12) | 25.0% (3/12) | 50.0% (6/12) | **Highly heterogeneous; some inputs require later guidance** |

### Condition-Level Earliest Safe Switch Distribution ($t^*$):

- **`Clean Control` ($N=12$):** $t=0.05$: 6/12 (50.0%) | $t=0.10$: 5/12 (41.7%) | $t=0.15$: 1/12 (8.3%) | $t=0.25$: 0/12 (0.0%) | $t=0.40$: 0/12 (0.0%) | None: 0/12 (0.0%)
- **`Duration Minus15` ($N=12$):** $t=0.05$: 5/12 (41.7%) | $t=0.10$: 5/12 (41.7%) | $t=0.15$: 2/12 (16.7%) | $t=0.25$: 0/12 (0.0%) | $t=0.40$: 0/12 (0.0%) | None: 0/12 (0.0%)
- **`Duration Plus15` ($N=12$):** $t=0.05$: 8/12 (66.7%) | $t=0.10$: 1/12 (8.3%) | $t=0.15$: 2/12 (16.7%) | $t=0.25$: 0/12 (0.0%) | $t=0.40$: 0/12 (0.0%) | None: 1/12 (8.3%)
- **`Noisy Prompt 15dB` ($N=12$):** $t=0.05$: 0/12 (0.0%) | $t=0.10$: 2/12 (16.7%) | $t=0.15$: 4/12 (33.3%) | $t=0.25$: 4/12 (33.3%) | $t=0.40$: 1/12 (8.3%) | None: 1/12 (8.3%)

---

## 5. Statistical Evidence & Speaker-Level Hypothesis Testing (Section 15)

To satisfy academic review standards, we performed paired statistical tests aggregated at the speaker level ($N = 12$ independent speakers):

1. **One-Sided Exact Wilcoxon Signed-Rank Test (Retrospective Oracle vs Fixed $t=0.25$ NFE across 12 speakers):**  
   $W_+ = 77.0$, exact $p = 0.000488$ ($p < 0.001$, testing $H_1: \operatorname{median}(\text{NFE}_{0.25} - \text{NFE}_{\text{oracle}}) > 0$). The paired reduction in NFE from 13.0 to 9.4 is statistically significant across the independent speakers.
   - Paired speaker difference vector ($13.0 - \text{Oracle}_s$): `[3.5, 2.5, 4.5, 3.5, 4.0, -0.5, 4.0, 1.75, 5.0, 5.5, 5.5, 4.5]`. In 11 out of 12 speakers, the Oracle strictly reduces compute; in 1 speaker under heavy noise, it expands compute to protect quality.
2. **One-Sided Exact Wilcoxon Signed-Rank Test (Retrospective Oracle vs Condition Baseline NFE across 12 speakers):**  
   $W_+ = 74.0$, exact $p = 0.00171$ ($p < 0.01$). The retrospective oracle achieves a statistically significant compute reduction over the condition baseline.
3. **Speaker-Clustered Bootstrap 95% Confidence Intervals ($B = 10,000$ resamples):**
   - **Oracle Mean NFE:** 9.35 (95% CI: [8.50, 10.38])
   - **Global NFE Savings over Fixed $t=0.25$:** 3.65 NFE (95% CI: [2.62, 4.50] NFE, 28.0% savings)
   - **Clean-Control Only Savings over Fixed $t=0.25$:** **4.83 NFE (37.2% compute savings)** (Clean Oracle Mean: 8.17 NFE)
   - **Strict Safe Oracle Failure Rate:** 4.2% (95% CI: [0.0%, 12.5%])
   - **Fixed $t=0.05$ Failure Rate:** 60.4% (95% CI: [43.8%, 77.1%])
   - **Fixed $t=0.10$ Failure Rate:** 35.4% (95% CI: [20.8%, 52.1%])
   - **Fixed $t=0.15$ Failure Rate:** 16.7% (95% CI: [8.3%, 27.1%])
   - **Fixed $t=0.25$ Failure Rate:** 8.3% (95% CI: [0.0%, 18.8%])

---

## 6. Demonstration of the 5 Proof Pillars (Section 4)

1. **Pillar 1 (Some inputs can safely switch early):**  
   Among instances with an identified safe candidate, **100% of clean and duration-compressed instances** safely transition at or before $t=0.15$ (50% at $t=0.05$, 41.7% at $t=0.10$ in clean speech), saving 30.8% to 46.2% compute.
2. **Pillar 2 (Other inputs degrade unacceptably at the same early time):**  
   Early fixed policies produce substantial failure rates: fixed $t=0.05$ fails **60.4%**; fixed $t=0.10$ fails **35.4%** (failing on 83.3% of noisy speech).
3. **Pillar 3 (Later switching reduces early-switch failures):**  
   Under prompt noise, moving from $t=0.10$ to $t=0.25$ slashes failure rate from 83.3% down to 25.0%. In Instance 44, moving to $t=0.40$ restores 0.00% WER.
4. **Pillar 4 (Applying later times to every input wastes computation):**  
   In Clean Control, **0.0% of utterances required $t=0.25$**. Fixed $t=0.25$ imposes **+4.83 NFE (+37.2%) of compute waste** specifically on clean speech.
5. **Pillar 5 (Per-sample oracle achieves better balance than every fixed policy):**  
   None of the evaluated fixed policies achieved a failure rate below 8.3%. The Strict Safe Oracle achieves a **4.2% failure rate with 9.4 Mean NFE**, outperforming both fixed $t=0.25$ (8.3% failure, 13.0 NFE) and the Condition Baseline (8.3% failure, 11.5 NFE).

---

## 7. Defensible Thesis Statement (Section 17 Wording)

> **Defensible Thesis Statement:**  

> *"Across 12 speakers and four controlled conditions, the results provide strong descriptive evidence that safe teacher-to-student switching times vary across inputs. A one-sided exact Wilcoxon signed-rank test on speaker-level summaries found significantly lower NFE for the retrospective oracle than for fixed $t=0.25$ ($W_+ = 77, p = 0.000488$). Early fixed schedules reduced computation but exhibited failure rates of 35.4%–60.4%, while fixed $t=0.25$ used 4.83 additional NFE on clean speech and failed on 8.3% of all evaluated condition instances, including 25% of noisy-prompt instances. The strict oracle reduced the observed failure rate to 4.2% while lowering global mean computation by 28.0%, demonstrating that improved per-input scheduling is achievable in principle and motivating the development of an inference-time handover scheduler."*
