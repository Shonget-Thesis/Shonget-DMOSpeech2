# Novel-Algorithm Comparison: Why SNR, Why Drift, Why LTT Against Other Algorithms

**Run:** Colab notebook `colab/DMOSpeech2_Novel_Algorithm_Comparison.ipynb`, Tesla T4, 23–24 October 2026, fast mode (`RUN_C2 = False`)
**Code:** `colab/run_novel_algorithm_comparison.py` at commit `381ec71` (the copy in the results ZIP is byte-identical)
**Protocol:** frozen in `novel_algorithms_preregistration.json` before the analysis ran
**Data:** the same 500 Seed-TTS test-en speakers (development 100 / calibration 200 / test 200), clean and noisy15 prompts, failure labels from `colab/results/gpu_grid_all1000.csv`
**Companion report:** [SIGNAL_SELECTION_VALIDATION_REPORT.md](SIGNAL_SELECTION_VALIDATION_REPORT.md) compared SNR and drift with alternative *signals*. This report compares the whole proposal with alternative *algorithms*.

## Purpose

The signal-selection study showed that SNR is the right gate signal and that no other teacher signal beats drift. This study asks the next question: **could a different algorithm do better than the proposed SNR + drift scheduler?** Three new candidate algorithms, each built on a specific finding from earlier tests, were compared with the proposal under identical rules:
- the same failure rule;
- the same Learn-Then-Test calibration (α = 0.05, δ = 0.10) on the calibration split;
- one look at the held-out test split.

| ID | Algorithm | Idea | Built on which finding |
|---|---|---|---|
| **P0** | **Proposal: SNR gate + drift stopping rule** (λ̂ = 0.50, frozen) | Definition 3.4 | — |
| P1 | Proposal with λ re-certified in this pipeline | Same rule, fresh LTT | — |
| B0 / B1 / B2 | Fixed Step 8 / SNR → Step 4 / SNR → Step 5 | Simple baselines | — |
| **C1a / C1b** | **Student-probe handover** | At step k the student takes its real first step from the teacher's guess; if teacher and student agree, hand over and reuse that step, otherwise keep the teacher. C1a probes gated (clean) prompts only; C1b probes every prompt | Drift measures the teacher, but failures happen in the student |
| **C3** | **Risk-budgeted SNR-band schedule** | One handover step per SNR band (<15, 15–20, 20–30, ≥30 dB), chosen from all 256 schedules by Pareto testing + LTT | Risk is driven by prompt quality, so a finer prompt-quality schedule might save more |
| C2 | Joint handover + restart level | Choose the handover step and the student's restart level | The student re-noises teacher work back to τ = 0.076 |

**C2 was not run in this pass.** It needs new audio synthesis and scoring, about 6 hours on a T4. It remains the one untested candidate (see "What remains").

## Test index

| Section | Test | Purpose | Why it helps | Outcome |
|---|---|---|---|---|
| Validity | Reproduce stored d₄ / d₅ / d₆ | New signals line up with the old failure labels | Without it, no comparison is trustworthy | Max difference 0.00003 |
| 1 | Held-out comparison of all policies | Does any new algorithm beat the proposal? | The direct answer to "why not another algorithm?" | No candidate beats a plain SNR gate |
| 2 | C1 signal quality (development AUC) | Can the student probe find safe handovers? | Shows *why* C1 succeeds or fails | Useless on noisy prompts (AUC 0.43) |
| 3 | C3 band-schedule search | Does a finer SNR schedule save more? | Tests whether one SNR threshold is enough | Best schedule = the existing 20 dB gate |
| 4 | Robustness across CPU and GPU labels | Which policy is safe on both label sets? | Separates "cheapest here" from "reliably safe" | The proposal and SNR → Step 5 are safe on both; SNR → Step 4 is not |

---

## 1. Held-out results (200 test speakers)

> **Test:** Every policy calibrated by LTT on the calibration split (or fixed in advance), then evaluated once on the test split.  
> **Purpose:** Check whether any new algorithm beats the proposed SNR + drift scheduler.  
> **Why it helps:** This is the direct evidence for "why this algorithm and not another one".

| Policy | Certified choice | Test failures | Mean NFE | ≤ 5% | Verdict vs. P0 |
|---|---|---|---|---|---|
| B0 Fixed Step 8 (DMOSpeech 2) | always Step 8 | 0/200 (0.0%) | 21.00 | ✅ | WORSE |
| B1 SNR → Step 4 (SNR-only rule) | fixed rule | 4/200 (2.0%) | 17.04 | ✅ | IMPROVES* |
| B2 SNR → Step 5 | fixed rule | 3/200 (1.5%) | 18.03 | ✅ | MATCHES |
| **P0 Proposal: SNR + drift** | λ̂ = 0.50 | **3/200 (1.5%)** | **18.15** | ✅ | reference |
| P1 Proposal, λ re-certified | λ = 1.20 | 4/200 (2.0%) | 17.04 | ✅ | IMPROVES* |
| **C1a Student probe (gated)** | λ = 0.83 | 4/200 (2.0%) | 17.07 | ✅ | IMPROVES* |
| **C1b Student probe (all prompts)** | λ = 0.21 | 0/200 (0.0%) | **23.97** | ✅ | **WORSE** |
| **C3 SNR-band schedule** | Steps 8 / 8 / 4 / 4 | 4/200 (2.0%) | 17.04 | ✅ | IMPROVES* |

\* **These four "IMPROVES" rows are the same policy: "SNR ≥ 20 dB → Step 4, else Step 8".**
- **P1:** the re-certified λ = 1.20 accepts every gated prompt at Step 4.
- **C1a:** the probe threshold 0.83 accepts nearly every gated prompt at Step 4; the tiny probe cost raises NFE to 17.07.
- **C3:** the best of 256 schedules is exactly the SNR-only rule.
- They beat P0 on NFE only because P0's λ̂ = 0.50 is more conservative. Failures are not significantly different (McNemar vs. P0: 2 vs. 1, p = 1.0).

**Finding:** no new algorithm found a policy better than the simple SNR gate. Every certified candidate either reduced to "SNR ≥ 20 dB → early step" or cost more.

---

## 2. Why the student probe (C1) failed

> **Test:** Development-split AUC of the student-probe disagreement vs. drift, for clean-prompt and noisy-prompt failures.  
> **Purpose:** Check whether the probe can identify handovers that are safe.  
> **Why it helps:** Explains the C1 results instead of just reporting them.

| Signal | AUC, Step-4 failure, gated clean prompts [95% CI] | AUC, Step-5 failure, noisy prompts | AUC, Step-5 failure, all prompts |
|---|---|---|---|
| Drift (proposal) | 0.474 [0.29, 0.67] | 0.410 | 0.162 |
| Student probe (C1) | 0.711 [0.46, 0.91] | 0.432 | 0.229 |

(0.5 = chance.)

- **Noisy prompts are where C1 had to win,** because all remaining savings sit there: about half of noisy prompts are safe at Step 5, but every policy sends them to Step 8. The probe scores **at chance** (0.43). Failing and passing noisy prompts get nearly the same probe value (0.357 vs. 0.366). So LTT could only certify a threshold that rejects almost every probe, and **C1b ended up costing 23.97 NFE, more than DMOSpeech 2's fixed Step 8 (21)**: it paid for probes and still fell back to Step 8.
- **Clean prompts:** the probe looks better than drift (0.71 vs. 0.47), but it rests on only 3 failures and the interval includes 0.5. Clean prompts are already near Step 4, so there is little to gain there. C1a therefore collapsed to SNR → Step 4.
- **Like drift, the probe tracks the prompt condition** (lower on noisy prompts; correlation with drift r = 0.55 on noisy prompts), not whether the output will fail.

**Finding:** asking the student directly does not predict failure any better than drift does. The obstacle is the same for both: clean-prompt failures are seed noise, and noisy-prompt failures are not visible in the trajectory.

---

## 3. Why one SNR threshold is enough (C3)

> **Test:** All 256 schedules assigning a handover step to each SNR band (<15, 15–20, 20–30, ≥30 dB); development Pareto front (29 schedules), LTT on calibration, one look at test.  
> **Purpose:** Check whether a finer prompt-quality schedule saves more than a single gate.  
> **Why it helps:** Tests whether the proposal's single 20 dB threshold leaves savings on the table.

The certified schedule was **Step 8 / Step 8 / Step 4 / Step 4**: below 20 dB use the reference, above 20 dB hand over early. That is **exactly the proposal's single 20 dB gate**.

**Finding:** a finer schedule finds nothing extra. Together with the signal-selection study (LTT independently calibrated the gate to 20.75 dB), this is the **third independent confirmation that 20 dB is the right gate**.

---

## 4. Robustness: cheapest here vs. safe everywhere

> **Test:** The same policies' held-out results on the CPU labels (phase 3) and the GPU labels (this run).  
> **Purpose:** Check which policies stay within the 5% limit on both label sets.  
> **Why it helps:** Separates a policy that happens to be cheapest on one label set from one that is reliably safe.

| Policy | CPU test risk / NFE | GPU test risk / NFE | Safe on both? |
|---|---|---|---|
| SNR → Step 4 (= B1, P1, C1a, C3) | **5.5%** / 17.04 | 2.0% / 17.04 | ❌ Exceeds 5% on CPU labels |
| SNR → Step 5 (B2) | 2.0% / 18.03 | 1.5% / 18.03 | ✅ |
| **Proposal P0 (λ̂ = 0.50)** | **2.5% / 18.15** | **1.5% / 18.15** | ✅ |

- The policy the new algorithms collapsed to (SNR → Step 4) is cheaper by about 1 NFE, but it **broke the 5% guarantee on the CPU labels**. Its calibration count (5/200, p = 0.062) sits right at the edge of certification.
- The proposal, and SNR → Step 5, are safe on both. Their extra margin is what kept them safe.
- So "cheapest on GPU labels" is not the same as "better". **The proposal is the more robust choice.**

---

## What this means for SNR, drift and LTT

| Component | Evidence from this study | Combined with the signal-selection study |
|---|---|---|
| **SNR gate** | Every certified algorithm, including the 256-schedule search, reduced to the 20 dB SNR gate. No algorithm beat it | ✅ **Justified:** SNR matches an oracle gate at the lowest cost; 20 dB confirmed three ways |
| **Drift stopping rule** | No candidate stopping algorithm (student probe, finer SNR bands) beat it; the probe failed exactly where a stopping signal could help (noisy prompts). P0 stays safe on both label sets | ⚠️ **Kept as the proposal's zero-cost stopping signal:** safe and no worse than any alternative, but not shown to add value over a fixed early step |
| **LTT calibration** | LTT refused to certify unsafe settings for every candidate: it forced C1b back to near-reference behavior instead of letting it fail | ✅ **Justified:** the guarantee held for every certified policy on test |

## What remains

- **C2 (joint handover + restart)** is the only candidate not yet tested. It targets the cause identified in this project: the student re-noises the teacher's work back to τ = 0.076. It is the remaining idea that could rescue noisy prompts. It needs `RUN_C2 = True`, about 6 hours on a T4 or an estimated 2–3 hours on an A100, and saves progress to Google Drive.
- **Novelty claims:** C1 and C3 did not improve on the proposal, so neither can serve as a novel-algorithm contribution. The contribution remains the risk-calibrated handover framework plus the empirical findings.

## Caveats

- **Post-hoc:** the test split was inspected in earlier analyses. The verdicts follow preregistered rules, but a winning candidate would still need fresh speakers. None won, so this does not change the conclusions.
- **Small clean-failure counts** (about 2% on GPU labels) give wide development AUC intervals for any stopping signal.
- **Synthetic noise only:** the noisy condition is 15 dB white noise.
- **"IMPROVES" is defined on NFE.** With failures not significantly different, a 1-NFE saving that breaks the guarantee on the other label set is not a real improvement.

## Files

| File | Location |
|---|---|
| Notebook | `colab/DMOSpeech2_Novel_Algorithm_Comparison.ipynb` |
| Runner | `colab/run_novel_algorithm_comparison.py` |
| Raw results (auto-generated report, summary JSON, features CSV, preregistration, run info) | `colab/results/novel_algorithms_comparisons/` |

---

## Summary

We tested whether any alternative could replace the parts of the proposed scheduler: nine prompt signals against SNR, four teacher signals against drift, three calibration frameworks against Learn-Then-Test (signal-selection study), and three new algorithms against the whole SNR + drift policy (student-probe handover with and without the gate, and a 256-schedule SNR-band search). SNR matched an oracle clean/noisy gate at the lowest cost, and every algorithm search, including the band schedule, converged on the same 20 dB SNR gate. LTT was required, because risk is not monotone in the threshold and a Hoeffding test cannot certify anything at 200 speakers. No alternative stopping method beat drift: the student probe failed on noisy prompts and, ungated, cost more than DMOSpeech 2 itself. The only policy that looked cheaper (SNR → Step 4) broke the 5% guarantee on the CPU labels, while the proposal stayed safe on both label sets. We therefore keep the SNR gate, because the evidence shows it is the right and cheapest gate; LTT, because it is the only framework that gives the guarantee at this sample size; and drift as the stopping signal, because it is zero-cost, safe and no alternative tested did better, while reporting honestly that its added value over a fixed early step is not established.
