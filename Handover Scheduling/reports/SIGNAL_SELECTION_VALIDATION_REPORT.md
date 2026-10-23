# Signal-Selection Validation: Why the SNR Gate, Why Drift, Why Learn-Then-Test

**Run:** Colab notebook `colab/DMOSpeech2_Signal_Selection_Validation.ipynb`, Tesla T4, 23 October 2026
**Code:** `colab/run_signal_selection_validation.py` at commit `38ac387` (the copy inside the results ZIP is byte-identical)
**Protocol:** frozen in `signal_selection_preregistration.json` at 10:48 UTC, before the analysis ran
**Data:** the same 500 Seed-TTS test-en speakers (development 100 / calibration 200 / test 200), clean and noisy15 prompts, failure labels from `colab/results/gpu_grid_all1000.csv`

## Purpose

The proposed adaptive scheduler (Definition 3.4) makes three design choices:

| Component | Proposal's choice | Question a panel will ask |
|---|---|---|
| Measurement 1: gate | Energy-percentile prompt SNR, θ = 20 dB | Why SNR and not another prompt-quality measure? |
| Measurement 2: stopping signal | Endpoint drift dᵢ, λ̂ = 0.50 | Why drift and not another teacher signal? |
| Calibration | Learn-Then-Test, exact binomial test | Why LTT and not another framework? |

This study tests each choice against alternatives under one protocol, fixed in advance:
- **Development:** AUC and the direction of each signal.
- **Calibration:** threshold by LTT (α = 0.05, δ = 0.10).
- **Test:** held-out robust risk and mean NFE, looked at once.

## Bottom line

| Question | Verdict | One-line reason |
|---|---|---|
| **Q1. Why SNR?** | ✅ **MATCHED, and equivalent to a perfect gate** | SNR does as well as the true clean/noisy label and is the cheapest signal that does (0.49 ms) |
| **Q2. Why drift?** | ⚠️ **MATCHED by a fixed Step 4** | No teacher signal, drift included, improves on simply handing over at Step 4 after the SNR gate |
| **Q3. Why LTT?** | ✅ **Justified** | Hoeffding cannot certify anything at n = 200; risk is not monotone in λ, which rules out CRC and RCPS; LTT reproduces λ̂ = 0.50 |

**Validity check passed:** the Colab teacher pass reproduced the stored d₄ / d₅ / d₆ with a maximum difference of 0.00003 (correlation 1.0000), so every new signal lines up with the existing failure labels.

---

## Q1. Why a prompt-SNR gate?

Each candidate gate signal was plugged into the same rule, "if the prompt looks safe, hand over early; else Step 8". Its threshold was certified by LTT on calibration and tested once.

| Signal | Kind | Median cost | Dev AUC, Step-5 failure [95% CI] | Separates noisy from clean (AUC) | Gate → Step 4: test failures / NFE | Gate → Step 5: test failures / NFE |
|---|---|---|---|---|---|---|
| **Prompt SNR (proposal)** | Signal processing | **0.49 ms** | **0.815** [0.75, 0.87] | **1.000** | 4/200 / **17.06** | 3/200 / **18.05** |
| Spectral flatness | Signal processing | 2.43 ms | 0.806 [0.75, 0.86] | 1.000 | 4/200 / 17.14 | 3/200 / 18.02 |
| Spectral-percentile SNR | Signal processing | 4.11 ms | 0.806 [0.75, 0.86] | 0.993 | 3/200 / 17.82 | 3/200 / 18.61 |
| Noise floor (dBFS) | Signal processing | 0.35 ms | 0.792 [0.73, 0.85] | 0.976 | 5/200 / 18.56 | 7/200 / 18.59 |
| Prompt UTMOS | Neural network | 38.25 ms | 0.777 [0.72, 0.84] | 0.937 | 4/200 / 17.66 | 3/200 / 18.50 |
| High-frequency ratio | Signal processing | 2.20 ms | 0.667 [0.61, 0.74] | 0.730 | 5/200 / 18.80 | 6/200 / 19.35 |
| Speaking rate | Metadata | ~0 | 0.537 | 0.500 | 3/200 / 20.80 | 4/200 / 20.85 |
| Target text length | Metadata | ~0 | 0.561 | 0.500 | 6/200 / 20.76 | 5/200 / 20.82 |
| Prompt duration | Metadata | ~0 | 0.503 | 0.500 | 0/200 / 21.00 | 0/200 / 21.00 |
| *Oracle (true clean/noisy label)* | — | — | 0.809 | 1.000 | 4/200 / 17.00 | 3/200 / 18.00 |

All policies stayed within α = 5% on test.

**Findings:**
1. **SNR is as good as a perfect gate.** It separates noisy from clean prompts perfectly (AUC 1.000) and lands within 0.06 NFE of the oracle in both templates.
2. **Only one alternative ties it:** spectral flatness (ΔNFE −0.08 / +0.03; McNemar 0 vs. 0, p = 1.0). It costs 5× more compute and has no physical unit.
3. **Every other quality measure is worse or more expensive.**
   - Spectral SNR, the noise floor and prompt UTMOS give up 0.5–1.5 NFE.
   - Prompt UTMOS also needs an extra neural network pass, which makes it 78× slower than SNR.
4. **Metadata signals can't do the job.** Duration, speaking rate and text length don't detect noise (AUC ≈ 0.5), so LTT falls back to about 21 NFE. This shows that **the gate must measure prompt quality**, and any signal will not do.
5. **The 20 dB threshold is confirmed independently.** Calibrated by LTT on its own, the SNR gate settled at **20.75 dB**, close to the proposal's hand-set 20 dB.
6. **Limit of every gate signal:** none predicts which prompts fail *within* a condition (AUC 0.49–0.68 within clean, 0.49–0.59 within noisy). The gate's value is entirely in separating clean from noisy prompts.

**Defensible statement:**
> "Among nine candidate prompt signals, the energy-percentile SNR matched an oracle clean/noisy gate (17.06 vs. 17.00 NFE, equal failures) at the lowest cost of any signal that did (0.49 ms). Only spectral flatness tied it, at five times the cost; metadata signals could not detect degraded prompts. LTT independently calibrated the SNR threshold to 20.75 dB, consistent with the 20 dB chosen on the development split."

---

## Q2. Why drift as the stopping signal?

With the SNR gate fixed at 20 dB, each teacher-trajectory signal was plugged into Definition 3.4's rule: "stop at the first of Steps 4 → 5 → 6 whose score passes λ". λ was certified by LTT on calibration.

| Signal | Dev AUC, Step-4 failure (gated) [95% CI] | Test failures | Mean NFE | Matched-random p |
|---|---|---|---|---|
| **Drift (proposal)** | 0.526 [0.32, 0.71] (direction **reversed**, see below) | 4/200 | 17.04 | 1.000 |
| Cosine change | 0.512 [0.34, 0.74] | 3/200 | 17.05 | 0.029 |
| Per-frame max drift | 0.739 [0.54, 1.00] | 4/200 | 18.39 | 0.815 |
| CFG gap | 0.704 [0.47, 0.90] | 4/200 | 17.50 | 0.788 |
| Velocity norm | 0.735 [0.37, 0.94] | 4/200 | 17.05 | 0.990 |
| *SNR gate → fixed Step 4* | — | 4/200 | 17.04 | — |
| *SNR gate → fixed Step 5* | — | 3/200 | 18.03 | — |
| *SNR gate → fixed Step 6* | — | 4/200 | 19.02 | — |
| *Proposal as frozen (λ̂ = 0.50)* | — | 3/200 | 18.15 | 0.522 |

**Findings:**
1. **Drift adds nothing beyond the gate.**
   - Calibrated by LTT, the drift rule stopped every gated prompt at Step 4, producing exactly the fixed-Step-4 policy (17.04 NFE, same 4 failures, McNemar 0 vs. 0).
   - As frozen in the proposal (λ̂ = 0.50), its step choices were indistinguishable from random (p = 0.52).
2. **No alternative teacher signal does better.** All four were also certified and safe on test, at equal or higher NFE than the fixed step.
   - Cosine change's choices beat random (p = 0.029), but by one failure, and the result is not significant after correcting for testing five signals (Holm-adjusted p ≈ 0.15).
3. **On development, drift pointed the wrong way.** Lower drift went with more Step-4 failures, the opposite of the proposal's assumption that unsettled = risky (raw AUC 0.47). Chapter 3 §5.2 found the same thing (AUC ≈ 0.474).
4. **Why no stopping signal can help on clean prompts:** clean-prompt failures occur at the 2.5% seed-noise floor (Chapter 3 §4.9.1). They are mostly evaluation-model jitter, so there is nothing for any signal to predict. All the dev AUC confidence intervals include 0.5.

**Defensible statement:**
> "Endpoint drift was compared with four other zero-cost teacher-trajectory signals under identical calibration. None improved on handing over at a fixed step after the SNR gate: once the gate removes noisy prompts, clean-prompt failures are at the seed-noise floor and cannot be predicted from the trajectory. The stopping signal is therefore reported as a tested negative result."

---

## Q3. Why Learn-Then-Test for calibration?

**Requirement:** a statement of the form *"the selected threshold adds failures to at most 5% of new inputs, with 90% confidence"*, for a threshold chosen from many candidates.

| Framework | What it guarantees | Fit |
|---|---|---|
| Naive selection (most aggressive threshold with calibration risk ≤ α) | Nothing | ❌ No guarantee; overfits |
| Cross-validation (Appendix D pilot) | No finite-sample guarantee | ❌ With 36 inputs it could not certify 5% |
| Split conformal prediction | Coverage of prediction sets | ❌ Different question |
| Conformal Risk Control | Risk ≤ α on average | ❌ Weaker guarantee; needs risk monotone in λ |
| RCPS | Risk ≤ α with probability 1 − δ | ❌ Needs risk monotone in λ |
| **Learn-Then-Test, exact binomial** | Risk ≤ α with probability 1 − δ for the selected threshold | ✅ No monotonicity needed; exact test for counts |

**Evidence from this run:**

1. **Hoeffding is infeasible at this sample size.** Certifying α = 5% at δ = 10% needs at least 461 calibration speakers with Hoeffding, even with zero failures, against 45 with the exact binomial. In 200 re-splits, Hoeffding certified **no threshold, every time**.
2. **Risk is not monotone in λ, on both label sets.** On calibration, a larger λ gave *fewer* failures at 3 of 20 steps:

   | Labels | Examples |
   |---|---|
   | CPU (original calibration) | λ 0.45 → 0.50: 5 → 4 failures; λ 0.55 → 0.60: 6 → 3; λ 0.90 → 0.95: 4 → 3 |
   | GPU | λ 0.45 → 0.50: 3 → 2; λ 0.65 → 0.70: 5 → 4; λ 0.70 → 0.75: 4 → 3 |

   This breaks the assumption behind CRC and RCPS. LTT does not need it.
3. **Re-split check** (200 random 50/50 re-splits of the 400 calibration + test speakers):

   | Labels | Method | Held-out risk > 5% | Mean held-out risk | Mean NFE | Most common λ |
   |---|---|---|---|---|---|
   | CPU | Naive selection | 3.5% | 3.54% | 17.04 | 1.20 |
   | CPU | Hoeffding | 0.0% | 0.00% | 21.00 | none certified |
   | CPU | **LTT (proposal)** | **3.0%** | **2.52%** | **18.10** | **0.50** |
   | GPU | Naive selection | 0.0% | 2.26% | 17.03 | 1.20 |
   | GPU | Hoeffding | 0.0% | 0.00% | 21.00 | none certified |
   | GPU | **LTT (proposal)** | **0.0%** | **1.97%** | **17.29** | **1.20** |

   - On CPU labels, LTT's most common choice is **λ = 0.50**, the proposal's certified value. It exceeded 5% on 3.0% of held-out halves, within its δ = 10% allowance.
   - Naive selection was slightly worse (3.5%), with a higher mean risk (3.54% vs. 2.52%).
   - On GPU labels failures are rarer, so naive selection and LTT behaved almost the same.
   - **Be honest about this:** the naive-vs-LTT gap is small on this data. The case for LTT rests on its *guarantee*, on Hoeffding's infeasibility and on the non-monotone risk, not on a large empirical gap.

**Is LTT a contribution?** No. LTT, binomial p-values and fixed-sequence testing are existing methods (Angelopoulos et al.; Bauer, 1991) and must be cited as such. The contribution is the **adaptation**:
- casting the handover point as a risk-controlled selection problem;
- defining a reference-relative added-failure risk (Definition 3.1);
- a data design that meets LTT's independence assumption (speaker-disjoint splits, one utterance per speaker);
- showing that the guarantee holds on held-out speakers.

CALM (Schuster et al., 2022) made the same kind of contribution for early exit in language models.

**Defensible statement:**
> "Learn-Then-Test was chosen because the added-failure risk is not monotone in the threshold (observed on both label sets), which rules out Conformal Risk Control and RCPS, and because a Hoeffding-based test cannot certify α = 5% with fewer than 461 calibration inputs. With the exact binomial test, 200 calibration speakers suffice, and repeated re-splits stayed within the δ = 10% allowance."

---

## What this means for the thesis

| Component | Status after this study |
|---|---|
| SNR gate | ✅ **Justified**: as good as an oracle gate, cheapest of the signals that match it, threshold confirmed by LTT (20.75 dB) |
| Drift | ⚠️ **Negative result**: no better than a fixed step or four alternatives; report it honestly as tested and not needed |
| LTT | ✅ **Justified**: required by the non-monotone risk and the sample size |
| Overall method | The efficiency gain over DMOSpeech 2 comes from the **SNR gate + LTT**; drift is an ablation |

## Caveats

- **Post-hoc validation:** the test split had been inspected in earlier analyses. Verdicts use the preregistered rules, but a fully fresh confirmation would need new speakers (e.g., the 166 unused Seed-TTS speakers with noisy15 prompts).
- **Small failure counts:** on GPU labels, clean failures are rare (about 2%), so development AUCs for the stopping signals have wide intervals (all include 0.5).
- **Prompt corruption:** the noisy condition is synthetic 15 dB white noise. Phase 12 showed reverberation degrades early handover too, and no signal (SNR or drift) predicted which reverberant prompts fail.
- **Cost figures** are median CPU times inside the Colab run. The UTMOS time includes GPU synchronization.

## Files

| File | Location |
|---|---|
| Notebook | `colab/DMOSpeech2_Signal_Selection_Validation.ipynb` |
| Runner | `colab/run_signal_selection_validation.py` |
| Raw results (auto-generated report, summary JSON, features CSV, preregistration, run info) | `colab/results/signal_selection/` |
