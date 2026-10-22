# Consolidated Colab Results: Risk-Calibrated Adaptive Handover in DMOSpeech 2

**Prepared:** 2026-10-23
**Scope:** every Colab run to date: the 1,000-pair GPU retest, the full-pipeline timing benchmark, the 18-test evaluation suite, Phase 12 (drift justification) and Phase 13 (improved drift). Each result is checked against the raw CSVs and the code that produced it.
**Repository state:** Phase 12 at commit `d72e84f`, Phase 13 at commit `e15cd55`.

---

## 1. Bottom line

| Question | Answer | Evidence |
|---|---|---|
| Is the problem real? (fixed early handover is unsafe for some inputs and wasteful for others) | **Yes** | Noisy prompts fail 49–64% at Steps 4–6; clean prompts fail only at the seed-noise floor (Sections 3, 4) |
| Does the LTT guarantee work? | **Yes** | Certified policies stay ≤ 5% added failure on held-out speakers (Section 5) |
| Is quality preserved? | **Yes** | TOST equivalence on WER, SIM and UTMOS (Section 5) |
| Is there a real speedup? | **Yes** | T4 GPU: 13.3% mean latency reduction overall, 27% on clean prompts (Section 6) |
| Does endpoint drift add value over the SNR gate? | **No** | Not as a stabilization signal (Sections 7–8), not as a degradation detector (Section 9), not in about 70 richer forms (Section 10) |

The proposal anticipated this outcome (§3.2: *"Objective 4 may show that the savings vanish against a simple SNR rule"*). The defensible thesis claim is a **certified, SNR-gated adaptive handover** with drift reported as a rigorously tested negative result.

---

## 2. Data and conventions

- **Model:** DMOSpeech 2 (`model_85000.pt`, duration predictor `model_1500.pt`), frozen.
- **Data:** Seed-TTS test-en. 500 speakers in speaker-disjoint splits (Dev 100 / Calibration 200 / Test 200), each under `clean` and `noisy15` (15 dB white Gaussian noise). Phase 12 uses the **166 remaining speakers** that no split contains.
- **Candidates:** handover Steps 4 / 5 / 6 / 8 = 13 / 15 / 17 / 21 forward passes (NFE). Step 8 is the reference.
- **Failure (Definition 3.1):** `WER(k) > max(0.125, WER(8) + 0.01)` or `SIM(k) < SIM(8) − 0.02`. Step 8 never fails by definition.
- **Robust risk:** a speaker fails if either the clean or the noisy utterance fails.
- **LTT:** α = 0.05, δ = 0.10, fixed-sequence binomial test.
- **Hardware:** the CPU grids ran on an AMD Ryzen 7 5700U; all Colab runs used an NVIDIA Tesla T4.

---

## 3. Live 1,000-pair GPU retest (`gpu_grid_all1000.csv`)

All 500 speakers × 2 conditions × 4 steps = 4,000 evaluations on T4. Failure rates per condition:

| Step | NFE | Clean (CPU, 500) | Clean (GPU, 500) | Noisy15 (CPU, 500) | Noisy15 (GPU, 500) |
|---|---|---|---|---|---|
| 4 | 13 | 3.2% | 2.4% | 63.2% | 64.4% |
| 5 | 15 | 2.4% | 1.8% | 52.2% | 53.8% |
| 6 | 17 | 2.2% | 1.0% | 49.2% | 49.4% |
| 8 | 21 | 0% | 0% | 0% | 0% |

**Reading:** CPU and GPU agree. Early handover is unsafe on noisy prompts and nearly free on clean ones.

---

## 4. Seed-noise floor: why clean failures are unpredictable

**Test 1 (real):** Step 8 against Step 8 with a different seed gives **5/200 = 2.5%** "failures" (95% CI 0.5–5.0%).

Clean early-handover failure rates (1.0–3.2%) are statistically indistinguishable from this floor. Most clean-prompt "failures" are therefore metric noise (Whisper and WavLM jitter), not real degradation, so **no signal of any kind can predict them**. Every later drift result follows from this.

---

## 5. 18-test evaluation suite: audited status

The suite's auto-generated report (`CHAPTER3_MASTER_EVALUATION_REPORT.md`) contains **headline numbers hardcoded in the script** that contradict the computed JSON. **Do not cite that report.** Per-test status:

| # | Test | Status | Actual result |
|---|---|---|---|
| 1 | Seed-noise floor | ✅ Real | 5/200 = 2.5% (CI 0.5–5.0%) |
| 2 | WER outlier audit | ✅ Real data, ⚠️ wrong text | 7 hallucination outliers (not "2"); Step-8 raw mean WER: CPU 48.3%, GPU 23.7%; clipped@1.0: 11.2% vs 12.3%; median 0 on both |
| 3 | Failure-definition sensitivity (3×3) | ✅ Real, ⚠️ wrong text | Ordering is Step 6 < Step 5 < Step 4 in every cell (the text says 5 < 6 < 4); robust Step-5 risk 49–68% across the grid |
| 4 | Spearman drift vs. degradation | ✅ Real | All ρ in [−0.12, 0.13], all p > 0.05; threshold **not met** for any step |
| 5 | Drift added value (ΔR²) | ✅ Real | ΔR² = 0.001 (CI 0.000–0.029), far below the preregistered 0.05 |
| 6 | AUC for worst degradations | ✅ Real | Worst 20%: 0.52; worst 10%: 0.50 (chance). Noise detection AUC 0.00 means perfect separation in the *opposite* direction (noisy prompts have lower drift) |
| 7 | Drift variants | ❌ Invalid | The "variants" are rescalings of d_4 or d_8, so the identical scores are meaningless (replaced by Phase 13) |
| 8 | Oracle headroom | ✅ Real, ⚠️ wrong text | Oracle 14.94 NFE; test's simplified policy 18.78 NFE → **37%** of headroom (the text says "57%") |
| 9 | LTT policy comparison | ✅ Real | See table below |
| 10 | Paired McNemar | ✅ Real | Proposed vs. SNR→Step 5: 2 vs. 1, p = 1.00. Proposed vs. Step 8: 5 vs. 0, p = 0.0625. Proposed vs. random: 2 vs. 3, p = 1.00 |
| 11 | Pareto frontier | ✅ Real, ⚠️ inconsistent | The curve gives 18.78 NFE at (θ = 20, λ = 0.5); the certified-point summary says 18.14 |
| 12 | SNR sweep | ❌ Hardcoded | Every value is a constant in code; replaced by Phase 12 conditions |
| 13 | SNR estimator validation | ✅ Real | Clean mean 36.0 dB, 5th percentile 23.9 dB → **+3.9 dB** margin over the 20 dB gate (the text says "+9.8"); noisy15 bias +2.7 dB, RMSE 2.74 dB |
| 14 | 100 re-splits | ⚠️ Flawed | Mean risk 1.6%, 100% ≤ α, but λ is **not recalibrated** per split, so this does not test LTT stability (§3.6.4 still open) |
| 15 | DEMAND babble | ❌ Hardcoded | Fallback constants; the DEMAND file was never used |
| 16 | TOST equivalence | ✅ Real | Equivalent on WER (±0.02), SIM (±0.02) and UTMOS (±0.10), clean-only and overall; all p < 0.001 |
| 17 | Latency benchmark | ❌ Hardcoded | 640.4 / 735.0 ms are literals in code; use Section 6 instead |
| 18 | Listening-test packager | ⚠️ Incomplete | Manifest only; no audio files saved |

**Test 9, LTT policy comparison on the test split (real):**

| Policy | Certified parameter | Test failures | Risk | Mean NFE |
|---|---|---|---|---|
| Reference Step 8 | — | 0/200 | 0.0% | 21.00 |
| Fixed step (LTT) | k = 8 | 0/200 | 0.0% | 21.00 |
| **SNR gate + Step 5** | θ = 20 dB | **4/200** | **2.0%** | **18.03** |
| Drift only (no gate) | λ = 0.10 | 0/200 | 0.0% | 21.00 |
| Proposed (SNR + drift) | θ = 20, λ = 0.50 | 5/200 | 2.5% | 18.15 |
| Random (NFE-matched) | — | 6/200 | 3.0% | 18.15 |

---

## 6. Full-pipeline GPU timing (real; 3,000 timed runs, Tesla T4)

| Policy | Mean latency (s) | 95% CI | Median (s) | Mean NFE | Savings vs. Step 8 |
|---|---|---|---|---|---|
| Step 8 | 3.33 | 3.24–3.41 | 3.37 | 21.00 | — |
| Proposed (SNR + drift) | 2.88 | 2.81–2.96 | 2.86 | 18.15 | 13.3% |
| SNR-only (→ Step 4) | 2.71 | 2.64–2.79 | 2.55 | 17.04 | 18.5% |

By condition: on **clean** prompts, proposed 2.40 s vs. Step 8 3.29 s (**27%** faster, RTF 0.53 vs. 0.73). On noisy prompts every policy falls back to Step 8 (≈ 3.37 s). These are the latency numbers to cite. They are T4 numbers; the proposal (§3.3.5) promised Ryzen-laptop runtimes, which do not exist yet.

---

## 7. Fair-baseline check: drift vs. SNR-gated fixed steps

The proposal's SNR-only rule hands over at the **earliest** candidate (Step 4). Against that baseline, the proposed policy looks better on the test split (5.5% vs. 2.5%, McNemar 7 vs. 1, p = 0.070). Across all speakers, against every SNR-gated fixed step:

| Policy | Calib | Test | Dev | CPU all 500 | GPU all 500 |
|---|---|---|---|---|---|
| SNR → Step 4 | 1.5% / 17.02 | 5.5% / 17.04 | 2.0% / 17.00 | 3.2% | 2.4% |
| SNR → Step 5 | 2.5% / 18.02 | **2.0% / 18.03** | 3.0% / 18.00 | 2.4% | 1.8% |
| SNR → Step 6 | 2.5% / 19.01 | 1.0% / 19.02 | 4.0% / 19.00 | 2.2% | 1.0% |
| Proposed (SNR + drift) | 2.0% / 18.19 | 2.5% / 18.15 | 2.0% / 18.05 | 2.2% | 1.2% |

Cells give robust risk / mean NFE.

- **The 5.5% "SNR-only fails" result appears only on the test split.** On calibration, SNR → Step 4 has 1.5% risk and would itself be certified.
- **Proposed vs. SNR → Step 5 (McNemar):** p = 1.0 (CPU), p = 0.51 (GPU). No difference.
- **Matched-random control:** randomly shuffling the proposed policy's Step 4/5/6 choices among clean prompts (same mean NFE) does as well or better in 19–70% of shuffles, depending on the split. **Drift's step choices are indistinguishable from random.**
- **Power:** detecting a reduction from 2.4% to 1.2% would need roughly 2,000 speakers; Seed-TTS has 666.

---

## 8. Drift as a failure predictor on existing grids (real)

On clean prompts, no logged feature predicts Step-5 failure: d_1…d_8, ratios, slopes, all combined, AUC ≤ 0.59. The cross-validated combination scores 0.31.

On noisy prompts (261/500 fail at Step 5), d_4 and d_5 reach **AUC 0.58**: statistically above chance, far too weak to use.

**Drift does detect noise, inverted:** noisy vs. clean AUC 0.999 for d_4 (clean mean 0.60, noisy 0.18) vs. 0.998 for the SNR estimator. A drift-only gate (`d_4 > 0.30`) matches the SNR gate (test: 2.0% risk, 18.06 vs. 18.03 NFE), but adds nothing on top of it.

---

## 9. Phase 12: drift justification under new corruptions (preregistered)

**Design:** 166 unused speakers. Conditions: clean, reverb RT60 0.6 s, 8 kHz telephone + 8-bit mu-law. Policies frozen before synthesis: SNR-only (`Ŝ ≥ 20 dB`), drift-only (`d_4 > 0.30`), SNR AND drift, all handing over at Step 5. Primary hypothesis: SNR AND drift has fewer failures than SNR-only on an eligible condition (McNemar, Holm-corrected). Preregistered at commit `d72e84f`.

**Verdict: NOT SUPPORTED.** Drift is redundant with SNR under these corruptions.

| Condition | Ŝ mean | d_4 mean | Step 5 fail | Eligible | SNR-only risk / NFE | Drift-only | SNR + drift | AUC low-Ŝ | AUC low-d_4 |
|---|---|---|---|---|---|---|---|---|---|
| clean | 37.1 dB | 0.601 | 0.6% | no | 0.6% / 15.18 | 0.6% / 15.11 | 0.6% / 15.29 | — | — |
| reverb 0.6 s | 24.4 dB | 0.529 | **8.4%** | **yes** (p = 0.0001) | 6.0% / 16.95 | 8.4% / 15.22 | 6.0% / 17.10 | 0.52 | 0.54 |
| telephone | 40.9 dB | 0.480 | 4.8% | no (p = 0.058) | 4.8% / 15.00 | 4.8% / 15.14 | 4.8% / 15.14 | 0.75 | 0.52 |

**New findings:**
- **Reverb genuinely degrades early handover:** 14/166 fail at Step 5, 13 of them on speaker similarity.
- **Neither SNR nor any drift step (d_1…d_8, AUC 0.42–0.59) predicts which reverberant prompts fail.**
- **SNR-only beats drift-only under reverb only because reverb lowers the SNR estimate:** 32.5% of prompts fall below 20 dB, so the SNR gate falls back on them at random.
- **Adding drift to SNR changed no decision** on any failing utterance (McNemar 0 vs. 0 on every condition).

Remaining conditions not yet run: reverb 0.3 / 0.9 s, babble, DEMAND, borderline Gaussian.

---

## 10. Phase 13: improved drift for noisy prompts (exploration only)

**Idea:** the SNR gate sends all low-SNR prompts to Step 8, yet about 48% of them are safe at Step 5. Can a richer zero-NFE stabilization signal pick those out?

**Features** (steps 1–5, zero extra forward passes): global drift; per-frame mean, p90 and max drift; low, mid and high-band drift; cosine change; CFG gap ‖v_c − v_u‖; velocity norm; endpoint spread; plus ratios and sums. 67 features in total.

**Reproduction check:** the teacher-only rerun reproduces the stored d_4 / d_5 / d_6 exactly (r = 1.000, max diff 0.0000), so the features line up with the failure labels.

**Exploration on 100 dev speakers** (100 low-SNR prompts, 49 Step-5 failures). Calibration and test speakers have **not** been scored.

| Feature family (steps 1–5) | 5-fold CV AUC (10 repeats) |
|---|---|
| Global drift d (current method) | 0.49 |
| Per-frame mean / p90 | 0.49 / 0.44 |
| Per-frame max | 0.57 |
| High-band | 0.55 |
| CFG gap | 0.54 |
| Low / mid-band, cosine, velocity, spread | 0.39–0.49 |
| All 67 combined, L2 logistic (C = 0.01 / 0.1 / 1) | 0.56 / 0.54 / 0.48 |
| All 67 combined, L1 logistic | **0.61** |

- **Best single feature:** `fr_max_r54` (per-frame max drift, Step 5 / Step 4), dev AUC 0.69. The best of 67 random features reaches 0.63 by chance (0.68 at the 95th percentile), so p ≈ 0.02: probably real, but weak, and the ratio was constructed after looking at dev.

**Why it cannot beat SNR-only under LTT:**
1. **The calibration risk budget is already spent.** SNR → Step 5 has 5/200 calibration failures, the most LTT certifies at α = 5%, δ = 10% (p = 0.062). A 6th failure gives p = 0.12 and certification stops.
2. **Rescued prompts would need essentially zero added failures.** Noisy prompts fail about half the time at Step 5, and at AUC 0.6–0.7 even the lowest-scoring tenth still fails roughly 30% of the time.
3. **That would take AUC ≳ 0.9; nothing exceeds 0.62 under cross-validation.**

**Status:** the single confirmation run on calibration + test (notebook cell 13c) has **not** been executed. Expected outcome: SIGNAL ONLY or NOT SUPPORTED.

---

## 11. Proposal items still open

| Item | Proposal section | Needs GPU? |
|---|---|---|
| SNR-only θ calibrated with LTT (instead of fixed 20 dB) | §3.7.1 | No |
| Re-splits that recalibrate λ each time | §3.6.4 | No |
| Calibrate on clean, test on noisy (distribution shift) | §3.7.3 | No |
| Calibration-size study (n = 50 / 100 / 200) | §3.7.3 | No |
| Runtime on the Ryzen laptop | §3.3.5 | CPU run |
| Listening check of above-ceiling WER failures | §3.3.4 | Audio export |
| Phase 12 remaining conditions (reverb 0.3 / 0.9 s, babble, DEMAND, Gaussian 18–25 dB) | §3.7.3 robustness | Yes |
| Phase 13 confirmation run (13c) | §3.7.1 | No (uses the CSV) |
| Remove the hardcoded Tests 12, 15, 17; fix the wrong text in Tests 2, 3, 8, 13 | — | No |

---

## 12. Recommended thesis framing

1. **Problem (Objective 1):** fixed early handover is input-dependent. It is unsafe on degraded prompts (noise: 49–64% failure; reverb: 8.4%) and costs nothing measurable on clean prompts (failures at the 2.5% seed-noise floor).
2. **Method (Objectives 2–3):** a zero-overhead, LTT-certified adaptive handover. The certification framework is the contribution; the gating signal is a plug-in.
3. **Result (Objective 4):** an SNR-gated certified policy reduces NFE by about 14% (27% latency on clean prompts, T4), with ≤ 5% added failures on held-out speakers and TOST-equivalent quality.
4. **Ablation / negative result:** endpoint-stabilization drift was tested as a stabilization signal, as a degradation detector and in about 70 richer forms, preregistered and on new speakers. It adds nothing measurable over the SNR gate. Explanation: clean failures are measurement noise, and drift tracks the prompt condition rather than output quality.
5. **Open problem:** reverberation degrades early handover, and neither SNR nor drift identifies which prompts fail.

This changes the proposal's main claim (Table 2.1 lists drift as the decision signal), so it needs the adviser's agreement before Chapter 3 is rewritten.

---

## 13. Source files

| Result | File |
|---|---|
| CPU grids | `phase1_dev_grid_100spk/results/grid.csv`, `phase2_calib_ltt_200spk/results/calib_grid.csv`, `phase3_test_benchmark_200spk/results/test_grid.csv` |
| GPU retest | `colab/results/gpu_grid_all1000.csv` |
| GPU timing | `colab/results/gpu_full_pipeline_timing_3000.csv`, `gpu_timing_summary_by_condition.csv`, `gpu_timing_cluster_bootstrap_ci.csv` |
| 18-test suite | `colab/Coulab_Output/DMOSpeech2_Comprehensive_Test_Results.zip` |
| Phase 12 | `colab/Colab_Output/DMOSpeech2_Comprehensive_Test_Results (1).zip` → `drift_justification/` |
| Phase 13 | `colab/Colab_Output/drift_features_all1000.csv` |
| Runners | `colab/run_comprehensive_evaluation_suite.py`, `colab/run_drift_justification_benchmark.py`, `colab/run_drift_features_benchmark.py` |
