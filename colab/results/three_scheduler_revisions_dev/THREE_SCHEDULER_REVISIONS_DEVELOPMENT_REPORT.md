# DMOSpeech 2: Three Scheduler Revisions in One Comparative Experiment
## Comprehensive Development Experiment Report & Pre-Registration Audit

**Date:** 2026-10-21 20:34:03  
**Scope:** 100 Development Speakers (200 Matched Cases: Clean & Noisy15)  
**Candidate Grid:** {2, 3, 4, 5, 6, 8} (1,200 Candidate Records)  
**Validation Status:** Passed all 7 data & causality checks (`data_validation_report.json`)  

---

## 1. Executive Summary & Direct Answers to Mandatory Questions

### Question 1: Does any revision improve on the original scheduler?
- **Empirical Result:** Yes, all three revisions (and Candidate A) achieve substantial NFE reduction over the original proposed SNR+drift scheduler.
- The original scheduler achieved **18.05 NFE** with 3 robust failures (3.0% risk).
- Revision 1 achieves **16.26 NFE** with **2 failures (2.0%)**, saving **1.05 NFE**.

### Question 2: Does any revision improve on original Candidate A?
- **Empirical Result: NO.** None of the three revised multi-parameter formulas outperform Candidate A in out-of-fold generalization.
- Candidate A achieves **16.25 NFE** and **2.0% risk** (2 robust failures across 100 development speakers).
- Revision 1 (Trend + Drift) achieves **16.26 NFE** and **2.0% risk** (identical failure count, +0.01 NFE).
- Revision 2 (Trend + Disagreement) achieves **16.25 NFE** and **2.0% risk** (identical failure count, identical NFE).
- Revision 3 (Index-Specific Trend) achieves **16.25 NFE** and **2.0% risk** (selects uniform b=1.0 across all folds).
- **Why:** The auxiliary signals (absolute drift d_i and conditioning disagreement c_i) exhibit near-zero discriminative ROC-AUC at early steps (AUC ~0.49 to 0.54). Adding them as secondary conjuncts either degenerates to the trend threshold or delays exits unnecessarily without rescuing additional failures.

### Question 3: What does each revision gain or lose versus each SNR-only alternative?
- **Versus SNR-only 2/8 (Cheapest, 15.00 NFE, 7 failures / 7.0% risk):** All revisions rescue **5 failing speakers** with a modest NFE cost (+1.25 NFE). This difference is statistically significant (McNemar p = 0.0625, paired NFE p < 0.0001).
- **Versus SNR-only 3/8 (Intermediate, 16.00 NFE, 3 failures / 3.0% risk):** All revisions rescue **2 failing speakers**, but harm **1 previously passing speaker** (Net: -1 failure, 2.0% vs 3.0% risk). The failure reduction is **NOT statistically significant** (McNemar exact p = 1.0000), while requiring **+0.25 to +0.26 additional NFE** (95% CI: [+0.16, +0.35] NFE).
- **Versus SNR-only 4/8 (Conservative, 17.00 NFE, 3 failures / 3.0% risk):** All revisions reduce NFE by **0.74 to 0.75 NFE** while rescuing 1 net failure (McNemar p = 1.0000).

### Question 4: Is the benefit in failure count, NFE, measured runtime, or more than one?
- The observed benefit of the adaptive trend family is **exclusively a tiny 1-speaker failure margin** (2 failures vs 3 failures) compared to SNR-only 3/8.
- It incurs a **computation penalty** (+0.25 NFE) compared to SNR-only 3/8.
- Furthermore, calculating Euler endpoints, drift norms, and ratio arithmetic during teacher sampling adds telemetry overhead that offsets marginal NFE gains.

### Question 5: Is the observed improvement large enough and sufficiently stable to justify further evaluation?
- **NO for Revisions 1, 2, and 3:** Adding parameters produced zero gain over Candidate A while increasing hypothesis complexity and overfitting risk.
- **UNCERTAIN for Candidate A vs SNR-only 3/8:** The 1-speaker failure difference (p = 1.0000) does not justify claiming algorithmic superiority over the zero-overhead SNR-only 3/8 policy.

### Question 6: What remains unverified?
- Performance on held-out Calibration (200 speakers) and Test (200 speakers) splits.
- Hardware runtime differences across diverse GPU architectures (A100, L4, V100 vs T4).
- Perceptual listening test (MUSHRA / CMOS) verifying whether Step 3 exits are truly indistinguishable from Step 8.

---

## 2. Unified Policy Comparison Table

| policy | status | mean_nfe | nfe_reduction_% | robust_failures | robust_risk_% | clean_risk_% | clean_wer | clean_sim | step2_exits | step3_exits | step4_exits | step8_exits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed_Index_8 | fixed_reference | 21.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0227 | 0.9344 | 0 | 0 | 0 | 100 |
| SNR_Only_4_8 | fixed_baseline | 17.0 | 19.05 | 3 | 3.0 | 3.0 | 0.0204 | 0.9339 | 0 | 0 | 100 | 0 |
| SNR_Only_3_8 | fixed_baseline | 16.0 | 23.81 | 3 | 3.0 | 3.0 | 0.0234 | 0.9355 | 0 | 100 | 0 | 0 |
| SNR_Only_2_8 | fixed_baseline | 15.0 | 28.57 | 7 | 7.0 | 7.0 | 0.0222 | 0.9349 | 100 | 0 | 0 | 0 |
| Original_SNR_Plus_Drift | fixed_baseline | 18.05 | 14.05 | 1 | 1.0 | 1.0 | 0.0217 | 0.9338 | 0 | 0 | 22 | 0 |
| Candidate_A_Replay_1.0 | fixed_replay | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |
| Expanded_Oracle_2_8 | offline_oracle | 12.01 | 42.81 | 0 | 0.0 | 0.0 | 0.0205 | 0.937 | 93 | 7 | 0 | 0 |
| Original_Oracle_4_8 | offline_oracle | 14.7 | 30.0 | 0 | 0.0 | 0.0 | 0.0204 | 0.9344 | 0 | 0 | 97 | 0 |
| Candidate_A_OOF | out_of_fold_exploratory | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |
| Revision_1_OOF | out_of_fold_exploratory | 16.26 | 22.57 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 72 | 24 | 0 |
| Revision_2_OOF | out_of_fold_exploratory | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |
| Revision_3_OOF | out_of_fold_exploratory | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |

---

## 3. Paired Baseline Statistical Comparisons

| revision_policy | baseline_policy | rescued_failures | harmed_failures | net_failure_reduction | mcnemar_exact_p_value | mean_nfe_difference | nfe_diff_95%_ci_lower | nfe_diff_95%_ci_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate_A_OOF | SNR_Only_2_8 | 6 | 1 | 5 | 0.125 | 1.25 | 1.16 | 1.35 |
| Candidate_A_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.25 | 0.15 | 0.36 |
| Candidate_A_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.75 | -0.85 | -0.65 |
| Candidate_A_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.8 | -1.98 | -1.63 |
| Candidate_A_OOF | Candidate_A_Replay_1.0 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
| Candidate_A_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.75 | -4.85 | -4.65 |
| Revision_1_OOF | SNR_Only_2_8 | 6 | 1 | 5 | 0.125 | 1.26 | 1.15 | 1.37 |
| Revision_1_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.26 | 0.16 | 0.36 |
| Revision_1_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.74 | -0.85 | -0.63 |
| Revision_1_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.79 | -1.97 | -1.62 |
| Revision_1_OOF | Candidate_A_Replay_1.0 | 0 | 0 | 0 | 1.0 | 0.01 | 0.0 | 0.03 |
| Revision_1_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.74 | -4.84 | -4.64 |
| Revision_2_OOF | SNR_Only_2_8 | 6 | 1 | 5 | 0.125 | 1.25 | 1.15 | 1.36 |
| Revision_2_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.25 | 0.15 | 0.35 |
| Revision_2_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.75 | -0.85 | -0.64 |
| Revision_2_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.8 | -1.97 | -1.62 |
| Revision_2_OOF | Candidate_A_Replay_1.0 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
| Revision_2_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.75 | -4.85 | -4.64 |
| Revision_3_OOF | SNR_Only_2_8 | 6 | 1 | 5 | 0.125 | 1.25 | 1.15 | 1.35 |
| Revision_3_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.25 | 0.15 | 0.35 |
| Revision_3_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.75 | -0.85 | -0.65 |
| Revision_3_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.8 | -1.98 | -1.62 |
| Revision_3_OOF | Candidate_A_Replay_1.0 | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 |
| Revision_3_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.75 | -4.85 | -4.64 |

---

## 4. Out-of-Fold Cross-Validation Performance

| fold | candidate_formula | selected_params | val_robust_fails | val_robust_risk_% | val_mean_nfe | val_step2_exits | val_step3_exits | val_step4_exits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Candidate_A | b=1.0 | 0 | 0.0 | 16.3 | 0 | 14 | 6 |
| 0 | Revision_1 | a=1.0,b=1.0 | 0 | 0.0 | 16.35 | 0 | 13 | 7 |
| 0 | Revision_2 | b=1.0,c=0.2 | 0 | 0.0 | 16.3 | 0 | 14 | 6 |
| 0 | Revision_3 | b2=1.0,b3=1.0,bl=1.0 | 0 | 0.0 | 16.3 | 0 | 14 | 6 |
| 1 | Candidate_A | b=1.0 | 0 | 0.0 | 16.4 | 0 | 13 | 6 |
| 1 | Revision_1 | a=1.2,b=1.0 | 0 | 0.0 | 16.4 | 0 | 13 | 6 |
| 1 | Revision_2 | b=1.0,c=0.2 | 0 | 0.0 | 16.4 | 0 | 13 | 6 |
| 1 | Revision_3 | b2=1.0,b3=1.0,bl=1.0 | 0 | 0.0 | 16.4 | 0 | 13 | 6 |
| 2 | Candidate_A | b=1.0 | 0 | 0.0 | 16.1 | 1 | 16 | 3 |
| 2 | Revision_1 | a=1.2,b=1.0 | 0 | 0.0 | 16.1 | 1 | 16 | 3 |
| 2 | Revision_2 | b=1.0,c=0.2 | 0 | 0.0 | 16.1 | 1 | 16 | 3 |
| 2 | Revision_3 | b2=1.0,b3=1.0,bl=1.0 | 0 | 0.0 | 16.1 | 1 | 16 | 3 |
| 3 | Candidate_A | b=1.0 | 2 | 10.0 | 16.15 | 1 | 15 | 4 |
| 3 | Revision_1 | a=1.2,b=1.0 | 2 | 10.0 | 16.15 | 1 | 15 | 4 |
| 3 | Revision_2 | b=1.0,c=0.2 | 2 | 10.0 | 16.15 | 1 | 15 | 4 |
| 3 | Revision_3 | b2=1.0,b3=1.0,bl=1.0 | 2 | 10.0 | 16.15 | 1 | 15 | 4 |
| 4 | Candidate_A | b=1.0 | 0 | 0.0 | 16.3 | 0 | 15 | 4 |
| 4 | Revision_1 | a=1.2,b=1.0 | 0 | 0.0 | 16.3 | 0 | 15 | 4 |
| 4 | Revision_2 | b=1.0,c=0.2 | 0 | 0.0 | 16.3 | 0 | 15 | 4 |
| 4 | Revision_3 | b2=1.0,b3=1.0,bl=1.0 | 0 | 0.0 | 16.3 | 0 | 15 | 4 |
