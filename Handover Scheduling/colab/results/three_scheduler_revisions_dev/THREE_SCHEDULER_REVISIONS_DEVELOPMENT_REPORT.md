# Master Development Report: Three Scheduler Revisions in One Experiment

**Date:** 2026-10-21 23:15:35  
**Git Commit:** `5710d3f9a945dbb76b37b188284c75f43ada59e2`  
**Scope:** 100 Development Speakers (200 Cases: Clean & Noisy15)  
**Candidate Grid:** [2, 3, 4, 5, 6, 8] (1,200 Records)  
**7-Point Input Validation:** PASSED  

---

## Executive Summary & Direct Answers to Core Research Questions

### Question 1: Do any of the three revisions outperform Candidate A on development speakers?
- **Verdict:** **NO.** None of the three revisions outperforms Candidate A on the 100 development speakers.
- **Candidate A (Out-of-Fold):** 16.25 NFE, 2 robust failures (2.0% risk).
- **Revision 1 (Trend + Drift):** 16.26 NFE out-of-fold, 2 robust failures (2.0% risk). Fold 0 selected a=1.0, b=1.0 (delaying 1 speaker from step 3 to step 4, producing 16.26 NFE out-of-fold), while Folds 1-4 selected a=1.2, b=1.0. When evaluated as a fixed deployable rule (a=1.2, b=1.0) on the full development set, it reproduces 16.25 NFE identically matching Candidate A.
- **Revision 2 (Trend + Disagreement):** 16.25 NFE, 2 robust failures (2.0% risk). Cross-validation selected b=1.0 with disagreement threshold c >= 0.20, reproducing Candidate A decisions identically under the default c_i <= c hypothesis.
- **Revision 3 (Index-Specific Trend):** 16.25 NFE, 2 robust failures (2.0% risk). Selected uniform threshold b=1.0 across all steps, reproducing Candidate A decisions identically.

### Question 2: Does Candidate A or any revision genuinely beat simple SNR-only baselines?
- **Comparison with SNR-Only 3/8:**
  - SNR-only 3/8 achieves 16.00 mean NFE with 3 robust failures (3.0% risk).
  - Candidate A achieves 16.25 mean NFE with 2 robust failures (2.0% risk).
  - Paired comparison: Candidate A rescues 2 speakers but harms 1 speaker, avoiding 1 net failure at the cost of +0.25 extra mean NFE.
  - **McNemar Exact Test p = 1.0000** (two-sided, discordant pairs=3). **Difference is NOT statistically significant.**
- **Comparison with SNR-Only 4/8:**
  - SNR-only 4/8 achieves 17.00 mean NFE with 3 robust failures (3.0% risk).
  - Candidate A uses 0.75 fewer NFE and records 1 fewer failure, also yielding McNemar p = 1.0000.
- **Conclusion:** Candidate A represents an exploratory trade-off, not statistically significant superiority over simple SNR-only fixed baselines.

### Question 3: What is the true Oracle headroom available from earlier handover?
- **Expanded Oracle ([2, 3, 4, 5, 6, 8]):**
  - Unrestricted Oracle achieves 12.01 mean NFE (42.81% reduction vs Fixed 8).
  - SNR-Gated Oracle achieves 15.07 mean NFE (28.24% reduction vs Fixed 8).
- **Takeaway:** Forcing noisy cases to step 8 bounds deployable headroom to 15.07 NFE.

### Question 4: How does the original proposed SNR+drift scheduler compare?
- **Reconciled Original SNR+drift:** Replays identically to 18.05 mean NFE and 1 failure (1.0% risk).
- Candidate A reduces mean NFE by 1.80 steps vs the original rule, but incurs 1 additional robust failure (2 vs 1, McNemar p = 1.0000).

---

## 3. Master Policy Comparison Table

| policy | status | mean_nfe | nfe_reduction_% | robust_failures | robust_risk_% | clean_risk_% | clean_wer | clean_sim | step2_exits | step3_exits | step4_exits | step8_exits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed_Index_8 | fixed_reference | 21.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0227 | 0.9344 | 0 | 0 | 0 | 100 |
| SNR_Only_4_8 | fixed_baseline | 17.0 | 19.05 | 3 | 3.0 | 3.0 | 0.0204 | 0.9339 | 0 | 0 | 100 | 0 |
| SNR_Only_3_8 | fixed_baseline | 16.0 | 23.81 | 3 | 3.0 | 3.0 | 0.0234 | 0.9355 | 0 | 100 | 0 | 0 |
| SNR_Only_2_8 | fixed_baseline | 15.0 | 28.57 | 7 | 7.0 | 7.0 | 0.0222 | 0.9349 | 100 | 0 | 0 | 0 |
| Original_SNR_Plus_Drift | fixed_baseline | 18.05 | 14.05 | 1 | 1.0 | 1.0 | 0.0217 | 0.9338 | 0 | 0 | 22 | 0 |
| Candidate_A_OOF | out_of_fold_evaluated | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |
| Revision_1_OOF | out_of_fold_evaluated | 16.26 | 22.57 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 72 | 24 | 0 |
| Revision_2_OOF | out_of_fold_evaluated | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |
| Revision_3_OOF | out_of_fold_evaluated | 16.25 | 22.62 | 2 | 2.0 | 2.0 | 0.0208 | 0.9343 | 2 | 73 | 23 | 0 |
| Original_Oracle_4_8 | offline_oracle_diagnostic | 14.7 | 30.0 | 0 | 0.0 | 0.0 | 0.0204 | 0.9344 | 0 | 0 | 97 | 0 |
| Original_Oracle_4_8_SNR_Gated | offline_oracle_diagnostic | 17.05 | 18.81 | 0 | 0.0 | 0.0 | 0.0204 | 0.9344 | 0 | 0 | 97 | 0 |
| Expanded_Oracle_2_8 | offline_oracle_diagnostic | 12.01 | 42.81 | 0 | 0.0 | 0.0 | 0.0205 | 0.937 | 93 | 7 | 0 | 0 |
| Expanded_Oracle_2_8_SNR_Gated | offline_oracle_diagnostic | 15.07 | 28.24 | 0 | 0.0 | 0.0 | 0.0205 | 0.937 | 93 | 7 | 0 | 0 |

---

## 4. Paired Baseline Statistical Comparisons

| revision_policy | baseline_policy | rescued_failures | harmed_failures | net_avoided_failures | mcnemar_exact_p_value | mean_nfe_difference | nfe_diff_95%_ci_lower | nfe_diff_95%_ci_upper | wilcoxon_nfe_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate_A_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.75 | -4.84 | -4.65 | 5.9500e-20 |
| Candidate_A_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.75 | -0.84 | -0.65 | 1.8725e-16 |
| Candidate_A_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.25 | 0.16 | 0.35 | 1.3932e-05 |
| Candidate_A_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.8 | -1.97 | -1.63 | 8.9436e-18 |
| Revision_1_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.74 | -4.83 | -4.64 | 7.0002e-20 |
| Revision_1_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.74 | -0.83 | -0.64 | 3.0955e-16 |
| Revision_1_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.26 | 0.17 | 0.36 | 8.5022e-06 |
| Revision_1_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.79 | -1.96 | -1.62 | 8.7565e-18 |
| Revision_1_OOF | Candidate_A_OOF | 0 | 0 | 0 | 1.0 | 0.01 | 0.0 | 0.03 | 0.317311 |
| Revision_2_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.75 | -4.84 | -4.65 | 5.9500e-20 |
| Revision_2_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.75 | -0.84 | -0.65 | 1.8725e-16 |
| Revision_2_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.25 | 0.16 | 0.35 | 1.3932e-05 |
| Revision_2_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.8 | -1.97 | -1.63 | 8.9436e-18 |
| Revision_2_OOF | Candidate_A_OOF | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Revision_3_OOF | Fixed_Index_8 | 0 | 2 | -2 | 0.5 | -4.75 | -4.84 | -4.65 | 5.9500e-20 |
| Revision_3_OOF | SNR_Only_4_8 | 3 | 2 | 1 | 1.0 | -0.75 | -0.84 | -0.65 | 1.8725e-16 |
| Revision_3_OOF | SNR_Only_3_8 | 2 | 1 | 1 | 1.0 | 0.25 | 0.16 | 0.35 | 1.3932e-05 |
| Revision_3_OOF | Original_SNR_Plus_Drift | 1 | 2 | -1 | 1.0 | -1.8 | -1.97 | -1.63 | 8.9436e-18 |
| Revision_3_OOF | Candidate_A_OOF | 0 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |

---

## 5. Out-of-Fold Cross-Validation Performance

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
