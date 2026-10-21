# Updated code bundle: independent recheck

**Verdict: substantial fixes are present, but “all blockers resolved / ready for the full Colab benchmark” is premature. Keep the existing development grid. Make the specific corrections below, then run the two-case smoke test before the 50-case timing screen.**

Reviewed the attached `three_scheduler_revisions_verification_bundle(1).zip` and `VERDICT_CODE_BUNDLE_AND_FIXES_REPORT.md`. Executed saved-record analysis and small validator checks only. No synthesis or GPU runtime was executed during this audit.

## What is now fixed

| Previous issue | Finding in this update |
| --- | --- |
| Random inputs in the actual benchmark | Actual benchmark now loads manifest prompt audio, text, and seeds. |
| Teacher-only timer | Actual benchmark now calls `run_student_completion`, whose supplied implementation includes three student calls and Vocos decoding, inside the timer. |
| Missing genuine d_1 before r_2 | Both new teacher loops calculate d_1 before computing r_2. |
| Condition-label gating in offline evaluation | Evaluator now passes measured prompt SNR. |
| Causality test bypassed actual policy functions | Positive test now calls actual policy closures through its replay container. |
| No conservative fallback when threshold search fails | Cross-validation now explicitly falls back to fixed index 8. |
| Aggregate-only timing exports | Benchmark code now exports case-level timing, seeds, exits, duration, and RTF. |
| Unbalanced policy order | Execution order rotates across cases. |

These are real code changes. They improve the implementation substantially.

## Offline results independently reproduced

All seven updated saved analysis CSVs reproduce from the supplied development grid: 100 fold assignments, 575 threshold-search rows, 20 fold results, 400 out-of-fold speaker decisions, **13** policy comparison rows, **19** paired comparison rows, and 100 case-audit rows. The report's repeated “14-policy” description is stale.

| Policy | Mean NFE per utterance | Failed speakers / 100 |
| --- | ---: | ---: |
| SNR-only 3/8 | 16.00 | 3 |
| Original SNR + absolute drift | 18.05 | 1 |
| Candidate A, out of fold | 16.25 | 2 |
| Revision 1, out of fold | 16.26 | 2 |
| Revision 2, out of fold | 16.25 | 2 |
| Revision 3, out of fold | 16.25 | 2 |

The scientific interpretation is unchanged: the three revisions have not improved on Candidate A in this development experiment. Candidate A versus SNR-only 3/8 rescues two speakers and harms one, with exact McNemar p = 1.0. This does not establish superiority, nor does it establish equivalence.

## Required corrections before the full timing screen

### 1. Restrict online exits to the tested candidate indices

In both `run_smoke_test_synthesis` and `run_conditional_gpu_timing`, adaptive exit logic runs whenever `i >= 2`. That includes **index 7**, whereas the offline policies search only **{2, 3, 4, 5, 6}**, with fallback 8.

For example, if a trajectory never qualifies through 6 but qualifies at 7, the new online loop exits at 7 while the evaluated offline policy returns 8. The development grid contains no index-7 quality outcome.

**Fix:** keep integrating the teacher as necessary, but evaluate early-exit rules only when `i in CANDIDATE_STEPS`. Apply this to both loops. Record index 8 explicitly as fallback. Test a constructed trace that first qualifies at 7 and confirm the policy still returns 8.

This is a potential discrepancy identified in code; it is not a claim that one of the currently saved 100 speakers actually exits at 7.

### 2. Remove unnecessary adaptive-feature overhead from simple baselines

The timed loop computes endpoint drift, trend ratio, and endpoint clones for **every policy**, including Fixed 8 and SNR-only. The drift helper calls `.item()`, which can force device synchronization, so these operations can affect measured GPU runtime.

SNR-only should use measured SNR and its selected fixed index; it does not need drift or trend. Fixed 8 needs neither decision feature. Giving them adaptive-feature overhead can make a more complex policy appear more competitive than it is.

**Fix:** calculate only the features required by each deployed policy. Preserve the same integration, CFG, selected-endpoint construction, student completion, and vocoder across policies. Debug traces can be collected in a separate untimed diagnostic run. Baselines should still perform the real synthesis work required by their own algorithms.

### 3. Make the smoke test enforce its claims

The current smoke function records `waveform_finite` but does not assert it. It prints “passed successfully” regardless of that value. It does not compare its online exit or traces with saved replay, does not instrument model calls, and tests only Candidate A. The caller does not check `smoke_res`; missing imports can return an empty result and permit continuation.

**Fix:** fail immediately for missing prerequisites, empty/non-finite waveforms, invalid exits, failed replay comparisons, or incorrect actual call counts. Save a smoke-test report. Exercise every policy intended for timing on one clean and one noisy case before allowing the screen to start. Expect teacher calls 2(s+1), student calls 3, and Vocos calls 1, using actual instrumentation to check the formula.

For comparison with historical CSV traces, first verify matching preprocessing, model settings, and RNG behavior. If those cannot be matched, compare the live policy to a full-trajectory replay from the same newly prepared inputs and document the historical compatibility limitation.

The warmup now invokes teacher, student, and Vocos, which is better than matrix multiplication. It still uses synthetic random inputs and a single teacher evaluation. Reuse prepared real smoke-test inputs to warm the actual timed path.

### 4. Fix clean-Colab setup and transfer the actual revised sources

Notebook Cell 1 constructs `ROOT_DIR`, `CONFIG`, and import paths **before** Cell 2 may clone the project and change directories. In a fresh session, those absolute paths remain pointed at the old directory. Existing-directory handling can also attempt Git commands from the wrong directory.

The attached report states the fixes are **uncommitted and unpushed**. A Git clone therefore cannot be assumed to contain these changes.

**Fix:** clone or locate the repository first, then resolve one absolute root and rebuild all configuration/import paths. Supply the revised runner and helpers to that checkout by an explicit upload or an intentionally updated repository version. Print the imported module path and source SHA256. Check dependencies, both checkpoints, dataset audio, and imports before starting GPU work. No Git push is required merely to fix this; uploading the corrected sources is sufficient.

## Remaining audit/reporting corrections

- **Absent causality report still passes:** `validate_inputs(..., causality_report=None)` assigns Check 7 a pass. I reproduced this. Require the report or label the check unverified and stop the full validation claim.
- **Absolute drift is not checked for finiteness:** the validator checks `q_A` and `q_C`, but not each row's required `d_<teacher_step>`. The report claims this is checked. Add explicit finite-drift validation for indices 2–6.
- **Provenance check is weaker than described:** it accepts any nonempty provenance labels totaling 1,200 rows; it does not enforce the claimed 800 reused / 400 newly synthesized breakdown or verify generation compatibility. Make the check and its description agree. Retain the explicit limitation about missing generation metadata.
- **Replay container is not a strict future-access blocker:** its indexing advances the simulated solver automatically. I confirmed that access sequence `[8, 2]` is allowed. That could represent paying for later computation, but it cannot establish a valid early exit at 2 after inspecting 8. Log the highest visited index and reject returned exits below it; preferably share one stepwise decision function with actual inference. The current test is useful, but its guarantee is overstated.
- **Frozen Revision 1 differs from its out-of-fold aggregate:** the fixed a=1.2, b=1.0 rule reproduces **16.25 NFE** on the full development set; **16.26** belongs to the fold-dependent selection procedure. Keep these rows and labels distinct.
- **Reported call counts are currently inferred:** `teacher_calls = 2*(exit_step+1)`, `student_calls=3`, and `vocos_calls=1` are assigned, not observed. They are expected counts; instrument the smoke test to verify them.
- **Preparation time needs CUDA synchronization** if it is to include GPU input-preparation work accurately. Keep it separate from teacher-through-Vocos synthesis latency as intended.
- **Do not report p = 0:** rounding tiny Wilcoxon p-values to six decimals produces zeros. Keep numerical precision and format small values scientifically or as inequalities.
- **Oracles are diagnostics:** the CSV labels them generically `offline_replay`. Mark them explicitly as non-deployable oracle diagnostics, since they consult outcome labels.

## Exactly what to do next

1. Preserve the current 1,200 records and reproduced development results.
2. Correct candidate-index gating, baseline feature overhead, smoke-test enforcement, and Colab source/path setup.
3. Repair the remaining validator/reporting claims without regenerating audio.
4. In Colab, run the two selected utterances across the six timed policies as a smoke test only. Save the assertions, waveform checks, exits, measured SNR, model-call counts, and replay agreement.
5. If that passes, run the planned **50 cases × six policies = 300 timing measurements** sequentially on the GPU. This is a development timing screen, not independent final solution validation.

**No full 1,000-utterance rerun is justified by this recheck. The immediate work is a bounded implementation correction followed by a small real-pipeline verification.**
