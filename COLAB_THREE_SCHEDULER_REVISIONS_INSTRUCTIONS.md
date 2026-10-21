# Colab Instructions: Three Scheduler Revisions in One Comparative Experiment

## Task for the implementer

Create a new executable notebook in the existing DMOSpeech 2 project:

`colab/DMOSpeech2_Three_Scheduler_Revisions_Comparison.ipynb`

Implement all three scheduler revisions below, compare each against the same baselines, and export a unified report. Run all further tests and analyses in Google Colab. Use the actual project implementation and saved development artifacts. This document supplies instructions, not a completed notebook or experimental results.

The objective is to find a scheduler with a better quality–computation trade-off than simple SNR-only policies. Improvement and algorithmic novelty are hypotheses requiring evidence.

## 1. Scope and workload

- Use the original **100 development speakers only**.
- Each speaker has clean and noisy15 conditions: **200 cases**.
- Reuse the expanded grid at indices **{2,3,4,5,6,8}**, comprising **1,200 candidate records**.
- Evaluate all three revisions and all baselines on these same cases.
- Do not use the old 200 calibration or 200 test speakers to select formulas, directions, thresholds, or stopping indices.
- Preserve original notebooks, frozen policies, and completed benchmark results.

This is one shared-data experiment. It does not require separately generating 1,000 cases for each revision. Reconstruct selected quality outcomes from the grid when matching records exist. Real online timing is a separate stage within the same notebook.

## 2. Colab execution and resource use

Log the assigned GPU model, available VRAM, host RAM, Python/PyTorch/CUDA versions, precision, checkpoints, and source hashes. Colab Pro does not guarantee a particular GPU or memory allocation. Host RAM and GPU VRAM are separate resources.

Use one synthesis model instance per runtime. Share candidate generation and telemetry collection across all formulas. Run GPU synthesis and policy timing sequentially on a shared GPU; three competing model copies are unnecessary and can distort timings or exceed memory.

CPU-side tabular analyses may run concurrently within Colab if resources permit. Use a GPU runtime for inference and timing. Keep metric evaluator loading memory-aware, using the existing project workflow rather than assuming every model fits simultaneously.

## 3. Required inputs and verification

Load the saved development artifacts:

- `development_expanded_grid.csv`
- `candidate_formula_fold_results.csv`
- `candidate_formula_search_log.csv`
- `early_handover_case_audit.csv`
- `expanded_policy_comparison.csv`
- Original development manifest, original analysis scripts, and fold assignments if available.
- Model/checkpoint/input provenance and compatibility anchor results for reused records.

Validate:

1. Exactly 100 development speaker IDs, two conditions per speaker, and six candidate indices per case.
2. Exactly 1,200 unique speaker/condition/candidate records, with no missing required quality or decision inputs.
3. Reference values match the corresponding index-8 records.
4. Failure flags reproduce the original definition.
5. NFE equals 2(s+1)+3.
6. New and reused rows have compatible generation, preprocessing, duration, noise, precision, and metric settings. Matching IDs or copied reference columns alone are insufficient evidence.
7. Signals used at index i come from computations available by index i. The offline CSV may contain future drift columns; the online policy must never access them early.

Retrieve missing original fold assignments where possible. If unavailable, generate new deterministic development folds and label them as a new exploratory evaluation. Do not claim to reproduce historical folds without their identities.

## 4. Controlled implementation settings

Keep checkpoints, target text, prompt preprocessing, duration prediction, injected noise, randomization, CFG convention, teacher time grid, student restart j=1, three student passes, student initialization/re-noising, vocoder, metric definitions, and failure margins consistent across policies.

Use generated frames only for trajectory scores, excluding prompt frames. Teacher endpoints use the actual implemented guided velocity:

`v_guided = v_cond + cfg_strength * (v_cond - v_uncond)`

`E_i = state_i + (1 - t_i) * v_guided_i`

The teacher grid remains:

`t_i = 1 - cos((pi/2) * i/16)`

Verify real source signatures before implementing wrappers. Do not substitute a direct teacher-state handover for the existing endpoint/re-noising procedure, especially at indices 2 and 3, which precede the fixed student restart time.

## 5. Common signal definitions

For endpoint E_i over generated frames:

`d_i = norm(E_i - E_(i-1)) / (norm(E_i) + epsilon)`

Trend ratio:

`r_i = d_i / (d_(i-1) + epsilon)`

Conditioning disagreement, using teacher outputs before guidance:

`c_i = norm(v_cond_i - v_uncond_i) / (norm(v_cond_i) + norm(v_uncond_i) + epsilon)`

Use the epsilon and norm conventions from the verified signal implementation. Log exact settings. To calculate r2, endpoint predictions at indices 0, 1, and 2 are required. Deriving the trend requires no extra teacher forward evaluations if these predictions are retained during the existing trajectory.

Low drift or disagreement does not guarantee correct speech. Trend can be small while absolute changes remain large. Conditioning disagreement may represent useful conditioning rather than danger. These limitations motivate the comparisons below.

## 6. Implement three distinct revisions

All revisions initially retain the same pre-generation gate:

`estimated_prompt_SNR < 20 dB -> teacher handover at index 8`

For eligible prompts, examine indices {2,3,4,5,6}, stopping at the earliest candidate that satisfies the policy's rule. Otherwise continue to index 8.

### Revision 1: Trend plus absolute drift

Stop when both conditions hold:

`d_i <= a AND r_i <= b`

Rationale: a decreasing change may still be too large; the absolute-drift condition limits its size while the trend checks contraction.

Select a and b using development training folds only. Keep the parameter search small and explicitly logged. This is a candidate hypothesis, not proof that imposing both conditions improves predictions.

### Revision 2: Trend plus conditioning-disagreement check

Initial candidate stopping rule:

`r_i <= b AND c_i <= c`

Rationale: the disagreement signal showed descriptive association with early candidate failures, but its standalone threshold was unstable. Combining it with trend may or may not yield more selective stopping.

The initial direction assumes high disagreement should delay stopping. If testing the opposite direction, list it as a separate variant selected only within training folds. Do not choose the direction using validation outcomes and then report those same outcomes as unbiased validation.

### Revision 3: Index-specific trend thresholds

Stop when:

`r_i <= b_i`

Rationale: score distributions and quality risk may vary across teacher indices, so a single threshold may be inappropriate.

For the initial low-parameter experiment, use three threshold groups:

- b2 for index 2.
- b3 for index 3.
- b_later shared by indices 4, 5, and 6.

Report this grouping explicitly rather than implying five independently selected thresholds. Any later expansion is another logged experiment. Avoid automatically fitting five independent thresholds to a cohort with very few failures.

These are separate decision rules, not three model-training jobs. No teacher/student weights are trained in this experiment.

## 7. Baselines required in the same run

| Policy | High-SNR behavior | Low-SNR behavior | Why included |
| --- | --- | --- | --- |
| Fixed index 8 | Always 8 | Always 8 | Quality/computation reference |
| SNR-only 4/8 | Always 4 | 8 | Original simpler baseline |
| SNR-only 3/8 | Always 3 | 8 | Stronger simple early-exit competitor |
| SNR-only 2/8 | Always 2 | 8 | Cheapest simple early-exit comparison |
| Original SNR + drift | Earliest 4/5/6 with d_i <=0.50, else 8 | 8 | Original proposed version |
| Original Candidate A | Earliest eligible candidate satisfying its implemented trend rule | 8 | Starting point for the revisions |
| Revision 1 | Trend + absolute-drift rule | 8 | New candidate |
| Revision 2 | Trend + disagreement rule | 8 | New candidate |
| Revision 3 | Index-specific trend rule | 8 | New candidate |

Retrieve Original Candidate A's true candidate order, comparison operator, and parameter selection from code. Distinguish a fixed threshold-1.00 replay from its historical fold-selected results. Never silently substitute a summary statistic for the actual algorithm.

Include an expanded-grid oracle as an optional offline headroom diagnostic, labeled non-deployable. It uses quality outcomes that an online scheduler cannot know.

Each revision must have explicit paired comparisons against SNR-only 2/8, 3/8, 4/8, original Candidate A, and the original scheduler. Do not compare different input subsets or give only the new formulas access to early candidates without also evaluating the simple 2/8 and 3/8 policies.

## 8. Shared generation and signal recovery

Use existing candidate audio/metrics when compatible. If signal columns needed by a revision are missing, recover teacher telemetry with a matched teacher-only pass when possible. Do not regenerate student outputs solely to obtain a missing score.

If a new candidate output is genuinely absent, generate it using the same matched conditions and evaluate it against the matched index-8 reference. Collect the signals for all three revisions during that same trajectory.

Candidate-grid generation can integrate the teacher to collect multiple endpoints. Those cached-grid timings cannot be used as deployment latency for an early-stopping policy.

## 9. Development selection protocol

Use the same deterministic five speaker-disjoint outer folds for all revisions. Both conditions of a speaker remain together.

Within each outer training set:

1. Generate a small, predefined threshold grid or score quantiles computed only from training speakers. Save the resulting numerical thresholds.
2. Use an inner speaker-disjoint selection procedure, where feasible, to select parameters for each revision. Select for low mean NFE subject to the empirical risk screening target; report if no setting qualifies.
3. Freeze that fold's selected rule and evaluate it on the outer validation speakers once.
4. Save every attempted parameter setting, its training/inner-validation outcomes, and the selected setting.

Use the prespecified 5% robust risk target as a development screen, not a claimed statistical certificate. Define tie handling and a reference fallback if no candidate qualifies before execution. Do not adjust the failure margins to make a revision win.

If choosing one formula family after comparing outer-fold results, describe that family selection as exploratory. All these development data have already informed the research; resplitting them does not create fresh independent confirmation.

Do not permit speaker IDs, WER, SIM, UTMOS, reference quality, future endpoints, or oracle labels as online features.

## 10. Quality evaluation and computation

Retain the original reference-relative failure definition:

`WER_candidate > max(0.125, WER_reference + 0.01) OR SIM_candidate < SIM_reference - 0.02`

Robust speaker failure is failure in either condition. Report clean and noisy outcomes separately as well.

Use the existing Whisper large-v3 WER evaluator, microsoft/wavlm-base-plus-sv speaker evaluator, and UTMOS evaluator when new audio needs scoring. Respect actual waveform sample rates and exact normalization settings. Passing the margins does not establish perceptual equivalence or perfect audio.

NFE(s) = 2(s+1)+3, producing {9,11,13,15,17,21} calls at indices {2,3,4,5,6,8}. This counts teacher and student calls, not vocoder evaluations. Formula arithmetic has runtime cost even when it requires no additional teacher calls.

## 11. One unified result table plus paired comparisons

For every policy, report:

- Speaker count, case count, and evaluation status (fixed, out-of-fold exploratory, or oracle).
- Robust failures and rate; clean/noisy failures separately.
- WER, SIM, and UTMOS where available, with clean/noisy breakdown.
- Mean NFE, reduction against index 8, and selected-index distribution.
- Additional calls versus each SNR-only baseline.
- Rescued and introduced failures relative to each baseline on matched speakers.
- Actual core synthesis latency and RTF if timed; otherwise mark explicitly “not measured.”

Provide a separate paired comparison table for each revision against the required baselines. Use speaker-level pairing and speaker-clustered intervals. If testing multiple comparisons, define correction families and report raw and adjusted p-values accurately. Development significance is exploratory, especially after repeated formula selection.

Report the trade-off, not just a winner label. An observed one-speaker improvement is uncertain. NFE differences do not by themselves establish runtime differences, and non-significance does not prove equivalence.

## 12. Conditional actual GPU timing in the same notebook

After quality/NFE screening, time promising revisions alongside the same relevant baselines using genuine online stopping. Report why a revision is included or excluded; do not quietly omit unfavorable outcomes from the quality table.

Load one real synthesis model and run each policy sequentially. No concurrent policy timing on the shared GPU. Freeze the decision rule for this timing screen before measurement.

Include applicable SNR estimation, actual teacher execution through selected handover, that policy's decision computations, fixed three-pass student completion, and Vocos decoding. Exclude preprocessing, duration prediction, loading, file writing, and offline evaluators consistently, documenting the boundary.

Use warm-up, balanced policy order, batch size 1, high-resolution wall time, and CUDA synchronization before the start and before the final timestamp. Count actual calls. Replay from cached final endpoints or precomputed selected indices is not an online policy timing test.

Each baseline computes only what it needs. Each revision computes only its required scores; do not charge every method for collecting all experimental telemetry.

If all nine deployable policies are timed on all 200 cases, the total is **1,800 timed generations**, separate from the 1,200 candidate-grid rows. If only promising policies are timed, report the actual policy list and workload. This is development timing, not independent confirmation.

Before linking cached quality to timed synthesis, verify output/decision equivalence on representative cases and record actual timing settings. Do not assume matching seed integers establish compatibility across different backends or software versions.

## 13. Notebook cell order

1. Purpose, exploratory status, configuration, and output directory.
2. Colab resource/environment logging and real project setup.
3. Load inputs and validate manifests, rows, references, failure flags, and provenance.
4. Reconstruct original policies and audit historical folds where possible.
5. Implement three revision functions and verify online feature availability.
6. Recover missing shared telemetry/outputs only where necessary.
7. Create/save speaker fold assignments and threshold search definitions.
8. Run selection and out-of-fold evaluation for all three revisions.
9. Produce the unified table and each revision's direct baseline comparisons.
10. Audit failed, rescued, harmed, and unnecessarily delayed cases.
11. Run conditional online GPU timing with real implementation.
12. Export CSVs, metadata, report, figures if useful, and downloadable archive.

Use resumable outputs for any expensive synthesis. Errors and skipped stages must be explicit. Use no mock model outputs or fabricated timing placeholders.

## 14. Required outputs

Save under `colab/results/three_scheduler_revisions_dev/`:

- `speaker_fold_assignments.csv`
- `revision_definitions.json`
- `threshold_search_log.csv`
- `revision_fold_results.csv`
- `out_of_fold_case_decisions.csv`
- `all_policy_comparison.csv`
- `paired_baseline_comparisons.csv`
- `failure_and_waiting_case_audit.csv`
- `data_validation_report.json`
- `experiment_metadata.json`
- `online_policy_timing.csv` if performed.
- `THREE_SCHEDULER_REVISIONS_DEVELOPMENT_REPORT.md`
- A downloadable archive of the results and exact analysis/scheduler source snapshot.

The final report must answer for each revision:

1. Does it improve on the original scheduler?
2. Does it improve on original Candidate A?
3. What does it gain or lose versus each SNR-only alternative?
4. Is the benefit in failure count, NFE, measured runtime, or more than one?
5. Is the observed improvement large enough and sufficiently stable to justify further evaluation?
6. What remains unverified?

## 15. Decision and next stage

The development goal is lower computation/runtime at comparable quality risk, or lower failure risk at comparable cost. Do not claim superiority merely because a formula is more complex or newly written.

If a candidate has a credible trade-off, freeze the complete rule, check related work, and plan independent calibration plus fresh held-out evaluation. The original 200 calibration and 200 test speakers have already been inspected; they are not fresh validation for a redesign informed by their results. Old certificates do not transfer automatically.

If no revision improves the trade-off, report that clearly. Do not automatically expand the search until a favorable result appears. The existing original results remain evidence for version 1.

## Final implementation instruction

Deliver one new Colab notebook implementing the complete shared-data comparison above. All three revisions and the same baselines must appear together in the results. Start with the 100 development speakers, reuse compatible saved candidates, and perform online GPU timing sequentially only when warranted. This task requests notebook implementation from these Markdown instructions; it does not authorize fabricated outputs or imply that an improved or novel algorithm has already been established.
