# Verdict: Is a Full GPU Retest Necessary?

## Decision

**Do not rerun all 4,000 GPU candidate evaluations at this stage.** The current GPU experiment is useful evidence. First audit the saved records and timing implementation. A suspicious timing pattern warrants checking the measurement; it does not establish that the entire experiment is invalid.

This verdict concerns the reported results in `CHAPTER3_MASTER_RESEARCH_REPORT(2).md`. The underlying code, audio files, and raw evaluation records have not been independently audited here.

## CPU–GPU differences

Small numerical differences between CPU and GPU execution can occur. They may occasionally change a drift-threshold decision, generated waveform, ASR transcription, or reference-relative failure classification.

However, GPU acceleration establishes faster execution only when timing is measured correctly. It does not automatically establish greater accuracy, reliability, or statistical validity. Matching rounded mean NFE values also does not prove identical decisions across hardware.

The reported CPU held-out failure rate of **5/200 (2.5%)** and GPU pooled failure rate of **6/500 (1.2%)** use different cohorts. Their difference cannot currently be attributed to hardware.

## Required checks

| Check | Action | New synthesis required? |
| --- | --- | --- |
| GPU timing | Inspect timer boundaries, CUDA synchronization, warm-up, and whether each candidate has its own measured runtime. | Only a small timing benchmark if existing measurements are inadequate. |
| CPU–GPU comparison | Match the original 200 test speakers by speaker, condition, candidate, and seed. Compare selected steps, drift values, metric differences, and failure agreement. | Usually no; use saved records. |
| GPU split results | Report development, calibration, and held-out test separately using the original 100/200/200 partition. Evaluate the frozen policy on the 200-speaker GPU test split. | No, if candidate outputs and telemetry were saved. |
| SNR-only comparison | Build the paired speaker-level pass/fail table for SNR-only versus the proposed scheduler and calculate the paired test. | No, if saved records are complete. |
| Reproducibility | Verify matching checkpoints, preprocessing, duration settings, seeds, dtype settings, metric configurations, and failure definitions. | Only for a configuration error or missing records. |

## Timing concern

The GPU table reports approximately **0.527 seconds for clean speech at every candidate**, although Step 4 requires 13 NFE and Step 8 requires 21 NFE. Nearly identical times are possible when other costs dominate, but this pattern needs an explanation.

Inspect whether the timer covers actual candidate synthesis, includes the same components for every policy, and waits for GPU work to complete. Check that a shared or cached timing value was not copied across candidates.

If a timing benchmark is needed, use the same representative utterances for every candidate, warm up the models, synchronize the GPU before and after each timed region (or use correctly synchronized CUDA events), and repeat measurements. Record audio duration, NFE, synthesis latency, and RTF. Measure the proposed scheduler as well as fixed candidates so its decision overhead is included.

**Do not infer a 13.6% runtime improvement from a 13.6% NFE reduction.** NFE savings are already a separate computational result; wall-clock savings require valid timing.

## Interpretation of the GPU policy results

Across the reported pooled 500-speaker GPU cohort:

| Policy | Mean NFE | NFE savings | Robust added failure |
| --- | ---: | ---: | ---: |
| SNR-only | 17.02 | 18.9% | 12/500 (2.4%) |
| Proposed scheduler | 18.15 | 13.6% | 6/500 (1.2%) |

The proposed scheduler uses more computation and has fewer observed added failures. Both pooled empirical rates are below 5%, so the GPU table does not support a broad claim that SNR-only necessarily fails the target.

Determine whether the proposed policy offers a worthwhile improvement through the paired comparison and the original held-out split. The pooled 500-speaker GPU result includes development and calibration speakers and therefore is supplementary evidence rather than a new independent 500-speaker test.

## When to rerun

- **Timing issue only:** repeat the timing benchmark; retain valid audio-quality evaluations.
- **Missing records for a few cases:** regenerate only the missing cases under the documented configuration.
- **Confirmed synthesis or evaluation bug:** identify which outputs the bug affects and rerun that scope.
- **No implementation issue and complete records:** regenerate the analysis tables without repeating synthesis.

Do not change the policy threshold after examining the GPU test results and then describe that same test set as untouched. Any subsequent tuning must be disclosed.

## Final verdict

**Keep the GPU results and audit them first. A complete GPU rerun is not currently justified.** Most outstanding checks can be completed from existing files. A small synchronized timing benchmark may be necessary, and broader reruns should depend on a confirmed error.
