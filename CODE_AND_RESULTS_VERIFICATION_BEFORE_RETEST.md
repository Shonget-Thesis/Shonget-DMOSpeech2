# Verify the Experiment Before Retesting

## Decision

**Do not rerun the full CPU or GPU experiment yet. First inspect the testing code and raw records.** The research report alone cannot establish that both implementations and their evaluations are equivalent. SNR-only performing well is not evidence that the test was wrong.

## Files to collect

Provide one ZIP containing the following project files, preserving relative paths:

1. **CPU and GPU benchmark scripts**, including `run_gpu_expanded_benchmark.py` and the CPU candidate-grid runner.
2. **Teacher sampling and student completion functions**, including `run_student_completion`, time grids, CFG computation, state extraction, restart settings, and random-noise handling.
3. **Scheduler implementation**, including prompt SNR estimation, endpoint drift, threshold checks, selected-step logic, and fallback behavior.
4. **Evaluation implementation**, including audio preprocessing, Whisper transcription and decoding settings, text normalization, WER, speaker embeddings/SIM, UTMOS, reference matching, and failure classification.
5. **Calibration and analysis scripts**, including fixed-sequence testing, threshold selection, bootstrap intervals, paired tests, and report/table generation.
6. **Raw CPU and GPU result CSV/JSON files**, including candidate-level outcomes and saved telemetry. Include speaker ID, condition, candidate step, seed, duration, selected step, drift values, metric values, reference IDs, failure flags, NFE, and available timing fields.
7. **Original speaker split manifests**, preserving the development/calibration/test membership.
8. **Run configurations and environment records**, including checkpoint identifiers, source commit, package versions, dtype settings, and available run logs.

If large audio files are impractical to include, provide the code and records first. Keep the audio available for the specific cases whose outcomes differ. Exclude credentials and access tokens.

## Verification questions

| Area | What must be checked |
| --- | --- |
| Inputs | Do CPU and GPU use the same speaker, prompt, target text, duration, and clean/noisy waveform? |
| Randomness | Are actual initial noise tensors and student restart noise comparable? Identical seed numbers alone do not establish identical random tensors across devices. |
| Teacher candidates | Does the shared GPU teacher trajectory reproduce the states used by separately evaluated CPU candidates? Check update order, time indexing, CFG, endpoint extraction, and tensor copying. |
| Student completion | Are restart level, input state transformation, student times, number of passes, and noise handling equivalent? |
| Scheduling | Are SNR and drift computed on the same waveform/frame regions with the same normalization, thresholds, and comparison operators? |
| Evaluation | Do ASR decoding, text normalization, resampling, speaker embeddings, and reference matching agree? |
| Failure labels | Are WER expressed in the correct units and SIM margins applied to the correctly matched Step-8 reference? Is robust failure aggregated by speaker? |
| Calibration | Were policy settings and threshold order fixed independently of calibration, and was the selected threshold frozen before testing? |
| Compute | Does stated NFE match actual model calls? Distinguish logical per-policy NFE from the shared computation used to generate the candidate grid. |
| Timing | Does the timed region include the intended stages and synchronize GPU execution? Are partial-stage and complete-generation timings labeled correctly? |

## Focused comparison

Start with the same 200 held-out speakers on CPU and GPU. Match records by speaker ID, condition, and candidate step. Identify cases where either the proposed or SNR-only policy changes pass/fail status.

For each changed case, trace the selected candidate, generated output, ASR transcript, WER, SIM, matched reference metrics, and the exact clause that triggers failure. Compare intermediate synthesis states or audio where needed to locate whether the difference originates in generation, evaluation, or record matching.

## What happens after the audit

- **Analysis or reporting error:** regenerate labels, statistics, and tables from valid saved outputs.
- **Timing error only:** run a small synchronized end-to-end timing benchmark.
- **Synthesis or evaluation configuration error:** correct it and rerun only the affected scope, which may be broad if the error is systematic.
- **No material error:** keep the results and report that the added value of trajectory probing over SNR-only is currently inconclusive.

## Final instruction

**Collect the code and raw records first. Verify the experiment, then decide the necessary rerun scope.** Do not repeat all 4,000 evaluations merely because a simpler baseline performs competitively, and do not tune the method on the held-out test set to force a favorable result.
