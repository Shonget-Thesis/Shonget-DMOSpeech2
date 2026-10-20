# Verdict: New Colab GPU Notebook for Earlier Handover Testing

## Decision

Run this new development experiment on **Google Colab with a CUDA GPU**, preferably the Tesla T4 used previously. Create a separate notebook:

`colab/DMOSpeech2_Early_Handover_Development_GPU.ipynb`

This document specifies the notebook to implement in the existing repository. It does not claim that the notebook has already been implemented or executed.

Purpose: determine whether handover before index 4 provides useful quality–computation headroom, and whether an adaptive rule can identify safe early exits. Beating SNR-only is a hypothesis to investigate, not an expected result to manufacture.

## Why this test

The current scheduler chooses indices 4, 5, 6, or 8. On prompts eligible for early stopping, SNR-only already chooses the cheapest of those options: index 4. Drift can delay stopping but cannot choose a cheaper index within that range.

Adding indices 2 and 3 creates a possibility of reducing computation below that baseline. This does not establish that earlier output quality is acceptable or that drift can predict it.

## Data and workload

- Use only the existing **100 development speakers** and their saved manifest.
- Evaluate both clean and noisy15 conditions: **200 cases**.
- Expanded candidate grid: **{2, 3, 4, 5, 6, 8}**.
- New indices 2 and 3 require **400 new candidate outputs** if matching old GPU records for 4/5/6/8 can be reused.
- The completed merged grid should contain **1,200 candidate rows**.
- If compatibility cannot be established, rerun the complete development grid: **1,200 candidate outputs**, not the whole 500-speaker experiment.
- Analyse high-SNR early-exit cases separately from low-SNR fallback cases. Do not let the gate's performance obscure whether drift adds information.

The noisy condition is included to characterize the expanded grid and check the existing gate. This experiment does not automatically authorize changing the SNR threshold or optimizing on calibration/test speakers.

## Notebook cells to implement

### 1. Purpose, configuration, and GPU check

Declare development-only exploratory status. Record the candidate grid, manifest path, output directory, precision, checkpoints, source snapshot, and original frozen policy parameters.

Require `torch.cuda.is_available()` and log GPU name and runtime versions. Use the repository's established environment setup and pinned requirements; do not introduce unrelated package upgrades. FP32 should match the previous GPU experiment.

### 2. Load the actual project implementation

Reuse verified project model loading, preprocessing, teacher solver, student completion, metric wrappers, and seed derivation. Inspect the functions and their signatures before adapting them. Do not create mock model calls or guess imports.

Keep the original benchmark and its outputs intact. Place experiment-specific changes in a separate module if needed and record its checksum.

### 3. Validate manifest and reusable records

Confirm 100 distinct development speaker IDs, two conditions per speaker, and no duplicate candidate keys.

Old rows can be reused only if checkpoints, source behavior, input audio/text, duration, preprocessing, noise realization, precision, seeds, student random tensors or generator convention, and metric definitions match. Prefer source/input hashes and small reproduced anchors at existing indices over merely matching filenames.

If anchors do not match, resolve the cause before mixing records. Never splice CPU quality records into the new GPU grid as though they are one experiment.

### 4. Verify the expanded sampler on a small development subset

Preserve all original semantics except the expanded teacher candidate list:

- Same teacher Sway-Euler grid and CFG convention.
- Same raw-prompt preprocessing and duration inputs.
- Same deterministic initial teacher noise and established candidate-independent student seeding.
- Same endpoint-to-student state construction, including re-noising.
- Fixed student restart j=1 and exactly three student calls.
- Same Vocos decoding and sample rate.

For d2, preserve the endpoint at index 1 and compare it with index 2. For d3, compare index 3 with index 2. Continue consecutive comparisons through index 6. Exclude prompt frames exactly as before.

Teacher time at indices 2 or 3 precedes the fixed student's first consistency time. The endpoint/re-noising procedure must remain the implemented one; do not silently substitute direct state handover or change student restart to make the new candidates work.

Check old candidate outputs and NFE on a small subset before launching the development grid. For a later online scheduler, verify early-stop versus candidate-grid equivalence at any selected exit paths.

### 5. Generate and checkpoint missing candidates

Run batch size 1 under inference mode. Compute the teacher trajectory once per case for candidate-grid collection if appropriate, then run student completion for each candidate with the established matched randomization.

Save each completed row immediately, support resume by unique case/candidate key, and keep errors explicit. Never substitute prompt audio for a missing target recording.

Cached trajectory collection is valid for quality-grid generation. Its elapsed time is **not** genuine online policy latency.

### 6. Evaluate quality

Use the established evaluators:

| Measure | Existing evaluator | Purpose |
| --- | --- | --- |
| WER | Whisper large-v3 and existing text normalization/WER wrapper | Check intended words |
| SIM | microsoft/wavlm-base-plus-sv with cosine similarity | Check speaker similarity against the target recording |
| UTMOS | Existing SpeechMOS UTMOS predictor | Supplemental naturalness estimate |

Honor the actual generated and target sample rates. Apply the original failure definition against each matched GPU Step-8 reference:

`WER_candidate > max(0.125, WER_reference + 0.01) OR SIM_candidate < SIM_reference - 0.02`

Keep these margins unchanged for this diagnostic. Retain robust speaker failure as the maximum across clean and noisy conditions. Passing indicates compliance with these margins, not perfect audio.

### 7. Compute oracle headroom before designing a new rule

For each case, identify the cheapest passing candidate in the expanded grid. Compare the original-grid oracle with the expanded-grid oracle to quantify headroom specifically from indices 2 and 3.

Report how many high-SNR cases pass at index 2, index 3, both, or neither. Check continuous WER/SIM differences as well as binary labels. Do not assume that quality improves monotonically with more teacher steps.

The oracle uses output-quality labels and is an offline diagnostic only. Those labels cannot be inputs to a deployed scheduler.

### 8. Examine whether early signals identify failures

Analyse d2 against index-2 failure and d3 against index-3 failure. Keep this distinct from whether additional waiting later rescues those failures.

Show case counts, distributions, and descriptive ROC-AUC where both passing and failing cases exist. Few failures mean uncertain estimates. Do not infer that the old lambda=0.50 applies at these earlier times.

If exploring multiple rules or thresholds, log the search and assess them with speaker-disjoint development folds where feasible. Clearly label all such results exploratory; they are not new independent certification.

### 9. Compare fair baselines

Reconstruct these policies from the candidate grid:

| Policy | Rule and reason |
| --- | --- |
| Fixed index 8 | Original quality/reference budget |
| Original SNR-only 4/8 | SNR >=20 chooses 4, otherwise 8; main existing competitor |
| SNR-only 2/8 | Tests whether an earlier simple fixed choice captures the gain |
| SNR-only 3/8 | Same fairness check for index 3 |
| Original SNR + drift | Existing {4,5,6,8}, lambda=0.50 method; unchanged version-1 comparator |
| Expanded adaptive candidate | Only if development evidence supports a concrete early-exit rule |
| Expanded oracle | Offline bound on available savings; not deployable |

Report robust failures, clean/noisy failure counts, WER, SIM, UTMOS, mean NFE, and exit counts. A smaller NFE alone is not success if quality risk increases beyond the chosen target.

### 10. Conditional development runtime check

Only after identifying a plausible expanded rule, benchmark that frozen exploratory rule against the simple alternatives on GPU.

Use genuine online teacher stopping and time applicable SNR estimation, teacher execution, decision work, three student passes, and Vocos decoding. Exclude and declare preprocessing, duration prediction, model loading, file writes, and quality evaluators consistently.

Use warm-up, balanced policy order, and CUDA synchronization before start and end timestamps. Count actual model calls. Never infer actual policy runtime from cached candidate student/vocoder timings.

This is a development runtime screen. It does not replace a future independent evaluation of a redesigned policy.

## NFE accounting

Use the existing convention, counting teacher indices 0 through s and three student calls:

| Handover index | Teacher calls | Student calls | Total NFE | Reduction versus index 8 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 6 | 3 | 9 | 57.14% |
| 3 | 8 | 3 | 11 | 47.62% |
| 4 | 10 | 3 | 13 | 38.10% |
| 5 | 12 | 3 | 15 | 28.57% |
| 6 | 14 | 3 | 17 | 19.05% |
| 8 | 18 | 3 | 21 | Reference |

Vocoder decoding is included in timing but excluded from this teacher/student NFE convention.

## Required saved outputs

Use a new directory such as `colab/results/early_handover_dev_gpu/`.

1. `development_expanded_grid.csv`: 1,200 complete candidate rows when merged, with source provenance for reused rows.
2. `early_signal_diagnostics.csv`: drift discrimination counts and summaries.
3. `expanded_policy_comparison.csv`: baseline, original, oracle, and any exploratory-rule outcomes.
4. `early_handover_case_audit.csv`: individual early-exit successes, failures, rescues, and harmful waiting.
5. `experiment_metadata.json`: source hashes, manifest/input provenance, checkpoints, seeds, versions, GPU, precision, and settings.
6. `EARLY_HANDOVER_DEVELOPMENT_REPORT.md`: findings and a proceed/stop recommendation.
7. If runtime screening is reached, save raw policy timing rows separately.

Grid rows should identify speaker, utterance, condition, split, candidate index, estimated SNR, available consecutive drifts, seed, WER, SIM, UTMOS, reference values, failure flags, NFE, duration, source run, and execution provenance. Missing values and failed evaluations must be explicit.

## Decision after the test

- **Earlier candidates rarely pass:** report little headroom in this direction and stop expanding this experiment automatically.
- **Earlier candidates pass, but simple SNR-only 2/8 or 3/8 is as effective:** adaptation has not demonstrated incremental value.
- **Earlier candidates pass, but drift cannot identify them:** do not assume threshold tuning fixes the signal; assess whether a different measurable signal is justified.
- **A rule identifies early exits with a promising risk–cost trade-off:** freeze its definition and plan independent calibration and fresh held-out evaluation before claiming improved performance.

The original 200 calibration and 200 test speakers have already been inspected. They can support explicitly exploratory comparisons, but must not be presented as fresh independent validation for a redesign informed by those results. The original lambda certificate does not automatically transfer to an expanded candidate policy.

## Instruction to the implementer

Create the separate Colab notebook named above using the real repository implementation. Implement the GPU development-grid experiment, checkpointed outputs, diagnostics, and fair baseline comparisons first. Make genuine online timing a conditional second stage. Preserve the original study artifacts and document all changes. Do not claim success, certification, or a better scheduler until the appropriate results exist.

## Extension: Compare three alternative scheduling formulas

Expand the same new notebook to collect signals and evaluate three distinct candidate decision rules. This extension supplements the earlier-index experiment above. It does not establish novelty or replace the completed original policy results.

### Shared definitions

Let E_i be the projected teacher endpoint at index i, restricted to generated frames. Let d_i = norm(E_i - E_(i-1)) / (norm(E_i) + epsilon), matching the original implementation. Calculate scores from predictions available at the current index only; future teacher predictions and output quality labels must never be used by the online stopping rule.

Keep the SNR < 20 dB -> index 8 gate for all three candidates initially. At eligible indices {2,3,4,5,6}, compare the candidate score with its development-selected threshold; use index 8 as fallback. The original lambda=0.50 is not automatically appropriate for these new scores or earlier indices.

### Candidate A: Drift trend

Score:

`q_A(i) = d_i / (d_(i-1) + epsilon)`

Hypothesis: the rate at which endpoint changes shrink may be more informative than a single absolute change. A low ratio can also occur when both changes are large; a near-zero denominator can make the ratio unstable. These are reasons to test the signal, not assume it is a reliable risk estimate.

To evaluate at index 2, preserve endpoints at indices 0, 1, and 2 so d1 and d2 are available. Log the original d_i alongside the ratio. If a combined absolute-drift condition is later explored, record it as a separate candidate variant with its own selection history rather than silently modifying this formula.

### Candidate B: Local endpoint stability

Divide generated frames into fixed contiguous windows. For window k, compute:

`r_(i,k) = norm(E_(i,k) - E_(i-1,k)) / (norm(E_(i,k)) + epsilon)`

Then:

`q_B(i) = percentile_90({r_(i,k)})`

Hypothesis: one global norm may hide a smaller region that is still changing. A high percentile of local changes might identify such cases more selectively.

Choose and record a fixed window length before looking at failure labels. A practical initial setting is 16 mel frames with non-overlapping windows and the remaining final window included. This is an experimental default, not an established optimal value. Document percentile interpolation, epsilon, short-output handling, and near-zero-energy behavior. If window length is tuned, include it in the logged development search and fold procedure.

### Candidate C: Conditioning disagreement

Using the separate teacher predictions already calculated for CFG:

`q_C(i) = norm(v_cond(i) - v_uncond(i)) / (norm(v_cond(i)) + norm(v_uncond(i)) + epsilon)`

Use the pre-guidance conditional and unconditional predictions over generated frames only.

Hypothesis: disagreement may contain information about inputs needing more teacher computation. However, strong disagreement may instead reflect useful conditioning. Determine the direction of association on development training folds; do not presume that larger values mean higher risk. If both stopping directions are explored, count them as separate logged variants.

### What makes these candidate algorithms

A score alone is not a complete scheduler. For each candidate, specify the score, candidate indices, comparison direction, threshold selection, SNR gate, earliest-acceptable-index rule, fallback, and fixed student completion procedure. Freeze these together before later confirmation.

These are proposed formulas for exploration. Their components may have prior uses. Claim novelty only after a focused literature comparison identifies a meaningful distinction; changing constants, combining familiar terms, or assigning a new name is insufficient by itself.

### Efficient parallel comparison on Colab

1. Load one synthesis model instance and generate one shared development candidate grid.
2. Compute and checkpoint A, B, C and original drift during the same teacher trajectory. These scores reuse teacher predictions and require no additional teacher forward calls solely to calculate them, although tensor operations and synchronization have costs.
3. Reuse each candidate output's quality evaluation for every rule that selects that output.
4. Run separate analysis functions for A, B, and C from the shared records. CPU-side analyses can run concurrently if resources permit. On one GPU, do not launch three competing full model copies or time policies concurrently.
5. For genuine policy timing, enable only that policy's required signal calculations. Charging every candidate for all three scores would distort deployment cost. Fixed and SNR-only baselines should perform none of this unnecessary probing.

If old audio/quality rows are reusable but the new signal traces are absent, recover traces with a matched teacher-only pass when possible. Do not regenerate student audio solely because a telemetry column is missing. Verify alignment before joining the recovered signals.

### Fair selection and comparisons

- Use the 100 development speakers only for candidate selection. All conditions belonging to a speaker stay in the same fold.
- Use a small predefined threshold search, with identical fold assignments and comparable tuning budgets across candidates. Report every explored formula, direction, and threshold family.
- Where feasible, use five speaker-disjoint development folds: choose parameters on four folds and evaluate decisions on the remaining fold. These are exploratory estimates, not independent certification.
- Include original SNR-only 4/8, SNR-only 2/8 and 3/8, the original SNR+drift scheduler, and the grid oracle. If other fixed early exits are competitive, include the corresponding simple SNR-gated alternatives too.
- Report whether the adaptive rule improves on the relevant simple policy at comparable risk or computation. Beating only the original 4/8 baseline after giving the adaptive rule earlier exits does not establish that the extra signal helps.
- Do not use WER, SIM, UTMOS, Step-8 quality, future endpoints, or oracle labels as online features. They are offline evaluation targets only.
- With few failure cases, report uncertainty and avoid selecting a winner solely from a tiny difference or the smallest observed failure count. The 5% risk target remains a screening objective until valid independent calibration is performed.

### Additional required outputs

Save `candidate_formula_scores.csv`, `candidate_formula_search_log.csv`, `candidate_formula_fold_results.csv`, and `candidate_formula_comparison.csv` alongside the existing output files. Include:

- Formula identifier and exact implementation settings.
- Per-case/per-index raw scores and original d_i.
- Speaker fold, selected direction, threshold, and exit index.
- Robust and condition-specific failures, WER/SIM changes, and NFE.
- Comparison with each relevant simple baseline.
- True runtime records for promising frozen exploratory rules if timing stage is reached.
- A transparent recommendation: proceed with a particular candidate, insufficient evidence, or no useful improvement found.

### Success criterion and next stage

The target is lower measured runtime at comparable quality risk, or lower failure risk at comparable cost. A more complicated or newly written formula is not itself a successful result.

After development selection, freeze the complete chosen policy, check related work, and arrange independent calibration and fresh held-out evaluation. Preserve the original method as a comparator and do not transfer its certificate to the new rule. No formula is guaranteed to outperform SNR-only.
