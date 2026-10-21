# Verdict: Fix the Three-Revisions Notebook Before the Colab Run

Reviewed attachments:

- DMOSpeech2_Three_Scheduler_Revisions_Comparison.ipynb
- DMOSpeech2_Three_Scheduler_Revisions_Comparison.md

## Decision

**Not ready for Run all yet.** Fix the confirmed syntax errors, reconcile the reconstructed original baseline, and verify the external runner implementation before launching the experiment. These pre-run corrections do not require new audio generation.

The experiment's scope remains valid: compare three scheduler revisions and shared baselines on the same 100 development speakers, with clean/noisy conditions, in Google Colab. The attached notebook delegates most substantive work to an external script; reviewing the notebook alone cannot certify that implementation.

## 1. Confirmed syntax errors

Three code cells contain unterminated string literals: notebook array indices 15, 17, and 19, corresponding to the cross-validation summary, paired-comparison summary, and case-audit sections.

For example, replace a literal newline inside an ordinary quoted string with:

```python
print("\n=== Selected Parameters Per Fold ===")
```

Apply the equivalent fix to all three affected print statements. Validate Python syntax for the code cells, accounting for legitimate Colab/IPython shell commands and magics separately. The static check found these errors without running inference.

## 2. Supply and verify the actual runner script

The notebook imports these operations from `run_three_scheduler_revisions_dev.py`:

- Input validation and case dictionary construction.
- Baseline and revision policy functions.
- Cross-validation and parameter selection.
- Unified comparisons and case audits.
- Conditional real GPU timing.
- Report generation.

That script was not included in the reviewed attachments. Ensure the exact file exists on the checked-out source snapshot and provide it for review, together with any local helper modules it depends on. A prose report describing its behavior does not replace the source.

Record the executed source commit and file hashes. Confirm that configuration values, including risk target, random seed, fold count, candidates, and timing sample, are passed to or consistently read by the functions. Several notebook imports currently call functions without passing the top-level configuration, so the settings cannot be verified from the notebook alone.

## 3. Replace the ineffective causality check

The current cell passes complete candidate records into three policies and prints a PASS message. It does not hide future values, despite its comment. This does not demonstrate that decisions use only current information.

Implement a meaningful check that blocks or raises on future step/feature access. In particular, a CSV row at index 2 may contain columns d3 through d6; reading them to choose an index-2 exit would leak future computation. Also forbid quality labels, WER, SIM, UTMOS, and reference quality as online inputs.

Test policy behavior at representative early-exit and fallback paths. Print PASS only after executable checks establish the claimed property. Report missing source/provenance evidence as unverified rather than passing it by default.

## 4. Reconcile the original scheduler baseline

The new audit document reports original SNR+drift at approximately **17.31 NFE and 3/100 failures**. The earlier checked development results were **18.05 NFE and 1/100 failures**.

Before interpreting any improvement:

1. Replay the exact original SNR gate, candidate indices {4,5,6}, drift column at each index, threshold 0.50, and index-8 fallback.
2. Preserve the original failure definition and matched GPU references.
3. Compare per-speaker selected indices, NFE, and failure flags with the saved original case audit.
4. Explain every discrepancy and regenerate dependent comparison tables from the corrected implementation.

Do not assume the new values represent an improved measurement. They may indicate changed settings, changed records, or an implementation error; the source and case records must establish the cause.

Other reported metric means also differ from the earlier development report. For example, Candidate A's new displayed clean WER/SIM values differ materially from the earlier grid-derived means. Trace these to the same 100 speakers and selected rows before accepting the new report's results. This is an analysis reconciliation task, not automatic evidence that synthesis must be repeated.

## 5. Make timing genuinely conditional and policy-specific

The visible timing condition checks CUDA availability and a boolean flag, not whether a credible revision was selected. The timing function receives no explicit selected revision parameters or out-of-fold decisions, so its exact policy binding is not visible in the notebook.

Required implementation details:

- Define a development-screen inclusion rule and export which policies qualify.
- Bind each timed run to an explicit policy definition and parameter set. If timing fold-selected rules, use the assigned speaker's fold rule and label that clearly; do not silently replace it with one globally selected setting.
- Measure real online stopping through teacher, relevant decision work, fixed student completion, and Vocos decoding.
- Use CUDA synchronization, warm-up, balanced policy order, matching inputs, actual call counts, and one policy at a time on the shared GPU.
- Charge each policy only for its required computations, without unnecessary multi-formula telemetry overhead.
- Declare exclusions consistently. Cached candidate endpoints or precomputed selected steps do not establish online scheduler latency.

The current configuration sets **50 timing cases**, fewer than the full 200 development cases. A small runtime screen is acceptable if explicitly labeled and selected without cherry-picking. Save the same case IDs for all policies, preferably using complete speaker clean/noisy pairs. Do not present that timing sample as the full development cohort or the independent test cohort.

## 6. Make setup and validation reliable

- Resolve a single project root and use absolute paths consistently; the current branch checkout/pull can execute outside the repository on repeated notebook runs.
- Verify shell-command exit status instead of printing environment readiness unconditionally.
- Use the project's known dependency setup and record versions. The shown notebook does not include dependency installation for its external runner.
- Pin a source snapshot for a reproducible experiment rather than silently changing it through git pull on each execution.
- Stop immediately if required input validation fails, for example by explicitly raising when `validation_passed` is false.
- Define whether provenance/causality checks are passed, failed, or not verified. Source labels and nonmissing columns do not by themselves prove compatibility or causality.
- Check the output directory after changing the working directory so output files are not accidentally split across two locations.

## 7. Describe the selection procedure accurately

The visible explanation calls evaluation on the 80 training speakers “inner selection,” but it describes direct training-risk minimization rather than an inner cross-validation procedure. The runner must show which method is actually implemented.

Use a small logged search, speaker-separated outer evaluation, and training-only parameter/direction selection. If there is no inner validation, describe that honestly rather than claiming nested cross-validation. Repeated inspection of these development data makes the results exploratory regardless of fold separation.

Save actual fold assignments and per-case out-of-fold decisions. Report every explored revision and baseline, including unfavorable outcomes. Explain whether the claimed 11-policy table includes oracle and fold-selected Candidate A variants; distinguish them from fixed deployable policies when counting timing runs.

## 8. Correct the audit document's status and claims

- Remove “pre-registered” unless supported by a genuine dated registration with the relevant scope. A versioned exploratory specification is a more accurate description when applicable.
- Distinguish prior local exploratory results from the planned Colab run. Do not describe pending notebook execution as completed empirical verification.
- Index 8 is the reference, not a proven upper bound on quality.
- Do not claim no leakage, full provenance verification, or online causality solely because the report lists PASS.
- Differences in NFE establish computation counts, not measured runtime improvements.
- Keep conclusions provisional until baseline reconstruction and metric aggregation agree with raw records.

## Minimal next steps

1. Fix the three broken string literals.
2. Supply the real runner script and helpers, and resolve the original-baseline discrepancy.
3. Implement effective causality checks and stop-on-failed-validation behavior.
4. Confirm source setup, selection, and exact policy definitions used for timing.
5. Run the Colab data-analysis stages first; launch conditional real timing only after those checks pass.

## Final verdict

The design can proceed after these concrete execution and comparison issues are resolved. The current notebook should not yet receive a blanket Run all approval. No new full audio-generation run is justified merely to fix the identified syntax, configuration, or analysis problems. The external implementation remains necessary for a complete pre-run audit.
