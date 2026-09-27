# Critical Verdict on the Fixed Handover-Timing Validation Results

## Reviewed document

**Source:** `EMPIRICAL_VALIDATION_RESULTS_SUMMARY.md`  
**Reported scope:** 12 speakers, 4 controlled conditions, 5 fixed switching policies, and 240 generation runs.

## Overall verdict

The new experiment provides **strong and promising preliminary evidence** that fixed teacher-to-student handover timing creates a computation-quality trade-off. However, the current report does **not yet constitute airtight or formal proof** because one extreme WER result contaminates the reference-based safety classification and several claims are stronger than the analysis supports.

The thesis direction remains viable and should continue. The report must first be corrected and recomputed before it is used as formal evidence in a proposal or defense.

## What the experiment successfully demonstrates

The global pattern is the type of evidence required to motivate adaptive handover scheduling:

| Fixed policy | Reported failure rate | Mean NFE | Interpretation |
|---|---:|---:|---|
| \(t=0.05\) | 58.3% | 7 | Efficient but highly unsafe |
| \(t=0.10\) | 33.3% | 9 | Efficient but unsafe for many cases |
| \(t=0.15\) | 12.5% | 11 | Strong static baseline but still fails under stress |
| \(t=0.25\) | 0% under the relative rule | 13 | Reference policy; safe by construction but more expensive |
| \(t=0.40\) | 31.2% | 16 | Expensive and not consistently better |
| Per-sample oracle | 0% | 9 | Reported quality-efficiency upper bound |

The condition-level results also show a useful distinction:

- \(t=0.15\) reportedly passes every clean and duration-perturbed case;
- the same \(t=0.15\) policy fails 50% of noisy cases;
- some noisy cases require \(t=0.25\), while other samples can switch at \(t=0.05\), \(0.10\), or \(0.15\);
- the reported earliest safe times vary across speakers and conditions.

This supports the central observation:

> An early fixed policy can save computation but fail on difficult inputs, while a later fixed policy can protect quality but perform unnecessary computation on easier inputs.

## Critical issue 1: the 2775% WER result

Instance 44 reports:

\[
\operatorname{WER}_{t=0.25}=2775\%.
\]

This single value dominates the reported averages:

- noisy \(t=0.25\) mean WER: **234.05%**;
- global \(t=0.25\) mean WER: **59.15%**.

Removing only that observation gives approximate diagnostic values of:

- **3.05%** mean WER for the remaining 11 noisy \(t=0.25\) instances;
- **1.37%** mean WER for the remaining 47 global \(t=0.25\) instances.

WER can exceed 100% when a transcript contains many insertions. Nevertheless, 2775% is extreme and must be audited manually before the result is accepted. Possible causes include:

- a genuinely repetitive or hallucinated generated output;
- an incorrect reference-output pairing;
- malformed or empty reference text;
- normalization or tokenization failure;
- repeated text produced by the ASR evaluator;
- a transcription-processing bug.

The audio, reference text, ASR hypothesis, token counts, and WER calculation for this instance must be included in the audit.

## Critical issue 2: the safety rule becomes invalid when the reference fails

The report defines an acceptable WER using:

\[
\operatorname{WER}_{i,t}
\le
\max\left(3\%,\operatorname{WER}_{i,0.25}+1\%\right).
\]

For instance 44, this becomes:

\[
\operatorname{WER}_{i,t}\le2776\%.
\]

Therefore, almost any output is classified as safe. This is why the instance is assigned an earliest safe time of \(t=0.05\), even though its reference generation is catastrophically poor.

This affects:

- earliest-safe labels;
- oracle decisions;
- fixed-policy failure rates;
- oracle mean NFE;
- oracle WER and similarity;
- compute-regret calculations.

## Required safety-rule correction

Use both a relative non-inferiority rule and an absolute quality ceiling:

\[
\operatorname{WER}_{i,t}
\le
\min\left(
W_{\max},
\operatorname{WER}_{i,0.25}+\Delta_W
\right).
\]

For example:

\[
W_{\max}=10\%, \qquad \Delta_W=1\%.
\]

The exact absolute ceiling must be justified before recomputing the results. It should not be selected merely to obtain a preferred outcome.

If the \(t=0.25\) reference itself exceeds the absolute quality ceiling, label the instance as:

> **Reference-generation failure**

Such an instance should be analyzed separately. It must not automatically make every alternative schedule safe.

## Critical issue 3: zero failure at \(t=0.25\) is partly tautological

Because every alternative is compared with \(t=0.25\), the reference necessarily passes its own relative comparison. Therefore, the report should separate:

1. **relative non-inferiority failure**, measuring whether an alternative is worse than the reference;
2. **absolute quality failure**, measuring whether the generated output is acceptable at all.

The claim that \(t=0.25\) has a 0% failure rate is defensible only for the relative criterion—not as proof that every \(t=0.25\) output is objectively good.

## Critical issue 4: formal hypothesis rejection is not yet shown

The report states that the null hypothesis was “formally rejected,” but it does not present:

- a statistical test;
- a confidence interval;
- a significance level;
- a predefined statistical rejection rule;
- a clustered analysis accounting for repeated speakers.

The current defensible wording is:

> The descriptive empirical results do not support the null hypothesis and provide evidence of a fixed handover-timing limitation.

“Formally rejected” should be used only after completing the declared statistical analysis.

## Critical issue 5: the observations are clustered

The 48 evaluated instances come from 12 speakers observed under four conditions. They are not 48 fully independent speakers.

The analysis should therefore:

- identify the independent speaker count as \(N=12\);
- use paired comparisons because every policy is evaluated on the same instances;
- cluster or bootstrap by speaker rather than treating all 48 rows as independent;
- report condition-specific results separately.

## Critical issue 6: negative compute regret is misleading

The report gives negative compute regret to \(t=0.05\) and \(t=0.10\). These policies use fewer evaluations than the oracle because they sometimes switch unsafely.

This should be renamed **NFE difference from the safe oracle**. Alternatively, define a combined regret function that penalizes both computation and quality failure. A method should not appear better than the oracle merely because it saves computation by producing unacceptable outputs.

## Critical issue 7: naturalness remains unmeasured

Spectral flux is not a direct naturalness metric. The report currently provides evidence about:

- intelligibility through WER;
- speaker identity through similarity;
- computation through NFE;
- execution time through latency.

It does not establish perceptual naturalness or the absence of audible artifacts. Add an accepted speech-quality metric, a blinded listening test, or both before making naturalness claims.

## Important baseline for the future solution

The condition-level results suggest a simple policy may already be competitive:

```text
If the prompt is noisy:
    switch at t = 0.25
Otherwise:
    switch at t = 0.15
```

Any future adaptive scheduling algorithm must be compared not only with fixed \(t=0.25\), but also with:

- fixed \(t=0.10\);
- fixed \(t=0.15\);
- the simple condition-based policy;
- the corrected per-sample oracle.

If the simple condition-based rule performs equally well, a complicated readiness formula would not be justified.

## Required corrections before claiming proof

1. Audit instance 44 and explain the 2775% WER.
2. Verify the reference text, ASR hypothesis, token counts, and WER implementation.
3. Add a predetermined absolute WER ceiling.
4. Mark failed reference generations separately.
5. Recompute all safety labels and earliest-safe times.
6. Recompute the oracle, failure rates, NFE differences, and latency results.
7. Separate relative non-inferiority from absolute quality failure.
8. Report median WER and robust summaries alongside the mean.
9. Add paired, speaker-clustered confidence intervals or hypothesis tests.
10. Add a naturalness or perceptual-quality evaluation.
11. Compare against the simple clean/noisy condition-based schedule.

## Safe thesis claim before correction

> Across 12 speakers and four controlled conditions, the descriptive results show that early fixed handover schedules reduce computation but violate the study's current quality criteria on a subset of inputs, particularly under noisy conditioning. Later schedules reduce some of these failures but require additional computation on clean inputs. These findings provide strong preliminary evidence of a fixed handover-timing trade-off, although an extreme reference WER result and the resulting reference-relative safety classification must be corrected before the limitation is claimed as formally established.

## Rating

| Category | Assessment |
|---|---:|
| Experimental scope | 8.5/10 |
| Observed timing variation | 8.5/10 |
| Evidence of a computation-quality trade-off | 8/10 |
| Current metric validity | 6/10 |
| Current statistical support | 5.5/10 |
| Current proof readiness | 7.5/10 |
| Potential after correction | **8.5–9/10** |

## Final verdict

The core empirical pattern is promising and likely meaningful: no aggressive fixed schedule is simultaneously cheap and safe across all tested conditions, while \(t=0.25\) appears unnecessarily expensive for many clean and duration-perturbed inputs. This is a credible foundation for an adaptive handover-scheduling thesis.

However, the report is not yet defense-ready proof. The 2775% WER result and the reference-relative safety rule currently make at least one oracle label invalid and may affect aggregate claims. Correcting the metric pipeline and recomputing the analysis is mandatory before stating that adaptive timing is proven necessary.
