# Latest Critical Verdict on the Fixed Handover-Timing Proof

## Reviewed document

**Source:** `FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF(1).md`  
**Reported scope:** 12 speakers, 4 conditions, 48 condition instances, and 240 generation runs.

## Overall verdict

The revised report is better organized and resolves several earlier inconsistencies. However, it still contains **two decisive validity errors**:

1. the claimed zero-failure oracle contradicts the two instances labeled `None`;
2. the reported Wilcoxon statistics are mathematically impossible for \(N=12\) paired observations.

Consequently, the experiment provides strong descriptive evidence of a fixed handover-timing limitation, but the report does not yet justify claims of formal empirical proof or speaker-level statistical significance.

## Improvements in this version

This revision correctly:

- identifies the 2775% WER event as a repetitive ASR hallucination associated with a failed generated output;
- uses a post-audit absolute WER ceiling;
- separates the 12 speakers from the 48 condition instances;
- reports exact failure counts in the master table;
- makes global fixed-policy failure counts consistent with the condition table;
- uses one oracle mean NFE value of approximately 9.4;
- includes a simple condition-based baseline;
- reports bootstrap intervals for several descriptive statistics.

These changes improve the credibility of the descriptive findings.

## Decisive issue 1: zero-failure oracle contradicts `None` cases

The earliest-safe distribution reports:

- Duration \(+15\%\): `None` for 1 of 12 instances;
- Noisy Prompt 15 dB: `None` for 1 of 12 instances.

Therefore, two of the 48 instances have no evaluated switching time that passes the stated safety criteria:

\[
\frac{2}{48}=4.17\%.
\]

The master table nevertheless reports:

\[
\operatorname{FailureRate}_{\text{oracle}}=0\%.
\]

Both statements cannot be correct.

### Required correction

The report must explicitly define how the oracle handles a `None` case:

1. **Strict safe oracle:** If no policy passes, record an oracle failure. The oracle failure rate becomes at least \(2/48=4.17\%\).
2. **Best-available oracle:** Choose the least harmful output even if it fails. It cannot be called a zero-failure oracle.
3. **Additional safe policy exists:** If an unreported switching time passed, add it to the evaluated candidate set and earliest-safe distribution.

Until one of these corrections is applied, the following claims are unsupported:

- “Oracle achieves 0% failure”;
- “Oracle resolves the dilemma”;
- “per-sample timing guarantees acceptable quality.”

## Decisive issue 2: Wilcoxon statistics are impossible for \(N=12\)

The report describes paired, speaker-level Wilcoxon signed-rank tests with \(N=12\), but reports:

\[
W=957
\]

and:

\[
W=708.
\]

For \(N=12\), the largest possible sum of signed ranks is:

\[
1+2+\cdots+12
=
\frac{12(13)}{2}
=78.
\]

Therefore, neither \(W=957\) nor \(W=708\) can result from a 12-pair Wilcoxon signed-rank test.

The reported values likely came from treating the 48 repeated condition instances as independent pairs. If so, the analysis is not speaker-clustered, contrary to the report.

The reported value:

\[
p=2.28\times10^{-8}
\]

is also incompatible with an ordinary exact two-sided signed-rank test based on only 12 independent pairs.

### Required correction

Choose and document one valid analysis:

- aggregate the four conditions within each speaker and run a paired test on 12 speaker-level values;
- use a speaker-clustered permutation test;
- use a mixed-effects or repeated-measures model;
- use a cluster bootstrap and make confidence intervals, rather than an invalid p-value, the primary evidence.

Report the exact paired vectors used in every statistical test.

## Additional issue 1: Pillar 4 overstates the result

The report says:

> In clean, compressed, and expanded speech, 0% of utterances required \(t=0.25\).

However, Duration \(+15\%\) contains:

- one \(t=0.25\) absolute failure;
- one `None` instance.

The accurate wording is:

> Among instances with an identified safe candidate, none of the clean or duration-perturbed instances had \(t=0.25\) as its earliest safe time.

That does not establish that \(t=0.25\) was unnecessary for the failed `None` instance.

## Additional issue 2: global and condition-specific NFE savings are mixed

The reported \(+3.65\) NFE difference is the global difference between fixed \(t=0.25\) and the reported oracle:

\[
13.0-9.35=3.65.
\]

It should not be described specifically as the amount wasted “on clean inputs.”

Using the Clean Control earliest-safe distribution:

\[
\overline{\operatorname{NFE}}_{\text{clean oracle}}
=
\frac{6(7)+5(9)+1(11)}{12}
=8.17.
\]

The clean-only difference is therefore:

\[
13-8.17=4.83\text{ NFE}.
\]

Report global and condition-specific savings separately.

## Additional issue 3: oracle bootstrap interval

The master table gives the zero-failure oracle a confidence interval of \([0,0]\). A simple empirical cluster bootstrap will always return zero when every recorded oracle label is zero, but this is not a meaningful population-level confidence interval for a true failure probability.

After correcting the `None` cases, report an appropriate clustered interval and clearly distinguish:

- empirical failure proportion on the evaluated dataset;
- uncertainty about performance on unseen speakers.

## Additional issue 4: naturalness remains unmeasured

The experiment evaluates:

- intelligibility using WER;
- speaker identity using similarity;
- computation using NFE;
- execution time using latency.

It does not directly measure perceived naturalness. Avoid naturalness claims unless an accepted speech-quality metric or blinded listening evaluation is added.

## What the experiment currently supports

The following descriptive findings are credible:

1. Earliest safe switching times vary across speakers and conditions.
2. Fixed \(t=0.05\) and \(t=0.10\) frequently fail the stated quality criteria.
3. Noise generally increases the amount of teacher guidance required.
4. Fixed \(t=0.25\) performs more computation than necessary for many clean samples.
5. No evaluated fixed policy achieves zero absolute failure across all 48 condition instances.
6. Per-instance scheduling has the potential to improve the computation-quality trade-off.

These findings are sufficient to continue developing the thesis and testing a scheduler. They are not yet sufficient for a zero-failure or formal significance claim.

## Defensible claim at the current stage

> Across 12 speakers and four controlled conditions, the descriptive results provide strong evidence that safe teacher-to-student switching times vary and that fixed schedules create a computation-quality trade-off. Early fixed schedules reduce computation but fail on subsets of difficult or noisy inputs, while later schedules increase computation and may also degrade some outputs. The oracle handling of instances without any safe candidate and the speaker-level statistical analysis must be corrected before claiming zero oracle failures or formal statistical significance.

## Required next actions

1. Identify the two `None` instances in the raw telemetry.
2. Decide whether the oracle records failure or selects the least harmful failed output.
3. Recompute oracle failure rate, NFE, latency, WER, similarity, and confidence intervals.
4. Aggregate each policy's results to 12 speaker-level paired values or use a proper repeated-measures method.
5. Rerun the statistical analysis and ensure every Wilcoxon statistic is within its valid range.
6. Separate global NFE savings from clean-only savings.
7. Replace the final thesis statement using the recomputed values.

## Rating

| Aspect | Rating |
|---|---:|
| Thesis concept | 8.5/10 |
| Experimental design | 8/10 |
| Descriptive evidence | 8/10 |
| Oracle validity | 5/10 |
| Statistical validity | 3/10 |
| Current defense readiness | **6/10** |
| Potential after correction | **8.5–9/10** |

## Final verdict

The handover-timing problem appears credible and worthy of a thesis. However, the newest report is not yet formally proven because its zero-failure oracle contradicts two `None` cases and its Wilcoxon statistics cannot come from the claimed \(N=12\) paired analysis.

Correct those two issues before presenting the report as statistically established. No additional audio generation is necessarily required at this stage; the immediate task is to repair and rerun the analysis pipeline using the existing telemetry.
