# Review Verdict on the Revised Fixed Handover-Timing Empirical Proof

## Reviewed document

**Source:** `FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`  
**Reported scope:** 12 speakers, 4 controlled conditions, 5 fixed switching policies, and 240 generation runs.

## Overall verdict

The revised report is better than the earlier version because it introduces an absolute WER ceiling, acknowledges the hallucinated reference output, and includes a simple condition-based baseline. However, it still contains multiple numerical contradictions and unsupported statistical claims.

The underlying experimental pattern provides **strong descriptive evidence** of a fixed handover-timing limitation. The report is not yet sufficiently internally consistent to claim that the problem is “fully grounded, statistically significant, and empirically proven.”

The thesis direction remains strong. The required next step is to regenerate every result table and conclusion from one corrected analysis pipeline—not to run another generation experiment immediately.

## What is now stronger

The report correctly attempts to address the earlier reference-contamination problem using:

\[
\operatorname{WER}_{i,t}
\le
\min\left(
W_{\max},
\max\left(3\%,\operatorname{WER}_{i,0.25}+1\%\right)
\right),
\]

with:

\[
W_{\max}=12.5\%.
\]

It also explicitly identifies reference-generation failures rather than assuming that every \(t=0.25\) output is objectively acceptable.

The reported pattern is useful:

- early switching is computationally cheap but fails on some inputs;
- clean and duration-perturbed samples generally do not require \(t=0.25\);
- noisy samples often require more teacher guidance;
- earliest safe times reportedly vary from \(t=0.05\) to \(t=0.25\);
- the oracle reportedly uses less computation than the safer static policies;
- a simple condition-based competitor is now included.

This is the correct structure for investigating whether one fixed handover time is inadequate.

## Critical inconsistency 1: \(t=0.25\) failure rate

The master table reports:

\[
\operatorname{FailureRate}(0.25)=8.3\%.
\]

Because there are 48 condition instances:

\[
8.3\%\approx\frac{4}{48}.
\]

However, the condition-by-condition table reports a 0% failure rate for \(t=0.25\) in every condition:

| Condition | Reported \(t=0.25\) failure rate |
|---|---:|
| Clean | 0% |
| Duration \(-15\%\) | 0% |
| Duration \(+15\%\) | 0% |
| Noisy 15 dB | 0% |

These condition values imply a global failure rate of 0%, not 8.3%.

If all four failures occurred under the noisy condition, the noisy failure rate should be:

\[
\frac{4}{12}=33.3\%.
\]

The raw classifications must determine which result is correct.

## Critical inconsistency 2: oracle mean NFE

The master table reports:

\[
\overline{\operatorname{NFE}}_{\text{oracle}}=9.4.
\]

Later sections repeatedly report:

\[
\overline{\operatorname{NFE}}_{\text{oracle}}=9.0.
\]

This causes several related values to disagree:

| Quantity | Master table | Later claims |
|---|---:|---:|
| Oracle mean NFE | 9.4 | 9.0 |
| \(t=0.25\) NFE difference | 3.65 | 3.96 or 4.0 |
| Compute reduction | approximately 27.7% | 30.5% or 30.8% |

Only one recomputed set of values should appear throughout the report.

## Critical inconsistency 3: corrected and old failure rates are mixed

The corrected master table reports:

| Policy | Corrected master-table failure rate |
|---|---:|
| \(t=0.05\) | 60.4% |
| \(t=0.10\) | 35.4% |
| \(t=0.15\) | 16.7% |

Sections 5 and 6 still use the earlier values:

| Policy | Older value still used |
|---|---:|
| \(t=0.05\) | 58.3% |
| \(t=0.10\) | 33.3% |
| \(t=0.15\) | 12.5% |

This indicates that the narrative was not regenerated after the corrected safety rule was applied.

## Critical inconsistency 4: “only safe fixed policy” is false

The report describes \(t=0.25\) as the only safe fixed policy, but the master table assigns it an 8.3% absolute failure rate.

If that value is correct, the accurate statement is:

> None of the evaluated fixed policies achieved a 0% absolute failure rate.

This may strengthen the motivation for adaptive timing, but it must be stated accurately.

## Critical inconsistency 5: noisy recovery claim

The report says that \(t=0.25\) “restores quality completely,” reducing the noisy failure rate to 0%.

However, the report also:

- acknowledges a noisy hallucination at \(t=0.25\);
- gives \(t=0.25\) an 8.3% global absolute failure rate;
- states that \(t=0.40\) repairs the hallucinated instance.

The safe wording is:

> Extending teacher guidance to \(t=0.25\) substantially reduces early-switch failures under noise but does not necessarily eliminate every absolute quality failure.

## Critical issue 6: statistical significance is not demonstrated

The report labels the results “statistically significant,” but it does not report:

- a named statistical test;
- a test statistic;
- a p-value;
- a confidence interval;
- an alpha level;
- a speaker-clustered bootstrap or paired analysis.

Meeting a descriptive decision rule is not equivalent to demonstrating statistical significance.

Until statistical analysis is added, remove:

- “statistically significant”;
- “overwhelmingly refuted”;
- “mathematically impossible”;
- “formally rejected.”

Use “strong descriptive evidence” instead.

## Critical issue 7: random-seed replication is unsupported

The report states that the pattern repeats across random seeds, but the experimental setup does not identify multiple generation or perturbation seeds per instance.

Four controlled input conditions are not equivalent to four random seeds.

Either provide:

- seed identifiers;
- the number of seeds;
- seed-level results and variability;

or remove the random-seed replication claim.

## Critical issue 8: the corrected ceiling was not genuinely predeclared

The value \(W_{\max}=12.5\%\) was introduced after the extreme 2775% WER result was identified. It is therefore a post-audit correction for this dataset, not a genuinely predeclared criterion.

The report should call it:

> A post-audit corrected evaluation criterion.

For confirmatory evidence, freeze this criterion and apply it unchanged to a new untouched evaluation set.

## Critical issue 9: independent sample size

The report contains 48 condition instances, but these come from 12 speakers repeated across four conditions.

Therefore:

- independent speaker count: \(N=12\);
- repeated condition instances: \(N=48\);
- model generations: \(N=240\).

Statistical analysis should use paired comparisons and cluster or bootstrap by speaker.

## Critical issue 10: naturalness remains unproven

Spectral flux is not a direct naturalness measure. The current experiment measures:

- intelligibility through WER;
- identity preservation through speaker similarity;
- computation through NFE;
- execution time through latency.

It does not establish equal perceived naturalness. A recognized speech-quality metric or blinded listening test is still needed for naturalness claims.

## Required single-source recomputation

Generate all tables and narrative values directly from one corrected JSON-analysis script. The script should produce:

1. absolute pass or failure for every instance-policy pair;
2. relative non-inferiority pass or failure;
3. reference-generation-failure flags;
4. earliest safe switch for every instance;
5. oracle NFE, WER, similarity, and latency;
6. global failure rates;
7. condition-specific failure rates;
8. NFE difference from the oracle;
9. medians and robust averages;
10. paired, speaker-clustered confidence intervals.

The report should import those generated values rather than manually repeating numbers in multiple sections.

## Required consistency checks

Before accepting the regenerated report, verify:

\[
\text{Global failures}
=
\sum_{c}\text{Failures within condition }c.
\]

Also verify:

\[
\overline{\operatorname{NFE}}_{\text{oracle}}
=
\frac{1}{48}\sum_{i=1}^{48}\operatorname{NFE}(t_i^*).
\]

Every percentage in the narrative must exactly match the master table.

## Defensible claim at the current stage

> Across 12 speakers and four controlled conditions, the results provide strong descriptive evidence that no evaluated fixed switching time consistently achieves both low computation and acceptable quality. Early fixed schedules save computation but fail on subsets of difficult or noisy inputs, while later schedules incur additional computation and may also degrade some outputs. Internal inconsistencies in the corrected failure rates, oracle computation, and statistical claims must be resolved before the fixed handover-timing limitation is declared formally established.

## Rating

| Aspect | Rating |
|---|---:|
| Thesis direction | 8.5/10 |
| Experimental design | 8/10 |
| Evidence of timing variation | 8.5/10 |
| Internal numerical consistency | 5/10 |
| Statistical evidence | 4/10 |
| Current defense readiness | **6.5/10** |
| Potential after clean recomputation | **8.5–9/10** |

## Final verdict

The revised report improves the evaluation framework, and the underlying handover-timing problem appears credible. Nevertheless, the document currently combines corrected master-table values with older narrative calculations. It therefore cannot yet support the exact headline claims of formal proof or statistical significance.

Recompute every metric and conclusion from one authoritative analysis pipeline. If the corrected results retain the same pattern—varying safe switch times, unsafe early fixed policies, excess computation from later schedules, and a lower-cost zero-failure oracle—the empirical foundation for the thesis will be strong.
