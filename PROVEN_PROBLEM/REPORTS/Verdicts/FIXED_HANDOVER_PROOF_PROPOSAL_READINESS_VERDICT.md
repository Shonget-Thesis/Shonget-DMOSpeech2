# Proposal-Readiness Verdict for the Fixed Handover-Timing Problem

## Reviewed document

**Source:** `FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF(3).md`

## Overall verdict

The fixed handover-timing problem is now **sufficiently supported for a thesis proposal**. The revised report correctly addresses the major methodological problems identified in earlier versions:

- the extreme 2775% WER case is audited;
- an absolute WER ceiling is applied;
- failed reference generations are identified;
- the two `None` cases are counted as strict-oracle failures;
- fixed-policy failure totals match their condition-level counts;
- global and clean-only computation savings are separated;
- the statistical comparison uses 12 speaker-level paired summaries;
- the oracle is explicitly described as retrospective;
- the Wilcoxon test is correctly identified as one-sided.

The experiment now provides strong evidence that safe switching times vary across inputs and that one fixed handover time produces a computation-quality trade-off within the evaluated speakers and conditions.

## Key validated results

| Policy | Failure rate | Mean NFE |
|---|---:|---:|
| Fixed \(t=0.05\) | 60.4% | 7.0 |
| Fixed \(t=0.10\) | 35.4% | 9.0 |
| Fixed \(t=0.15\) | 16.7% | 11.0 |
| Fixed \(t=0.25\) | 8.3% | 13.0 |
| Fixed \(t=0.40\) | 35.4% | 16.0 |
| Condition baseline | 8.3% | 11.5 |
| Strict retrospective oracle | 4.2% | 9.4 |

The strict oracle reduces global mean computation by approximately:

\[
13.0-9.35=3.65\text{ NFE},
\]

or approximately 28.0% relative to fixed \(t=0.25\).

For clean-control samples, the retrospective oracle uses approximately 8.17 NFE, giving:

\[
13.0-8.17=4.83\text{ NFE},
\]

or approximately 37.2% clean-only computation savings.

## Statistical result

Using the supplied 12-speaker difference vector, the one-sided exact Wilcoxon signed-rank result is:

\[
W_+=77,qquad p=0.000488.
\]

This supports the directional alternative:

\[
H_1:
\operatorname{median}
\left(
\operatorname{NFE}_{0.25}-\operatorname{NFE}_{\text{oracle}}
\right)>0.
\]

The report should describe the test as exploratory if the one-sided alternative was selected after examining the data. A future confirmatory evaluation should declare the direction beforehand.

## Three final corrections

### 1. Correct the denominator of the \(t=0.25\) failure claim

The report currently states that fixed \(t=0.25\) “fails on 8.3% of adverse inputs.” However:

\[
8.3\%=\frac{4}{48},
\]

which is the failure rate across all evaluated condition instances.

The condition-specific rates are:

- Duration \(+15\%\): \(1/12=8.3\%\);
- Noisy Prompt 15 dB: \(3/12=25.0\%\);
- Duration \(+15\%\) and noisy combined: \(4/24=16.7\%\).

Use:

> Fixed \(t=0.25\) fails on 8.3% of all evaluated condition instances, including 25% of noisy-prompt instances.

### 2. Correct the condition-status labels

Duration \(+15\%\) should not be labeled generally as requiring \(t\ge0.25\), because 11 of its 12 instances have an identified safe time at or before \(t=0.15\), while one instance has no safe candidate.

Recommended labels:

| Condition | Recommended status |
|---|---|
| Clean Control | Early switching sufficient for all evaluated instances |
| Duration \(-15\%\) | Early switching sufficient for all evaluated instances |
| Duration \(+15\%\) | Mostly early-safe, with one unresolved failure |
| Noisy Prompt 15 dB | Highly heterogeneous; some inputs require later guidance |

### 3. Define how `None` cases contribute to oracle NFE

The strict oracle records two failures but still reports a mean of 9.4 NFE. The report must specify which candidate time contributes to NFE and latency for those failed cases.

A reproducible rule could be:

> If no candidate passes all safety criteria, record an oracle failure and select the candidate with the smallest predefined quality-loss score for NFE and latency accounting.

For example:

\[
t_i^{\text{fallback}}
=
\arg\min_t
\left[
\lambda_W\widetilde{WER}_{i,t}
+
\lambda_S\left(1-\widetilde{SIM}_{i,t}\right)
\right].
\]

The weights, normalization, and tie-breaking rule must be declared. Alternatively, use a fixed fallback such as \(t=0.25\) or \(t=0.40\). The report must document whichever rule actually produced the reported 9.4 NFE.

## Final defensible thesis statement

> Across 12 speakers and four controlled conditions, the results provide strong descriptive evidence that safe teacher-to-student switching times vary across inputs. A one-sided exact Wilcoxon signed-rank test on speaker-level summaries found significantly lower NFE for the retrospective oracle than for fixed \(t=0.25\) (\(W_+=77\), \(p=0.000488\)). Early fixed schedules reduced computation but exhibited failure rates of 35.4%–60.4%, while fixed \(t=0.25\) used 4.83 additional NFE on clean speech and failed on 8.3% of all evaluated condition instances, including 25% of noisy-prompt instances. The strict oracle reduced the observed failure rate to 4.2% while lowering global mean computation by 28.0%, demonstrating that improved per-input scheduling is achievable in principle and motivating the development of an inference-time handover scheduler.

## What happens next

After applying the three corrections, freeze the problem-validation stage. The next research phase is implementing an **Adaptive Teacher-to-Student Handover Scheduler**.

The scheduler must be evaluated against:

- fixed \(t=0.10\);
- fixed \(t=0.15\);
- fixed \(t=0.25\);
- the condition-based baseline;
- the retrospective strict oracle.

The oracle is not the proposed solution. It is the upper-bound target showing what per-input scheduling could potentially achieve.

## Rating

| Area | Rating |
|---|---:|
| Problem evidence | 9/10 |
| Numerical consistency | 8.5/10 |
| Statistical presentation | 8.5/10 |
| Proposal readiness | **9/10** |
| Current final-thesis readiness | **7/10** |
| Overall thesis potential | **9/10** |

## Final conclusion

The fixed handover-timing problem is adequately established for a thesis proposal. No additional problem-discovery experiment is immediately necessary. Correct the three remaining documentation issues, freeze the validation report, and proceed to implementing and testing the inference-time scheduler.
