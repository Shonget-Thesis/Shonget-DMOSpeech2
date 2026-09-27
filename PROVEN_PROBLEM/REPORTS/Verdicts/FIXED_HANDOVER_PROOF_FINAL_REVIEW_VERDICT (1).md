# Final Review Verdict on the Fixed Handover-Timing Empirical Proof

## Reviewed document

**Source:** `FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF(2).md`  
**Reported scope:** 12 speakers, 4 controlled conditions, 48 condition instances, and 240 generation runs.

## Overall verdict

This version is substantially stronger than the previous revisions. The major contradictions involving the extreme WER outlier, reference contamination, `None` cases, oracle failure rate, global-versus-condition failure counts, and speaker-level analysis have largely been corrected.

The results now provide a strong empirical foundation for proposing a thesis on adaptive teacher-to-student handover scheduling. Three qualifications should still be incorporated into the final wording:

1. the reported Wilcoxon result is a **one-sided exact test** and should be identified as such;
2. the strict oracle is a retrospective analytical bound, not a deployable scheduling policy;
3. the word “unavoidable” is stronger than the experiment can establish.

With these wording corrections, the fixed handover-timing problem is sufficiently supported for a thesis proposal.

## What is now correctly resolved

### Extreme WER outlier

The 2775% WER case is explicitly audited as a repeated ASR-output hallucination associated with a failed \(t=0.25\) generation. The revised evaluation no longer permits this failed reference to raise the acceptable WER threshold to an unusable value.

### Absolute quality ceiling

The report now applies a post-audit absolute ceiling:

\[
W_{\max}=12.5\%.
\]

This prevents a catastrophically poor reference generation from automatically making poor alternatives appear safe.

### Strict-oracle failure rate

The two instances with no safe candidate are now counted as failures:

\[
\operatorname{FailureRate}_{\text{oracle}}
=
\frac{2}{48}
=4.17\%.
\]

This is consistent with the earliest-safe distributions:

- one `None` case under Duration \(+15\%\);
- one `None` case under Noisy Prompt 15 dB.

### Global and condition-level failure counts

The fixed-policy failure totals now agree with the condition tables:

| Policy | Global failures | Global rate |
|---|---:|---:|
| \(t=0.05\) | 29/48 | 60.4% |
| \(t=0.10\) | 17/48 | 35.4% |
| \(t=0.15\) | 8/48 | 16.7% |
| \(t=0.25\) | 4/48 | 8.3% |
| \(t=0.40\) | 17/48 | 35.4% |
| Strict safe oracle | 2/48 | 4.2% |

### Global and clean-only computation savings

The report now separates global savings from clean-only savings:

- Global oracle mean: approximately 9.35 NFE;
- Global difference from fixed \(t=0.25\): 3.65 NFE or approximately 28.0%;
- Clean-control oracle mean: approximately 8.17 NFE;
- Clean-control difference from fixed \(t=0.25\): 4.83 NFE or approximately 37.2%.

This distinction is methodologically correct.

## Remaining correction 1: identify the Wilcoxon test as one-sided

The report supplies the following 12-speaker difference vector:

\[
[3.5,2.5,4.5,3.5,4.0,-0.5,4.0,1.75,5.0,5.5,5.5,4.5].
\]

For this vector, the verified exact results are:

### One-sided alternative

For:

\[
H_1:\operatorname{median}
\left(
\operatorname{NFE}_{0.25}-\operatorname{NFE}_{\text{oracle}}
\right)>0,
\]

the exact result is:

\[
W_+=77,qquad p=0.00048828125.
\]

This matches the report's \(W=77\), \(p=0.00049\).

### Two-sided alternative

For a conventional two-sided test, the exact result is:

\[
W=1,qquad p=0.0009765625.
\]

Both tests indicate a statistically significant paired difference. The report must state which alternative was used.

### Recommended reporting

> A one-sided exact Wilcoxon signed-rank test across 12 speaker-level paired summaries found lower NFE for the retrospective oracle than for fixed \(t=0.25\), \(W_+=77\), \(p=0.000488\).

If the directional alternative was chosen only after observing the results, describe this test as exploratory. A confirmatory evaluation should specify the direction before examining the validation results.

## Remaining correction 2: distinguish the oracle from the proposed algorithm

The strict safe oracle selects the earliest acceptable switching time after evaluating all candidate schedules. It therefore uses future or counterfactual information unavailable during an ordinary single inference run.

The oracle demonstrates:

> A better computation-quality operating point exists in the evaluated data.

It does not yet demonstrate:

> A deployable algorithm can identify that switching time during inference.

Replace wording such as:

> A per-sample transition policy resolves this trade-off.

With:

> The retrospective per-sample oracle demonstrates that a better trade-off is achievable in principle, motivating the development of an inference-time handover scheduler.

The future scheduler must be evaluated separately on unseen samples.

## Remaining correction 3: replace “unavoidable”

The study evaluated:

- one DMOSpeech 2 configuration;
- 12 speakers;
- four controlled conditions;
- five candidate switching times;
- one post-audit safety definition.

This is sufficient to establish a meaningful limitation within the evaluation scope, but not a universal mathematical impossibility.

Replace:

> an unavoidable computation-quality trade-off

With:

> a significant computation-quality trade-off across the evaluated speakers and conditions.

## What the study now supports

The evidence supports the following conclusions:

1. Safe teacher-to-student switching times vary across the evaluated speakers and conditions.
2. Fixed \(t=0.05\) and \(t=0.10\) reduce computation but produce high failure rates.
3. Noise generally requires additional teacher guidance.
4. Fixed \(t=0.25\) uses more computation than necessary for every evaluated clean-control sample.
5. Fixed \(t=0.25\) is not universally safe under the post-audit absolute criteria.
6. No evaluated fixed policy matches the strict oracle's observed combination of failure rate and computation.
7. Per-input timing is a defensible target for a computational scheduling algorithm.

## What remains unproven

The study does not yet prove that:

- a deployable scheduler can predict the oracle time;
- the results generalize beyond the evaluated speakers and conditions;
- naturalness is preserved, because perceptual naturalness was not directly measured;
- a complex scheduler will outperform simpler rules on unseen data;
- the retrospective oracle's performance can be achieved without testing multiple schedules.

## Final defensible thesis statement

> Across 12 speakers and four controlled conditions, the results provide strong descriptive evidence that safe teacher-to-student switching times vary across inputs. A one-sided exact Wilcoxon signed-rank test on speaker-level summaries found significantly lower NFE for the retrospective oracle than for fixed \(t=0.25\) (\(W_+=77\), \(p=0.000488\)). Early fixed schedules reduced computation but exhibited failure rates of 35.4%–60.4%, while fixed \(t=0.25\) used 4.83 additional NFE on clean speech and retained an 8.3% overall failure rate. The strict oracle reduced the observed failure rate to 4.2% while lowering global mean computation by 28.0%, demonstrating that improved per-input scheduling is achievable in principle and motivating the development of an inference-time handover scheduler.

## Required next research stage

The next stage is to build an **Adaptive Teacher-to-Student Handover Scheduler** that predicts whether the sample should switch or continue teacher guidance during a single inference run.

It must be compared with:

- fixed \(t=0.10\);
- fixed \(t=0.15\);
- fixed \(t=0.25\);
- the simple condition-based baseline;
- the retrospective strict oracle.

Evaluation should use unseen speakers or utterances and should report:

- failure rate;
- unsafe-early rate;
- mean NFE;
- actual latency;
- WER;
- speaker similarity;
- a naturalness metric or blinded listening evaluation;
- regret relative to the retrospective oracle.

## Rating

| Area | Rating |
|---|---:|
| Problem relevance | 9/10 |
| Experimental design | 8.5/10 |
| Numerical consistency | 8.5/10 |
| Statistical evidence | 8/10 after identifying the one-sided test |
| Proposal readiness | **9/10** |
| Current final-thesis readiness | **7/10** |
| Overall thesis potential | **9/10** |

## Final verdict

The fixed handover-timing problem is now sufficiently supported for a thesis proposal. The evidence demonstrates substantial variation in safe switch times and a statistically supported computational gap between fixed \(t=0.25\) and the retrospective oracle.

The oracle is not the final solution. It establishes the achievable target. The actual thesis contribution will be the inference-time scheduling algorithm and evidence that it approaches this target better than strong fixed and simple condition-based baselines.
