# Proving the Fixed Handover-Timing Problem in DMOSpeech 2

## 1. Scope of this document

This document focuses only on determining whether DMOSpeech 2's **fixed teacher-to-student switching time** creates a genuine computational limitation.

It does not propose an adaptive algorithm, state-cushioning mechanism, or transition formula. A solution should be designed only after the problem is established.

The study must not begin with the assumption that the handover is broken. Instead, it must test whether using one switching time for every input produces an avoidable trade-off between speech quality and computation.

## 2. Research problem to be tested

> DMOSpeech 2 uses a fixed teacher-to-student switching time for every input. However, different utterances and input conditions may require different amounts of teacher guidance. A fixed schedule may therefore either switch too early and reduce quality or switch too late and perform unnecessary computation.

## 3. Research hypothesis

### Null hypothesis

> A single fixed switching time can provide the best acceptable quality-efficiency balance across all evaluated utterances and conditions.

### Alternative hypothesis

> No single fixed switching time provides the best acceptable quality-efficiency balance across all evaluated utterances and conditions because their earliest safe switching times differ.

Rejecting the null hypothesis establishes the fixed handover-timing problem.

## 4. What must be demonstrated

The problem is supported only when the experiment demonstrates all of the following:

1. Some inputs can safely switch early and save computation.
2. Other inputs experience unacceptable degradation at the same early time.
3. Later switching restores acceptable quality for those difficult inputs.
4. Applying the later time to every input performs unnecessary computation on easy inputs.
5. A per-sample oracle achieves a better quality-computation balance than every safe fixed policy.

In simple terms:

> One fixed bell either dismisses difficult samples too early or keeps easy samples under teacher guidance longer than necessary.

## 5. Current pilot evidence

Existing experiments already show the expected trade-off at a preliminary scale:

| Input condition | Observed useful switch | Main observation |
|---|---:|---|
| Clean inputs | \(t=0.10\) | Similar quality with 9 instead of 13 NFE |
| Duration \(+15\%\) | \(t=0.10\) | 0% WER with less computation |
| Noisy prompt at 15 dB | \(t=0.25\) | WER improved from 18.21% to 7.50% compared with \(t=0.10\) |
| Noisy prompt, speaker similarity objective | \(t=0.40\) | Similarity increased to 0.6821, although WER did not improve over \(t=0.25\) |

These results indicate that:

- fixed \(t=0.10\) is efficient on clean inputs but unsafe under the tested noise condition;
- fixed \(t=0.25\) is more robust under noise but uses unnecessary computation on many clean inputs;
- fixed \(t=0.40\) costs more and does not consistently improve every quality measure.

This is **preliminary evidence**, not yet a general proof, because the stress evaluation used only five utterances and the clean sweep used ten.

## 6. Required validation experiment

### 6.1 Evaluation set

Use at least 20–30 diverse utterances for the next validation stage. A larger final evaluation is preferable when computational resources permit.

The set should include variation in:

- speaker identity;
- target-text length;
- predicted duration;
- phonetic complexity;
- short and long utterances;
- prompt recording quality.

Avoid placing several nearly identical utterances from one speaker entirely within the same evaluation split.

### 6.2 Input conditions

Evaluate every utterance under the following controlled conditions:

1. clean prompt;
2. noisy prompt;
3. predicted duration \(-15\%\);
4. predicted duration \(+15\%\).

Change only one factor at a time. Use the same target text, prompt, initial latent noise, checkpoint, vocoder, and evaluation pipeline across switch-time comparisons.

### 6.3 Fixed switching policies

Generate every sample using:

\[
t_{\text{switch}}\in\{0.05,0.10,0.15,0.25,0.40\}.
\]

These are fixed policies, not an adaptive solution.

| Fixed policy | Purpose |
|---|---|
| \(t=0.05\) | Very aggressive early switch |
| \(t=0.10\) | Strong early-switch baseline |
| \(t=0.15\) | Balanced early baseline |
| \(t=0.25\) | DMOSpeech 2 reference schedule |
| \(t=0.40\) | Extended teacher-guidance baseline |

## 7. Measurements

For every generated output, record:

- WER or CER;
- speaker similarity;
- a naturalness or perceptual-quality metric;
- teacher evaluations;
- student evaluations;
- total NFE;
- model-only latency;
- end-to-end latency;
- random seed and experimental condition.

If possible, add a small blinded listening evaluation. WER and speaker similarity alone do not prove equal naturalness.

Latency measurements should include warm-up runs and repeated timed runs. Report the median and variability rather than relying on one execution.

## 8. Predetermined safety criteria

Define the acceptance criteria before examining the new results. One possible rule is:

\[
\operatorname{WER}_{i,t}\leq
\max\left(3\%,\operatorname{WER}_{i,0.25}+1\%\right),
\]

\[
\operatorname{SIM}_{i,t}\geq
\operatorname{SIM}_{i,0.25}-0.02,
\]

and, when a naturalness metric is available,

\[
Q_{i,t}\geq Q_{i,0.25}-\epsilon_Q.
\]

A switch time is safe only if it passes **all** predetermined quality criteria.

These margins must be justified using prior work, evaluator reliability, or a pilot non-inferiority analysis. They should not be adjusted after observing which policy wins.

## 9. Earliest safe switch

For sample \(i\), define its earliest safe switching time as:

\[
t_i^*=\min\{t:t\text{ satisfies every safety criterion for sample }i\}.
\]

An example result is:

| Sample | Earliest safe time |
|---|---:|
| Easy sample A | 0.05 |
| Easy sample B | 0.10 |
| Medium sample C | 0.15 |
| Difficult sample D | 0.25 |
| Noisy sample E | 0.40 |

Meaningful and repeatable variation in \(t_i^*\) is necessary evidence that different samples require different amounts of teacher guidance.

## 10. Fixed-policy failure rate

For each fixed time \(t\), calculate:

\[
\operatorname{FailureRate}(t)=
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}[t\text{ violates at least one quality criterion for }i].
\]

This reveals whether aggressive fixed policies are unsafe for a meaningful proportion of inputs.

## 11. Computation and regret

Calculate mean computation for each fixed policy:

\[
\overline{\operatorname{NFE}}(t)=
\frac{1}{N}\sum_{i=1}^{N}\operatorname{NFE}_i(t).
\]

Then compare every fixed policy with the per-sample oracle:

\[
\operatorname{ComputeRegret}(t)=
\frac{1}{N}\sum_{i=1}^{N}
\left[\operatorname{NFE}_i(t)-\operatorname{NFE}_i(t_i^*)\right].
\]

For a policy that switches earlier than the safe oracle time, also report its unsafe-early rate:

\[
\operatorname{UnsafeEarlyRate}(t)=
\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[t<t_i^*].
\]

## 12. Required result table

| Policy | Failure rate | Unsafe-early rate | Mean NFE | Compute regret | Median latency | WER | SIM | Naturalness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed \(0.05\) | — | — | 7 | — | — | — | — | — |
| Fixed \(0.10\) | — | — | 9 | — | — | — | — | — |
| Fixed \(0.15\) | — | — | 11 | — | — | — | — | — |
| Fixed \(0.25\) | — | — | 13 | — | — | — | — | — |
| Fixed \(0.40\) | — | — | 16 | — | — | — | — | — |
| Per-sample oracle | 0% by definition | 0% | — | 0 | — | — | — | — |

## 13. Example of evidence that would prove the problem

| Policy | Failure rate | Mean NFE |
|---|---:|---:|
| Fixed \(0.05\) | 30% | 7 |
| Fixed \(0.10\) | 12% | 9 |
| Fixed \(0.15\) | 5% | 11 |
| Fixed \(0.25\) | 0% | 13 |
| Per-sample oracle | 0% | 8.5 |

This pattern establishes the problem because early fixed policies are efficient but unsafe, while the safe fixed policy performs excess computation. The oracle demonstrates that safety and lower computation are simultaneously achievable through per-sample timing.

## 14. Result that would not support adaptive timing

| Policy | Failure rate | Mean NFE |
|---|---:|---:|
| Fixed \(0.10\) | 0% | 9 |
| Per-sample oracle | 0% | 8.5 |

If fixed \(t=0.10\) safely handles nearly every sample and the oracle saves little additional computation, the evidence would support optimizing the fixed schedule rather than developing a complex per-sample switching method.

This outcome must be reported honestly; it would not mean the experiment failed. It would mean that the simpler solution is sufficient.

## 15. Statistical evidence

The final analysis should include:

- confidence intervals for failure rates;
- paired bootstrap confidence intervals for WER, similarity, naturalness, NFE, and latency differences;
- a predeclared non-inferiority margin for quality;
- effect sizes, not only p-values;
- results separated by clean and stressed conditions;
- seed-level variability where multiple noise seeds are used.

The main claim should be based on paired comparisons because every switching policy is evaluated on the same utterances.

## 16. Decision rule for declaring the problem established

The fixed handover-timing problem may be declared empirically established when:

1. earliest safe times vary meaningfully across samples or conditions;
2. early fixed policies produce a nontrivial and repeatable failure rate;
3. the safest fixed policy performs measurable excess computation;
4. the oracle has lower computation than every comparably safe fixed policy;
5. quality remains within predetermined non-inferiority limits;
6. the pattern repeats across utterances, speakers, and random seeds;
7. counted NFE savings correspond to measurable latency savings.

If these conditions are not met, the study must not claim that adaptive handover timing is necessary.

## 17. Defensible wording at the current stage

Based on the existing pilot results, the appropriate statement is:

> Preliminary experiments indicate that fixed teacher-to-student handover timing creates a measurable computation-quality trade-off across the evaluated clean and stressed inputs. Early switching reduces computation on clean inputs, whereas the same early policy degrades intelligibility under noisy conditioning. Larger paired evaluations are required to establish the generality and statistical reliability of this limitation.

Do not currently claim that:

- the teacher-to-student handover is broken;
- a collision occurs during state transfer;
- an adaptive method is already proven necessary;
- speech naturalness is unchanged;
- the pilot percentages generalize to DMOSpeech 2 as a whole.

## 18. Next stage after proof

Only after the decision rule is satisfied should the research proceed to designing a computational method that predicts the safe switching time during a single inference run.

That later method must be evaluated against fixed \(t=0.05\), \(0.10\), \(0.15\), \(0.25\), \(0.40\), and the per-sample oracle. The problem-validation results and the solution-evaluation results must remain separate.

## Final verdict

The current evidence is sufficient for a **pilot problem demonstration**, but not yet for a final generalized claim. The decisive proof requires showing, on a larger paired evaluation, that no single fixed switch time can simultaneously match the oracle's safety and computational efficiency.
