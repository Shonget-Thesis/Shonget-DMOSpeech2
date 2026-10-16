# Expanded Handover Scope: Problem-Validation Experiment

## 1. Purpose

This experiment determines the correct scope of the proposed thesis before the proposal.

The central question is:

> Is the relevant DMOSpeech 2 limitation caused only by using one fixed teacher stopping point, or does the fixed student restart level also affect the quality-computation trade-off?

This document validates the **problem scope**. It does not assume that both parameters must become adaptive.

---

## 2. The Two Handover Parameters

### Teacher stopping point \(s\)

Controls when the teacher stops its trajectory and produces the endpoint estimate passed to the student.

### Student restart index \(j\)

Controls where the student starts on its own time/noise grid after receiving the teacher endpoint.

The current reference configuration is treated as:

\[
k_{\mathrm{ref}}=(s=0.25,j=1),
\]

subject to verification against the actual public code and configuration.

The earlier problem-validation experiment varied \(s\) while keeping \(j\) fixed. Therefore, it established preliminary evidence only for teacher stopping-time variability. It did not determine whether student restart level also matters.

---

## 3. Possible Outcomes

The experiment must distinguish among three defensible outcomes.

### Outcome A: Teacher-only limitation

Different teacher stopping points affect quality and computation, but changing the student restart level provides no consistent additional benefit.

Thesis scope:

> Adaptive teacher stopping with a fixed student restart level.

### Outcome B: Fixed restart also matters

Different restart levels produce consistent quality or computation differences even when teacher stopping time is controlled.

Thesis scope:

> Joint teacher-stopping and student-restart configuration.

### Outcome C: Interaction between stopping and restart

The best restart level depends on the selected teacher stopping point. For example, one restart level may be safer after an early teacher stop, while another may be cheaper after a later teacher stop.

Thesis scope:

> An adaptive handover policy with a teacher-step-dependent restart map \(j=\rho(i)\).

If the restart results are inconsistent, weak or unpredictable, choose Outcome A.

---

## 4. Research Questions for This Pilot

1. When student restart level is held fixed, how does teacher stopping time affect WER, speaker similarity, speech quality and computational cost?
2. When teacher stopping time is held fixed, how does student restart level affect the same outcomes?
3. Is there a repeatable interaction between teacher stopping time and student restart level?
4. Does allowing joint \((s,j)\) selection provide a meaningful advantage over selecting only \(s\) with fixed \(j\)?

---

## 5. Pilot Hypotheses

### Teacher stopping effect

- \(H_{0,T}\): Teacher stopping point does not materially affect the quality-computation outcomes when restart level is controlled.
- \(H_{1,T}\): Teacher stopping point materially affects the quality-computation outcomes when restart level is controlled.

### Student restart effect

- \(H_{0,S}\): Student restart level does not materially affect the quality-computation outcomes when teacher stopping point is controlled.
- \(H_{1,S}\): Student restart level materially affects the quality-computation outcomes when teacher stopping point is controlled.

### Interaction effect

- \(H_{0,I}\): The effect of student restart level does not depend on teacher stopping point.
- \(H_{1,I}\): The effect of student restart level depends on teacher stopping point.

These hypotheses are for development-stage scope selection. Final thesis hypotheses may be simplified after the scope is chosen.

---

## 6. Experimental Design

Use a paired factorial experiment in which teacher stopping point and student restart index are varied independently.

Initial configuration grid:

| Teacher stopping argument \(s\) | Restart \(j=1\) | Restart \(j=2\) |
| ---: | :---: | :---: |
| 0.07 | Test | Test |
| 0.10 | Test | Test |
| 0.15 | Test | Test |
| 0.25 | Test | Test |

This produces eight configurations per input.

Only use restart indices confirmed to be valid in the public DMOSpeech 2 implementation. Do not add unsupported restart levels.

Reference:

\[
k_{\mathrm{ref}}=(0.25,1).
\]

---

## 7. Data Requirements

Use development data only.

Minimum recommended pilot:

- at least 10–12 speaker-disjoint development speakers;
- at least one clean case per speaker;
- one controlled noisy condition per speaker if noise robustness remains in scope; and
- identical conditions across all eight configurations.

More speakers are preferable. The purpose is to identify a repeatable configuration pattern, not to claim final generalization.

Do not reuse calibration or final test speakers for design decisions.

---

## 8. Paired-Control Requirements

For every input, hold constant:

- prompt audio;
- prompt and target text;
- duration prediction;
- initial Gaussian latent;
- model and vocoder checkpoints;
- inference precision;
- random seed;
- hardware and software environment; and
- evaluation pipeline.

Only \(s\) and \(j\) may change.

If inference is stochastic, use the same predetermined seeds for every configuration and analyze results at the paired input-seed level.

---

## 9. Metrics

Record for every run:

### Quality

- WER;
- speaker similarity;
- UTMOS, if available;
- absolute quality-threshold failures;
- reference-relative added failures; and
- ASR transcript for forensic checking.

### Computation

- teacher conditional calls;
- teacher unconditional calls;
- student calls;
- total forward passes or NFE;
- warmed-up wall-clock latency;
- real-time factor; and
- peak memory, if available.

### Configuration

- stopping argument \(s\);
- actual teacher stopping time;
- teacher step index;
- student restart index \(j\);
- corresponding student noise level \(\tau\);
- input ID;
- speaker ID;
- condition; and
- seed.

---

## 10. Added-Failure Criterion

For input \(x\) and configuration \(k=(s,j)\), compare against the matching reference run:

\[
f(x,k)=1
\]

when either:

\[
\mathrm{WER}(x,k)>
\max\left(c,\mathrm{WER}(x,k_{\mathrm{ref}})+m_W\right),
\]

or:

\[
\mathrm{SIM}(x,k)<
\mathrm{SIM}(x,k_{\mathrm{ref}})-m_S.
\]

Otherwise, set \(f(x,k)=0\).

Freeze \(c\), \(m_W\), and \(m_S\) before analyzing the pilot.

Always report raw WER and SIM in addition to the binary failure result.

---

## 11. Required Comparisons

### Comparison A: Prove or reject the teacher-only problem

Within \(j=1\), compare:

\[
(0.07,1),\ (0.10,1),\ (0.15,1),\ (0.25,1).
\]

This reproduces the original fixed-teacher-scheduling analysis.

Question:

> Does the cheapest acceptable teacher stopping point vary across inputs when student restart remains fixed?

### Comparison B: Test the student-restart problem

Within every fixed \(s\), compare:

\[
(s,1)\quad\text{against}\quad(s,2).
\]

Question:

> Does changing only the restart level consistently alter quality or cost?

### Comparison C: Test the interaction

Determine whether the difference between \(j=1\) and \(j=2\) changes with \(s\).

Question:

> Is one restart level safer or cheaper at particular teacher stopping points?

### Comparison D: Test the value of expanded scope

For every input, identify:

1. the cheapest non-failing configuration with fixed \(j=1\); and
2. the cheapest non-failing configuration across both \(j=1\) and \(j=2\).

Calculate:

\[
\Delta C_{\mathrm{restart}}
=
C(k^*_{j=1})-C(k^*_{\mathrm{joint}}).
\]

This measures the maximum additional opportunity created by varying restart level.

If \(\Delta C_{\mathrm{restart}}\) is usually zero or negligible, expanded scope is not justified.

---

## 12. Per-Input Decision Table

Create a table like:

| Input | Best non-failing configuration with fixed \(j=1\) | Best non-failing joint \((s,j)\) | Extra saving from restart choice | Restart choice repeatable? |
| --- | --- | --- | ---: | --- |
| Input 1 |  |  |  |  |
| Input 2 |  |  |  |  |

The expanded scope is useful only if the joint choice provides meaningful additional benefit and can be converted into a rule available during inference.

---

## 13. Aggregate Summary Table

Report:

| Configuration | Failure rate | Median WER | Mean SIM | Mean NFE | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| \((0.07,1)\) |  |  |  |  |  |
| \((0.07,2)\) |  |  |  |  |  |
| \((0.10,1)\) |  |  |  |  |  |
| \((0.10,2)\) |  |  |  |  |  |
| \((0.15,1)\) |  |  |  |  |  |
| \((0.15,2)\) |  |  |  |  |  |
| \((0.25,1)\) |  |  |  |  |  |
| \((0.25,2)\) |  |  |  |  |  |

Report clean and noisy conditions separately before reporting any combined result.

---

## 14. Analysis Strategy

### Descriptive analysis

- failure rate and confidence interval per configuration;
- distributions of WER, SIM, NFE and latency;
- win/tie/loss counts for \(j=2\) relative to \(j=1\) within each \(s\);
- frequency of the best restart level at each handover point; and
- additional oracle savings enabled by joint selection.

### Paired analysis

Use paired comparisons because every configuration is evaluated on the same input and seed.

For a sufficiently sized pilot, use an appropriate paired non-parametric test such as the Wilcoxon signed-rank test for continuous metrics. Do not rely only on averages.

### Interaction analysis

A formal two-factor repeated-measures or mixed-effects analysis may be used when the sample size is adequate. For the initial pilot, interaction plots and paired effect differences are acceptable for deciding whether expanded scope deserves further investigation.

Statistical significance alone is insufficient. Require a meaningful computational improvement without an unacceptable increase in failure risk.

---

## 15. Scope-Selection Rules

### Select teacher-only adaptive scheduling when:

- the cheapest acceptable \(s\) varies between inputs;
- changing \(j\) offers little or inconsistent additional benefit;
- \(j=2\) increases quality failures; or
- no inference-time rule can reliably choose between restart levels.

Final scope:

> Adapt teacher stopping time while holding student restart at the verified baseline value.

### Select a fixed alternative restart when:

- one restart index consistently outperforms the original across stopping points;
- it is non-inferior in quality; and
- it reduces cost or improves robustness.

Final scope:

> Adapt teacher stopping time using one empirically selected but globally fixed student restart level.

This is not per-input restart adaptation.

### Select the expanded joint scope when:

- the preferred restart level depends consistently on teacher stopping point;
- the joint configuration provides meaningful additional cost savings;
- quality remains within frozen acceptance criteria;
- the relationship generalizes across development speakers and conditions; and
- a deterministic map \(j=\rho(i)\) can be frozen before calibration.

Final scope:

> Select teacher stopping per input and use a teacher-step-dependent student restart map.

### Do not select full per-input dual adaptation unless:

- observable inference signals can predict both decisions;
- the dataset is large enough to calibrate the expanded policy;
- the improvement clearly exceeds the added complexity; and
- sufficient time remains for independent calibration and held-out evaluation.

---

## 16. Required Integrity Checks

- Verify that \(j=1\) and \(j=2\) use the intended student time indices.
- Count actual teacher and student forward passes.
- Confirm identical seeds and initial latents across paired configurations.
- Confirm no speaker leakage between development, calibration and test sets.
- Confirm all configuration failures use the matching reference input and seed.
- Inspect catastrophic WER values and transcripts for ASR repetition loops.
- Preserve raw values even when robust summaries are added.
- Do not choose the final scope from the final test set.

---

## 17. What the Pilot Can Prove

The pilot can support one of these conclusions:

### Teacher-only conclusion

> Preliminary evidence indicates that teacher stopping requirements vary across inputs, while student restart variation provides no consistent additional advantage. The study will therefore focus on adaptive teacher stopping with a fixed restart level.

### Expanded-scope conclusion

> Preliminary evidence indicates that both teacher stopping time and student restart level affect the quality-computation trade-off, and that the preferred restart level depends systematically on the handover point. The study will therefore investigate a joint handover configuration policy.

The pilot cannot establish final generalization or prove that the final adaptive solution works on unseen speakers. Those claims require frozen calibration and held-out testing.

---

## 18. Recommended Execution Order

1. Verify baseline \(s\), \(j\), \(\tau\), time grids and NFE accounting.
2. Freeze the development speakers, conditions, seeds and quality margins.
3. Run all eight \((s,j)\) configurations per input.
4. Calculate raw metrics and added failures.
5. Test the teacher effect with \(j=1\).
6. Test the restart effect within every \(s\).
7. examine the \(s\times j\) interaction.
8. Compare fixed-restart and joint oracle opportunities.
9. Select teacher-only, fixed-alternative-restart or joint scope.
10. Freeze the restart decision or restart map.
11. Update Chapters 1 and 3 to match the selected scope.
12. Only afterward calibrate the final adaptive policy and evaluate it on untouched test speakers.

---

## 19. Required Deliverable

Create:

`EXPANDED_HANDOVER_SCOPE_PILOT_REPORT.md`

The report must conclude with exactly one selected scope:

1. teacher-only adaptation;
2. teacher adaptation with a newly selected fixed restart;
3. teacher adaptation with a deterministic restart map; or
4. full dual adaptation, only if strongly justified.

The safest default is teacher-only adaptation. Expanded scope should be selected only when the pilot demonstrates repeatable additional value.

