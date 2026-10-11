# Adaptive Scheduling: Proof Status and Remaining Validation

## Overall verdict

The adaptive scheduling problem has **strong preliminary empirical support**, but it is not yet fully validated for final thesis conclusions.

The completed experiments support the claim that a single fixed handover time is not optimal for every evaluated input. However, the adaptive scheduler still requires evaluation on an independent, unseen test set before its effectiveness and generalizability can be claimed confidently.

---

## Evidence already established

The fixed-handover experiment evaluated 48 controlled instances and found different earliest quality-preserving switching times:

| Earliest safe switching time | Number of instances |
|---|---:|
| $t=0.05$ | 20 |
| $t=0.10$ | 13 |
| $t=0.15$ | 9 |
| $t=0.25$ | 6 |

This supports the central scheduling premise:

$$
\text{Different evaluated inputs required different safe handover times.}
$$

Aggressive fixed schedules also produced failures:

| Fixed schedule | Failure rate | Mean NFE |
|---|---:|---:|
| $t=0.05$ | 58.33% | 7 |
| $t=0.10$ | 33.33% | 9 |
| $t=0.15$ | 12.50% | 11 |
| $t=0.25$ | 0% relative to itself | 13 |

These results indicate that:

- the original $t=0.25$ schedule is safe relative to its own output but computationally conservative for many instances;
- one globally early switch saves computation but fails on some inputs;
- per-input scheduling is empirically justified on the evaluated dataset.

The calibrated hybrid scheduler achieved:

- zero reference-adjusted failures;
- mean base NFE of 11.21;
- mean effective NFE of 11.48 after probe overhead;
- approximately 11.7% effective NFE savings relative to the 13-NFE reference;
- median WER of 0;
- mean speaker similarity of approximately 0.8044.

The current defensible claim is:

> On the evaluated 48-instance dataset, the adaptive scheduler preserved quality relative to each instance's fixed-handover reference while reducing effective neural function evaluations by approximately 11.7%.

---

## Why the evidence is not yet final

### 1. Possible calibration–evaluation overlap

If the scheduler thresholds and decision rules were developed using the same 48 instances on which performance was reported, the results measure calibration performance rather than independent generalization.

A valid final evaluation requires:

$$
\text{calibration set}\ne\text{held-out test set}.
$$

The scheduling policy must be frozen before the held-out test results are examined.

### 2. Limited evaluation coverage

The current 48 instances provide useful feasibility evidence, but they do not guarantee generalization to:

- unseen speakers;
- new sentences;
- different random seeds;
- other languages;
- emotional speech;
- additional noisy or out-of-domain prompts.

### 3. Suspicious 2775% WER outlier

One $t=0.25$ output reportedly produced a WER of 2775%, heavily distorting the mean WER for that policy. The sample must be audited to determine whether it represents:

- repeated or looping generated speech;
- an ASR failure;
- incorrect WER normalization;
- corrupted reference or hypothesis text;
- or a genuine catastrophic baseline output.

The result must not be silently removed. Any correction or exclusion requires a predetermined, documented rule.

### 4. Reference-relative safety is not absolute quality

The scheduler reported zero reference-adjusted failures, but four outputs violated the strict absolute quality ceiling.

An output can pass a relative test by matching a poor $t=0.25$ reference. Therefore, the final thesis must report both:

- non-inferiority relative to the original fixed reference; and
- compliance with absolute quality requirements.

### 5. The scheduler must beat the strongest fixed alternative

The appropriate competitor is not only DMOSpeech 2's reported fixed $t=0.25$ schedule. The scheduler must also be compared with the best global fixed schedule discovered using calibration data.

The current likely competitor is fixed $t=0.15$:

- fixed $t=0.15$: 11 mean NFE and 12.5% failures;
- calibrated adaptive scheduler: 11.48 effective NFE and zero reference-adjusted failures.

This is promising, but it must be reproduced on unseen data with statistical uncertainty reported.

### 6. NFE savings must translate into real efficiency

Reduced NFE does not automatically guarantee proportional deployment improvement. The final evaluation must also measure:

- end-to-end latency;
- Real-Time Factor;
- scheduler and diagnostic-probe overhead;
- teacher and student runtime separately;
- peak memory, if deployment efficiency is claimed.

---

## Required final validation

### Step 1: Freeze the scheduler

Freeze all:

- thresholds;
- decision rules;
- fallback behavior;
- quality ceilings;
- feature definitions;
- candidate switch times.

Do not modify the scheduler after examining held-out results.

### Step 2: Construct an unseen test set

Use speakers, texts, and random seeds that were not used to design the scheduler. Include the same predetermined condition categories where appropriate.

Repeated seeds from the same speaker–text pair must not be treated as completely independent observations.

### Step 3: Compare the required methods

Evaluate:

1. DMOSpeech 2 fixed $t=0.25$;
2. best global fixed schedule selected using calibration data;
3. proposed adaptive scheduler;
4. per-sample oracle, used only as an upper-bound reference.

### Step 4: Report quality and efficiency

At minimum, report:

- WER or CER;
- speaker similarity;
- naturalness estimator, if available;
- strict absolute failure rate;
- reference-adjusted failure rate;
- teacher, student, and effective total NFE;
- latency and RTF;
- probe overhead;
- oracle regret or excess NFE.

### Step 5: Perform paired statistical analysis

Use paired comparisons because every method should process the same input, text, and seed. Report:

- effect sizes;
- confidence intervals;
- paired significance tests;
- speaker- or utterance-level aggregation where appropriate.

---

## Final acceptance criteria

The adaptive scheduling thesis receives strong final support if the frozen scheduler demonstrates on held-out data that:

$$
\text{adaptive failure rate}
<
\text{best-fixed failure rate}
$$

or achieves statistically supported non-inferior quality with lower effective computation, while also satisfying:

$$
\text{mean effective NFE}<13.
$$

The adaptive method should ideally provide a better quality–efficiency trade-off than every tested global fixed schedule.

If the best fixed schedule matches or outperforms the adaptive scheduler in both quality and effective computation, the evidence would not justify per-input adaptation.

---

## Readiness assessment

| Component | Current readiness |
|---|---:|
| Proposal-level problem justification | 85–90% |
| Evidence that safe switching times vary | Strong on evaluated data |
| Preliminary scheduler feasibility | Strong |
| Final generalization evidence | Incomplete |
| Overall final-thesis evidence | Approximately 65–75% |

These percentages are practical readiness estimates, not statistical probabilities.

---

## Defensible problem statement

> DMOSpeech 2 employs a predefined teacher-to-student handover schedule despite input-dependent differences in the earliest quality-preserving switching time. Aggressive fixed schedules reduce computation but introduce failures for some inputs, whereas conservative fixed schedules perform unnecessary teacher evaluations for others. This study therefore investigates a training-free adaptive per-input handover scheduler that aims to improve the quality–efficiency trade-off while preserving intelligibility, speaker similarity, and acoustic quality.

## Final conclusion

The scheduling problem is sufficiently supported for proposal and continued thesis development. It is not yet fully validated because the scheduler must still survive a frozen-policy evaluation on independent unseen data, the WER outlier must be audited, absolute quality ceilings must be enforced, and real latency savings must be demonstrated.

Research is rarely proven with absolute certainty. The goal is to produce repeatable evidence that survives fair held-out evaluation and comparison against the strongest fixed baseline.
