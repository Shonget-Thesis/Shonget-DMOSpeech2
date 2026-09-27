# Verdict 6: Adaptive Scheduler Evaluation Report Version 6

## Overall Assessment

Version 6 successfully fixes the three major problems identified in Version 5:

- the verified probe count is now 13, producing an effective mean of approximately 11.48 NFE;
- clean-speech savings are correctly separated into 23.1% base savings and 15.4% net savings after probe overhead;
- the pitch conclusion now acknowledges that a non-significant result does not independently prove statistical equivalence.

**Development-report rating: 9/10.**

The report is credible for proposal defense, methodology documentation, and presentation of pilot or development-set results. It is not yet the conclusive experiment for the thesis because the scheduler has not been evaluated on a genuinely held-out set of unseen speakers.

## What Is Now Strong

The report now provides:

- a clearly qualified implementation-specific fixed baseline;
- mathematically consistent strict and reference-adjusted criteria;
- a causal inference-time decision policy;
- explicit accounting for conditional probe execution;
- separate Hybrid and Pure-DSP failure results;
- disclosure and explanation of the single DSP failure;
- effective rather than base-only computation savings;
- pitch measurements from the outputs actually selected by the hybrid policy;
- an explicit development-set scope statement;
- a defensible research gap relative to DMOSpeech 2.

The main empirical result is now stated appropriately:

> On the 48-instance development benchmark, the calibrated hybrid scheduler matched the fixed $t=0.25$ implementation's reference-adjusted failure rate while reducing effective mean computation from 13.00 to approximately 11.48 NFE, representing 11.7% net savings. No statistically significant difference in pitch variation was detected.

## Remaining Correction 1: Per-Speaker Effective NFE List

The reported per-speaker effective NFE values are:

```text
[11.25, 11.75, 11.75, 10.75, 11.25, 11.75,
 11.75, 11.75, 11.25, 11.25, 11.75, 11.25]
```

These values sum to 137.50 and have a mean of:

$$
\frac{137.50}{12}=11.4583,
$$

not 11.4791 or 11.48.

The list appears to add $0.25$ NFE to each speaker, which accounts for only 12 probes. Since the report now states that 13 probes occurred, the speaker containing the additional probe should receive another:

$$
\frac{1}{4}=0.25
$$

in its speaker-level mean.

The corrected speaker values should sum to:

$$
11.4791\times12\approx137.75.
$$

The correct speaker must be identified from telemetry rather than selected arbitrarily. After updating the list, recalculate the speaker-level Wilcoxon test. The expected conclusion will probably remain unchanged because all speaker-level NFE values remain below 13, but the test must be calculated from the correct data.

## Remaining Correction 2: Equivalence Terminology

The primary section title has been corrected to **Paired Pitch-Variation Comparison**, but the subsection still says:

> Paired Comparison & Statistical Equivalence Analysis

That subsection should be renamed to one of the following:

- **Paired Pitch-Variation Statistical Analysis**;
- **Paired Difference Analysis**;
- **Paired Pitch-Variation Comparison Results**.

The reported Wilcoxon result is:

$$
p=0.6782.
$$

This means the experiment did not detect a statistically significant difference. It does not prove equivalence. Formal equivalence would require a predefined practical margin and an equivalence procedure such as TOST.

The existing interpretation is otherwise appropriate:

> No statistically significant difference in $CV_{F0}$ was detected between the hybrid scheduler and fixed $t=0.25$ on the development set. The results provide no evidence of pitch flattening, but do not independently establish statistical equivalence.

## What Version 6 Proves

Version 6 provides development-set evidence that:

1. switching-time requirements differ across evaluated inputs and conditions;
2. an automatic inference-time policy can choose among several handover points;
3. the calibrated hybrid policy can reduce effective NFE on the development set;
4. the development-set reliability can match fixed $t=0.25$ under the reference-adjusted criterion;
5. the trajectory probe can prevent at least one failure missed by the acoustic-only policy;
6. no significant pitch-variation difference was detected between the hybrid and fixed policies.

## What Version 6 Does Not Yet Prove

The report does not yet prove:

- zero-failure performance on unseen speakers;
- generalization beyond the four evaluated conditions;
- formal equivalence of prosodic diversity;
- universal superiority over every possible fixed schedule;
- that the reported thresholds are optimal outside the development benchmark.

## Final Decision

Version 6 is complete enough to serve as a strong development or pilot report after correcting the per-speaker NFE list and the remaining equivalence heading.

The scheduler and all thresholds should now be frozen. The next major research step is a held-out evaluation using speakers that were not involved in threshold calibration. Adaptive and fixed policies should use identical seeds, and every executed probe should be counted in effective NFE.

If the frozen scheduler maintains meaningful compute savings with approximately 0–5% additional failures on unseen speakers, the work will become a strong and defensible final BS Computer Science thesis contribution.
