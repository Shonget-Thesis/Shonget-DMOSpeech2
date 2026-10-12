# Student Restart-Level Pilot: Required Steps Before the Proposal

## Purpose

Determine whether the student restart level should:

1. remain fixed at the DMOSpeech 2 baseline value; or
2. depend on the teacher-to-student handover point through a restart map \(j=\rho(i)\).

This is a **development pilot**, not the final thesis evaluation. Its purpose is to select the design that will be proposed and later calibrated and tested.

---

## Step 1: Verify the Baseline Implementation

Before changing the restart level:

- identify the baseline student restart index and corresponding noise level \(\tau\);
- confirm the available student restart indices in the public code;
- verify the student time grid;
- record the repository commit and configuration used;
- use the same prompt, target text, duration, initial latent and random seed across configurations; and
- confirm that \((s=0.25,j=1)\) reproduces the intended reference output.

Do not begin the pilot until the baseline restart behavior is known exactly.

---

## Step 2: Use Development Data Only

Use speakers assigned only to the development split.

Do not use:

- calibration speakers;
- final test speakers; or
- results previously designated as final held-out evidence.

The pilot may influence the design, so its samples cannot later be presented as untouched test data.

Use clean and noisy prompts if both conditions are within the thesis scope.

---

## Step 3: Define the Configuration Grid

Evaluate every supported combination of teacher stopping argument and student restart index.

Initial grid:

| Teacher stopping argument | Restart \(j=1\) | Restart \(j=2\) |
| ---: | :---: | :---: |
| \(s=0.07\) | Test | Test |
| \(s=0.10\) | Test | Test |
| \(s=0.15\) | Test | Test |
| \(s=0.25\) | Test | Test |

Reference configuration:

\[
k_{\mathrm{ref}}=(s=0.25,j=1).
\]

Do not add more restart levels until the exact supported student grid has been verified.

---

## Step 4: Run a Strictly Paired Experiment

For each development input:

1. fix one random seed;
2. reuse the same initial Gaussian latent across every configuration;
3. reuse the same duration prediction;
4. use identical teacher, student and vocoder checkpoints;
5. synthesize all eight \((s,j)\) combinations;
6. save the generated audio and raw telemetry; and
7. repeat with additional predetermined seeds if inference is stochastic.

Only \(s\) and \(j\) should change between paired runs.

---

## Step 5: Record Metrics

For every configuration, record:

- WER;
- speaker similarity;
- UTMOS, if available;
- teacher forward passes;
- student forward passes;
- total NFE;
- wall-clock inference time;
- real-time factor;
- selected teacher time;
- restart index \(j\);
- restart noise level \(\tau\);
- clean or noisy condition;
- input and speaker IDs;
- random seed; and
- ASR transcript.

Use multiple warmed-up timing repetitions before interpreting latency.

---

## Step 6: Calculate Added Failure

Compare every configuration against the matching reference output produced from the same input and seed.

Use the Chapter 3 added-failure definition:

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

Freeze \(c\), \(m_W\), and \(m_S\) before interpreting the pilot.

Report absolute quality results alongside reference-relative failures. A reference-relative pass does not necessarily mean that the audio has good absolute quality.

---

## Step 7: Compare Restart Levels Within Each Handover Point

For each stopping argument \(s\), compare \(j=1\) against \(j=2\) on the same inputs.

Create a summary like:

| Handover | Restart | Mean WER | Mean SIM | Failure rate | Mean NFE | Median latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.07 | 1 |  |  |  |  |  |
| 0.07 | 2 |  |  |  |  |  |
| 0.10 | 1 |  |  |  |  |  |
| 0.10 | 2 |  |  |  |  |  |
| 0.15 | 1 |  |  |  |  |  |
| 0.15 | 2 |  |  |  |  |  |
| 0.25 | 1 |  |  |  |  |  |
| 0.25 | 2 |  |  |  |  |  |

Also report per-input results. Means alone can hide catastrophic failures.

---

## Step 8: Determine Whether a Stable Restart Pattern Exists

Check whether the preferred restart index follows a repeatable relationship with the handover point.

Possible useful finding:

- earlier handovers consistently work better with \(j=1\); and
- later handovers remain non-inferior with \(j=2\) while using less computation.

Possible negative finding:

- the better restart index changes unpredictably between samples;
- computational savings are negligible;
- \(j=2\) increases the failure rate; or
- any advantage disappears under noisy prompts.

Do not create an adaptive rule from isolated wins or one favorable sample.

---

## Step 9: Apply the Design Decision Rule

### Choose a fixed restart level when:

- one restart level is consistently safer;
- restart adaptation provides no meaningful savings;
- the preferred level is inconsistent across inputs; or
- there is no observable signal that can select the appropriate restart level during inference.

Then use:

\[
\rho(i)=1
\]

for every handover point, assuming \(j=1\) is the verified baseline.

### Choose a restart map when:

- a restart level is non-inferior in WER and SIM at a particular handover point;
- it provides meaningful NFE or latency savings;
- the result is consistent across development samples and conditions; and
- the map can be fixed before calibration and final testing.

Example only:

\[
\rho(i)=
\begin{cases}
1, & i\text{ is an early handover step},\\
2, & i\text{ is a later handover step}.
\end{cases}
\]

Use the actual observed mapping, not this example, if the evidence supports one.

### Do not yet build a second learned controller

If the best restart index varies per utterance without a stable relationship to the handover point, a separate input-dependent restart controller would be required. That would add a new research problem, additional features and another calibration burden. Keep it outside the main thesis unless the development evidence is exceptionally strong.

---

## Step 10: Freeze the Selected Restart Design

After analyzing the development pilot:

1. choose either fixed \(j\) or a deterministic restart map \(\rho(i)\);
2. document the evidence supporting the choice;
3. freeze the choice before risk calibration;
4. update Chapter 3 to match the final decision; and
5. do not change the restart design using calibration or test outcomes.

The stabilization threshold \(\lambda\) should then be calibrated while the restart design is already fixed.

---

## Step 11: Integrate It With the Main Adaptive Scheduler

The final inference sequence becomes:

1. estimate prompt SNR;
2. run teacher inference;
3. calculate endpoint-stabilization scores at candidate steps;
4. select the first teacher step satisfying the frozen handover rule;
5. obtain the student restart index using the frozen map \(j=\rho(i)\);
6. run the student from that restart level; and
7. decode the waveform.

The teacher-stopping policy remains risk-calibrated. The restart map is a design choice established using development data unless a separate calibration procedure is formally introduced.

---

## Step 12: Required Pilot Report

Produce a Markdown report containing:

1. repository and checkpoint identifiers;
2. development speakers and conditions;
3. configuration grid;
4. paired per-input results;
5. aggregate WER, SIM, UTMOS, NFE and latency;
6. added-failure rates;
7. the number of times each restart level wins, loses or ties;
8. clean-versus-noisy analysis;
9. the selected restart decision;
10. exact frozen map \(\rho(i)\), if applicable; and
11. limitations of the pilot.

Suggested filename:

`STUDENT_RESTART_LEVEL_PILOT_REPORT.md`

---

## Final Decision Standard

The pilot is successful even if adaptive restart is rejected.

- If a stable benefit exists, include the frozen restart map in the proposed method.
- If no stable benefit exists, keep the baseline restart level and report that the pilot justified simplifying the algorithm.

The purpose is not to force adaptive restart to succeed. The purpose is to determine, before the proposal, which design is supported by evidence.

