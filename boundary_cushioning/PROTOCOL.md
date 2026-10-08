# Controlled teacher-to-student handover study

## Question

Does changing acceptance of the first student estimate improve generated speech, and does the adaptive TAGS rule outperform a matched fixed blend? The study tests these questions; a nonzero difference between teacher and student estimates is not assumed to be an audible defect.

## Data and configurations

- The existing first-100-pair benchmark is development evidence. Its corrected comparison contains 96 completed pairs. Do not use those speakers to judge final performance.
- Build a new manifest from the local cross-sentence list using reference speakers absent from all 100 pilot manifest entries. Select 100 pairs from 20 speakers (five per speaker), with distinct reference utterances within each speaker and reference duration at most four seconds. Keep the deterministic selection rule and selected manifest. This avoids trimming audio while leaving its full transcript in place.
- Use seeds `1001,2002,3003` for objective evaluation. The primary configuration reproduces the pilot's teacher grid of 8, stopping threshold 0.07, student start index 1, sway -1, teacher stochasticity 1, and guidance strength 2. The app's default teacher grid of 16 is a separate secondary study and must have a separate results directory.
- Apply the same mono conversion, 24 kHz resampling, four-second reference trim, text, duration prediction, checkpoint, and vocoder to every method for a given case.

## Controlled arms

For each prompt and seed, compute the teacher endpoint estimate `T`, first student estimate `S`, and `D = S - T` once. Reuse the same later student noise across arms.

| Arm | Accepted estimate after first student step |
| --- | --- |
| `baseline` | `S` |
| `baseline_copy` | `S`; its final waveform must exactly equal `baseline` |
| `fixed` | `T + 0.8596 * D` |
| `proposed` | `T + w * D`, where `w = clip(1 - 0.26 * tanh(||D|| / (||T|| + eps)), 0.74, 1)` |

The fixed weight and adaptive formula are frozen from the pilot. Results from a new sample must never change either rule. If `baseline_copy` differs from `baseline`, stop and resolve the control failure before interpreting any comparison.

## Outcomes

**Primary outcome:** blinded pairwise listener preference for generated speech naturalness, comparing `proposed` against both `baseline` and `fixed`. Randomize A/B order, hide method names, permit ties, and keep the answer key separate from listener files. Before inspecting held-out outcomes, choose a balanced 40-case subset across the 20 speakers and three seeds, and collect at least three independent ratings per trial. A separate blinded single-clip baseline naturalness rating can test whether teacher-student disagreement predicts baseline quality. These tasks provide no reference speech, so they cannot measure speaker fidelity. Agent-generated labels are not listener ratings.

**Secondary outcomes:** utterance-level ASR word error rate, speaker similarity if a validated evaluator is available, waveform duration, and measured generation time. Report the number of missing scores. Raw spectral flux over the first 15% of generated audio is exploratory; it is neither a decibel measurement nor a direct measure of the internal handover.

## Analysis and interpretation

- Compare methods within the same prompt and seed. Report paired effects and intervals that account for speaker clustering. Report wins, losses, ties, and failures, not selected examples alone.
- Test whether normalized disagreement `rho = ||S - T|| / (||T|| + eps)` predicts baseline listener ratings and the paired benefit of `proposed`. Association alone does not prove the mechanism.
- Call `proposed` a perceptual improvement only if every planned A/B trial has at least three independent ratings, the duplicate-baseline control passes, and the lower bound of the speaker-clustered 95% interval for proposed preference exceeds 0.5 against **both** `baseline` and `fixed`. Report the two effect sizes and their uncertainty even when either condition fails. Check intelligibility and runtime before claiming a practical gain.
- Rate baseline naturalness on all 100 distinct prompts at synthesis seed `1001`, with at least three independent ratings per clip. Treat its association with disagreement as exploratory. A preferred intervention can establish a relative limitation of full acceptance, but these outcomes alone do not establish a particular audible artifact or a mathematical discontinuity.
- If results do not support these claims, report the null result and revise the thesis focus. Do not describe a proven boundary shock or C1 continuity from this experiment.
