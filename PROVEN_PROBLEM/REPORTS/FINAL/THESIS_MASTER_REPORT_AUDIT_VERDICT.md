# Thesis Master Report Audit Verdict

## Overall Verdict

The consolidated master report contains nearly all experimental work completed so far:

- the DMOSpeech 2 research gap;
- fixed-switch diagnostic experiments;
- five fixed switching schedules;
- the retrospective oracle;
- the pre-diffusion and hybrid schedulers;
- effective NFE accounting;
- WER, SIM, latency, and $CV_{F0}$ evaluation;
- speaker-level statistical tests;
- development-set limitations.

However, it is **not yet a fully correct final master report**. Several statements contradict either the DMOSpeech 2 paper or the report's own tables.

**Current rating: 7/10.**

The experimental foundation is valuable, but the framing presently makes the result sound stronger than the evidence permits.

## 1. DMOSpeech 2 Is Not Universally Hardcoded to $t=0.25$

The report states that standard DMOSpeech 2 hardcodes the switch to $t=0.25$. This is inaccurate. DMOSpeech 2 describes $t_{\text{switch}}$ as a predefined but adjustable parameter. It discusses using lower values for efficiency and higher values for diversity. Its principal reported teacher-guided configuration uses $t_{\text{switch}}=0.25$, but this is not a universal hardcoded requirement.

Use:

> DMOSpeech 2 performs teacher-to-student handover at a predefined, globally selected switching point. Although its reported teacher-guided configuration uses $t_{\text{switch}}=0.25$, it does not provide automatic per-utterance switch selection.

## 2. This Is Not an Exact Reproduction of the Paper's Sampling Schedule

The study uses:

$$
4\text{ teacher steps}+9\text{ student steps}=13\text{ NFE}.
$$

The DMOSpeech 2 paper reports approximately 14 teacher steps and 2 student steps for its 16-step $t_{\text{switch}}=0.25$ configuration. Although the report sometimes says “under our implementation,” other passages still describe the 13-NFE configuration as standard DMOSpeech 2.

Add this qualification:

> This study evaluates a DMOSpeech 2-derived implementation using Euler integration on a 16-point swayed grid. Its NFE schedule differs from the configuration reported in the original paper; consequently, the results should be interpreted as an evaluation of our implementation rather than an exact reproduction.

## 3. “No Static Threshold Can Work” Is Too Strong

The study evaluates five discrete fixed thresholds on 48 development instances. This cannot prove that every possible static threshold is incapable of achieving a good trade-off.

Replace:

> No single static handover threshold can achieve both compute efficiency and acoustic safety.

With:

> Among the five evaluated fixed switching schedules, no single schedule achieved the best observed efficiency–reliability trade-off across all tested inputs and acoustic conditions.

Also replace:

- “exhaustive empirical demonstration” with “controlled empirical evaluation”;
- “conclusively validated” with “provided evidence for”;
- “proof pillars” with “empirical findings” or “evidence pillars.”

## 4. Pillar 1 Contradicts the Results Table

Pillar 1 claims that clean speech safely switches at $t=0.05$ or $t=0.10$ for 100% of speakers. Table 2 reports:

| Clean-Speech Schedule | Failures |
| :--- | :---: |
| $t=0.05$ | 6/12 |
| $t=0.10$ | 1/12 |
| $t=0.15$ | 0/12 |

The correct conclusion is:

> Clean inputs generally support earlier handover than $t=0.25$, but $t=0.05$ and $t=0.10$ are not universally safe. Fixed $t=0.15$ was the earliest evaluated schedule with zero failures across all 12 clean instances.

This contradiction must be corrected before proposal submission.

## 5. The Reference-Adjusted Rule Is Not an Absolute Safety Ceiling

Section 2.2 says that the protocol introduced an absolute WER ceiling of 12.5%. However, the primary rule is:

$$
\mathrm{WER}\leq\max(12.5\%,\mathrm{refWER}+1\%).
$$

If reference WER is extremely high, the permitted threshold also becomes extremely high. For example, a reference WER near 2775% produces a permitted threshold near 2776%. Therefore, this is not an absolute safety criterion.

The two criteria should be described as:

- **Reference-adjusted rule:** baseline-relative non-degradation;
- **Strict 12.5% rule:** absolute intelligibility threshold.

Do not call 0/48 reference-adjusted failures complete acoustic safety. A more accurate statement is:

> The hybrid introduced no additional failures relative to the fixed baseline, but both policies retained four failures under the strict absolute criterion.

## 6. The Hybrid May Not Have Solved Instance 44

Instance 44 reportedly has 2775% WER at fixed $t=0.25$, while $t=0.40$ produces 0% WER. The hybrid noise guardrail normally selects $t=0.25$ when estimated SNR is below 18 dB.

Unless the hybrid regenerated a different output or selected another threshold, it may reproduce the failing $t=0.25$ result. Its 0/48 reference-adjusted failure count does not guarantee that Instance 44 became intelligible because the relative criterion may accept severe degradation when the reference is already corrupted.

The report must disclose for Instance 44:

- estimated SNR;
- selected hybrid switching point;
- hybrid WER and SIM;
- strict pass/fail status;
- whether the hybrid output is identical to the fixed-policy output.

If the hybrid retains the 2775% WER result, it has not completely solved absolute handover safety.

## 7. Additional Teacher Guidance Is Not Monotonically Better

The noisy-prompt results are:

| Switching Point | Failure Rate |
| :--- | :---: |
| $t=0.10$ | 83.3% |
| $t=0.15$ | 58.3% |
| $t=0.25$ | 0% reference-adjusted |
| $t=0.40$ | 25.0% |

Therefore, the finding is not simply that difficult prompts require progressively more teacher guidance.

Use:

> Difficult prompts benefit from additional teacher guidance up to an appropriate switching region, but excessively late switching can also degrade performance. Under the evaluated implementation, $t=0.25$ was more reliable than both earlier and later alternatives.

This non-monotonic behavior strengthens the motivation for adaptive scheduling.

## 8. Oracle Accounting Contains an Apparent Impossibility

The reported oracle reduction vector includes $-0.5$:

$$
13-\mathrm{NFE}_{\mathrm{oracle}}=-0.5.
$$

This implies an oracle mean of 13.5 NFE for one speaker. But fixed $t=0.25$ uses 13 NFE and reportedly passes the reference-adjusted criterion for all instances. If $t=0.25$ is an available safe candidate, the oracle should never require more than 13 NFE.

Possible causes include:

- inconsistent NFE accounting;
- an extra evaluation added to the oracle;
- different seeds or generated outputs;
- incorrect speaker aggregation;
- an oracle implementation error.

Until this is resolved, 9.35 NFE should not be described as a fully validated lower bound.

## 9. “Pure Acoustic DSP Scheduler” Is Not Entirely Acoustic

The scheduler uses both acoustic SNR and duration information. It is therefore not purely acoustic.

Recommended names:

- **Pre-Diffusion Input-Feature Scheduler**;
- **SNR-and-Duration Scheduler**;
- **Zero-Probe Input Scheduler**.

“Zero neural overhead” can remain if the duration predictor is already part of the normal inference pipeline.

## 10. Duration Terminology Must Be Unified

The report alternates among character count, phoneme length, predicted duration, explicit speed multipliers, and speaking rate. These are related but not interchangeable.

Use one primary deployable definition:

$$
\delta_{\mathrm{dur}}
=
\frac{|D_{\mathrm{requested}}-\widehat D_{\mathrm{natural}}|}
{\widehat D_{\mathrm{natural}}}.
$$

Then define:

- how $\widehat D_{\mathrm{natural}}$ is generated;
- how $D_{\mathrm{requested}}$ is obtained;
- what happens when the user provides no speed multiplier;
- whether character-based speaking rate is an additional fallback feature.

## 11. Teacher-to-Student Process Description Needs Correction

The report says that the teacher trajectory is extrapolated to $t=1$ before the student takes over. This is potentially misleading.

Use:

> The teacher performs the initial denoising or flow-matching steps until $t_{\text{switch}}$. The resulting intermediate latent state is then passed to the student, which completes generation.

## 12. Dataset Provenance Is Ambiguous

The scope describes the data as “Common Voice / Seed-TTS.” The exact relationship must be stated clearly.

Specify whether:

- the speakers and prompts came from Common Voice;
- Seed-TTS was used only as an evaluation protocol;
- samples came from both datasets;
- reference audio and target texts came from different sources.

Do not combine dataset names with a slash without explaining their roles.

## 13. Defensible Current Conclusion

The strongest conclusion presently supported is:

> Across 48 development instances and five evaluated fixed schedules, the earliest acceptable teacher-to-student switching point varied by input and acoustic condition. A calibrated inference-time hybrid policy matched the fixed $t=0.25$ implementation under the baseline-relative acceptance criterion while reducing effective mean computation from 13.00 to 11.48 NFE. However, both methods retained four failures under the strict absolute criterion, and held-out evaluation is still required.

## Is This Everything Currently Available?

Yes. The master report consolidates nearly all current experimental evidence. It is not yet everything required for the final thesis.

The following remain outstanding:

- corrected oracle accounting;
- explicit hybrid audit of Instance 44;
- exact dataset provenance;
- precise disclosure of the implementation's difference from the paper;
- frozen-threshold held-out evaluation on unseen speakers;
- preferably an ablation table covering SNR-only, duration-only, cosine-only, SNR-plus-duration, and full hybrid policies;
- preferably a small human-listening evaluation because Whisper can hallucinate;
- a related-work review supporting the novelty claim.

## Final Decision

The report is structurally comprehensive, but it currently overstates both the diagnosed limitation and the solution's safety.

The defensible contribution is not:

> A scheduler that completely solves teacher-to-student handover safety.

It is:

> A calibrated input-adaptive scheduler that reduces computation without introducing additional baseline-relative failures on a 48-instance development benchmark.

That remains a valid and meaningful thesis contribution. Correct the internal contradictions, freeze the scheduler, and conduct held-out evaluation before presenting the method as a final proven solution.
