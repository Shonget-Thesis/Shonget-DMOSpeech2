# Remaining Collision-Validation Tests and Stop Rule

## Current evidence

The completed baseline experiment used:

- 12 speakers;
- 2 target texts per speaker;
- 3 random seeds;
- 72 unchanged baseline trials;
- the original Seed-TTS/F5-TTS-base teacher-to-DMOSpeech 2 student pipeline;
- a fixed handover time of $t_h=0.25$;
- WER, speaker similarity, pitch variation, spectral flux, and trajectory-mismatch measurements.

The current evidence does **not** support the broad claim that the normal teacher-to-student handover is a harmful collision:

- teacher and student disagreement exists;
- the cross-model mismatch was not abnormally greater than both within-model references;
- mismatch did not significantly predict worse WER;
- high-mismatch outputs were not worse than low-mismatch outputs;
- the observed speaker-similarity relationship did not support degradation;
- informal listening did not reveal an obvious harmful transition.

Therefore, simply repeating the same normal-condition experiment with more selected samples is unlikely to establish the original claim. However, three legitimate investigations remain.

---

## 1. Audit the diagnostic implementation

Before abandoning the collision hypothesis, verify that the diagnostic measurements were implemented correctly and that the comparisons were scientifically fair.

### Same-state verification

Confirm that the teacher and student vectors were evaluated using exactly the same:

- handover state $x_h$;
- continuous time $t_h$;
- text conditioning;
- speaker prompt conditioning;
- duration and frame mask;
- latent/noise realization;
- tensor shape, scale, and parameterization.

The teacher and student outputs must represent the same mathematical quantity. If one predicts velocity while the other is interpreted using a different parameterization, their vector difference is invalid.

### Checkpoint identity

The JSON records the teacher as:

> `UniModel/real_unet (Seed-TTS / F5-TTS Base)`

Verify the exact repository, architecture, configuration, commit, and checkpoint. The thesis must not ambiguously call the model Seed-TTS if the implementation actually loads an F5-TTS-derived teacher.

### Timestep-comparability audit

The recorded intervals were:

$$
\Delta t_T=0.0659,
\qquad
\Delta t_S=0.2168.
$$

Teacher-to-teacher and student-to-student consecutive changes are therefore measured over unequal intervals. Determine whether dividing mismatch by $\Delta t$ is mathematically appropriate for the solver and vector-field outputs. If possible, evaluate both models on matched artificial diagnostic time offsets while retaining the original baseline output path.

### Statistical-direction audit

Verify that every one-sided Wilcoxon test uses the intended alternative hypothesis:

$$
H_1:M_{TS}>M_{TT}
$$

and:

$$
H_1:M_{TS}>M_{SS}.
$$

Check whether the reported $W$ statistic and $p$-value correspond to that direction. Recalculate the tests independently from the saved per-trial values.

### Metric audit

Confirm that:

- WER normalization handles punctuation, capitalization, contractions, and written numbers consistently;
- speaker similarity uses the same reference convention for all samples;
- corrupted text encoding, such as malformed apostrophes, is corrected before WER calculation;
- no speaker or sentence is counted as fully independent across repeated seeds;
- correlations are also examined at the speaker or prompt–text level.

### Audit decision

- If the audit finds a meaningful error, correct it and rerun the predetermined analysis.
- If the implementation is correct, retain the negative normal-condition verdict and proceed only to a targeted stress pilot.
- Do not change definitions or thresholds merely to obtain statistical significance.

---

## 2. Targeted stress-condition pilot

The existing experiment supports only this conclusion:

> A general harmful handover collision was not detected under the tested ordinary baseline conditions.

It does not establish that the handover is robust under every inference condition. A targeted pilot may test the narrower question:

> Under which challenging conditions, if any, does teacher-to-student handover become unstable or quality-relevant?

### Predetermined pilot size

Use a small but fixed pilot:

$$
5\text{ speakers}\times2\text{ texts}\times3\text{ seeds}=30
$$

matched units per condition.

Use speakers and texts selected before examining stress-test results. Reuse the same prompt, text, and seed combinations across conditions.

### Candidate stress conditions

Test only two or three conditions initially:

1. **Earlier handover:** reduce teacher guidance before the student takes control.
2. **Fewer remaining student steps:** reduce the student's opportunity to recover after handover.
3. **Duration perturbation:** introduce small, predetermined duration errors.

Other optional conditions include:

- long or syntactically complex sentences;
- noisy or shortened reference prompts;
- out-of-domain speakers;
- emotional speech;
- mismatched languages.

### Required controls

For every stress trial, hold constant:

- speaker prompt;
- target text;
- random seed;
- initial latent/noise;
- model checkpoints;
- vocoder;
- sampler and solver, except for the deliberately changed factor.

### Measurements

For each condition, record:

- $C_{TS}$ and $M_{TS}$;
- WER or CER;
- speaker similarity;
- an available validated non-intrusive quality score;
- runtime and teacher/student function evaluations;
- failure rate across seeds.

The key evidence is not merely that a stressful setting worsens speech. The study must show that handover mismatch becomes more quality-relevant under that setting:

$$
\text{larger handover mismatch}
\longrightarrow
\text{worse output under the stress condition}.
$$

### Defensible conclusion if successful

If a repeatable relationship appears only under stress, the appropriate claim is:

> The default handover is stable under the tested ordinary conditions but exhibits input- or configuration-dependent robustness limitations under specific challenging transition conditions.

This is a robustness claim, not proof that the normal DMOSpeech 2 baseline is universally defective.

---

## 3. Controlled causal-intervention experiment

The current 72-run analysis is primarily observational: it asks whether naturally occurring mismatch correlates with quality. Correlation may be weak because mismatch is not the cause of the errors.

A stronger test deliberately changes only the handover transition while holding all other factors constant.

### Experimental conditions

For every matched prompt, text, and seed, compare:

1. original direct handover;
2. a predetermined perturbation that increases transition disruption;
3. a predetermined alignment or smoothing intervention that reduces disruption.

### Causal pattern required

Evidence would be substantially stronger if:

$$
\text{increased disruption}
\longrightarrow
\text{worse speech}
$$

and:

$$
\text{reduced disruption}
\longrightarrow
\text{recovery toward or beyond baseline quality}.
$$

The response should be consistent across seeds and inputs and should show a meaningful effect size—not merely a statistically detectable but negligible difference.

### Important limitation

This experiment can establish that sufficiently strong handover perturbations cause damage. It does not automatically prove that the **natural baseline mismatch** is harmful. The intervention magnitude must remain plausible and the natural baseline must still show quality relevance before claiming that Adaptive TAGS solves an existing default-system problem.

---

## Recommended order

1. Audit the existing implementation and recompute the saved 72-run statistics.
2. If valid, preserve the negative normal-condition result.
3. Run a 30-unit pilot on two or three predetermined stress conditions.
4. If the pilot reveals a repeatable failure, expand only that condition to the full dataset.
5. Optionally use a controlled intervention to examine causality.
6. Apply the stop rule below before developing another collision-cushion solution.

---

## Final stop rule

Continue the collision/robustness direction only if at least one rigorously defined condition produces all of the following:

1. measurable cross-model handover disagreement;
2. abnormal disagreement relative to fair controls;
3. a repeatable association with worse objective quality;
4. worse high-mismatch than low-mismatch outputs;
5. consistency across multiple speakers, texts, and seeds.

If the implementation audit is correct and the predetermined stress pilot still finds no quality-relevant relationship, stop the collision path. Do not add new thresholds, metrics, or conditions solely to search for a positive result.

The appropriate pivot would then be:

> Adaptive teacher-to-student handover scheduling for reducing teacher computation while preserving baseline speech quality.

This pivot can reuse the current models, dataset, seeds, baseline outputs, WER pipeline, speaker-similarity pipeline, and handover instrumentation.

---

## Final verdict

More testing remains scientifically possible, but only in a controlled and predetermined form. The strongest immediate action is an implementation audit, followed by one small targeted stress-testing round. Repeating ordinary baseline trials or searching indefinitely for a favorable metric would not strengthen the thesis.

The present 72-run result must remain part of the thesis record:

> Under the evaluated normal baseline configuration, numerical teacher–student disagreement existed, but a general harmful collision was not empirically supported.
