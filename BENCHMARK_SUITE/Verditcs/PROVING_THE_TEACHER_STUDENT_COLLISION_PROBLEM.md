# Proving the Teacher-to-Student Collision Problem

## Problem-Validation Protocol Only

## 1. Purpose

This study must first determine whether the normal teacher-to-student handover creates a meaningful transition problem. No Adaptive TAGS formula, cushioning method, or proposed solution should be evaluated during this phase.

The central question is:

> Does switching from the teacher vector field to the student vector field create an abnormal transition, and is the strength of that transition associated with worse synthesized speech?

Testing a solution before establishing this relationship would create circular reasoning. A solution changing the output does not automatically prove that the original system contains a defect.

---

## 2. Operational definition of collision

The term **teacher-to-student collision** will not mean a literal crash. It will be defined operationally as:

> An abnormally large or directionally inconsistent change between teacher and student vector-field predictions at the shared handover state that is associated with measurable or perceptible output degradation.

This definition requires three elements:

1. measurable teacher–student disagreement;
2. evidence that the disagreement is abnormal rather than expected model variation;
3. evidence that greater disagreement predicts worse speech.

---

## 3. Measure the transition directly

At the exact teacher-to-student handover state $x_h$ and handover time $t_h$, evaluate both models using the same state and timestep:

$$
v_T=f_T(x_h,t_h),
$$

$$
v_S=f_S(x_h,t_h).
$$

### 3.1 Directional agreement

$$
\operatorname{Cosine}_{TS}=
\frac{v_T\cdot v_S}
{\lVert v_T\rVert_2\lVert v_S\rVert_2+\epsilon}.
$$

Interpretation:

- a value near 1 means the models point in similar directions;
- a lower value indicates stronger directional disagreement;
- a negative value means the predictions point in opposing directions.

### 3.2 Relative teacher–student velocity mismatch

$$
M_{TS}=
\frac{\lVert v_S-v_T\rVert_2}
{\lVert v_T\rVert_2+\epsilon}.
$$

A larger $M_{TS}$ means that the student proposes a more substantially different transition from the teacher at the shared handover state.

These quantities establish whether a measurable cross-model difference exists. They do not, by themselves, prove that the difference is harmful.

---

## 4. Establish whether the mismatch is abnormal

Teacher and student predictions are expected to differ because they are different models. The cross-model mismatch must therefore be compared with normal within-model changes.

### 4.1 Teacher continuity

Measure the natural change between consecutive teacher predictions:

$$
M_{TT}=
\frac{\lVert v_T(x_{h+1},t_{h+1})-v_T(x_h,t_h)\rVert_2}
{\lVert v_T(x_h,t_h)\rVert_2+\epsilon}.
$$

### 4.2 Student continuity

Measure the natural change between consecutive student predictions:

$$
M_{SS}=
\frac{\lVert v_S(x_{h+1},t_{h+1})-v_S(x_h,t_h)\rVert_2}
{\lVert v_S(x_h,t_h)\rVert_2+\epsilon}.
$$

### 4.3 Cross-model handover

Measure teacher–student disagreement at the same state and time:

$$
M_{TS}=
\frac{\lVert v_S(x_h,t_h)-v_T(x_h,t_h)\rVert_2}
{\lVert v_T(x_h,t_h)\rVert_2+\epsilon}.
$$

The collision hypothesis gains support if:

$$
M_{TS}>M_{TT}
$$

and

$$
M_{TS}>M_{SS}
$$

consistently across samples, with a statistically significant paired difference.

This comparison separates an unusual cross-model transition from ordinary trajectory evolution.

---

## 5. Connect the mismatch to output quality

For every baseline teacher-to-student generation, record one row containing:

| Sample | Speaker | Seed | $M_{TS}$ | Cosine agreement | WER | Speaker similarity | Naturalness | Listener artifact rating |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 |  |  |  |  |  |  |  |  |

Then test whether stronger mismatch is associated with worse baseline output.

### Required relationships

$$
M_{TS}\leftrightarrow\operatorname{WER},
$$

$$
M_{TS}\leftrightarrow-\operatorname{SIM},
$$

$$
M_{TS}\leftrightarrow-\operatorname{Naturalness},
$$

$$
M_{TS}\leftrightarrow\operatorname{ArtifactRating}.
$$

Use Spearman rank correlation because the measurements may not be normally distributed and may contain outliers.

The collision is relevant only if greater mismatch consistently corresponds to worse intelligibility, speaker identity, naturalness, or listener-rated artifacts.

Without this relationship, the experiment would show only that the teacher and student are different—not that their difference is a problem.

---

## 6. Compare low- and high-mismatch baseline samples

Divide baseline samples using a rule declared before inspecting quality outcomes:

- **Low mismatch:** bottom 25% of $M_{TS}$;
- **Medium mismatch:** middle 50%;
- **High mismatch:** top 25%.

Compare only the original baseline outputs:

| Baseline group | Mean/median WER | Mean SIM | Naturalness | Audible-artifact rate |
|---|---:|---:|---:|---:|
| Low mismatch |  |  |  |  |
| Medium mismatch |  |  |  |  |
| High mismatch |  |  |  |  |

Evidence for the problem becomes stronger if the high-mismatch group performs significantly worse than the low-mismatch group.

The high/low groups must be determined from $M_{TS}$ alone. They must not be selected after examining which samples sound bad.

---

## 7. Use diagnostic controls, not proposed solutions

Evaluate three unmodified inference paths:

1. **Teacher-only continuation**;
2. **Student-only generation**;
3. **Teacher-to-student direct handover**.

These are diagnostic controls, not proposed solutions.

All comparisons should use the same:

- prompt audio;
- target text;
- predicted duration;
- initial latent noise;
- random seed;
- checkpoint versions;
- vocoder;
- evaluation pipeline;
- comparable neural-function-evaluation budget where possible.

If the teacher-to-student path produces an abnormal transition or quality degradation that is absent from both continuous controls, the evidence can be attributed more specifically to switching models.

Teacher-only and student-only outputs should not be treated as the proposed replacement architecture. Their purpose is causal diagnosis.

---

## 8. Do not locate the collision using waveform time

A handover at diffusion coordinate $t=0.07$ does not mean that the corresponding artifact occurs at 0.07 seconds in the final speech waveform.

Therefore, this reasoning is invalid:

> A spectral spike at 0.07 seconds proves a handover collision.

The correct approach is:

1. measure teacher–student mismatch in latent or vector-field space at the actual model switch;
2. evaluate the complete synthesized output using WER, speaker similarity, naturalness, and listening results;
3. determine whether latent mismatch predicts the observed degradation.

Spectral flux may be retained as a secondary descriptive metric, but it must not be the only proof of collision.

---

## 9. Blinded perceptual validation

Objective metrics cannot fully establish that the transition is audibly problematic. Conduct a small blinded listening evaluation using baseline outputs only.

Select samples using their predetermined mismatch groups:

- high-mismatch baseline samples;
- low-mismatch baseline samples.

Listeners should not be shown the group labels or collision hypothesis. Ask them to rate:

1. intelligibility;
2. naturalness;
3. presence of clicks, roughness, unstable onset, or other artifacts;
4. speaker consistency.

The problem gains perceptual support if high-mismatch baseline outputs receive consistently worse ratings than low-mismatch baseline outputs.

---

## 10. Repeatability requirements

The pattern should be evaluated across:

- multiple speakers;
- different target texts;
- short and long utterances;
- several random seeds;
- clean prompts;
- noisy or challenging prompts, if included in the declared scope.

A single spectacular failure cannot establish a general collision problem. The important result is a repeatable relationship between transition mismatch and degradation.

---

## 11. Required proof chain

The collision problem will be considered empirically supported only if the following chain is established:

### Condition 1: Existence

Teacher and student predictions show measurable disagreement at the same handover state.

### Condition 2: Abnormality

Cross-model disagreement is significantly greater than normal within-teacher and within-student trajectory variation.

### Condition 3: Quality association

Greater cross-model disagreement predicts worse baseline WER, speaker similarity, naturalness, or listener-rated artifacts.

### Condition 4: Repeatability

The relationship occurs across speakers, texts, and random seeds rather than in one isolated example.

### Condition 5: Perceptual relevance

Blinded listeners rate high-mismatch baseline outputs worse than low-mismatch baseline outputs.

No proposed cushioning solution is required to establish these five conditions.

---

## 12. Decision criteria

### Collision problem supported

The problem is supported if:

- $M_{TS}$ is consistently and significantly greater than $M_{TT}$ and $M_{SS}$;
- $M_{TS}$ or cosine disagreement significantly predicts one or more predefined quality outcomes;
- high-mismatch baseline samples perform worse than low-mismatch baseline samples;
- the relationship replicates across seeds and speakers;
- blinded listeners detect worse quality or more artifacts in the high-mismatch group.

### Collision problem not supported

The problem is not supported if:

- teacher–student disagreement is comparable to ordinary within-model variation;
- mismatch does not predict output degradation;
- high- and low-mismatch groups have comparable quality;
- the relationship disappears across seeds;
- listeners cannot distinguish the groups.

In that case, the teacher–student difference should be interpreted as normal model variation rather than a defect requiring correction.

---

## 13. Defensible problem statement

> DMOSpeech 2 transfers a teacher-derived intermediate state directly to a distilled student model. Because the teacher and student represent separately learned vector fields, their predictions at the shared handover state may differ. It remains unknown whether this cross-model disagreement exceeds normal within-model trajectory variation and whether greater disagreement is associated with degraded intelligibility, speaker similarity, or perceived naturalness. This study first characterizes the teacher-to-student transition before evaluating any handover-control mechanism.

---

## 14. Main research question

> Does teacher-to-student vector-field disagreement at the handover point exceed normal within-model trajectory variation and predict measurable or perceptible degradation in synthesized speech?

## 15. Specific research questions

1. How large is the teacher–student vector-field disagreement at the shared handover state?
2. Is the cross-model disagreement significantly greater than consecutive within-teacher and within-student changes?
3. Does greater teacher–student disagreement correlate with worse WER, speaker similarity, naturalness, or listener-rated artifacts?
4. Do high-mismatch baseline samples exhibit worse quality than low-mismatch baseline samples?
5. Is the mismatch–degradation relationship repeatable across speakers, texts, and noise seeds?

---

## Final verdict

The problem-validation phase must use the unmodified teacher-to-student baseline. It must first establish that teacher–student disagreement is abnormal, repeatable, and connected to actual degradation.

The central test is:

$$
\boxed{\text{Does larger }M_{TS}\text{ consistently predict worse baseline speech?}}
$$

If the answer is yes, the collision is a defensible research problem. If the answer is no, the measured teacher–student difference is not sufficient evidence of a defect.

