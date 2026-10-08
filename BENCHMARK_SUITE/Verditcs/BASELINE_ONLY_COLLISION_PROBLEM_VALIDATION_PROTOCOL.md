# Baseline-Only Validation of the Seed-TTS-to-Student Handover Problem

## 1. Scope and purpose

This protocol focuses exclusively on determining whether the **original, unmodified Seed-TTS-to-DMOSpeech 2 student handover** exhibits an empirically meaningful transition problem.

In this protocol, **Seed-TTS is the teacher** and the distilled **DMOSpeech 2 model is the student**. Before collecting results, the researchers must verify and record the exact teacher architecture, repository revision, checkpoint name, and configuration loaded by the baseline implementation. If the code actually loads a different teacher, the document must use that model's real name instead of Seed-TTS.

This phase is for **problem validation only**. It must not evaluate or discuss whether any proposed method solves the problem.

### Strict scope rule

All synthesized audio evaluated in this experiment must come from the unchanged teacher-to-student baseline.

The following must not be applied during problem validation:

- Adaptive TAGS;
- fixed cushioning or interpolation;
- cosine weighting;
- SLR caps;
- orthogonal projection;
- temporal boundary windowing;
- calibrated gates;
- adaptive switch-time scheduling;
- modified solvers;
- any other proposed correction.

Teacher and student networks may be queried diagnostically on saved baseline states, but those queries must not modify the state, switching rule, or final baseline output.

---

## 2. Research problem

The evaluated DMOSpeech 2 baseline performs the initial trajectory using Seed-TTS and then transfers the intermediate state directly to the distilled DMOSpeech 2 student. Because Seed-TTS and the student represent separately learned vector fields, their predictions at the shared handover state may differ.

The research problem is:

> It remains unknown whether teacher-to-student vector-field disagreement at the unmodified handover exceeds ordinary within-model trajectory variation and whether stronger disagreement is associated with measurable or perceptible degradation in the resulting baseline speech.

The study must not begin by assuming that the handover is defective. It must test that claim empirically.

---

## 3. Main research question

> Does the unmodified teacher-to-student baseline exhibit abnormal cross-model transition disagreement, and does greater disagreement predict worse synthesized-speech quality?

## 4. Specific research questions

1. How large is the teacher–student vector-field disagreement at the baseline handover state?
2. Is cross-model disagreement greater than normal changes observed within the teacher and within the student?
3. Does stronger disagreement predict worse baseline WER, speaker similarity, naturalness, prosody, or listener-rated artifacts?
4. Are high-disagreement baseline outputs worse than low-disagreement baseline outputs?
5. Is the mismatch–degradation relationship repeatable across speakers, texts, and random seeds?

---

## 5. Experimental unit and recommended sample

Use at minimum:

- 10–12 speakers;
- 2 target sentences per speaker;
- 3 predetermined random seeds per prompt–text pair.

For 12 speakers, this produces:

$$
12\times2\times3=72
$$

matched baseline trials.

If time is severely limited, a smaller predetermined pilot may be used, but its findings must be labeled preliminary. Do not select samples because they already sound bad.

---

## 6. Baseline-only generation procedure

For each speaker, sentence, and seed:

1. Load the original Seed-TTS teacher checkpoint and the original DMOSpeech 2 student checkpoint used by the baseline.
2. Enable inference/evaluation mode for both networks.
3. Fix the prompt audio, target text, predicted duration, latent noise, seed, sampler grid, switch time, checkpoint versions, and vocoder.
4. Run the normal teacher phase until the original handover time $t_h$.
5. Cache the exact baseline handover state $x_h$.
6. Query the teacher and student on the cached state for diagnostic measurements only.
7. Continue the original student phase without changing $x_h$ or either model output.
8. Save the final unmodified baseline audio and all diagnostic measurements.

Only one final audio output is required per trial: the normal teacher-to-student baseline output.

---

## 7. Direct cross-model measurement

At the cached baseline handover state $x_h$ and time $t_h$, evaluate:

$$
v_T=f_T(x_h,t_h),
$$

$$
v_S=f_S(x_h,t_h).
$$

### 7.1 Directional agreement

$$
C_{TS}=
\frac{v_T\cdot v_S}
{\lVert v_T\rVert_2\lVert v_S\rVert_2+\epsilon}.
$$

### 7.2 Relative cross-model mismatch

$$
M_{TS}=
\frac{\lVert v_S-v_T\rVert_2}
{\lVert v_T\rVert_2+\epsilon}.
$$

### 7.3 Norm ratio

$$
R_{TS}=
\frac{\lVert v_S\rVert_2}
{\lVert v_T\rVert_2+\epsilon}.
$$

These measurements show whether the two models differ at the same baseline state. They do not independently establish that the difference is harmful.

---

## 8. Within-model reference measurements

The existence of $M_{TS}>0$ is expected because the teacher and student are different models. The cross-model mismatch must be compared with ordinary within-model variation.

Use cached states already visited by the baseline trajectory. Do not generate alternative audio paths.

### 8.1 Within-Seed-TTS change

For two consecutive teacher states immediately before handover:

$$
M_{TT}=
\frac{\lVert v_T(x_h,t_h)-v_T(x_{h-1},t_{h-1})\rVert_2}
{\lVert v_T(x_{h-1},t_{h-1})\rVert_2+\epsilon}.
$$

### 8.2 Within-student change

For the first two student states already produced after handover:

$$
M_{SS}=
\frac{\lVert v_S(x_{h+1},t_{h+1})-v_S(x_h,t_h)\rVert_2}
{\lVert v_S(x_h,t_h)\rVert_2+\epsilon}.
$$

These are diagnostic references extracted from the same unchanged baseline run. They are not Seed-TTS-only or student-only replacement synthesis systems.

### Important comparability requirement

Because the teacher and student may use unequal timestep intervals, raw consecutive-step differences can be misleading. Where possible, normalize by timestep distance:

$$
\widetilde M=\frac{M}{\Delta t}.
$$

Report both raw and time-normalized measurements.

---

## 9. Validation tests: not only the three-way comparison

The Seed-TTS→student, Seed-TTS→Seed-TTS, and student→student comparison is only the **numerical abnormality test**. It cannot prove practical harm by itself. The baseline problem-validation study must use several linked tests.

### Essential tests

1. **Same-state cross-model disagreement:** Measure Seed-TTS and student predictions at exactly the same cached baseline handover state.
2. **Three-way abnormality comparison:** Compare Seed-TTS→student mismatch against Seed-TTS→Seed-TTS and student→student variation, with timestep normalization.
3. **Mismatch–quality association:** Test whether larger handover mismatch corresponds to worse WER/CER, speaker similarity, naturalness, or artifact ratings in the unchanged baseline audio.
4. **High-versus-low mismatch comparison:** Define groups using mismatch values before examining quality, then test whether high-mismatch baseline outputs are worse.
5. **Seed and input repeatability:** Check whether the result persists across random seeds, speakers, and texts rather than appearing in isolated examples.

The minimum defensible evidence is Tests 1–3 plus repeatability. Without the mismatch–quality association, the study can show only that the models differ—not that the difference is a problem.

### Strong supporting tests

6. **Blinded listening:** Determine whether listeners hear lower naturalness, instability, or artifacts in high-mismatch baseline samples.
7. **Local boundary-profile analysis:** On cached states from the unchanged baseline trajectory, query both models at several points immediately before and after $t_h$. Determine whether disagreement is locally elevated around the handover rather than being equally large everywhere. These queries are diagnostic only and must not create alternative output paths.
8. **Pre-vocoder and post-vocoder consistency:** Examine the baseline acoustic representation before vocoding and the resulting waveform after vocoding. This helps determine whether the observed degradation is already present at the model transition or is mainly introduced by the vocoder.
9. **Confound-controlled analysis:** Test whether the mismatch–quality relationship remains after accounting for predicted duration, utterance length, speaker, text, seed, vector norm, and speaking rate.

These supporting tests strengthen causal interpretation, but none may modify the baseline trajectory or apply a proposed correction.

---

## 10. Numerical abnormality test

For every matched baseline trial, test whether:

$$
M_{TS}>M_{TT}
$$

and:

$$
M_{TS}>M_{SS}.
$$

Use one-sided paired Wilcoxon signed-rank tests because every trial provides all three measurements and the distributions may be non-normal.

Report:

- sample count;
- median and interquartile range;
- paired median differences;
- exact or appropriately calculated $p$-values;
- effect sizes;
- confidence intervals where feasible;
- number of trials in which $M_{TS}$ exceeds each within-model reference.

Statistical significance alone is insufficient. The magnitude of the difference must also be practically meaningful.

---

## 11. Baseline output-quality measurements

Evaluate the final unchanged baseline output using predefined metrics:

### Primary metrics

- Word Error Rate or Character Error Rate;
- speaker similarity;
- validated naturalness score such as UTMOS, if available;
- blinded listener naturalness rating;
- blinded listener artifact rating.

### Secondary metrics

- pitch variation and prosodic measurements;
- energy variation;
- spectral flux as a descriptive measure only;
- duration and speaking-rate measurements.

Spectral flux must not be treated as a standalone collision detector. It may reflect natural consonants, vocal onset, prosody, noise, silence, or vocoder behavior.

---

## 12. Mismatch–quality association

For each baseline trial, create one analysis row:

| Trial | Speaker | Text | Seed | $C_{TS}$ | $M_{TS}$ | $M_{TT}$ | $M_{SS}$ | WER | SIM | Naturalness | Artifact rating |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 |  |  |  |  |  |  |  |  |  |  |  |

Use Spearman rank correlations to test:

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

The central problem claim requires evidence that stronger baseline transition mismatch is associated with worse baseline speech.

The previously reported relationship between shock ratio and baseline peak flux,

$$
r=0.084,\qquad p=0.4159,
$$

must not be presented as supporting evidence because it was weak and statistically non-significant.

---

## 13. Predetermined high- and low-mismatch analysis

Create groups using $M_{TS}$ only, before inspecting output quality:

- **Low mismatch:** bottom quartile;
- **Medium mismatch:** middle two quartiles;
- **High mismatch:** top quartile.

Then compare the unchanged baseline outputs:

| Baseline group | Median WER | Mean/median SIM | Naturalness | Audible-artifact rate |
|---|---:|---:|---:|---:|
| Low mismatch |  |  |  |  |
| Medium mismatch |  |  |  |  |
| High mismatch |  |  |  |  |

The collision problem gains support if the high-mismatch group performs consistently and significantly worse than the low-mismatch group.

Do not create the groups based on WER, flux, listening results, or knowledge of which outputs sound defective.

---

## 14. Baseline-only blinded listening test

The listening test must use only unchanged baseline outputs.

Select baseline samples from the predetermined mismatch groups:

- approximately 10 high-mismatch outputs;
- approximately 10 low-mismatch outputs.

Randomize their order and hide all mismatch labels from listeners.

Ask listeners to rate:

1. intelligibility;
2. naturalness;
3. presence of clicks, roughness, unstable onset, or other artifacts;
4. speaker consistency.

Do not tell listeners that certain samples are expected to contain a collision. This avoids demand bias.

The collision problem gains perceptual support if high-mismatch baseline samples receive significantly worse blinded ratings.

---

## 15. Repeatability analysis

Because several random seeds are used for each prompt–text pair, determine whether:

- $M_{TS}$ remains similar across seeds;
- high-mismatch classification is stable;
- mismatch–quality relationships repeat across seeds;
- the finding is not driven by one speaker or sentence.

Use speaker-level aggregation or mixed-effects analysis where possible so that repeated trials from the same speaker are not incorrectly treated as fully independent observations.

---

## 16. What counts as support for the problem?

The teacher-to-student collision problem is supported only if the evidence establishes the following chain:

### A. Existence

Teacher and student predictions differ measurably at the shared baseline handover state.

### B. Abnormality

Cross-model mismatch is significantly and meaningfully greater than time-normalized within-teacher and within-student variation.

### C. Quality relevance

Greater cross-model mismatch predicts worse predefined baseline quality outcomes.

### D. Group difference

High-mismatch baseline outputs are worse than low-mismatch baseline outputs.

### E. Repeatability

The relationships persist across speakers, texts, and seeds.

### F. Perceptual relevance

Blinded listeners rate high-mismatch baseline outputs as less natural or more artifact-prone.

---

## 17. Interpretation matrix

| Experimental result | Defensible conclusion |
|---|---|
| $M_{TS}$ is not greater than within-model variation | No abnormal cross-model transition is supported |
| $M_{TS}$ is greater, but unrelated to quality | Numerical discontinuity exists, but practical harm is unestablished |
| $M_{TS}$ is greater and predicts objective degradation | Objective collision problem supported |
| Objective relationship plus worse blinded ratings | Strong practical collision problem supported |
| Relationship occurs only in a small subgroup | Collision is an input-dependent robustness problem, not a universal defect |
| Relationship disappears across seeds | Initial finding is unstable and requires rejection or further validation |

---

## 18. Claims prohibited during this phase

Until the complete evidence chain is satisfied, do not claim that:

- DMOSpeech 2 has a proven collision;
- direct handover is universally unstable;
- a nonzero teacher–student displacement is automatically harmful;
- cosine similarity proves vector identity;
- a chosen angle threshold defines collision;
- spectral flux proves an audible seam;
- a proposed method solves the problem;
- the problem is severe across all inputs.

This phase must report negative findings honestly if the collision hypothesis is not supported.

---

## 19. Defensible problem statement

> The evaluated DMOSpeech 2 baseline transfers a Seed-TTS-derived intermediate state directly to its distilled student model. Although this handover is functional, Seed-TTS and the student represent separately learned vector fields, and the magnitude and quality implications of their disagreement at the shared handover state have not been fully characterized. This study evaluates only the unmodified Seed-TTS-to-student baseline to determine whether cross-model disagreement exceeds ordinary within-model trajectory variation and whether greater disagreement is associated with degraded intelligibility, speaker similarity, naturalness, or perceived transition quality.

---

## 20. Final decision rule

The core problem claim will be accepted only when baseline-only evidence supports:

$$
M_{TS}>M_{TT},M_{SS}
$$

and:

$$
\text{greater }M_{TS}\longrightarrow\text{worse baseline speech quality}.
$$

If only the first relationship is observed, the study may claim a numerical cross-model discontinuity but not a harmful collision.

If neither relationship is observed, the collision problem is not empirically supported and should not be presented as a defect requiring correction.

## Final reminder

> Generate and evaluate only the unchanged teacher-to-student baseline during problem validation. Diagnostic teacher and student queries may measure the baseline transition, but they must not alter the state, produce competing solution outputs, or introduce any cushioning mechanism.
