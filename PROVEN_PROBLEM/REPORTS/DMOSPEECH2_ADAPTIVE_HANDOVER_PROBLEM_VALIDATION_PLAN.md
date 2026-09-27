# DMOSpeech 2 Adaptive Handover Problem-Validation Plan

**Research focus:** Determine whether DMOSpeech 2's fixed teacher-to-student switching policy creates a robustness or efficiency limitation before developing a new adaptive handover method.

**Important boundary:** This study does not claim that the normal handover is broken. Previous controlled testing did not establish a general handover collision. The new investigation concerns whether a fixed switching time is suboptimal across different inputs and conditions.

---

## 1. Two Defensible Handover Research Options

### Option 2 — Robustness or Stress-Testing Thesis

Research question:

> Under what conditions does teacher-to-student handover become unstable?

The potential problem is:

> The default handover is stable under ordinary conditions but may lack robustness under challenging transition conditions.

Possible stress conditions include:

- earlier or later switching;
- different latent-noise seeds;
- duration underestimation or overestimation;
- out-of-domain speakers;
- long target sentences;
- emotional or expressive speech;
- noisy prompt audio;
- mismatched languages;
- fewer remaining student steps.

This problem becomes defensible only if specific conditions repeatedly cause degradation and changing the switching time helps recover quality.

### Option 3 — Adaptive Efficiency Thesis

Research question:

> Can the system determine when teacher guidance is no longer necessary?

The potential problem is:

> DMOSpeech 2 uses a fixed handover time for every input despite possible differences in sentence complexity, speaker characteristics, duration, and teacher-student trajectory convergence.

This problem becomes defensible if:

- some samples can safely transition earlier;
- other samples require additional teacher guidance;
- the fixed switching point wastes computation or reduces quality;
- an adaptive policy improves the quality-latency trade-off.

---

## 2. Core Switch-Time Sweep

For every prompt-target pair, generate speech using several transition times:

$$
t_{\text{switch}}\in\{0.05,\ 0.10,\ 0.15,\ 0.25,\ 0.40,\ 0.50\}.
$$

The paper configuration at approximately $t_{\text{switch}}=0.25$ serves as the fixed-policy reference.

| Switch time | Meaning |
| :---: | :--- |
| 0.05 | Teacher stops extremely early |
| 0.10 | Early transition |
| 0.15 | Moderately early transition |
| 0.25 | DMOSpeech 2 fixed reference |
| 0.40 | Extended teacher guidance |
| 0.50 | Late transition |

For every output, record:

- WER or CER;
- speaker similarity;
- UTMOS or another naturalness measure;
- pitch/prosodic diversity;
- actual wall-clock latency;
- number of teacher and student evaluations;
- total neural function evaluations;
- teacher-student velocity agreement at candidate switch points.

All conditions must use the same prompt, target text, duration, initial noise, checkpoint, vocoder, and evaluation pipeline.

---

## 3. Proving a Robustness Problem

### Step 1 — Establish Normal Performance

Evaluate clean, ordinary inputs using the default switch:

$$
t_{\text{switch}}=0.25.
$$

This confirms that DMOSpeech 2 normally performs well and prevents the thesis from falsely claiming a universal handover failure.

### Step 2 — Apply Controlled Stress Conditions

Change one factor at a time. Recommended initial factors are:

1. long target text;
2. noisy prompt audio;
3. predicted-duration perturbation.

For duration perturbation, predetermined levels might include:

$$
-15\%,\ -10\%,\ 0\%,\ +10\%,\ +15\%.
$$

Do not combine multiple stresses during the first experiment because the cause of degradation would become ambiguous.

An example screening result might look like this:

| Condition | Default-switch WER |
| :--- | :---: |
| Clean prompt | 2% |
| Mild noise | 4% |
| Strong noise | 18% |
| Long sentence | 15% |
| Duration $-15\%$ | 22% |

This table alone would show model degradation, but it would not prove a handover problem. The entire model might simply be sensitive to those conditions.

### Step 3 — Determine Whether Switch Timing Controls the Failure

For each condition that fails at the default switch, compare early, default, and late switching:

$$
t_{\text{switch}}\in\{0.10,\ 0.25,\ 0.40\}.
$$

Example:

| Noisy-prompt condition | Early switch | Default switch | Late switch |
| :--- | :---: | :---: | :---: |
| WER | 25% | 18% | **6%** |
| Speaker similarity | 0.68 | 0.73 | **0.82** |

If later switching repeatedly restores quality, the failure is plausibly connected to insufficient teacher guidance before transition.

### Robustness Decision Rule

The handover-robustness problem is supported only if:

1. The default transition performs well under ordinary conditions.
2. It repeatedly degrades under predefined challenging conditions.
3. Changing only the transition time significantly restores quality.
4. The recovery repeats across utterances and noise seeds.
5. The effect is not explained solely by general duration, prompt, language, or model failure.

If every tested switch time fails under a particular stress, that result indicates general model sensitivity—not handover robustness.

---

## 4. Proving an Adaptive-Efficiency Problem

### Step 1 — Determine the Earliest Safe Switch Per Sample

For each utterance, test all candidate transition times. Define quality constraints before examining the results. Example limits are:

$$
\mathrm{WER}\leq3\%,
$$

$$
\Delta\mathrm{SIM}\leq0.02,
$$

$$
\Delta\mathrm{Naturalness}\leq0.10
$$

relative to the fixed-reference output or another predefined quality anchor.

The earliest candidate satisfying every constraint becomes the utterance's **earliest safe switch**.

Example:

| Sample | 0.05 | 0.10 | 0.15 | 0.25 | 0.40 | Earliest safe switch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Easy sentence | Fail | Pass | Pass | Pass | Pass | **0.10** |
| Medium sentence | Fail | Fail | Pass | Pass | Pass | **0.15** |
| Difficult sentence | Fail | Fail | Fail | Pass | Pass | **0.25** |
| Very difficult sentence | Fail | Fail | Fail | Fail | Pass | **0.40** |

Meaningful variation among samples supports the claim that one fixed transition time is not universally optimal.

### Step 2 — Quantify Wasted Teacher Computation

Compare the paper's fixed switch with an oracle that selects each sample's earliest safe switch.

Example:

| Policy | Average teacher evaluations | WER | Latency |
| :--- | :---: | :---: | :---: |
| Fixed switch | 14 | 2.0% | 8.0 s |
| Oracle earliest-safe switch | 8 | 2.1% | 5.5 s |

If many inputs can safely transition earlier, the fixed policy spends unnecessary teacher computation without a meaningful quality benefit.

### Step 3 — Identify Samples Requiring Later Switching

Test whether samples that fail at $t=0.25$ recover when the teacher continues to $t=0.40$ or $t=0.50$.

A particularly strong result would show both:

- easy inputs can switch earlier and save computation;
- difficult inputs require later switching to preserve quality.

This would demonstrate that a fixed transition cannot efficiently serve both input types.

### Step 4 — Predict the Safe Switch During Inference

Testing every switch time is acceptable only for establishing the oracle labels. A deployable method must select the switch using signals available during a single inference run.

Possible signals include:

$$
a_t=\cos(\mathbf{v}_T,\mathbf{v}_S),
$$

$$
d_t=\frac{\|\mathbf{v}_T-\mathbf{v}_S\|_2}{\|\mathbf{v}_T\|_2+\epsilon},
$$

as well as:

- target-text length;
- predicted duration;
- teacher velocity magnitude;
- change in teacher velocity;
- latent-state stability;
- speaker or prompt characteristics.

A threshold-based controller might use:

$$
\text{Switch if }a_t\geq\tau_a
\quad\text{and}\quad
d_t\leq\tau_d.
$$

These signals are candidates, not yet validated predictors. They must be tested against the oracle earliest-safe switch on held-out samples.

### Adaptive-Efficiency Decision Rule

The adaptive-efficiency problem is supported only if:

1. Earliest safe transition times vary meaningfully among samples.
2. A substantial proportion of samples can transition earlier than $0.25$.
3. Some difficult samples benefit from extended teacher guidance, if such cases exist.
4. The per-sample oracle improves the quality-latency trade-off over the fixed policy.
5. Observable signals predict safe transition decisions on held-out samples.

---

## 5. Staged Low-Cost Experiment

### Phase 1 — Clean Switch-Time Sweep

Use 10–20 predetermined clean utterances:

$$
10\text{ samples}\times6\text{ switch times}=60\text{ outputs}
$$

for the 10-sample version.

This phase primarily tests adaptive efficiency.

Possible conclusions:

- Different samples have different earliest safe switches: adaptive efficiency is supported.
- Every sample performs best at the same fixed time: adaptive efficiency is not supported.
- Switch time makes little practical difference: no meaningful switch-timing problem is demonstrated.

### Phase 2 — Focused Stress Test

Select three predefined stress factors, such as:

- long target text;
- noisy prompt;
- duration perturbation.

Test three important switch times:

$$
t_{\text{switch}}\in\{0.10,\ 0.25,\ 0.40\}.
$$

For 10 inputs and three stresses:

$$
3\text{ stresses}\times10\text{ samples}\times3\text{ switch times}=90\text{ outputs}.
$$

This phase primarily tests robustness.

To reduce computation further, Phase 2 should be performed only if Phase 1 or a smaller screening run suggests that switch timing materially affects quality.

---

## 6. Interpretation Matrix

| Experimental finding | Supported interpretation |
| :--- | :--- |
| Different samples require different safe switch times | Adaptive-efficiency problem supported |
| Clean inputs tolerate earlier switching | Fixed policy wastes teacher computation |
| Difficult inputs improve with later switching | Fixed policy can transition prematurely |
| Stress failures are rescued by later switching | Handover-robustness limitation supported |
| Every transition time fails under stress | General model robustness issue, not handover-specific |
| Outputs are similar across all transition times | Switch timing may not be consequential |
| The fixed $0.25$ transition is consistently best | No strong adaptive-switching problem demonstrated |
| Alignment signals fail to predict safe switching | Proposed predictor is unsuitable even if oracle variation exists |

---

## 7. Recommended Combined Thesis Direction

Begin with adaptive efficiency because it does not require the normal handover to be defective.

### Proposed Problem Statement

> DMOSpeech 2 employs a fixed teacher-to-student transition time for all inputs. However, utterances may differ in the amount of teacher guidance required before safe student continuation. The efficiency and quality implications of this fixed transition policy have not been independently characterized.

### Proposed Main Objective

> To characterize input-dependent teacher-student convergence and develop an adaptive transition-time policy that reduces unnecessary teacher computation while preserving speech quality and extending teacher guidance when challenging inputs require it.

### Possible Thesis Title

> **Input-Adaptive Teacher-to-Student Transition Timing for Efficient and Robust Flow-Matching Speech Synthesis**

### Research Questions

1. Do different utterances have meaningfully different earliest safe teacher-to-student transition times?
2. How much teacher computation can be removed without materially reducing intelligibility, speaker similarity, naturalness, or prosodic diversity?
3. Which observable trajectory and input features best predict the earliest safe transition?
4. Does an adaptive transition policy improve the quality-latency trade-off over DMOSpeech 2's fixed transition?
5. Under challenging inputs, can extended teacher guidance improve robustness compared with the fixed transition?

---

## 8. Final Recommendation

One switch-time experiment can screen both thesis options:

- **Option 2:** Is fixed handover insufficiently robust under difficult conditions?
- **Option 3:** Does fixed handover use unnecessary teacher computation on easy inputs?

Start with the 10-sample, six-switch-time clean sweep. This requires approximately 60 outputs and provides the fastest evidence about whether switch timing matters. Do not build a new adaptive algorithm until the sweep demonstrates meaningful sample-to-sample variation or a repeatable robustness effect.

If the sweep supports both early-switch savings and late-switch recovery, combine the two directions into one thesis:

> An input-adaptive transition policy selects earlier handover for converged samples and preserves teacher guidance for difficult samples, improving efficiency and robustness relative to a fixed switching policy.

If the sweep shows that $t=0.25$ is consistently optimal or that switching time has little effect, discontinue the adaptive-handover direction and pursue the few-step inference-collapse problem instead.
