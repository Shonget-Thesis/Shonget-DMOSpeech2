# Current Verdict on the DMOSpeech 2 Handover Problem

## Overall verdict

The **handover direction is viable as a thesis**, but the problem must be framed accurately.

The current evidence does **not** prove that the normal teacher-to-student handover is broken, unstable, or harmful. Instead, it provides preliminary evidence that DMOSpeech 2's **fixed switching schedule may be computationally overconservative for ordinary inputs and suboptimal across different input conditions**.

The defensible research problem is therefore:

> DMOSpeech 2 uses the same fixed teacher-to-student switching time for every utterance, even though different generation trajectories and input conditions may require different amounts of teacher guidance. This fixed policy may perform unnecessary teacher computation on easy inputs and may not provide the best quality-efficiency trade-off under challenging conditions.

## What the experiments currently support

### 1. Some clean samples can switch earlier

The 10-sample switch-time sweep found that, under the study's current acceptance criteria:

- five samples could switch at \(t=0.05\);
- five samples required \(t=0.10\);
- all ten samples could switch before the reference \(t=0.25\) schedule;
- the per-sample oracle reduced mean counted NFE from 13 to 8 while approximately preserving WER and speaker similarity.

This is preliminary evidence that the fixed \(t=0.25\) schedule can use more teacher computation than necessary on the tested clean samples.

### 2. Switching time matters under stressed conditions

The stress test found that different switch times produced different results under duration errors and noisy prompts. In the noisy condition, for example, the switch time with the lowest WER was not the same as the switch time with the highest speaker similarity.

This supports the argument that there may not be one universally optimal switching time for every condition.

### 3. The preliminary controller reduced counted computation

On 10 held-out clean utterances, Adaptive TAGS produced:

| Metric | Fixed \(t=0.25\) | Adaptive TAGS | Difference |
|---|---:|---:|---:|
| Mean counted NFE | 13.0 | 8.2 | \(-36.9\%\) |
| Mean WER | 2.54% | 2.54% | No observed aggregate change |
| Mean speaker similarity | 0.8525 | 0.8474 | \(-0.0051\) |
| Measured latency | 37.15 | 34.56 | Approximately \(-7.0\%\) |

These results support a preliminary **efficiency claim**, but not yet a complete superiority claim.

## What has not yet been proven

### The handover itself is not proven to be defective

The results do not demonstrate a collision, discontinuity, audible seam, or systematic quality loss caused by passing the teacher state directly to the student. Therefore, the thesis should not claim that DMOSpeech 2 has a broken handover mechanism.

### Adaptive switching is not yet proven necessary

Every held-out controller decision selected either \(t=0.05\) or \(t=0.10\), while no sample used the \(t=0.25\) fallback. The evaluation did not compare Adaptive TAGS against strong static alternatives such as fixed \(t=0.10\) and fixed \(t=0.15\).

An examiner could therefore ask:

> Why is an adaptive controller necessary if switching every sample at \(t=0.10\) produces equivalent results?

Until this comparison is performed, the study has stronger evidence for **earlier switching** than for **adaptive switching**.

### Robustness of the controller is unproven

The noisy samples exhibited high cosine alignment even when early switching produced poor WER. A controller based mainly on a cosine threshold may therefore incorrectly classify a noisy transition as safe.

The existing stress test establishes that switch timing matters, but it does not establish that the current Adaptive TAGS controller selects the correct time under those stresses.

### Naturalness has not been established

WER measures intelligibility, while speaker similarity measures identity preservation. Neither directly proves naturalness, smoothness, or the absence of an audible transition artifact. Claims about full naturalness require an appropriate perceptual metric or listening evaluation.

## Essential next experiment

Evaluate the following methods on the same held-out samples:

| Method | Purpose |
|---|---|
| Fixed \(t=0.05\) | Aggressive static baseline |
| Fixed \(t=0.10\) | Strong early-switch competitor |
| Fixed \(t=0.15\) | Balanced static competitor |
| Fixed \(t=0.25\) | DMOSpeech 2 reference schedule |
| Adaptive TAGS | Proposed method |
| Per-sample oracle | Upper bound for adaptive selection |

For every method, report:

- WER;
- speaker similarity;
- a naturalness or perceptual-quality measure;
- counted NFE, including any student probing calls;
- repeated model-only and end-to-end latency;
- unsafe early-switch rate;
- regret relative to the per-sample oracle;
- paired confidence intervals or a non-inferiority analysis.

The actual controller should also be evaluated under noisy prompts, duration errors, and other challenging conditions—not only the individual fixed switch times.

## Recommended thesis claim

> Preliminary experiments indicate that the fixed \(t=0.25\) teacher-to-student schedule is computationally conservative for the tested clean Seed-TTS utterances. A threshold-based controller reduced counted NFE by 36.9% on 10 held-out utterances while preserving the observed aggregate WER, although measured latency improved by only approximately 7.0% and mean speaker similarity decreased slightly. General superiority over strong fixed early-switch baselines and robustness under stressed conditions remain to be established.

## Rating

| Aspect | Current assessment |
|---|---:|
| Viability of the handover thesis direction | Good |
| Evidence that the fixed schedule may be inefficient | Preliminary but credible |
| Evidence that the normal handover is broken | Not established |
| Evidence that adaptation is better than an earlier fixed switch | Not established yet |
| Proposal readiness | **7.5/10** |
| Final-defense readiness | **5.5/10** |

## Final conclusion

Adaptive TAGS is **not a bad thesis concept**. It has become a plausible efficiency and robustness study. However, its novelty cannot rest merely on inserting a mathematical bridge between teacher and student, because DMOSpeech 2 already performs a teacher-to-student transition successfully.

The meaningful contribution must be demonstrated as a **data-driven switching policy that chooses better transition times than both the original schedule and strong fixed early-switch alternatives**. If Adaptive TAGS beats or provides a better quality-efficiency trade-off than fixed \(t=0.10\) and \(t=0.15\), particularly across clean and stressed inputs, the thesis becomes substantially stronger. If a single earlier fixed switch performs equally well, the contribution should instead be reframed as handover schedule optimization rather than adaptive control.
