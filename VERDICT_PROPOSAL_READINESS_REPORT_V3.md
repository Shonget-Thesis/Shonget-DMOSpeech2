# Verdict: Proposal Readiness of the Latest Scheduling Report

Reviewed document: `CHAPTER3_MASTER_RESEARCH_REPORT(3).md`

## Decision

**Suitable for adviser consultation now. Revise before formal proposal presentation.** The study has substantial experimental work and a plausible contribution, but the current report overstates several conclusions. A full experimental rerun is not presently justified.

This verdict reviews the report's stated results and internal reasoning. The implementation, raw records, and audio have not been independently verified.

## Summary of the reported findings

| Finding | Defensible interpretation |
| --- | --- |
| Proposed mean NFE is 18.15 versus 21 for Step 8 | A reported 13.6% reduction in model evaluations. This is not automatically a 13.6% runtime reduction. |
| CPU held-out added failures: 5/200 (2.5%) | Meets the stated one-sided 90%-confidence verification criterion: binomial p approximately 0.0623. |
| GPU held-out added failures: 3/200 (1.5%) | Encouraging reference-relative reliability on the same test cohort, subject to verification of the evaluation pipeline. |
| Clean switching decisions agree for 200/200 CPU–GPU cases | Supports consistent decisions on those tested cases; does not prove identical audio or universal hardware invariance. |
| GPU SNR-only: 4/200 failures, proposed: 3/200 | A small descriptive difference. The reported exact paired test p = 1.0 does not establish superiority. |
| GPU timing measures student completion plus vocoder only | Does not establish total GPU synthesis latency or a fair speedup against complete CPU synthesis. |

The failure measure concerns added degradation relative to each matched Step-8 output. Zero added failure does not establish good absolute speech quality. In particular, noisy prompts routed to Step 8 inherit the reference output and its quality limitations.

## Main weakness: the simpler baseline is competitive

SNR-only switches clean prompts early and sends noisy prompts to Step 8. The proposed method adds trajectory drift checks to select among teacher stopping points.

On the GPU held-out split, SNR-only uses **17.04 NFE**, compared with **18.15 NFE** for the proposed method, and has **4 failures rather than 3**. The proposed method therefore spends additional computation for one fewer observed failure. The evidence does not establish that this difference is a consistent advantage.

The GPU calibration table also reports **5/200 SNR-only failures**. Under the report's stated single-policy binomial criterion, this corresponds to p approximately **0.0623**, below delta = 0.10. Provided that the SNR-only policy was fixed independently of calibration and the records are valid, it can satisfy the same certification criterion. Therefore, the report cannot broadly claim that SNR-only is insufficient or that trajectory probing is necessary.

The CPU comparison is more favorable descriptively (11 versus 5 failures), but its reported two-sided McNemar p = 0.0703 also does not establish superiority at the conventional 0.05 level. GPU results expose an uncertainty already present in the CPU evidence.

## Required corrections before formal presentation

1. **Fix the diagram's held-out p-value.** CPU 5/200 corresponds to approximately 0.0623. The value 0.0264 belongs to calibration 4/200.
2. **Remove the 17.1× CPU–GPU speedup claim.** It compares complete CPU generation with only GPU student/vocoder time. Remove the complete-system real-time claim until teacher computation is included.
3. **Replace quality-invariance and perceptual-equivalence claims.** Non-significant metric differences do not establish equivalence. Report the actual comparisons and their uncertainty.
4. **Correct the cross-split threshold comparison.** Development currently uses lambda = 0.60 under a row labeled lambda = 0.50. Recalculate or clearly distinguish the policies.
5. **Revise the SNR-only discussion.** State that it is a competitive baseline and that the added benefit of trajectory probing is inconclusive in the GPU held-out results.
6. **Remove untested causal explanations.** Do not attribute audio differences specifically to vocoder arithmetic or claim that drift proves acoustic stabilization without direct evidence.
7. **Clarify statistical units and hardware settings.** Respect pairing of clean/noisy cases within speakers, state the exact Windows edition, and retain precision and microbenchmark claims only when verified.

## Next action without wasting runs

**Audit saved records before generating more audio.** Trace speakers whose CPU/GPU failure classifications changed. Compare configuration, generated audio, ASR transcriptions, similarity scores, and matched reference scores.

- If a bug is identified, rerun the affected scope.
- If the records are valid, retain both sets of results and disclose the uncertainty.
- If total GPU latency is required, run a small synchronized timing benchmark including teacher, scheduler, student, and vocoder.

Do not repeatedly tune the method on the existing held-out test set to force an advantage. Any redesign requires development data and a fresh final evaluation or a clearly disclosed exploratory analysis.

## Defensible proposal framing

> This study investigates whether risk-calibrated teacher-to-student handover scheduling can reduce model evaluations while controlling reference-relative quality degradation in DMOSpeech 2. It compares the proposed policy with fixed schedules and a simpler SNR-only policy, and evaluates the added value of trajectory drift probing.

The current results support feasibility against fixed Step 8. They do not yet establish superiority over SNR-only. Determining whether the additional mechanism is worthwhile remains a legitimate research question, and an inconclusive or negative comparison must be reported honestly.

## Final verdict

**Continue with adviser consultation. Correct the report before formal proposal presentation. No full rerun is currently necessary.** The work has a defensible experimental question and encouraging compute savings, while the trajectory component's incremental contribution remains unresolved.
