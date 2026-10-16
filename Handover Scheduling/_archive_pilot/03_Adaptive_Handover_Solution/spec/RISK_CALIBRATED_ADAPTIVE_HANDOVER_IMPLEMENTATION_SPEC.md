# Risk-Calibrated Adaptive Handover: Implementation and Testing Specification

## 1. Purpose

This document converts the proposed Chapter 3 methodology into an implementation plan for the public DMOSpeech 2 inference pipeline.

The proposed method is a **risk-calibrated, per-utterance teacher-to-student handover policy**. It attempts to stop teacher guidance at the earliest safe candidate step using:

1. an endpoint-stabilization signal;
2. a prompt-SNR safety gate;
3. a threshold calibrated on separate data; and
4. a safe fallback to the reference DMOSpeech 2 configuration.

This document is an implementation specification, not evidence that the method already works.

---

## 2. Core Research Problem

DMOSpeech 2 uses a predetermined teacher-to-student handover configuration for every input. Preliminary experiments suggest that the cheapest acceptable handover differs between inputs.

The proposed algorithm therefore seeks to solve:

\[
\min_{\pi}\; \mathbb{E}_{x\sim D}[C(\pi(x))]
\quad\text{subject to}\quad
R(\pi)\leq\alpha,
\]

where:

- \(x\) is an input utterance and its conditioning information;
- \(\pi(x)\) is the selected handover configuration;
- \(C(\pi(x))\) is its computational cost;
- \(R(\pi)\) is the probability of an added failure relative to the reference; and
- \(\alpha\) is the maximum tolerated added-failure risk.

---

## 3. Proposed Solution

### 3.1 Name

**Risk-Calibrated Adaptive Teacher-to-Student Handover**

### 3.2 Main idea

During teacher inference, estimate the teacher's final endpoint at each candidate step. Compare the current estimate with the preceding estimate.

- If the endpoint estimate has changed only slightly, the teacher trajectory is considered sufficiently stable.
- If the prompt is also sufficiently clean, the method may hand generation to the student.
- If either condition fails, teacher inference continues.
- If no early candidate passes, the method executes the reference handover.

### 3.3 Decision rule

At a candidate step \(i\), switch early when:

\[
d_i\leq\lambda
\quad\text{and}\quad
\widehat{\mathrm{SNR}}\geq\theta.
\]

Here:

- \(d_i\) is the endpoint-stabilization score;
- \(\lambda\) is the calibrated stabilization threshold;
- \(\widehat{\mathrm{SNR}}\) is the prompt-SNR estimate; and
- \(\theta\) is the minimum permitted prompt-SNR threshold.

The policy selects the **first** candidate step satisfying both conditions. If none qualifies, it selects the reference step.

---

## 4. Endpoint-Stabilization Signal

### 4.1 Endpoint estimate

At teacher time \(t_i\), compute the deterministic endpoint estimate:

\[
\widehat{x}_1(t_i)
=
x_{t_i}+(1-t_i)v(x_{t_i},t_i),
\]

where \(v(x_{t_i},t_i)\) is the deterministic guided teacher velocity.

Do not add a new teacher forward pass solely to compute this estimate. Reuse the guided velocity already evaluated for the Euler update.

### 4.2 Stabilization score

For \(i\geq1\):

\[
d_i=
\frac{
\left\|\widehat{x}_1(t_i)-\widehat{x}_1(t_{i-1})\right\|_M
}{
\left\|\widehat{x}_1(t_i)\right\|_M+\varepsilon
}.
\]

The norm must be computed only over generated, non-prompt frames selected by mask \(M\). Add a small numerical constant \(\varepsilon\), such as \(10^{-8}\), to prevent division by zero.

### 4.3 Interpretation

- Small \(d_i\): consecutive endpoint estimates are similar; the teacher prediction may have stabilized.
- Large \(d_i\): the predicted endpoint is still changing; retain teacher guidance.

This is a proposed observable proxy for readiness. Its association with safe handover must be tested on the development split.

---

## 5. Prompt-SNR Gate

### 5.1 Proposed estimator

1. Divide the prompt waveform into 25 ms frames.
2. Use a 10 ms hop.
3. Compute the energy of every frame.
4. Estimate signal energy using the mean energy of the upper half of frames.
5. Estimate noise energy using the mean energy of the lowest 10% of frames.
6. Compute:

\[
\widehat{\mathrm{SNR}}
=10\log_{10}
\left(
\frac{E_{\mathrm{upper\ half}}+arepsilon}
{E_{\mathrm{lowest\ 10\%}}+\varepsilon}
\right).
\]

### 5.2 Function skeleton

```python
def estimate_prompt_snr(waveform, sample_rate, eps=1e-8):
    frame_length = round(0.025 * sample_rate)
    hop_length = round(0.010 * sample_rate)

    frames = frame_waveform(waveform, frame_length, hop_length)
    energy = frames.square().mean(dim=-1)

    sorted_energy = energy.sort().values
    noise_count = max(1, round(0.10 * len(sorted_energy)))
    upper_start = len(sorted_energy) // 2

    noise_energy = sorted_energy[:noise_count].mean()
    signal_energy = sorted_energy[upper_start:].mean()

    return 10.0 * torch.log10(
        (signal_energy + eps) / (noise_energy + eps)
    )
```

The estimator must first be checked against prompts with known added-noise levels. It is a proxy rather than a guaranteed physical SNR estimator.

---

## 6. Candidate Handover Configurations

Use the public implementation's actual time grid. Chapter 3 currently proposes:

| Stopping argument \(s\) | Actual teacher time \(t_h\) | Teacher evaluations | Cost with \(j=1\) | Cost with \(j=2\) |
| ---: | ---: | ---: | ---: | ---: |
| 0.07 | 0.0761 | 5 | 13 | 12 |
| 0.10 | 0.1181 | 6 | 15 | 14 |
| 0.15 | 0.1685 | 7 | 17 | 16 |
| 0.25 | 0.2929 | 9 | 21 | 20 |

The reference is currently defined as:

\[
k_{\mathrm{ref}}=(s=0.25,j=1).
\]

The exact cost identity proposed in Chapter 3 is:

\[
C(s,j)=2(k(s)+1)+(4-j).
\]

The factor of two counts the teacher's conditional and unconditional evaluations. The remaining term counts student evaluations.

### Required verification before experimentation

The paper reportedly describes a 14-teacher-plus-2-student configuration, while the current Chapter 3 cost reconstruction from the public code gives a different count. Before experiments:

1. record the exact repository commit;
2. record all configuration and checkpoint hashes;
3. instrument actual teacher and student forward calls;
4. compare measured calls against the cost formula; and
5. resolve which configuration reproduces the published baseline.

Do not report computational savings until this discrepancy is resolved.

---

## 7. Student Restart Map

Let \(j=\rho(i)\) determine the student's restart index after stopping at teacher step \(i\).

The proposed student grid is:

\[
\tau\in\{0,0.0761,0.2929,0.6173\}.
\]

Initially use the reference restart index unless the restart pilot demonstrates that an alternative is non-inferior in both WER and SIM under the predetermined margins.

Recommended safe initial implementation:

```python
def restart_index_for_step(step):
    return 1
```

Only introduce a variable restart map after the main switching-time implementation has been verified.

---

## 8. Inference Algorithm

### 8.1 Pseudocode

```text
Input:
    prompt audio
    prompt text
    target text
    calibrated stabilization threshold lambda
    development-set SNR gate theta
    ordered candidate steps
    reference step
    restart map rho

Estimate prompt SNR once.
previous_endpoint = None

For every teacher-grid step i:
    Evaluate guided teacher velocity v_i.
    Compute deterministic endpoint estimate endpoint_i.

    If previous_endpoint exists:
        Compute stabilization score d_i on generated frames only.

    If i is the reference step:
        Select i and stop teacher inference.

    Else if i is a candidate step:
        If d_i <= lambda AND estimated SNR >= theta:
            Select i and stop teacher inference.

    Perform the normal teacher Euler update.
    previous_endpoint = endpoint_i.

Construct the final teacher endpoint using the baseline stochastic convention.
Start the student at rho(i).
Run the remaining student steps.
Decode using Vocos.
Return audio and telemetry.
```

### 8.2 Python-oriented skeleton

```python
@torch.no_grad()
def adaptive_handover_sample(
    prompt_waveform,
    prompt_text,
    target_text,
    lambda_threshold,
    snr_threshold,
    candidate_steps,
    reference_step,
    restart_map,
    seed,
):
    set_all_seeds(seed)
    snr_hat = estimate_prompt_snr(prompt_waveform, sample_rate)

    previous_endpoint = None
    selected_step = reference_step
    selected_score = None
    telemetry = []

    for i in teacher_grid_steps:
        velocity = evaluate_guided_teacher_velocity(state, time_grid[i])
        endpoint = state + (1.0 - time_grid[i]) * velocity

        score = None
        if previous_endpoint is not None:
            score = masked_relative_rms(
                endpoint - previous_endpoint,
                endpoint,
                generated_frame_mask,
            )

        qualifies = (
            i in candidate_steps
            and i != reference_step
            and score is not None
            and score <= lambda_threshold
            and snr_hat >= snr_threshold
        )

        telemetry.append({
            "step": i,
            "time": float(time_grid[i]),
            "stabilization_score": to_float(score),
            "snr_estimate_db": to_float(snr_hat),
            "qualifies": bool(qualifies),
        })

        if qualifies or i == reference_step:
            selected_step = i
            selected_score = score
            teacher_endpoint = construct_baseline_endpoint(
                state, velocity, time_grid[i]
            )
            break

        state = euler_update(state, velocity, time_grid[i], time_grid[i + 1])
        previous_endpoint = endpoint.detach()

    restart_index = restart_map[selected_step]
    student_output = run_student_from_endpoint(
        teacher_endpoint,
        restart_index=restart_index,
        seed=seed,
    )
    waveform = vocoder(student_output)

    return waveform, {
        "selected_step": selected_step,
        "selected_time": float(time_grid[selected_step]),
        "selected_score": to_float(selected_score),
        "snr_estimate_db": to_float(snr_hat),
        "restart_index": restart_index,
        "trace": telemetry,
    }
```

Adapt function names to the actual DMOSpeech 2 repository. Preserve the baseline's stochastic-term and seed behavior exactly.

---

## 9. Added-Failure Definition

For input \(x\) and candidate configuration \(k\), define an added failure relative to the reference output as:

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

Otherwise:

\[
f(x,k)=0.
\]

The draft mentions preliminary values:

- \(c=12.5\%\) absolute WER ceiling;
- \(m_W=1\) percentage point; and
- \(m_S=0.02\).

These values are **not yet final**. They must be justified and frozen before inspecting calibration outcomes.

By construction, the reference configuration has zero added failure relative to itself. This does not mean the reference has perfect absolute quality.

---

## 10. Dataset Separation

Use speaker-disjoint splits:

| Split | Permitted use |
| --- | --- |
| Development | Select candidates, study signals, choose SNR gate, decide restart map and freeze margins |
| Calibration | Select and certify \(\lambda\) only |
| Test | Final evaluation only; no tuning |

Recommended safeguards:

- At most one utterance per speaker in calibration and test where feasible.
- Never tune \(\lambda\), \(\theta\), margins or candidates using test results.
- If any parameter is changed after examining the test split, that split is no longer a valid final test set.
- Use the same random draws for all configurations of the same input.
- Run all networks in evaluation mode.

---

## 11. Development-Set Procedure

For every development input and condition:

1. Run every candidate \((s,j)\) configuration with the same seed.
2. Run the reference configuration.
3. Record WER, SIM, UTMOS, teacher calls, student calls, total forward passes, wall-clock time and peak memory.
4. Record \(d_i\) at every candidate teacher step.
5. Record the estimated prompt SNR.
6. Calculate the added-failure indicator for every candidate.
7. Identify the cheapest non-failing configuration for each input. This is the diagnostic oracle.
8. Examine whether \(d_i\) and SNR are associated with the oracle decision.

Do not continue to calibration if the stabilization score carries no useful information about safe earlier handover. A negative result should be reported rather than hidden.

---

## 12. Threshold Calibration

### 12.1 Threshold grid

Create a fixed ordered grid:

\[
\Lambda=\{\lambda_1,\lambda_2,\ldots,\lambda_G\},
\]

ordered from conservative to aggressive. Freeze this grid before examining calibration outcomes.

For each \(\lambda_g\), run the policy on all \(n\) calibration inputs and compute:

\[
\widehat{R}_g=
\frac{1}{n}\sum_{l=1}^{n}f(x_l,\pi_{\lambda_g}(x_l)).
\]

### 12.2 Binomial p-value

For the null hypothesis

\[
H_g:R(\pi_{\lambda_g})>\alpha,
\]

compute:

\[
p_g=
P\left(\operatorname{Bin}(n,\alpha)
\leq n\widehat{R}_g\right).
\]

In code, if `failures` is the integer number of added failures:

```python
from scipy.stats import binom

def calibration_p_value(failures, n, alpha):
    return binom.cdf(failures, n, alpha)
```

### 12.3 Fixed-sequence selection

```python
def calibrate_threshold(calibration_results, lambdas, alpha, delta):
    last_certified = None

    for lambda_value in lambdas:  # conservative -> aggressive
        failures = count_added_failures(
            calibration_results,
            lambda_value,
        )
        p_value = calibration_p_value(
            failures,
            len(calibration_results),
            alpha,
        )

        if p_value <= delta:
            last_certified = lambda_value
        else:
            break

    return last_certified  # None means reference-only fallback
```

If no threshold is certified, deploy the reference configuration for every input.

### 12.4 Important sample-size warning

The risk guarantee can be impossible with a small calibration set. Chapter 3 notes that with \(n=50\), \(\alpha=0.05\), and \(\delta=0.05\), even zero observed failures cannot certify a threshold.

Choose \(\alpha\), \(\delta\), and calibration size before running calibration. Do not select them afterward merely to obtain certification.

---

## 13. Required Baselines

Implement and evaluate:

1. **Fixed reference:** \(s=0.25,j=1\).
2. **Fixed alternatives:** \(s\in\{0.07,0.10,0.15\},j=1\).
3. **SNR-only policy:** earliest candidate when SNR passes the gate; otherwise reference.
4. **Proposed policy:** calibrated stabilization threshold plus SNR gate.
5. **Proposed without SNR gate:** set \(\theta=-\infty\).
6. **Oracle diagnostic:** cheapest non-failing candidate selected after seeing all outcomes.
7. **Teacher-only and student-only:** quality-cost context, if supported by the implementation.

The oracle is not deployable. It is only an upper bound on achievable savings within the candidate set.

---

## 14. Required Telemetry

Save one structured record per input, seed, condition and method:

```json
{
  "input_id": "...",
  "speaker_id": "...",
  "split": "development|calibration|test",
  "condition": "clean|noisy",
  "seed": 0,
  "method": "fixed_reference|fixed_007|snr_only|proposed|oracle",
  "lambda": null,
  "theta": null,
  "snr_estimate_db": null,
  "selected_stopping_argument": null,
  "selected_teacher_time": null,
  "selected_teacher_step": null,
  "student_restart_index": null,
  "stabilization_trace": [],
  "teacher_conditional_calls": 0,
  "teacher_unconditional_calls": 0,
  "student_calls": 0,
  "total_forward_passes": 0,
  "wall_clock_seconds": 0.0,
  "real_time_factor": 0.0,
  "peak_gpu_memory_mb": 0.0,
  "wer": null,
  "sim": null,
  "utmos": null,
  "reference_wer": null,
  "reference_sim": null,
  "added_failure": null,
  "output_audio_path": "...",
  "asr_transcript": "..."
}
```

Keep raw audio, ASR transcripts, metrics, configuration files and environment metadata.

---

## 15. Test Plan

### Phase A: Baseline reconstruction

- Reproduce reference audio deterministically with fixed seeds.
- Count actual model forward calls.
- Verify teacher and student time grids.
- Verify how the stochastic endpoint term is applied.
- Reconcile paper and code cost/configuration differences.

### Phase B: Signal instrumentation

- Add endpoint-estimate logging.
- Add generated-frame masking.
- Add stabilization-score logging.
- Confirm that instrumentation does not alter baseline audio.
- Verify no extra model forward calls are introduced.

### Phase C: Candidate mapping on development data

- Run all fixed configurations.
- Compute the cheapest non-failing configuration per input.
- Test whether stabilization scores predict early-switch safety.
- Select the candidate set, SNR gate and restart behavior.

### Phase D: Calibration

- Freeze \(c,m_W,m_S,\theta,\Lambda,\alpha,\delta\).
- Run every threshold policy on the calibration split.
- Apply fixed-sequence binomial testing.
- Select \(\widehat{\lambda}\), or reference fallback if none certifies.

### Phase E: Final test

- Run the frozen policy once on untouched speakers.
- Compare against every baseline.
- Report quality, added-failure risk and computational cost separately.
- Do not retune after seeing test results.

### Phase F: Robustness

- Calibrate on clean and test on noisy prompts.
- Calibrate on mixed conditions and test again.
- Repeat using different calibration sizes.
- If possible, repeat calibration/test splits and measure the fraction satisfying \(R\leq\alpha\).

---

## 16. Evaluation Outputs

Report at minimum:

### Quality

- WER distribution;
- SIM distribution;
- UTMOS distribution;
- absolute threshold failures; and
- added-failure rate relative to reference with confidence intervals.

### Computation

- teacher conditional calls;
- teacher unconditional calls;
- student calls;
- total forward passes;
- wall-clock time after warm-up;
- real-time factor; and
- peak GPU memory.

### Adaptive behavior

- frequency of each selected handover point;
- selections by clean/noisy condition;
- relationship between \(d_i\), SNR and the selected point;
- fraction of oracle savings achieved; and
- fallback rate.

Reduced forward-pass count must not be reported as reduced latency unless repeated timing measurements also support the claim.

---

## 17. Unit and Integrity Checks

Implement the following checks before full experiments:

1. **Mask test:** modifying prompt frames must not affect the generated-frame norm calculation.
2. **Zero-change test:** identical consecutive endpoints produce \(d_i\approx0\).
3. **Scale test:** multiplying both endpoint estimates by the same positive scalar should approximately preserve the relative score.
4. **Reference fallback test:** an impossibly strict \(\lambda\) must reproduce the reference output exactly under the same seed.
5. **Aggressive-threshold test:** a very large \(\lambda\) should select the earliest candidate when the SNR gate passes.
6. **SNR-gate test:** a prompt below \(\theta\) must reach the reference regardless of \(d_i\).
7. **No-overhead test:** enabling telemetry must not increase teacher or student forward-call counts.
8. **Seed test:** identical inputs, configurations and seeds must reproduce identical outputs.
9. **Split-leakage test:** no speaker may occur across prohibited split boundaries.
10. **Metric test:** reference-relative failure calculations must use the matching input and seed.

---

## 18. Claims Allowed After Each Stage

| Stage | Defensible claim |
| --- | --- |
| Implementation only | The proposed policy has been implemented according to the specification |
| Development mapping | The selected signals are or are not associated with safe earlier handover on development data |
| Calibration | A threshold was or was not certified under the selected \(\alpha,\delta\) assumptions |
| Held-out test | The frozen policy achieved the measured quality, risk and cost on unseen test inputs |
| Robustness tests | The method did or did not retain its behavior under the tested distribution shifts |

Do not claim universal safety, per-utterance guarantees or general robustness. The proposed guarantee is marginal over inputs drawn from the same distribution assumed by calibration.

---

## 19. Unresolved Items Before Final Execution

- [ ] Record DMOSpeech 2 repository commit hash.
- [ ] Record teacher, student, vocoder and evaluation checkpoint hashes.
- [ ] Reconcile the paper's reported step count with the public code.
- [ ] Verify the exact teacher and student time grids.
- [ ] Confirm the stochastic endpoint construction.
- [ ] Confirm classifier-free guidance call counting.
- [ ] Select and justify \(c,m_W,m_S\).
- [ ] Select and justify \(\alpha,\delta\).
- [ ] Determine calibration size before running calibration.
- [ ] Define and freeze the \(\lambda\) grid.
- [ ] Select \(\theta\) using development data only.
- [ ] Record corpus size, speakers and split identifiers.
- [ ] Record the additive-noise source and license.
- [ ] Record GPU, driver, CUDA, PyTorch and package versions.
- [ ] Record all seeds.
- [ ] Decide whether the restart pilot remains in scope.
- [ ] Verify that the endpoint score uses deterministic velocity only.
- [ ] Verify that the final endpoint preserves baseline stochastic behavior.

---

## 20. Final Implementation Summary

The proposed system does not cushion or blend the teacher and student states. Instead, it decides **when** to perform the existing handover.

Its operational rule is:

> Stop at the earliest candidate step where the teacher endpoint estimate is sufficiently stable and the prompt passes the SNR gate; otherwise, fall back to the reference handover.

Its research contribution is strengthened by calibrating the stabilization threshold so that the expected rate of added failures is bounded, under the stated calibration assumptions, while attempting to reduce inference cost.

