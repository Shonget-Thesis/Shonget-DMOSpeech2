# Exact Implementation and Code Changes Report: Full GPU Pipeline Timing Benchmark

**Author:** Antigravity (Pair Programming Assistant)  
**Date:** October 20, 2026  
**Status:** Implemented in Local Working Tree (Uncommitted & Unpushed per Strict User Command)  
**Target Reference:** [`VERDICT_TIMING_IMPLEMENTATION_FIXES_BEFORE_RUN.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/VERDICT_TIMING_IMPLEMENTATION_FIXES_BEFORE_RUN.md) & [`VERDICT_GPU_FULL_PIPELINE_TIMING_RERUN.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/VERDICT_GPU_FULL_PIPELINE_TIMING_RERUN.md)  
**Archive Package:** [`REPLICATION_AND_VERIFICATION_BUNDLE.zip`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/REPLICATION_AND_VERIFICATION_BUNDLE.zip)  

---

## Executive Summary of Completed Corrections

In response to the peer-review verdict in [`VERDICT_TIMING_IMPLEMENTATION_FIXES_BEFORE_RUN.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/VERDICT_TIMING_IMPLEMENTATION_FIXES_BEFORE_RUN.md), all five required corrections were implemented, tested, and empirically validated:

1. **Baseline Drift Overhead Eliminated (Correction 1):** Removed all drift calculations, intermediate endpoint projections, and `.item()` GPU synchronization calls from Fixed Step 8, SNR-Only, and Proposed noisy fallback paths. Drift is strictly evaluated only for Proposed on clean speech at candidate indices 4, 5, and 6.
2. **Speaker-Level Wilcoxon Aggregation (Correction 2):** Corrected the unit of analysis for paired inference. Paired latency differences for clean and noisy conditions are averaged per speaker ($\overline{\Delta t}(s)$), and Wilcoxon signed-rank testing is conducted on the $N=200$ independent held-out test speakers with prespecified hypothesis directions. Both Aggregate Ratio Savings and Mean Per-Speaker Percentage Savings are explicitly defined and reported.
3. **Sampling Equivalence & Drift Reproduction Empirically Verified (Correction 3):** Traced and resolved the single-case drift discrepancy (traced to silence-trimming in `preprocess_ref_audio_text`; restoring `audio_tensor = torch.from_numpy(prompt_np)` bitwise reproduced historical drift $d_4=0.6979, d_5=0.5673, d_6=0.3635$ and Step 6 selection). Executed [`Handover Scheduling/scratch/verify_sampling_equivalence.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/scratch/verify_sampling_equivalence.py) across all 5 representative exit paths (Step 4 exit, Step 5 exit, Step 6 exit, clean low-SNR fallback Step 8, and noisy fallback Step 8). Output endpoints, stabilization drifts, and synthesized 24 kHz audio waveforms yielded **0.00000000e+00 absolute discrepancy** (bitwise identical!).
4. **Historical CPU Contingency Table Corrected (Correction 4):** Completely excised the unverified proxy table (185/8/6/1). Anchored all reporting strictly to the authoritative held-out CPU records (`benchmark_results.json` and `test_grid.csv`): 188 Both Pass, 4 Both Fail, 7 SNR Fail / Proposed Pass ($b$), 1 Proposed Fail / SNR Pass ($c$), total 200 speakers (SNR failure rate 5.5%, Proposed failure rate 2.5%, exact two-sided McNemar $p = 0.0703125$, directional one-sided $p = 0.03515625$).
5. **Real Codebase & Parameter Verification (Correction 5):** Fixed keyword arguments in `DMOInference` (`student_checkpoint_path`, `duration_predictor_path`), standardized reference audio preprocessing matching `src/infer.py`, verified actual `WavLMSimilarityEvaluator.compute_similarity` code, and recorded SHA256 checksums.

---

## 1. Correction 1: Elimination of Baseline Drift Overhead

### 1.1 Specification Rules
Per Section 26 of [`VERDICT_TIMING_IMPLEMENTATION_FIXES_BEFORE_RUN.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/VERDICT_TIMING_IMPLEMENTATION_FIXES_BEFORE_RUN.md):

| Policy / Path | Drift Computation | Endpoint Projections | GPU `.item()` Calls |
| --- | --- | --- | --- |
| **Fixed Step 8** | **None** | Only at index 8 for student input | **None** |
| **SNR-Only (Clean, $\ge 20$ dB)** | **None** | Only at index 4 for student input | **None** |
| **SNR-Only (Noisy, $< 20$ dB)** | **None** | Only at index 8 for student input | **None** |
| **Proposed (Noisy, $< 20$ dB)** | **None** (Step-8 fallback) | Only at index 8 for student input | **None** |
| **Proposed (Clean, $\ge 20$ dB)** | Consecutive drift at 4, 5, 6 | Index 3 (for $d_4$), 4, 5, 6 | Only at decision steps 4, 5, 6 |

### 1.2 Implemented Code in `colab/run_gpu_full_pipeline_timing_benchmark.py`

```python
def run_early_stop_teacher(
    tts: Any,
    step_cond: torch.Tensor,
    text: torch.Tensor,
    y0: torch.Tensor,
    prompt_len: int,
    prompt_snr: float,
    policy: str,
    lambda_thresh: float = 0.50,
    snr_thresh: float = 20.0,
    cfg_strength: float = CFG_STRENGTH,
) -> Tuple[torch.Tensor, int, int, Dict[int, float], float]:
    """
    Teacher Euler solver with genuine online early stopping per policy.
    Zero drift overhead for baselines (Point 1 of Verdict):
    - Fixed Step 8: No drift work, no intermediate endpoint projections.
    - SNR-Only: No drift work, no intermediate endpoint projections.
    - Proposed (SNR < 20 dB): Follows fixed Step 8 fallback with zero drift work.
    - Proposed (SNR >= 20 dB): Compares consecutive endpoints strictly at 4, 5, 6
      (preserving endpoint at 3 for d4, and previous endpoint for d5 and d6).
    """
    device = step_cond.device
    state = y0.clone()
    drift_trace: Dict[int, float] = {}
    prev_endpoint = None
    teacher_calls = 0
    decision_drift = -1.0

    # Determine execution path
    if policy == "step8":
        target_step = 8
        is_dynamic = False
    elif policy == "snr_only":
        target_step = 4 if prompt_snr >= snr_thresh else 8
        is_dynamic = False
    elif policy == "proposed":
        if prompt_snr < snr_thresh:
            target_step = 8  # Fallback due to noise: zero drift work
            is_dynamic = False
        else:
            target_step = None
            is_dynamic = True
    else:
        raise ValueError(f"Unknown policy: {policy}")

    with torch.inference_mode():
        # Fast path for deterministic target_step (step8, snr_only, proposed noisy fallback)
        if not is_dynamic:
            for i in range(target_step + 1):
                t_curr_val = SWAY_TEACHER_GRID[i]
                t_curr = torch.tensor(t_curr_val, device=device, dtype=torch.float32)

                pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=False, drop_text=False)
                null_pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=True, drop_text=True)
                teacher_calls += 2
                v_curr = pred + (pred - null_pred) * cfg_strength

                if i == target_step:
                    endpoint = state + (1.0 - t_curr_val) * v_curr
                    return endpoint, target_step, teacher_calls, {}, -1.0

                dt = SWAY_TEACHER_GRID[i + 1] - t_curr_val
                state = state + dt * v_curr

        # Dynamic path: Proposed policy on clean speech (prompt_snr >= 20 dB)
        # Evaluates candidate decisions only at indices 4, 5, 6
        for i in range(9):
            t_curr_val = SWAY_TEACHER_GRID[i]
            t_curr = torch.tensor(t_curr_val, device=device, dtype=torch.float32)

            pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=False, drop_text=False)
            null_pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=True, drop_text=True)
            teacher_calls += 2
            v_curr = pred + (pred - null_pred) * cfg_strength

            # Index 3: Preserve endpoint to compute d_4 at the next step
            if i == 3:
                prev_endpoint = state + (1.0 - t_curr_val) * v_curr

            # Indices 4, 5, 6: Consecutive drift evaluations
            elif i in [4, 5, 6]:
                endpoint_curr = state + (1.0 - t_curr_val) * v_curr
                d_i = compute_stabilization_drift(endpoint_curr, prev_endpoint, prompt_len)
                drift_trace[i] = d_i

                if d_i <= lambda_thresh:
                    return endpoint_curr, i, teacher_calls, drift_trace, d_i

                # Retain immediately previous endpoint for next consecutive candidate
                prev_endpoint = endpoint_curr

            # Index 8: Final fallback
            elif i == 8:
                endpoint_curr = state + (1.0 - t_curr_val) * v_curr
                decision_drift = drift_trace.get(6, -1.0)
                return endpoint_curr, 8, teacher_calls, drift_trace, decision_drift

            # State evolution
            dt = SWAY_TEACHER_GRID[i + 1] - t_curr_val
            state = state + dt * v_curr

    return endpoint_curr, 8, teacher_calls, drift_trace, decision_drift
```

---

## 2. Correction 2: Correct Wilcoxon Unit of Analysis & Reporting

### 2.1 Speaker-Level Aggregation Methodology
To address the statistical dependence between clean and noisy observations from the same speaker:
1. For each speaker $s \in \{1 \dots 200\}$ in the held-out test split, compute the condition-specific latency difference vs Step 8:
   $$\Delta t_{\text{clean}}(s) = t_{\text{step8, clean}}(s) - t_{\text{pol, clean}}(s)$$
   $$\Delta t_{\text{noisy}}(s) = t_{\text{step8, noisy}}(s) - t_{\text{pol, noisy}}(s)$$
2. Aggregate to a single prespecified speaker-level latency difference:
   $$\overline{\Delta t}(s) = \frac{\Delta t_{\text{clean}}(s) + \Delta t_{\text{noisy}}(s)}{2}$$
3. Execute the paired Wilcoxon signed-rank test on the $N=200$ independent speaker differences with pre-specified directions:
   * **Proposed vs. Step 8:** `alternative='greater'` ($H_1: \overline{\Delta t} > 0$, Proposed is faster).
   * **SNR-Only vs. Step 8:** `alternative='greater'` ($H_1: \overline{\Delta t} > 0$, SNR-Only is faster).
   * **Proposed vs. SNR-Only:** `alternative='two-sided'` ($H_1: \overline{\Delta t} \neq 0$).

### 2.2 Formal Definitions of Percentage Latency Savings
The analysis suite computes and distinguishes both metrics:
1. **Aggregate Ratio Savings:**
   $$\text{Ratio Savings} = \frac{\overline{t}_{\text{step8}} - \overline{t}_{\text{pol}}}{\overline{t}_{\text{step8}}} \times 100\%$$
2. **Mean of Per-Speaker Percentage Savings:**
   $$\text{Mean Per-Speaker Savings} = \frac{1}{N} \sum_{s=1}^N \left( \frac{\overline{\Delta t}(s)}{\overline{t}_{\text{step8}}(s)} \times 100\% \right)$$

---

## 3. Correction 3: Empirical Sampling Equivalence Verification & Drift Reproduction

### 3.1 Root Cause & Resolution of the Single-Case Drift Discrepancy

Per [VERDICT_TIMING_RUN_DRIFT_AND_STATISTICS_BLOCKERS.md](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/VERDICT_TIMING_RUN_DRIFT_AND_STATISTICS_BLOCKERS.md), an empirical trace was conducted on clean utterance common_voice_en_103675-common_voice_en_103677 (input seed 507950035).

* **Root Cause Identified:** In the preliminary verification script, udio was loaded from 
ef_audio (the output of preprocess_ref_audio_text). preprocess_ref_audio_text uses pydub silence-stripping, which removed 5,770 samples (240 ms) of non-speech audio from the prompt, truncating it from 155,648 to 149,878 samples and altering prompt_len from 609 to 586 mel frames. This shifted the temporal boundary $[ 	ext{prompt\_len} : ]$ where stabilization drift is calculated.
* **Historical Pipeline Alignment:** In the authoritative historical benchmark scripts ([
un_test_grid.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/phase3_test_benchmark_200spk/run_test_grid.py#L159-L167) and [colab/run_gpu_expanded_benchmark.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/colab/run_gpu_expanded_benchmark.py#L273-L281)), preprocess_ref_audio_text was only used to extract normalized text (
ef_text_clean), while udio_tensor was always constructed directly from 	orch.from_numpy(prompt_np).float().unsqueeze(0) with RMS scaling.
* **Historical Reproduction:** When prepare_case_inputs in 
un_gpu_full_pipeline_timing_benchmark.py and erify_sampling_equivalence.py was restored to this exact historical standard:
  -  = 0.697909$ (Historical CSV: **.6979$**)
  -  = 0.567322$ (Historical CSV: **.5673$**)
  -  = 0.363520$ (Historical CSV: **.3635$**)
  - Because  > 0.50$ and  > 0.50$, the online scheduler correctly evaluated steps 4, 5, and 6, selecting **Step 6** ( \le 0.50$), exactly reproducing the historical decision and requiring 14 teacher forward calls ( 	imes (6 + 1)$).

### 3.2 Comprehensive 5-Case Representative Exit-Path Verification Log

Per Section 44 of [VERDICT_TIMING_RUN_DRIFT_AND_STATISTICS_BLOCKERS.md](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/VERDICT_TIMING_RUN_DRIFT_AND_STATISTICS_BLOCKERS.md), [Handover Scheduling/scratch/verify_sampling_equivalence.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/scratch/verify_sampling_equivalence.py) was executed across 5 representative test items covering all exit paths:

`
================================================================================
VERIFYING SAMPLING EQUIVALENCE & EXIT PATHS ACROSS 5 REPRESENTATIVE CASES
================================================================================
Loading model checkpoints on CPU...
Download Vocos from huggingface charactr/vocos-mel-24khz

================================================================================
TEST CASE: Historical Drift Case (Exit at Step 6, d4=0.6979, d5=0.5673, d6=0.3635)
Utterance: common_voice_en_103675-common_voice_en_103677 | Condition: clean | Expected Exit: Step 6
================================================================================
  Prompt SNR: 35.24 dB (Gate Threshold = 20.0 dB)
  Candidate-Grid Teacher Pass:
    Grid drift d_4: 0.697909
    Grid drift d_5: 0.567322
    Grid drift d_6: 0.363520
  [step8     ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [snr_only  ] Selected Exit Step: 4 | NFE: 10
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [proposed  ] Selected Exit Step: 6 | NFE: 14
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    -> Proposed selected step EXACTLY matched expected Step 6!
    -> Drifts evaluated: {4: 0.6979085206985474, 5: 0.5673217177391052, 6: 0.36352041363716125}
    Audio Waveform Difference vs Grid: max=0.00000000e+00

================================================================================
TEST CASE: Exit at Step 5 (d4=0.7634 > 0.50, d5=0.4121 <= 0.50)
Utterance: common_voice_en_1205005-common_voice_en_1205007 | Condition: clean | Expected Exit: Step 5
================================================================================
  Prompt SNR: 38.28 dB (Gate Threshold = 20.0 dB)
  Candidate-Grid Teacher Pass:
    Grid drift d_4: 0.763377
    Grid drift d_5: 0.412118
    Grid drift d_6: 0.296331
  [step8     ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [snr_only  ] Selected Exit Step: 4 | NFE: 10
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [proposed  ] Selected Exit Step: 5 | NFE: 12
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    -> Proposed selected step EXACTLY matched expected Step 5!
    -> Drifts evaluated: {4: 0.7633772492408752, 5: 0.4121180772781372}
    Audio Waveform Difference vs Grid: max=0.00000000e+00

================================================================================
TEST CASE: Exit at Step 4 (d4=0.2776 <= 0.50)
Utterance: common_voice_en_15265-common_voice_en_15266 | Condition: clean | Expected Exit: Step 4
================================================================================
  Prompt SNR: 23.38 dB (Gate Threshold = 20.0 dB)
  Candidate-Grid Teacher Pass:
    Grid drift d_4: 0.277617
    Grid drift d_5: 0.247597
    Grid drift d_6: 0.192901
  [step8     ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [snr_only  ] Selected Exit Step: 4 | NFE: 10
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [proposed  ] Selected Exit Step: 4 | NFE: 10
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    -> Proposed selected step EXACTLY matched expected Step 4!
    -> Drifts evaluated: {4: 0.27761712670326233}
    Audio Waveform Difference vs Grid: max=0.00000000e+00

================================================================================
TEST CASE: Exit at Step 8 Clean Fallback (Native prompt SNR = 18.28 dB < 20 dB)
Utterance: common_voice_en_137148-common_voice_en_137152 | Condition: clean | Expected Exit: Step 8
================================================================================
  Prompt SNR: 18.28 dB (Gate Threshold = 20.0 dB)
  Candidate-Grid Teacher Pass:
    Grid drift d_4: 0.583844
    Grid drift d_5: 0.482167
    Grid drift d_6: 0.331067
  [step8     ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [snr_only  ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [proposed  ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    -> Proposed selected step EXACTLY matched expected Step 8!
    -> Zero drift evaluations performed (immediate SNR fallback)!
    Audio Waveform Difference vs Grid: max=0.00000000e+00

================================================================================
TEST CASE: Exit at Step 8 Noisy Fallback (Injected noise SNR ~ 17.71 dB < 20 dB)
Utterance: common_voice_en_103675-common_voice_en_103677 | Condition: noisy15 | Expected Exit: Step 8
================================================================================
  Prompt SNR: 17.71 dB (Gate Threshold = 20.0 dB)
  Candidate-Grid Teacher Pass:
    Grid drift d_4: 0.188647
    Grid drift d_5: 0.192947
    Grid drift d_6: 0.185336
  [step8     ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [snr_only  ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    Audio Waveform Difference vs Grid: max=0.00000000e+00
  [proposed  ] Selected Exit Step: 8 | NFE: 18
    Endpoint Difference vs Grid: max=0.00000000e+00, mean=0.00000000e+00
    -> Proposed selected step EXACTLY matched expected Step 8!
    -> Zero drift evaluations performed (immediate SNR fallback)!
    Audio Waveform Difference vs Grid: max=0.00000000e+00

================================================================================
ALL 5 REPRESENTATIVE EXIT PATH CHECKS PASSED WITH ZERO DISCREPANCY!
================================================================================
`

* **Conclusion:** Intermediate endpoints, drift values, and output audio are bit-for-bit identical (.00000000	ext{e}+00$) to the candidate-grid sampler across all 5 paths. Zero drift work is performed during fallback. Historical drift values ( = 0.6979, d_5 = 0.5673, d_6 = 0.3635$) and Step 6 selection are reproduced exactly.

---

## 4. Correction 4: Trace and Removal of the Unexplained p-Value

The previous informal note $(p=0.453)$ was an error and has been completely removed. All statistical comparisons in Chapter 3 are anchored to the following verified contingency tables from [`Handover Scheduling/scratch/audit_verdict.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/scratch/audit_verdict.py):

| Evaluation Cohort | Both Pass | Both Fail | SNR Fail, Proposed Pass ($b$) | Proposed Fail, SNR Pass ($c$) | McNemar 2-Sided $p$ | McNemar 1-Sided $p$ (Proposed Better) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU Test Held-Out (200 spk)** | **188** | **4** | **7** | **1** | **$p = 0.0703125$** | **$p = 0.03515625$** |
| **GPU Test (200 spk)** | 195 | 2 | 2 | 1 | **$p = 1.0000$** | **$p = 0.5000$** |
| **GPU Pooled (500 spk)** | 486 | 4 | 8 | 2 | **$p = 0.1094$** | **$p = 0.0547$** |

* **CPU Held-Out Verification:** Directly extracted from `Handover Scheduling/phase3_test_benchmark_200spk/results/test_grid.csv`. SNR-Only yielded 11/200 robust failures (5.5%), while Proposed Adaptive ($\lambda=0.50$) yielded 5/200 robust failures (2.5%), rescuing 7 speakers while failing only 1 that SNR-Only passed. Under the prespecified hypothesis that Proposed reduces robust failures ($b > c$), the directional exact McNemar test yields $p = 0.03515625$ ($p < 0.05$), while the conservative two-sided test yields $p = 0.0703125$.
* Note on Previous Discrepancy: An intermediate audit script (`scratch/audit_verdict.py`) had reported 185/8/6/1 based on GPU proxy classification. The authoritative ground-truth CPU held-out table is strictly 188 / 4 / 7 / 1, as verified in `benchmark_results.json` and `test_grid.csv`.

---

## 5. Correction 5: Real Verified Code & Working-Tree Alignment

### 5.1 Real Code of `WavLMSimilarityEvaluator.compute_similarity` in `core/metrics.py`
```python
    def compute_similarity(
        self,
        candidate_audio: Union[str, np.ndarray],
        ground_truth_audio: Union[str, np.ndarray],
        sample_rate: int = 24000,
        gt_sr: Optional[int] = None,
    ) -> float:
        """
        Computes cosine similarity between synthesized audio and ground-truth target recording.
        Properly honors gt_sr if ground_truth_audio is provided as a numpy array.
        """
        target_gt_sr = gt_sr if gt_sr is not None else sample_rate
        emb_cand = self.get_embedding(candidate_audio, sample_rate)
        emb_gt = self.get_embedding(ground_truth_audio, target_gt_sr)
        sim = F.cosine_similarity(emb_cand, emb_gt).item()
        return float(sim)
```

### 5.2 Corrected `DMOInference` Keyword Arguments
In both the benchmark runner and equivalence test, `DMOInference` is initialized strictly according to [`src/infer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/infer.py):
```python
tts = DMOInference(
    student_checkpoint_path=ckpt_model,
    duration_predictor_path=ckpt_duration,
    device=str(device),
)
```

### 5.3 SHA256 Checksums of Benchmark Files
| File Path | SHA256 Checksum |
| --- | --- |
| `Handover Scheduling/colab/run_gpu_full_pipeline_timing_benchmark.py` | `bf20d1fdb12b788169e9658e20819b61d5c96690cd9a726f61b1f3904259e41f` |
| `Handover Scheduling/colab/DMOSpeech2_Full_Pipeline_Timing_Benchmark.ipynb` | `e6b22dc0a2014bd77f5c00c8d9cf2fe962c194c8ce35fa01cf9d822867b68156` |
| `Handover Scheduling/core/sampler.py` | `9b3a06e35ec4b2dd07e9dfa7cdc6a7a5d476af5cfb3d0b8654e79d440d10117b` |
| `Handover Scheduling/core/metrics.py` | `6848a416da291b3c7b89f963b43b59006ef1b6167dd757d7173ef19ce9bb032a` |
| `Handover Scheduling/scratch/verify_sampling_equivalence.py` | `e368ed2984aec3220ee9512c7a49b00a55a26c48651c7bf1b8dab051947a1d6f` |
| `Handover Scheduling/data/splits/test_manifest.json` | `b61bddde6bc4d89400d42d902793b3a8477155cbaed294f50e02568d47107416` |

---

## 6. Repository Status
* **Git Status:** 0 staged files, 0 commits, 0 pushes. All changes remain safely uncommitted in the local working tree.
* **Archive Bundle:** Packaged in [`REPLICATION_AND_VERIFICATION_BUNDLE.zip`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/REPLICATION_AND_VERIFICATION_BUNDLE.zip) (0.94 MB).
