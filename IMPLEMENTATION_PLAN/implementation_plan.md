# Implementation Plan: DMOSpeech 2 + Smooth Transition (TAGS-Flow) 4-Step Evaluation

This plan outlines the implementation and empirical evaluation of our proposed continuous transition manifold $w(t)$ and dynamic cadence scaling factor $\alpha$ directly integrated into the DMOSpeech 2 inference engine, focusing on testing the accelerated **4-step inference budget**.

## User Review Required

> [!IMPORTANT]
> **Zero Model Retraining:** All tests will use the existing pretrained DMOSpeech 2 checkpoints (`model_85000.pt` and `model_1500.pt`) located in `ckpts/`. We will create an isolated, non-destructive inference solver module (`src/infer_tags.py`) so that the original `src/infer.py` and existing baseline scripts remain untouched and intact.

---

## Proposed Changes

### Core Inference Module

#### [NEW] [infer_tags.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/infer_tags.py)
* Inherits or wraps `DMOInference` from [src/infer.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/infer.py).
* Implements the unified **Master Trajectory Evolution Operator**:
  $$\mathbf{x}_{t+h} = \mathbf{x}_t + h \cdot \alpha \cdot \left[ (1 - w(t)) \cdot \mathbf{v}_{\text{teacher}} + w(t) \cdot \mathbf{v}_{\text{student}} \right]$$
* Implements the **Cosine Transition Manifold ($w(t)$)** over a configurable transition window $[t_{\text{start}}, t_{\text{end}}]$:
  $$w(t) = \frac{1}{2} \left[ 1 - \cos\left(\pi \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}\right) \right]$$
* Implements **Dynamic Cadence Velocity Scaling ($\alpha$)**:
  $$\alpha = \text{clamp}\left( \frac{T_{\text{ref}} / |X_{\text{ref}}|}{\hat{T}_{\text{pred}} / |X_{\text{target}}|}, 0.80, 1.20 \right)$$
* Provides a dedicated 4-step solver loop `generate_tags_4step()`:
  * **Step 1 ($t = 0.00 \to 0.25$):** Teacher dominant ($w \approx 0$).
  * **Step 2 ($t = 0.25 \to 0.50$):** Smooth Transition Handshake ($w(t)$ cosine cross-fade).
  * **Step 3 ($t = 0.50 \to 0.75$):** Student acceleration ($w \approx 1$).
  * **Step 4 ($t = 0.75 \to 1.00$):** Student finalization to clean mel-spectrogram.

---

### Benchmark & Experimentation Scripts

#### [NEW] [test_tags_4step.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/test_tags_4step.py)
* Automated comparison script that runs 3 test configurations on the same input sentence and reference audio:
  1. **Config A: Baseline DMOSpeech 2 (Pure Student 4-Step)** — measures baseline 4-step quality without teacher guidance.
  2. **Config B: Baseline DMOSpeech 2 (Default Hybrid ~8 Steps)** — measures the existing hard-cutoff reference at higher compute.
  3. **Config C: Our Proposed Method (TAGS-Flow 4-Step)** — evaluates whether our smooth transition $w(t)$ + $\alpha$ achieves clean, non-slurred audio in only 4 steps.
* Generates three labeled WAV files in the workspace:
  * `output_baseline_student_4step.wav`
  * `output_baseline_hybrid_8step.wav`
  * `output_proposed_tags_4step.wav`
* Measures and logs wall-clock execution time (seconds on your AMD Ryzen 7 CPU) for each run.

---

## Verification Plan

### Automated Verification
* Run `test_tags_4step.py` via powershell:
  ```powershell
  python test_tags_4step.py
  ```
* Verify that all 3 audio files generate successfully without memory errors or crashes on your 16GB RAM laptop.
* Compare CPU inference latency (seconds) across all configurations.

### Manual Verification
* Listen to the resulting `.wav` files side-by-side to assess:
  1. **Phase Roughness & Boundary Buzzing:** Confirm whether Config C eliminates the harsh transition artifacts present in the naive baseline.
  2. **Pacing & Consonant Clarity:** Confirm whether $\alpha$ prevents slurring or rushing on target syllables.
  3. **Voice Similarity:** Confirm that the prompt speaker's voice timbre is preserved across the 4-step generation.
