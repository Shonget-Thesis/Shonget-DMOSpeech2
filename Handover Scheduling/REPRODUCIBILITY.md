# Reproducibility Specification & Environment Record

This document records the exact environment, commit references, model checksums, precision policies, and objective-to-task mappings required to guarantee reproducible execution of the handover-scheduling method and evaluation in Chapter 3.

---

## 1. Repository & Commit Information

- **Repository:** `Shonget-Thesis/Shonget-DMOSpeech2`
- **Current Repository Commit Hash:** `ac269f197ad59a5ea6161b3f960771930a3de8b7`
- **Upstream Repository:** `yl4579/DMOSpeech2`
- **Upstream Base Commit Hash:** `10c2255a59b96c311c317536603a18672e69ac84`
- **Working Directory:** `Handover Scheduling/`

---

## 2. Checkpoint SHA-256 Hashes

| Checkpoint Name | File / Location | Checkpoint Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **DMOSpeech 2 Model (Teacher & Student)** | `ckpts/model_85000.pt` | 4,019,822,332 bytes | `e143188f7c46b69abb60786937d79b89dac9777d3c4c1e91a97634bcd39ebf2c` |
| **Duration Predictor (GRPO-finetuned)** | `ckpts/model_1500.pt` | 270,615,794 bytes | `6bd50428f3b5f0f25c1a60a0053f40ee47c817bda03c8ee120100f22680ae4c2` |
| **Whisper large-v3 ASR** | `~/.cache/huggingface/hub/models--openai--whisper-large-v3/.../model.safetensors` (Commit `06f233f`) | 3,087,130,976 bytes | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| **WavLM-large Speaker Verifier** | `~/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/.../pytorch_model.bin` (Commit `feb593a`) | 404,547,053 bytes | `e906bce2fa42fb497a1d1a9ecf81548adb7e03b12a5644e32d2f42f0d6500fad` |
| **SpeechMOS UTMOS Predictor** | `~/.cache/torch/hub/checkpoints/utmos22_strong_step7459_v1.pt` (`tarepan/SpeechMOS:v1.2.0`) | 411,179,119 bytes | `38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff` |

---

## 3. Environment & Software Versions

- **Operating System:** Windows 11 (AMD64)
- **Python Version:** `3.12.6` (tags/v3.12.6:a4a2d2b, Sep 6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)]
- **PyTorch Version:** `2.5.1+cu121`
- **torchaudio Version:** `2.5.1+cu121`
- **transformers Version:** `5.14.1`
- **PyTorch CUDA Build:** `12.1`
- **cuDNN Version:** `90100` (bundled in PyTorch binary)
- **Evaluation Packages:**
  - `jiwer`: `4.0.0`
  - `speechbrain`: `1.1.1`
  - `librosa`: `0.11.0`
  - `soundfile`: `0.14.0`
  - `scipy`: `1.17.1`
  - `pytest`: `9.1.1`
  - `vocos`: `0.1.0`
  - `pandas`: `3.0.3`
  - `pyarrow`: `25.0.0`

---

## 4. Hardware Configuration & Execution Device

- **CPU:** AMD Ryzen 7 5700U with Radeon Graphics (8 physical cores, 16 logical processors)
- **GPU:** AMD Radeon(TM) Graphics (Device ID: `0x164C`, AdapterRAM: 536,870,912 bytes / 512 MB)
- **GPU Driver Version:** `31.0.14076.3`
- **Execution Target:** The system GPU is an integrated AMD Radeon processor without native NVIDIA CUDA support (`torch.cuda.is_available() == False`). All synthesis and evaluation pipelines execute on the host CPU using evaluation mode (`torch.inference_mode()`) with isolated, seeded local `torch.Generator` instances for bitwise reproducibility across runs.

---

## 5. Numerical Precision

- **Precision Policy:** Full 32-bit floating point (`FP32` / `torch.float32`) is enforced uniformly across:
  - Teacher ODE integration
  - Endpoint projection $\hat{x}_1(t_i)$
  - Student reverse process and denoising steps
  - Duration predictor and vocoder
  - Metric computation (Whisper, WavLM, and UTMOS)

---

## 6. Chapter 3 Objective-to-Task Mapping

| Research Objective | Implementation Tasks | Description |
| :--- | :--- | :--- |
| **Objective 1:** Characterize the effect of teacher handover timing | **Tasks 3–4** | Characterize candidate handover times $\{4, 5, 6, 8\}$ across clean and noisy conditions on development data; evaluate failure rates with Clopper–Pearson CIs; compute oracle step distribution and correlation with drift $d_i$ and prompt SNR. |
| **Objective 2:** Design and implement the input-adaptive handover scheduler | **Tasks 1–2** | Single deterministic sampler (`core/sampler.py`), single evaluation and failure module (`core/metrics.py`) implementing Definition 3.1. |
| **Objective 3:** Calibrate the scheduler's risk threshold | **Task 5** | Learn-Then-Test (LTT) risk calibration (`obj3_ltt/calibrate.py`, `verify.py`) with fixed-sequence testing bounded by $(\alpha=0.05, \delta=0.10)$; repeated-split stability verification. |
| **Objective 4:** Compare quality, risk, and computational efficiency | **Task 6** | Held-out benchmark on test split (`obj4_benchmark/run.py`) evaluating NFE forward-pass savings, bootstrap 95% CIs, Wilcoxon signed-rank tests, runtime, RTF, and distribution-shift robustness. |

---

## 7. Pilot Studies Archival Status

All prior pilot experiments, exploratory runs, heuristic 4-fold cross-validation logs, and preliminary reports have been preserved and moved to:
`Handover Scheduling/_archive_pilot/`
Full details of each archived pilot study, its historical location, and its status as preliminary exploration are indexed in `_archive_pilot/MANIFEST.md`.
