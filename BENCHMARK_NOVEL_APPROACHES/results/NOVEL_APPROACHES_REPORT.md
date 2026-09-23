# Benchmark of 3 Novel Handover Approaches vs. Baseline & Original TAGS

**Dataset:** 96 Cross-Sentence Pairs from LibriSpeech `test-clean`  
**Causal Isolation:** Enforced with `generator.eval()` & `teacher.eval()`  
**ASR Engine:** `openai/whisper-base` + `jiwer`  

---

## 1. Executive Summary Table

| Method | Mean Weight $\bar{w}$ | Peak Flux ↓ | Mean Flux ↓ | WER ↓ | Peak Wins | Δ Peak vs. Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Hard Handover)** | `1.0000` | **60.33** | **11.71** | **11.49%** | 18/96 | — |
| **Original TAGS ($w=1-0.26\tanh$)** | `0.8612` | **59.92** | **11.66** | **11.16%** | 9/96 | `-0.41` |
| **Approach 1: Orthogonal Projection** | `0.6000` | **61.70** | **11.78** | **10.32%** | 30/96 | `+1.37` |
| **Approach 2: Boundary Windowing** | `0.6000` | **60.22** | **11.69** | **11.49%** | 20/96 | `-0.11` |
| **Approach 3: Calibrated Gate** | `0.8258` | **60.37** | **11.70** | **11.37%** | 19/96 | `+0.04` |

---

## 2. Per-Sample Detailed Comparison (First 15)

| ID | Shock ρ | Base Peak | Orig TAGS | App 1 (Ortho) | App 2 (Boundary) | App 3 (Calibrated) | Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `0.653` | `76.6` | `75.6` | `70.5` | `76.5` | `75.4` | **ORTHO** |
| 2 | `0.571` | `53.0` | `57.1` | `61.5` | `53.0` | `57.0` | **BOUND** |
| 3 | `0.678` | `64.3` | `63.1` | `61.9` | `64.2` | `60.2` | **CALIB** |
| 4 | `0.580` | `38.3` | `37.6` | `38.5` | `37.8` | `37.8` | **ORIG** |
| 5 | `0.546` | `41.0` | `40.8` | `40.4` | `40.9` | `40.7` | **ORTHO** |
| 6 | `0.538` | `102.1` | `93.6` | `94.2` | `102.0` | `97.9` | **ORIG** |
| 7 | `0.668` | `71.8` | `71.1` | `67.2` | `71.7` | `74.2` | **ORTHO** |
| 8 | `0.593` | `44.8` | `44.5` | `50.0` | `44.4` | `44.9` | **BOUND** |
| 9 | `0.555` | `0.5` | `0.5` | `0.5` | `0.5` | `0.5` | **BOUND** |
| 10 | `0.464` | `104.5` | `99.5` | `97.0` | `104.3` | `105.2` | **ORTHO** |
| 11 | `0.565` | `49.5` | `50.5` | `52.9` | `49.3` | `50.3` | **BOUND** |
| 12 | `0.524` | `39.1` | `40.5` | `39.2` | `39.1` | `40.6` | **BASE** |
| 13 | `0.531` | `61.1` | `60.3` | `69.4` | `61.0` | `59.8` | **CALIB** |
| 14 | `0.486` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | **BASE** |
| 15 | `0.685` | `3.1` | `4.7` | `5.5` | `3.1` | `5.4` | **BOUND** |

---

## 3. Analysis & Thesis Discussion
- **Approach 1 (Orthogonal Projection):** Damps only the lateral drift while preserving 100% of forward trajectory momentum.
- **Approach 2 (Boundary Windowing):** Selectively targets the reference-to-generation seam ($t_{boundary} = cond_mask.sum()$) rather than dampening the whole sequence uniformly.
- **Approach 3 (Calibrated Dynamic Range):** Expands the dynamic adaptation range from $\pm 0.02$ to $\pm 0.17$, giving the model real room to react dynamically to high vs. low shock sentences.
