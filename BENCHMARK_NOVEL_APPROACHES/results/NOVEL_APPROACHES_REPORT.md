# Benchmark of 3 Novel Handover Approaches vs. Baseline & Original TAGS

**Dataset:** 1 Cross-Sentence Pairs from LibriSpeech `test-clean`  
**Causal Isolation:** Enforced with `generator.eval()` & `teacher.eval()`  
**ASR Engine:** `openai/whisper-base` + `jiwer`  

---

## 1. Executive Summary Table

| Method | Mean Weight $\bar{w}$ | Peak Flux ↓ | Mean Flux ↓ | WER ↓ | Peak Wins | Δ Peak vs. Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Hard Handover)** | `1.0000` | **76.64** | **15.25** | **0.00%** | 0/1 | — |
| **Original TAGS ($w=1-0.26\tanh$)** | `0.8508` | **75.56** | **15.11** | **0.00%** | 0/1 | `-1.08` |
| **Approach 1: Orthogonal Projection** | `0.6000` | **70.46** | **15.07** | **0.00%** | 1/1 | `-6.18` |
| **Approach 2: Boundary Windowing** | `0.6000` | **76.50** | **15.22** | **0.00%** | 0/1 | `-0.14` |
| **Approach 3: Calibrated Gate** | `0.7518` | **75.40** | **15.11** | **0.00%** | 0/1 | `-1.24` |

---

## 2. Per-Sample Detailed Comparison (First 15)

| ID | Shock ρ | Base Peak | Orig TAGS | App 1 (Ortho) | App 2 (Boundary) | App 3 (Calibrated) | Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `0.653` | `76.6` | `75.6` | `70.5` | `76.5` | `75.4` | **ORTHO** |

---

## 3. Analysis & Thesis Discussion
- **Approach 1 (Orthogonal Projection):** Damps only the lateral drift while preserving 100% of forward trajectory momentum.
- **Approach 2 (Boundary Windowing):** Selectively targets the reference-to-generation seam ($t_{boundary} = cond_mask.sum()$) rather than dampening the whole sequence uniformly.
- **Approach 3 (Calibrated Dynamic Range):** Expands the dynamic adaptation range from $\pm 0.02$ to $\pm 0.17$, giving the model real room to react dynamically to high vs. low shock sentences.
