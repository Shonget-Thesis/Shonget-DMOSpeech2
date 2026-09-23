# 5-Way Handover Benchmark: Fixed-W Extension Report

**Protocol:** Strict Causal Noise Isolation with `eval()` mode (0.0 bit-for-bit diff when w=1.0)  
**ASR Model:** `openai/whisper-base` + `jiwer` WER  
**Fixed Weight Value:** w = 0.8596 (matched to Adaptive TAGS empirical mean weight)  
**Sample Count:** 95 LibriSpeech test-clean cross-sentence pairs  

---

## 1. Executive Summary — 5-Way Comparison

| Method | Avg Weight | Avg Peak Flux ↓ | Avg Mean Flux ↓ | Avg WER ↓ | Peak Wins |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Hard)** | `1.0000` | `59.84` | `11.62` | `11.6%` | 37/95 |
| **Cosine Ablation** | `0.0142` | `69.64` | `13.90` | `12.9%` | 29/95 |
| **Adaptive TAGS (tanh)** | `0.8611` | `59.43` | `11.57` | `11.2%` | 17/95 |
| **SLR Cap (rho=1.50)** | `1.0000` | `59.84` | `11.62` | `11.6%` | 0/95 |
| **Fixed w=0.8596** | `0.8596` | `59.55` | `11.58` | `11.2%` | 12/95 |

*Note: Baseline and SLR Cap are mathematically and numerically bit-for-bit identical on all samples where w_slr = 1.0 (verified max diff = 0.0).*

---

## 2. The Reviewer's Core Question: Adaptive TAGS vs. Fixed w=0.8596

| Metric | Adaptive TAGS | Fixed w=0.8596 | Difference (Fixed − TAGS) |
|:---|:---:|:---:|:---:|
| Avg Peak Flux | `59.4272` | `59.5497` | `+0.1225` |
| Avg WER | `11.24%` | `11.18%` | `-0.06%` |
| TAGS wins (lower peak flux) | 41/95 (43.2%) | — | — |
| Fixed wins (lower peak flux) | — | 38/95 (40.0%) | — |

### Paired Difference: Fixed_w Peak Flux − TAGS Peak Flux
- **Mean difference:** `+0.1225` (positive = TAGS achieves lower flux than fixed damping)
- **Std deviation:** `0.8367`
- **95% Confidence Interval:** `[-0.0457, +0.2908]`

---

## 3. High-Shock Analysis (Top Quartile)

| Subset | Baseline Avg Peak | TAGS Avg Peak | Fixed Avg Peak | TAGS Advantage (Fixed − TAGS) |
|:---|:---:|:---:|:---:|:---:|
| High-shock samples (ρ ≥ 0.645, n=24) | `63.25` | `61.99` | `62.14` | `+0.15` |

---

## 4. Per-Sample Detail (First 20 Samples)

| ID | Spk | Shock ρ | Base Peak | TAGS w | TAGS Peak | Fixed Peak | TAGS WER | Fixed WER | Peak Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4992 | `0.653` | `76.6` | `0.851` | `75.6` | `75.6` | `0.0%` | `0.0%` | **COSINE** |
| 2 | 4992 | `0.571` | `53.0` | `0.866` | `57.1` | `57.3` | `27.3%` | `27.3%` | **BASE** |
| 3 | 4992 | `0.678` | `64.3` | `0.847` | `63.1` | `63.9` | `10.0%` | `10.0%` | **TAGS** |
| 4 | 4992 | `0.580` | `38.3` | `0.864` | `37.6` | `37.7` | `0.0%` | `0.0%` | **COSINE** |
| 5 | 4992 | `0.546` | `41.0` | `0.871` | `40.8` | `40.9` | `14.3%` | `14.3%` | **COSINE** |
| 6 | 4992 | `0.538` | `102.1` | `0.872` | `93.6` | `93.6` | `20.0%` | `24.0%` | **FIXED** |
| 7 | 4992 | `0.668` | `71.8` | `0.848` | `71.1` | `71.1` | `0.0%` | `0.0%` | **COSINE** |
| 8 | 4992 | `0.593` | `44.8` | `0.862` | `44.5` | `44.5` | `7.7%` | `7.7%` | **TAGS** |
| 9 | 4992 | `0.555` | `0.5` | `0.869` | `0.5` | `0.5` | `13.3%` | `13.3%` | **BASE** |
| 10 | 4992 | `0.464` | `104.5` | `0.887` | `99.5` | `105.0` | `0.0%` | `0.0%` | **COSINE** |
| 11 | 4992 | `0.565` | `49.5` | `0.867` | `50.5` | `50.6` | `20.0%` | `20.0%` | **BASE** |
| 12 | 4992 | `0.524` | `39.1` | `0.875` | `40.5` | `40.4` | `15.0%` | `15.0%` | **BASE** |
| 13 | 4992 | `0.531` | `61.1` | `0.874` | `60.3` | `60.5` | `15.8%` | `15.8%` | **TAGS** |
| 14 | 672 | `0.486` | `0.0` | `0.883` | `0.0` | `0.0` | `10.0%` | `10.0%` | **BASE** |
| 15 | 672 | `0.685` | `3.1` | `0.845` | `4.7` | `4.7` | `0.0%` | `0.0%` | **BASE** |
| 16 | 672 | `0.597` | `25.7` | `0.861` | `27.2` | `27.2` | `21.4%` | `21.4%` | **BASE** |
| 17 | 672 | `0.524` | `51.3` | `0.875` | `51.6` | `51.6` | `23.5%` | `23.5%` | **BASE** |
| 18 | 672 | `0.639` | `87.8` | `0.853` | `86.9` | `86.8` | `5.6%` | `5.6%` | **FIXED** |
| 19 | 672 | `0.660` | `72.5` | `0.850` | `67.1` | `67.0` | `16.7%` | `16.7%` | **FIXED** |
| 20 | 672 | `0.495` | `91.2` | `0.881` | `83.6` | `84.0` | `5.9%` | `5.9%` | **TAGS** |

---

## 5. Summary Findings

1. **Resolution of Discrepancy:** SLR Cap and Baseline match with `0.0000` difference whenever `w_slr = 1.0`, confirming full causal isolation.
2. **TAGS vs. Fixed w=0.8596:**
   - TAGS beats Fixed-W on **41/95** samples (43.2%).
   - Net peak flux difference: **+0.1225** [95% CI: `-0.0457` to `+0.2908`].
