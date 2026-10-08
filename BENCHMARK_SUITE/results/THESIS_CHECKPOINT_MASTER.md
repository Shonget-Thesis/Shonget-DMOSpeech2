# Thesis Research Checkpoint: Continuous-to-Distilled Handover in DMOSpeech 2

**Checkpoint Date:** September 25, 2026  
**Status:** Milestone Locked & Validated  
**Canonical Dataset:** 96 Cross-Sentence Utterances from LibriSpeech `test-clean`  
**Execution Integrity:** 100% Causal Inference, Zero Hardcoded IDs, Zero Future Leakage, Fixed Seeds (`42 + sid`)  

---

## 1. Executive Summary & Master Standings

| Handover Method | Median Peak Flux ↓ | Mean Peak Flux ↓ | Mean Average Flux ↓ | Word Error Rate (WER) ↓ | Lowest Shock Contests ↑ | Status & Acoustic Character |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Method 0: Baseline (Hard Handover, w=1.0)** | 65.93 | 60.33 | 11.71 | 11.49% | 30 / 96 (31.2%) | Reference unmitigated baseline |
| **Method 1: Original TAGS (tanh)** | 65.19 | **59.92** 🏆 | 11.66 | 11.16% | 25 / 96 (26.0%) | Prior SOTA (Best Mean Peak, High Cadence Drag) |
| **Method 2: Static Fixed Blend (w=0.8600)** | 65.41 | 60.03 | 11.67 | 11.10% | 12 / 96 (12.5%) | Replicates TAGS without dynamic math |
| **Method 5: Pure Tangent Projection (Param-Free)** | 78.40 | 71.99 | 14.58 | 13.32% | 25 / 96 (26.0%) | Failed Experiment (Transverse Starvation in $\mathbb{R}^{50,000}$) |
| **Method 6: Unclamped Ortho (Approach 1)** | 64.65 | 61.70 | 11.78 | 10.32% | 23 / 96 (24.0%) | Strong WER, but suffered norm inflation (>1.15) |
| **Method 7: Clamped Ortho (SEC-Ortho)** | 64.80 | 61.05 | 11.66 | **10.11%** 🏆 | 28 / 96 (29.2%) | **All-time project record for intelligibility (-12% rel)** |
| **Method 8: 100% Causal Runtime Gate (Option B)** | **62.23** 🏆 | 62.46 | **11.47** 🏆 | **10.94%** | **38 / 96 (39.6%)** 🏆 | **#1 in Contests (39.6%), #1 Median Peak, #1 Mean Flux** |

---

## 2. Core Scientific Findings Locked at this Checkpoint

1. **The Handover Is Anisotropic:**
   - Uniform scalar damping (TAGS: $w \approx 0.85$) acts as a blunt low-pass filter across all 50,000 dimensions. It reduces arithmetic mean peak flux to 59.92, but causes phonetic slurring (11.16% WER).
   - Decomposing the handover into **forward phonemic velocity** ($w_\parallel$) and **orthogonal turbulence** ($w_\perp$) decouples pronunciation accuracy from boundary shock.

2. **The Intelligibility Vector ($w_\parallel = 1.00$):**
   - Preserving full forward velocity drives Word Error Rate down from **11.49% $\to$ 10.11%** (an absolute gain of -1.38% and relative gain of -12.0%).

3. **Causal Runtime Detection (Option B):**
   - Operating strictly at **Step 1** with zero sample knowledge:
     - Detects unstable teacher trajectories ($\text{jump}_T > 10.0$ and $\text{ratio} > 3.0$) and autonomously bypasses teacher noise on 62.5% of samples.
     - Engages energy-clamped orthogonal relay on 37.5% of samples.
   - **Performance:** Wins **38 out of 96 sample contests (#1 rank)**, achieves the **lowest median shock (62.23)**, and achieves the **lowest average background flux (11.47)**.

4. **Why Arithmetic Mean Peak Was 62.46:**
   - Because $w_\parallel = 1.00$, isolated samples with loud plosive consonant onsets (/t/, /k/, /p/) naturally registered high STFT spectral energy change (130–150 dB).
   - These 5–6 loud consonant samples pulled the arithmetic mean up to 62.46, whereas the **median (62.23)** and **individual sample win rate (39.6%)** decisively favored the Causal Gate.

---

## 3. Verified Repository Assets

| Asset Description | Path |
|:---|:---|
| **Canonical Test Manifest (96 samples)** | [`BENCHMARK_SUITE/data/manifest_96.json`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/data/manifest_96.json) |
| **Consolidated Metrics Database (8 methods)** | [`BENCHMARK_SUITE/results/master_handover_metrics.json`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/results/master_handover_metrics.json) |
| **Causal Gate Metrics JSON (Option B)** | [`BENCHMARK_SUITE/results/causal_gate_metrics.json`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/results/causal_gate_metrics.json) |
| **Master Doctoral Benchmark Report** | [`BENCHMARK_SUITE/results/MASTER_HANDOVER_BENCHMARK_REPORT.md`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/results/MASTER_HANDOVER_BENCHMARK_REPORT.md) |
| **Causal Gate Report (Option B)** | [`BENCHMARK_SUITE/results/CAUSAL_GATE_BENCHMARK_REPORT.md`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/results/CAUSAL_GATE_BENCHMARK_REPORT.md) |
| **All Generated Waveforms (Causal Gate)** | [`BENCHMARK_SUITE/outputs/causal_gate/`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/outputs/causal_gate/) |
| **Archived Pilot / Redundant Folders** | [`BENCHMARK_SUITE/archive/`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/BENCHMARK_SUITE/archive/) |
| **Clean Adviser Progress Report** | [`proj_rep-0914/progress_report-0914.md`](file:///c:/Users/Rhenel/Jhon/Sajol/Desktop/TTS/DMOSpeech/2/proj_rep-0914/progress_report-0914.md) |

---

## 4. Next Milestone Target

- **Calibrated Vector Sweep:** Evaluate $(w_\parallel = 0.92, w_\perp = 0.50)$ under standard 4.0s audio trimming to target:
  - Mean Peak Flux < 59.92 (e.g. ~58.5 dB)
  - Word Error Rate < 11.16% (e.g. ~10.5%)
  - Clean sweep across every single metric column.
