# 📊 Speech Trajectory Cushioning Benchmark Repository: Baseline vs. Constant vs. Adaptive

This directory contains the numerical benchmarks, comparative telemetry, and intelligibility analysis comparing all three speech trajectory synthesis regimes across 5 diverse voices:
1. **Baseline DMOSpeech 2** (Fixed $w = 1.0$)
2. **Constant Cushion DMOSpeech 2** (Fixed $w = 0.85$)
3. **Adaptive TAGS-Flow DMOSpeech 2** (Dynamic $w_{\text{adaptive}} = 1.0 - 0.26 \cdot \tanh(r)$)

---

## 📑 Contents

| File | Type | Description |
| :--- | :---: | :--- |
| 🎨 [SPECTROGRAM_VISUAL_AUDIT.md](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/SPECTROGRAM_VISUAL_AUDIT.md) | **Visual Audit & Gallery** | **Side-by-side Mel-Spectrogram image plots (0–12 kHz)** for all 5 samples with highlighted handover boundary line ($t=0.07\text{s}$). |
| 🔬 [BASELINE_VS_CONSTANT_VS_ADAPTIVE_REPORT.md](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/BASELINE_VS_CONSTANT_VS_ADAPTIVE_REPORT.md) | **Markdown Table** | **Complete Tri-Way comparative audit** across all 3 systems (boundary flux, dynamic $w$, reduction %, audio links, and Whisper ASR transcripts). |
| 📊 [BASELINE_VS_ADAPTIVE_5_SAMPLES_REPORT.md](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/BASELINE_VS_ADAPTIVE_5_SAMPLES_REPORT.md) | **Markdown Table** | Head-to-head comparative table between Baseline and Adaptive TAGS-Flow. |
| 📝 [FIVE_SAMPLES_BENCHMARK_REPORT.md](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/FIVE_SAMPLES_BENCHMARK_REPORT.md) | **Markdown Report** | Adaptive benchmark report highlighting shock ratio telemetry across the 5 acoustic prompts. |
| 🔢 [five_samples_constant_results.json](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/five_samples_constant_results.json) | **JSON (Raw)** | Complete Constant Cushion ($w=0.85$) telemetry (flux, spikes, runtime, Whisper output). |
| 🔢 [five_samples_benchmark_results.json](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/five_samples_benchmark_results.json) | **JSON (Raw)** | Complete Adaptive Cushion numerical records ($\|\mathbf{x}_{\text{teacher}}\|$, $\|\Delta\mathbf{x}\|$, ratio $r$, dynamic $w$, flux, Whisper output). |
| 🔢 [five_samples_baseline_results.json](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/five_samples_baseline_results.json) | **JSON (Raw)** | Complete Baseline DMOSpeech 2 numerical records (flux, spike, generation time, Whisper output). |
| 🔢 [adaptive_benchmark_results.json](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/ADAPTIVE_TAGS/metricsresults/adaptive_benchmark_results.json) | **JSON (Raw)** | Initial benchmark record comparing Baseline ($w=1.0$), Constant ($w=0.85$), and Adaptive ($w=0.8504$). |

---

## 🔬 Key Empirical Findings Across 5 Diverse Voices

* **Intelligibility Stability:** **0.0% Word Error Rate (100% transcript match)** across all 15 audio generations (Baseline, Constant, and Adaptive) validated with OpenAI Whisper ASR.
* **Acoustic Turbulence Attenuation:**
  - On dramatic theatrical speech (Sample 5), Constant cushion reduced boundary flux by **100.0%** (`18.60` $\to$ `0.007`) and Adaptive by **91.3%** (`18.60` $\to$ `1.62`).
  - On folksy country speech (Sample 2), Constant cushion reduced boundary flux by **96.6%** (`16.45` $\to$ `0.55`) and Adaptive by **3.9%** (`16.45` $\to$ `15.81`).
  - On serene feminine speech (Sample 1), Adaptive cushion achieved an optimal **18.2% reduction** (`24.72` $\to$ `20.22`), maintaining natural smoothness.
* **Dynamic Safety:** The adaptive factor dynamically adjusted between $w_{\text{adaptive}} = 0.8291$ and $0.8793$, remaining safely bounded within the theoretical guarantee $[0.74, 1.00]$.
