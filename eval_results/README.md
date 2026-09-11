# 📊 Empirical Evaluation: DMOSpeech 2 vs. Proposed Smooth Manifold

This directory archives the finalized objective evaluation, acoustic metrics, spectrogram visuals, and audio waveforms comparing **Baseline DMOSpeech 2** against the **Proposed $\mathcal{C}^1$-Continuous Smooth Manifold**.

---

## 🎧 Audio Assets

| System | Audio File | Description |
| :--- | :--- | :--- |
| **Baseline DMOSpeech 2** | [`01_baseline_dmospeech2.wav`](01_baseline_dmospeech2.wav) | 3 Teacher + 3 Student steps with unregularized hard cutoff at $t=0.076$. |
| **Proposed Smooth Manifold** | [`02_proposed_smooth_manifold.wav`](02_proposed_smooth_manifold.wav) | 3 Teacher + 3 Student steps with $\mathcal{C}^1$ continuous manifold handover. |

---

## 📈 Quantitative Acoustic Benchmark

| Metric | Baseline DMOSpeech 2 | Proposed Smooth Manifold | Thesis Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Transition Flux (Acoustic Turbulence)** | **19.88** | **13.07** | 🏆 **34.3% Reduction** (Significantly smoother transition dynamics) |
| **Spectral Centroid (Vocal Timbre)** | 2,621.8 Hz | **2,739.0 Hz** | 🤝 **Natural Timbre Preserved** (<120 Hz difference) |
| **Spectral Flatness (Noise Floor)** | 0.04690 | **0.04749** | 🤝 **Harmonic Structure Maintained** |
| **High-Frequency Ratio (>4kHz)** | 24.62% | **23.92%** | 🤝 **Preserves Consonant Crispness** |
| **Whisper ASR Intelligibility** | *"Hello. Demo Speech 2 is successfully running on this system."* | *"Hello. Demospeech 2 is successfully running on this system."* | 🏆 **100% Intelligible Match** (Natural pronunciation) |

---

## 🖼️ Visual Spectrogram Comparisons

- **Baseline Spectrogram:** [`01_baseline_spectrogram.png`](01_baseline_spectrogram.png)
- **Proposed Smooth Spectrogram:** [`02_proposed_smooth_spectrogram.png`](02_proposed_smooth_spectrogram.png)
- **Side-by-Side Comparison:** [`comparison_baseline_vs_proposed_spectrogram.png`](comparison_baseline_vs_proposed_spectrogram.png)

---

## 🧮 Computational Step Logs
- **Baseline Calculations (JSON):** [`baseline_computational_step_log.json`](baseline_computational_step_log.json)
- **Proposed Calculations (JSON):** [`proposed_computational_step_log.json`](proposed_computational_step_log.json)
- **Final Comparative Metrics (JSON):** [`final_comparative_report.json`](final_comparative_report.json)
