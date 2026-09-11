# 🏛️ Proposed Thesis Subsystem: Shonget-Matching / TAGS-Flow Inference Engine

**Author:** 
**Architecture Reference:** Figure 1, Block (c) in `IMPLEMENTATION_PLAN/SHONGET_MATCHING_ARCHITECTURE.jpg`  
**Specification:** `IMPLEMENTATION_PLAN/SYSTEM_ARCHITECTURE_TAGS_FLOW_ASSISTIVE.md`  

---

## 📌 Theoretical Formulation

This package houses the standalone implementation of the **Shonget-Matching Flow Solver** (also referred to algorithmically as **TAGS-Flow: Trajectory-Aware Guided Sampling**):

$$\mathbf{x}_{t+h} = \mathbf{x}_t + h \cdot \alpha \cdot \left[ (1 - w(t)) \cdot \mathbf{v}_{\text{teacher}}(\mathbf{x}_t, t) + w(t) \cdot \mathbf{v}_{\text{student}}(\mathbf{x}_t, t) \right]$$

### Key Components:
1. **Teacher Generator (F5-TTS DiT-Base):** Establishes initial pitch, vocal timbre, and phoneme grounding.
2. **Student Generator (DMOSpeech 2 Distilled 4-Step):** Provides rapid, low-latency trajectory completion.
3. **$\mathcal{C}^1$-Continuous Handshake Manifold ($w(t)$):**
   $$w(t) = \frac{1}{2} \left[ 1 - \cos\left( \pi \frac{t - t_1}{t_2 - t_1} \right) \right]$$
   Guarantees $\left. \frac{dw}{dt} \right|_{t_1} = \left. \frac{dw}{dt} \right|_{t_2} = 0$, mathematically eliminating the boundary shock and phase discontinuity present in baseline DMOSpeech 2.

---

## 📂 Source Code Overview

- **[`smooth_dmospeech.py`](smooth_dmospeech.py):** Core inference class `SmoothDMOSpeechInference` implementing the $\mathcal{C}^1$ manifold handover between Teacher and Student.
- **[`run_inference.py`](run_inference.py):** Clean, standalone runner script that synthesizes speech and saves the output waveform to `output_smooth.wav`.

---

## 🚀 How to Run

From the project root:
```powershell
.\venv\Scripts\Activate.ps1
python proposed_system/run_inference.py
```

Generated evaluation assets and comparative spectrograms are archived in [`../eval_results/`](../eval_results/).
