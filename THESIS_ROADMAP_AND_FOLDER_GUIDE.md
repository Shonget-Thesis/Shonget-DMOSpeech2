# Thesis Project Navigation Guide & Folder Separation

**Thesis Title:** Input-Adaptive Teacher-to-Student Transition Timing for Efficient Multi-Stage Diffusion Speech Synthesis (DMOSpeech 2)  
**Author:** Rhenel Jhon Sajol  
**Last Updated:** October 2026  

---

## 🗺️ Project Structure: Two Distinct Research Phases

To avoid any confusion between the **earlier exploratory experiments** and the **current proven thesis work**, this repository is strictly divided into two phases:

```
DMOSpeech 2 Workspace
│
├── 📁 PROVEN_PROBLEM/                ◄─── 🏆 [PHASE 2: CURRENT OFFICIAL THESIS WORK]
│   │                                      The 240-run Handover Timing Problem &
│   │                                      The Calibrated Hybrid Scheduler (π*) Solution
│   │
│   ├── 📁 REPORTS/
│   │   ├── 📁 FINAL/                 ◄─── THE SINGLE MASTER REPORT FOR DEFENSE / CONSULTATION
│   │   │   └── THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md
│   │   ├── 📁 Problem/               ◄─── The 240-run diagnostic proof report
│   │   │   └── FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md
│   │   └── 📁 Solution/              ◄─── The adaptive scheduler benchmark report
│   │       └── ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md
│   ├── 📁 results/                   ◄─── Raw verified telemetry JSONs (240 runs + 48 hybrid)
│   └── 📁 audio_problem_proof/       ◄─── The 240 synthesized .wav audio files
│
│ ═════════════════════════════════════════════════════════════════════════════════════════
│
├── 📁 BENCHMARK_100_SAMPLES/         ◄─── 📦 [PHASE 1: LEGACY / ARCHIVED EXPERIMENTS]
├── 📁 BENCHMARK_SUITE/               ◄─── Explored boundary shock & tanh damping (Sept 20-25)
└── 📁 SLR_BOUNDED_HANDOVER/               (Disproved by counterfactual control in Protocol 3)
```

---

## 🟢 PHASE 2: Current Official Thesis Work (USE THIS FOLDER)

**Folder Location:** [`PROVEN_PROBLEM/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/)

This folder contains **all the active, proven, and defendable research** for your thesis proposal and defense:

### 1. The Master Combined Report (Problem + Solution)
* 📄 **[`PROVEN_PROBLEM/REPORTS/FINAL/THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/FINAL/THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md)**  
  * **What it contains:** Combines the empirical problem proof (240 generation runs across 12 speakers and 4 conditions) with the methodological solution (Calibrated Hybrid Scheduler $\pi^*$), full comparative evaluation tables, paired Wilcoxon statistical tests ($W=0.0, p=0.000244$), and thesis defense statements.
  * **Status:** Fully audited, 100% matched to raw data, pushed to GitHub commit `a244f2a`.

### 2. Standalone Problem Proof Report
* 📄 **[`PROVEN_PROBLEM/REPORTS/Problem/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/Problem/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md)**  
  * Focuses purely on proving that a fixed switching point ($t=0.25$) is inefficient on clean speech (wasting 37.2% compute) and dangerous on noisy speech if switched too early.

### 3. Standalone Solution Evaluation Report
* 📄 **[`PROVEN_PROBLEM/REPORTS/Solution/ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/Solution/ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md)**  
  * Focuses on the formulation, algorithms, and 48-instance evaluation of the Pre-Diffusion Input-Feature Scheduler and the Calibrated Safety-Guarded Hybrid Scheduler ($\pi^*$).

### 4. Raw Underlying Telemetry Data
* 📊 **[`PROVEN_PROBLEM/results/fixed_handover_timing_proof.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/fixed_handover_timing_proof.json)** (186 KB) — All 240 diagnostic runs.
* 📊 **[`PROVEN_PROBLEM/results/calibrated_hybrid_scheduler_telemetry.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/calibrated_hybrid_scheduler_telemetry.json)** (47 KB) — All 48 hybrid scheduler evaluations.
* 🎵 **[`PROVEN_PROBLEM/audio_problem_proof/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/audio_problem_proof/)** — The 240 actual generated `.wav` audio files.

---

## 🟡 PHASE 1: Legacy / Archived Experiments (DO NOT MIX WITH THESIS)

**Folder Locations:**
* [`BENCHMARK_100_SAMPLES/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/BENCHMARK_100_SAMPLES/)
* [`BENCHMARK_SUITE/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/BENCHMARK_SUITE/)
* [`SLR_BOUNDED_HANDOVER/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/SLR_BOUNDED_HANDOVER/)

### What was done here (Sept 20–25, 2026):
* We explored whether the handover boundary caused an acoustic "boundary shock" at fixed $t=0.25$.
* We tested **Adaptive TAGS ($\tanh$)** ($w = 1 - 0.26\tanh$), **SLR Cap**, **SEC-Ortho**, etc.

### Why this phase was archived:
1. **The Counterfactual Test (Protocol 3):** When compared against a pure teacher counterfactual ([`HANDOVER_PROBLEM_REPORT_3_VERDICT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/HANDOVER_PROBLEM_REPORT_3_VERDICT.md)), the actual handover at $t=0.25$ was found to be mathematically smooth (diffusion re-noising acts as a low-pass filter) and achieved lower WER and lower spectral flux than the pure teacher.
2. Damping the vector with $\tanh$ or orthogonal projection did not improve speech intelligibility and produced slight phonetic drag.
3. Therefore, this phase was preserved for historical completeness, but set aside in favor of the **Timing Problem**.

---

## 📌 Summary for Your Adviser / Defense

| Research Aspect | Phase 1 (Archived / Legacy) | Phase 2 (Current Official Thesis) |
| :--- | :--- | :--- |
| **Core Question** | *"Is the handover vector collision causing boundary shock?"* | *"Why are we always using 4 teacher steps for every sentence?"* |
| **Investigated Factor** | Boundary Vector Damping ($w = 1 - 0.26\tanh$) | **Handover Timing / Steps ($t_{\text{switch}} \in \{0.05, 0.10, 0.15, 0.25, 0.40\}$)** |
| **Primary Dataset** | 96 LibriSpeech pairs | **12 Common Voice speakers $\times$ 4 conditions (240 runs)** |
| **Primary Result** | Hypothesis not supported (handover was already smooth) | **Problem proven:** Clean speech wastes 37% compute; noisy speech fails if early |
| **Final Solution** | Adaptive TAGS $\tanh$ (marginal benefit) | **Calibrated Hybrid Scheduler ($\pi^*$)** ($-15.4\%$ net compute on clean, 0% failure penalty) |
| **Where to Look** | `BENCHMARK_100_SAMPLES/`, `BENCHMARK_SUITE/` | 👉 **`PROVEN_PROBLEM/`** |
