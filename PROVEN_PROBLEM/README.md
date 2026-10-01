# PROVEN_PROBLEM: The 240-Run Handover Timing Problem & Causal Adaptive Scheduler

**Document Status:** Primary Workspace for the Current Official Master's Thesis  
**Master Thesis Report:** [`REPORTS/FINAL/THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/FINAL/THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md)  

---

## 🏆 Quick Directory Map

This folder contains the complete, self-contained implementation, evaluation, audio files, and telemetry for the **Handover Timing Problem and Adaptive Scheduler Solution**:

```
PROVEN_PROBLEM/
│
├── 📁 REPORTS/
│   ├── 📁 FINAL/
│   │   └── 🏆 THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md  <── THE MAIN THESIS REPORT
│   ├── 📁 Problem/
│   │   └── FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md                     <── Standalone problem proof (240 runs)
│   └── 📁 Solution/
│       └── ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md                      <── Standalone solution evaluation
│
├── 📁 results/
│   ├── fixed_handover_timing_proof.json                                  <── 240 diagnostic runs telemetry (186 KB)
│   └── calibrated_hybrid_scheduler_telemetry.json                        <── 48 hybrid scheduler instances (47 KB)
│
├── 📁 audio_problem_proof/                                               <── 240 generated .wav audio files
│
└── 🐍 Python Execution Scripts:
    ├── run_proving_fixed_timing_problem.py                               <── Generates 240 runs
    └── run_adaptive_scheduler_evaluation.py                              <── Evaluates Calibrated Hybrid Scheduler
```

---

## 💡 What Makes this Folder Different from Older Folders?

* **Older Folders (`BENCHMARK_100_SAMPLES/`, `BENCHMARK_SUITE/`):**
  * Investigated whether the handover boundary caused an acoustic vector shock at fixed $t=0.25$ and tested $\tanh$ damping.
  * Disproved in Protocol 3: The handover is already smooth at $t=0.25$, and vector damping caused slight WER drag.
* **This Folder (`PROVEN_PROBLEM/`):**
  * Focuses on the **timing / steps problem**: Why use 4 teacher steps for every utterance when clean speech only needs 1 or 2 steps, and noisy speech crashes if switched early?
  * Validated across **240 generation runs** (12 speakers $\times$ 4 conditions $\times$ 5 fixed policies).
  * Solved via the **Calibrated Safety-Guarded Hybrid Scheduler ($\pi^*$)**, saving **15.4% net compute on clean speech** with **0% failure penalty** ($W=0.0, p=0.000244$).
