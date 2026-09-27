# Adaptive Handover Study: Document & Report Index

This directory consolidates all the newly generated reports, validation blueprints, and master defense documentation for the thesis:
**"Input-Adaptive Teacher-to-Student Transition Timing for Efficient Flow-Matching Speech Synthesis"**.

---

## Recommended Reading Order

```
  1. FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md  <── [Phase 1: Proof of Problem (240 runs)]
                      │
                      ├──► 2. ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md <── [Phase 2: Invented Schedulers Benchmark]
                      │
                      ├──► 3. ADAPTIVE_TRANSITION_THESIS_DEFENSE_MASTER.md (Complete Defense Master)
                      │
                      ├──► 4. SWITCH_TIME_SWEEP_REPORT.md (Phase 1: Proof of 38.5% Compute Waste)
                      │
                      ├──► 5. ADAPTIVE_CONTROLLER_EVALUATION_REPORT.md (Held-Out Evaluation)
                      │
                      └──► 6. PHASE2_STRESS_TEST_REPORT.md (Adverse Robustness)
```

---

## File Summaries

### 1. [`FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md) *(The Primary Problem Proof)*
- **What it is:** The decisive empirical validation experiment strictly testing the Null Hypothesis ($H_0$) across 12 distinct speakers, 4 conditions, and 5 fixed policies ($N=48$ instances, 240 generation runs).
- **Key Finding:** Refutes the Null Hypothesis. Early fixed policies are unsafe (failure rates of 33.3% to 58.3%), while the standard fixed policy ($t=0.25$) is computationally wasteful (Compute Regret = +3.96 NFE on 100% of clean speech). The Per-Sample Oracle achieves 0.0% failure rate with 9.0 NFE.

### 2. [`ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md) *(The Methodological Evaluation)*
- **What it is:** The complete comparative evaluation of the two invented inference-time adaptive schedulers (SNR-Aware and Cosine Agreement Probe) against all 5 fixed baselines, the condition heuristic, and the strict retrospective oracle on all 48 benchmark instances.
- **Key Finding:** Both adaptive schedulers achieve statistically significant reductions in computation over fixed $t=0.25$ ($W=0.0, p=0.000244$), slashing mean NFE from 13.00 down to 10.21 ($-21.5\%$) with zero forward overhead and minimal regret (+0.85 NFE above oracle).

### 3. [`ADAPTIVE_TRANSITION_THESIS_DEFENSE_MASTER.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/ADAPTIVE_TRANSITION_THESIS_DEFENSE_MASTER.md) *(Capstone Document)*
- **What it is:** The complete, unified defense chapter.
- **Key Contents:** Theoretical motivation of two-phase flow matching, refutation of "boundary seam shock", empirical proofs of compute waste, the online adaptive controller architecture, and scripted committee defense Q&A.

### 3. [`SWITCH_TIME_SWEEP_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/SWITCH_TIME_SWEEP_REPORT.md) *(Phase 1 Study)*
- **What it is:** Empirical benchmark across 6 switch points ($t \in \{0.05, 0.10, 0.15, 0.25, 0.40, 0.50\}$) across 10 diverse sentences (60 runs).
- **Key Finding:** Proves that switching at $t=0.10$ achieves identical WER (1.67%) and speaker similarity (0.8507) as DMOSpeech 2's fixed $t=0.25$, proving that the fixed schedule wastes **38.5% of inference compute** with zero return.

### 4. [`ADAPTIVE_CONTROLLER_EVALUATION_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/ADAPTIVE_CONTROLLER_EVALUATION_REPORT.md) *(Online System Validation)*
- **What it is:** Generalization evaluation of the online adaptive controller on **10 completely unseen held-out samples**.
- **Key Finding:** The online controller achieved a **36.92% global NFE reduction** (8.2 NFE vs 13.0 NFE) with **identical WER (2.54% vs 2.54%)**.

### 5. [`PHASE2_STRESS_TEST_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/PHASE2_STRESS_TEST_REPORT.md) *(Phase 2 Adverse Robustness)*
- **What it is:** Robustness stress testing under 4 conditions (Clean, Duration $-15\%$, Duration $+15\%$, Noisy Prompt 15dB) $\times$ 3 switches ($t \in \{0.10, 0.25, 0.40\}$).
- **Key Finding:** Proves static $t=0.25$ fails in both directions: on clean/expanded inputs, early handover ($t=0.10$) achieves **0.00% WER** (saving 30.8% compute); on noisy inputs, sustaining teacher guidance to $t=0.25$ slashes WER from 18.21% down to **7.50%**.

### 6. [`DMOSPEECH2_ADAPTIVE_HANDOVER_PROBLEM_VALIDATION_PLAN.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/REPORTS/DMOSPEECH2_ADAPTIVE_HANDOVER_PROBLEM_VALIDATION_PLAN.md) *(Protocol Specification)*
- **What it is:** The comprehensive experimental design specification detailing hypotheses, metrics, and execution steps.

---

## Associated Data & Audio Locations (Relative to `PROVEN_PROBLEM/`)
- Raw JSON Metrics: `../results/`
  - [`fixed_handover_timing_proof.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/fixed_handover_timing_proof.json) *(The 240-Run Proof Telemetry)*
  - [`switch_time_sweep_summary.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/switch_time_sweep_summary.json)
  - [`adaptive_controller_eval_summary.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/adaptive_controller_eval_summary.json)
  - [`phase2_stress_test_summary.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/phase2_stress_test_summary.json)
- Synthesized Audio:
  - Problem Proof Audio (240 files): `../audio_problem_proof/`
  - Phase 1 Sweep Audio: `../audio_sweep/`
  - Adaptive Controller Audio: `../audio_adaptive_eval/`
  - Phase 2 Stress Audio: `../audio_stress_phase2/`
