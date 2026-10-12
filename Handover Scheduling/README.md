# Master Workspace Map: Handover Scheduling in DMOSpeech 2

**Author:** Rhenel Jhon Sajol  
**Thesis Topic:** Input-Adaptive Teacher-to-Student Handover Scheduling in DMOSpeech 2  
**Organization:** Topic-Based Modular Structure (3 Core Research Pillars + Figures & Chapter Guides)

---

## 📂 Topic-Based Folder Structure

The entire workspace has been restructured into **three self-contained research pillars**, each with its own `spec/protocols/`, `reports/`, `code/`, `telemetry/`, and `audio/`:

```
Handover Scheduling/
│
├── 📄 README.md                                      <── This Directory Roadmap
│
├── 🖼️ figures/                                        <── Publication-Ready Diagrams
│   ├── handover_scheduling_architecture.jpg          <── OFFICIAL THESIS FIGURE 1 (Selected Architecture)
│   └── proposed_solution_architecture.jpg            <── Detailed Mathematical Pipeline Diagram
│
├── 📁 01_Problem_Validation/                          <── PILLAR 1: Proving Fixed Handover Fails
│   ├── 📄 reports/
│   │   └── FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md   <── Problem Proof Report (87.5% safe early exit)
│   ├── 💻 code/
│   │   └── run_proving_fixed_timing_problem.py        <── Runner script (48 instances / 240 runs)
│   ├── 📈 telemetry/
│   │   └── fixed_handover_timing_proof.json           <── Raw telemetry metrics JSON
│   └── 🎧 audio/
│       └── audio_problem_proof/                       <── 240 diagnostic .wav synthesis files
│
├── 📁 02_Student_Restart_Validation/                  <── PILLAR 2: Proving Student Restart τ is Fixed at j=1
│   ├── 📄 reports/
│   │   ├── EXPANDED_HANDOVER_SCOPE_48_INSTANCES_PILOT_REPORT.md <── THE 48-INSTANCE REPORT (Proves j=1 is fixed)
│   │   ├── EXPANDED_HANDOVER_SCOPE_PILOT_REPORT.md    <── Initial 2-case preliminary preview
│   │   └── STUDENT_RESTART_LEVEL_PILOT_STEPS.md       <── Scope investigation notes
│   ├── 📋 protocols/
│   │   └── EXPANDED_HANDOVER_SCOPE_PROBLEM_VALIDATION.md <── Governing testbed design protocol
│   ├── 💻 code/
│   │   ├── run_expanded_scope_48_instances.py         <── Full 48-instance grid runner script
│   │   ├── run_expanded_handover_scope_validation.py  <── 2-instance pilot validation script
│   │   └── run_student_restart_pilot.py               <── Auxiliary pilot script
│   ├── 📈 telemetry/
│   │   ├── expanded_handover_scope_48_telemetry.json  <── 384 runs telemetry JSON
│   │   └── expanded_handover_scope_telemetry.json     <── Preliminary telemetry JSON
│   └── 🎧 audio/
│       ├── audio_expanded_scope_48/                   <── 384 generated .wav files across all (s, j) grids
│       └── audio_expanded_scope/                      <── Preliminary pilot audio
│
└── 📁 03_Adaptive_Handover_Solution/                  <── PILLAR 3: Proposed Risk-Calibrated Adaptive Solution
    ├── 📐 spec/
    │   └── RISK_CALIBRATED_ADAPTIVE_HANDOVER_IMPLEMENTATION_SPEC.md <── ALGORITHM & FORMULAS SPECIFICATION
    ├── 📄 reports/
    │   └── OFFICIAL_48_INSTANCE_BENCHMARK_REPORT.md   <── UNIFIED 48-INSTANCE MASTER REPORT (Verdict 1 & Figure 1)
    ├── 💻 code/
    │   ├── adaptive_handover_scheduler.py             <── THE OFFICIAL ADAPTIVE SCHEDULING ALGORITHM (Connects to DMOSpeech 2)
    │   └── benchmark_runners/                         <── Dedicated benchmark execution scripts
    ├── 📈 telemetry/
    │   ├── four_fold_cv_oof_telemetry.json            <── Cross-validation out-of-fold telemetry
    │   └── risk_calibrated_48_benchmark_telemetry.json <── 48-instance benchmark telemetry
    └── 🎧 audio/
        └── audio_48_benchmark/                        <── Synthesized benchmark evaluation audio
```

---

## ⚡ Direct File Finder

| What are you looking for? | Exactly Where to Open |
| :--- | :--- |
| **The Official Adaptive Scheduling Algorithm** | [`03_Adaptive_Handover_Solution/code/adaptive_handover_scheduler.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/03_Adaptive_Handover_Solution/code/adaptive_handover_scheduler.py) |
| **Official 48-Instance Master Benchmark Report** | [`03_Adaptive_Handover_Solution/reports/OFFICIAL_48_INSTANCE_BENCHMARK_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/03_Adaptive_Handover_Solution/reports/OFFICIAL_48_INSTANCE_BENCHMARK_REPORT.md) |
| **The Handover Implementation & Algorithm Spec** | [`03_Adaptive_Handover_Solution/spec/RISK_CALIBRATED_ADAPTIVE_HANDOVER_IMPLEMENTATION_SPEC.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/03_Adaptive_Handover_Solution/spec/RISK_CALIBRATED_ADAPTIVE_HANDOVER_IMPLEMENTATION_SPEC.md) |
| **The Student Restart Validation (48 Instances)** | [`02_Student_Restart_Validation/reports/EXPANDED_HANDOVER_SCOPE_48_INSTANCES_PILOT_REPORT.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/02_Student_Restart_Validation/reports/EXPANDED_HANDOVER_SCOPE_48_INSTANCES_PILOT_REPORT.md) |
| **The Fixed Timing Problem Proof Report** | [`01_Problem_Validation/reports/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/01_Problem_Validation/reports/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md) |
| **Official Architecture Diagram (Figure 1)** | [`figures/handover_scheduling_architecture.jpg`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/figures/handover_scheduling_architecture.jpg) |


