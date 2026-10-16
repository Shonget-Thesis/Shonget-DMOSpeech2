# Manifest of Archived Pilot Results

This manifest documents all experimental outputs, scripts, and reports moved from active development into `Handover Scheduling/_archive_pilot/`. In accordance with Chapter 3 methodology requirements, these artifacts serve as **preliminary and historical studies only** and are not cited as final thesis evidence.

---

## Inventory of Archived Components

### 1. `01_Problem_Validation/`
- **Original Location:** `Handover Scheduling/01_Problem_Validation/`
- **Contents:**
  - `code/run_proving_fixed_timing_problem.py`: Problem proof runner script on initial 8 instances.
  - `reports/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`: Problem validation report.
  - `telemetry/fixed_handover_timing_proof.json`: Telemetry data for the 8 instances.
  - `audio/`: Generated WAV files across fixed switching times ($t \in \{0.07, 0.10, 0.15, 0.25\}$).
- **Why Considered Preliminary:**
  - Small exploratory sample size (8 instances).
  - Evaluated using legacy evaluation tools: `openai/whisper-base` on CPU for ASR and Resemblyzer for speaker similarity (contrasting with Chapter 3's requirement of Whisper large-v3 and WavLM-large + ECAPA-TDNN).
  - Lacked speaker-disjoint splits and rigorous statistical calibration.

### 2. `02_Student_Restart_Validation/`
- **Original Location:** `Handover Scheduling/02_Student_Restart_Validation/`
- **Contents:**
  - `code/run_student_restart_pilot.py`: Pilot student restart exploration script.
  - `code/run_expanded_handover_scope_validation.py`: Handover scope validation runner.
  - `code/run_expanded_scope_48_instances.py`: 48-instance expanded scope runner.
  - `protocols/EXPANDED_HANDOVER_SCOPE_PROBLEM_VALIDATION.md`: Validation protocol document.
  - `reports/EXPANDED_HANDOVER_SCOPE_PILOT_REPORT.md`: Early pilot scope report.
  - `reports/STUDENT_RESTART_LEVEL_PILOT_STEPS.md`: Investigation into restart steps.
  - `reports/EXPANDED_HANDOVER_SCOPE_48_INSTANCES_PILOT_REPORT.md`: Report on 48 instances.
  - `telemetry/expanded_handover_scope_telemetry.json` & `expanded_handover_scope_48_telemetry.json`: Telemetry logs.
  - `audio/`: Synthesized waveforms across various restart levels.
- **Why Considered Preliminary:**
  - Intermediate studies exploring student restart configurations and expanded scope.
  - Evaluated duration perturbation conditions ($\pm 15\%$ duration) which are out of scope for the Chapter 3 handover-scheduling focus.
  - Used legacy metric models (Whisper-base and Resemblyzer) without finite-sample risk bounds.

### 3. `03_Adaptive_Handover_Solution/`
- **Original Location:** `Handover Scheduling/03_Adaptive_Handover_Solution/`
- **Contents:**
  - `code/adaptive_handover_scheduler.py`: Prototype adaptive scheduler and SNR estimator.
  - `code/benchmark_runners/run_risk_calibrated_48_benchmark.py`: Pilot 48-instance benchmark runner.
  - `spec/RISK_CALIBRATED_ADAPTIVE_HANDOVER_IMPLEMENTATION_SPEC.md`: Prototype design specification.
  - `spec/calibrated_handover_config.json`: Heuristic configuration file.
  - `telemetry/four_fold_cv_oof_telemetry.json`: Out-of-fold cross-validation telemetry.
  - `telemetry/risk_calibrated_48_benchmark_telemetry.json`: Benchmark telemetry over 48 instances.
  - `audio/`: Output audio files across adaptive, fixed, and reference arms.
- **Why Considered Preliminary:**
  - Employs 4-fold cross-validation on 48 instances rather than speaker-disjoint Learn-Then-Test (LTT) risk calibration.
  - Used legacy evaluation models (`openai/whisper-base` and Resemblyzer).
  - Inadvertently reported solver steps ($12.00 \to 9.65$) rather than neural forward passes (actual NFE $21 \to 16.29$).
  - Search grid stopped at $\lambda = 0.60$ (the boundary in all folds).
  - Stated uncomputed risk bounds ($\delta \le 0.05$) without finite-sample statistical guarantees.

### 4. `OFFICIAL_48_INSTANCE_BENCHMARK_REPORT.md`
- **Archived Location:** `Handover Scheduling/_archive_pilot/OFFICIAL_48_INSTANCE_BENCHMARK_REPORT.md`
- **Original Location:** `Handover Scheduling/03_Adaptive_Handover_Solution/reports/OFFICIAL_48_INSTANCE_BENCHMARK_REPORT.md`
- **Why Considered Preliminary & Corrections Required:**
  - Errata noted regarding solver steps vs. forward passes (NFE), telemetry discrepancy in switching distribution ($15/12/6/15$ vs $18/11/4/15$), reversed parameter description in Section 2, boundary truncation at $\lambda = 0.60$, and differing metric values. Full correction header documented per Task 7.
