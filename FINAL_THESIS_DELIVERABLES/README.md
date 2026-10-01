# Final Thesis Deliverables: Handover Timing & Adaptive Schedulers

**Author:** Rhenel Jhon Sajol  
**Thesis Topic:** Input-Adaptive Teacher-to-Student Transition Timing for Efficient Diffusion Speech Synthesis (DMOSpeech 2)  
**Deliverable Status:** Canonical Final Deliverables Package (Separated from all historical/exploratory experiments)  

---

## 📁 Package Contents

This folder contains **only the final, verified results, metrics, and documentation** for the 240-run Handover Timing Problem and the Calibrated Hybrid Scheduler Solution:

```
FINAL_THESIS_DELIVERABLES/
│
├── 🏆 THESIS_PROBLEM_AND_SOLUTION_CONSOLIDATED_MASTER_REPORT.md  <── THE MAIN MASTER THESIS REPORT
├── 📄 README.md                                                  <── Package Guide & Master Scorecard (This file)
│
├── 📁 reports/
│   ├── FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md                 <── Problem proof report (240 generation runs)
│   └── ADAPTIVE_SCHEDULER_EVALUATION_REPORT.md                  <── Solution benchmark report (48 instances)
│
├── 📁 metrics_and_telemetry/
│   ├── fixed_handover_timing_proof.json                         <── Raw telemetry for all 240 diagnostic runs (186 KB)
│   └── calibrated_hybrid_scheduler_telemetry.json               <── Raw telemetry for all 48 scheduler instances (47 KB)
│
└── 📁 code/
    ├── run_proving_fixed_timing_problem.py                      <── Generates the 240 diagnostic runs across 12 speakers
    └── run_adaptive_scheduler_evaluation.py                     <── Implements & evaluates the Calibrated Hybrid Scheduler
```

---

## 🏆 Final Master Benchmark Scorecard (All 48 Test Instances)

Every metric below has been audited and verified 100% against the raw telemetry JSON files in `metrics_and_telemetry/`:

| Method / Policy | Base NFE | Probe Overhead | Effective NFE | Net Savings vs Baseline | Ref-Adjusted Failures (%) | Strict Ceiling Failures (%) | Median WER | Pitch Var ($CV_{F0}$) | Operational Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed $t=0.05$** | 7.00 | 0.00 | 7.00 | $-6.00$ | 26 / 48 (**54.2%**) | 27 / 48 (56.2%) | 0.00% | 0.428 | Aggressively Unsafe |
| **Fixed $t=0.10$** | 9.00 | 0.00 | 9.00 | $-4.00$ | 13 / 48 (**27.1%**) | 14 / 48 (29.2%) | 0.00% | 0.439 | Unsafe Under Stress |
| **Fixed $t=0.15$** | 11.00 | 0.00 | 11.00 | $-2.00$ | 5 / 48 (**10.4%**) | 7 / 48 (14.6%) | 0.00% | 0.443 | Earliest 0% fail on clean |
| **Fixed $t=0.25$ (Baseline)** | 13.00 | 0.00 | 13.00 | *baseline* | **0 / 48 (0.0%)** | 4 / 48 (8.3%) | 0.00% | 0.445 | Static Default (+4.8 NFE on clean) |
| **Fixed $t=0.40$** | 16.00 | 0.00 | 16.00 | $+3.00$ | 10 / 48 (**20.8%**) | 12 / 48 (25.0%) | 0.00% | 0.449 | Overburn (Phonetic distortion) |
| **Pre-Diffusion Input (Zero-Probe)**| 10.92 | 0.00 | 10.92 | $-2.08$ | 1 / 48 (2.1%)* | 5 / 48 (10.4%) | 0.00% | 0.446 | Fast Acoustic DSP Heuristic |
| **Proposed Hybrid Scheduler ($\pi^*$)**| **11.21** | **+0.27** | **11.48** | **$-1.52$ (−11.7%)** | **0 / 48 (0.0%)** | **4 / 48 (8.3%)** | **0.00%** | **0.448** | **Proposed Causal Solution** |
| *Ref-Adjusted Oracle (Target)* | *9.04* | *0.00* | *9.04* | *$-3.96$ (−30.5%)* | *0 / 48 (0.0%)* | *4 / 48 (8.3%)* | *0.00%* | — | *Theoretical Upper Bound* |

*\*Note on Instance 21: Pre-Diffusion scheduler dispatched $t=0.10$ into phonetic drift ($15.38\%$ WER). The Hybrid Scheduler's trajectory probe measured $\rho_1 = 0.7584 < 0.78$, caught the divergence, and fell back to $t=0.15$ ($7.69\%$ WER), rescuing the utterance.*

---

## 📊 Key Validated Findings

1. **Zero Reliability Penalty:** The proposed hybrid scheduler achieves **0 / 48 failures (0.0%)** under baseline-relative criteria, matching the conservative fixed baseline while reducing compute.
2. **Statistically Significant Compute Reduction:** Paired one-sided Wilcoxon signed-rank test across all 12 independent speakers:
   $$W = 0.0, \quad p = 0.000244 \quad (p < 0.001)$$
   The scheduler achieves strictly lower computation on **100% of tested speakers**.
3. **Clean Speech Acceleration:** On clean speech, the scheduler achieves **11.00 effective NFE (−15.4% net compute savings)**.
4. **Preserved Prosody:** Fundamental frequency coefficient of variation ($CV_{F0}$) shows no statistically significant pitch flattening ($0.4479$ vs $0.4453, p = 0.6782, \text{95% CI: } [-0.0076, +0.0128]$).

---

## 🎧 Audio Files Location
The 240 synthesized `.wav` audio files corresponding to these evaluations are stored at:
`c:\Users\Rhenel Jhon Sajol\Desktop\TTS\DMOSpeech 2\PROVEN_PROBLEM\audio_problem_proof\`
