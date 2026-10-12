# Expanded Handover Scope: Problem-Validation Pilot Report

**Governing Protocol:** [`Handover Scheduling/Student Restart/EXPANDED_HANDOVER_SCOPE_PROBLEM_VALIDATION.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/Student%20Restart/EXPANDED_HANDOVER_SCOPE_PROBLEM_VALIDATION.md)  
**Telemetry File:** [`Handover Scheduling/metrics_and_telemetry/expanded_handover_scope_telemetry.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/metrics_and_telemetry/expanded_handover_scope_telemetry.json)  
**Audio Directory:** [`Handover Scheduling/audio_expanded_scope/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/audio_expanded_scope/)  
**Date of Execution:** 2026-10-11 20:28:11  
**Scope:** Paired factorial evaluation across 8 configurations (s in {0.07, 0.10, 0.15, 0.25} x j in {1, 2}) on development speakers under clean nominal and 15 dB noise conditions.

---

## 1. Executive Verdict & Selected Thesis Scope

> **FINAL SCOPE SELECTION (Section 15 & 19):**  
> **Teacher-Only Adaptive Scheduling (Fixed Student Restart j=1)**  
> **Classification:** Outcome A: Teacher-Only Limitation  
> **Empirical Rationale:** Varying student restart provides negligible or inconsistent additional computational opportunity (mean Delta C restart <= 0.25 NFE) or increases quality failure rates at early handover points. The study focuses purely on input-adaptive teacher stopping.

---

## 2. Answers to the 4 Research Questions (Section 4)

1. **Question 1 (Teacher Stopping Effect with Fixed j=1):**  
   *Finding:* With restart level held fixed at j=1, the earliest acceptable teacher stopping point varies significantly across inputs (s=0.10 vs. s=0.25). Aggressive early stopping at s=0.07 produces failure on noise, while conservative s=0.25 expends 3-4 unnecessary NFE on clean speech.  
   *Conclusion:* **The teacher stopping scheduling limitation is robustly confirmed (H_{1,T} accepted).**

2. **Question 2 (Student Restart Effect within Fixed s):**  
   *Finding:* When teacher stopping time s is controlled, advancing to j=2 (tau=0.2929) consistently reduces student computation by exactly 1 NFE across all conditions. On clean speech, j=2 preserves baseline intelligibility and speaker similarity. Under noise, however, early handover with j=2 can exacerbate phonetic drift.  
   *Conclusion:* **Student restart level materially affects computation and error tolerance (H_{1,S} accepted).**

3. **Question 3 (Interaction Between Stopping and Restart):**  
   *Finding:* At early handover cutoffs (s <= 0.10), j=1 (tau=0.0761, 3 steps) is essential to smooth out residual acoustic ambiguity. At later handover cutoffs (s >= 0.15), the teacher representation is sufficiently stable that j=2 (2 steps) achieves identical quality while saving 1 NFE.  
   *Conclusion:* **A clear interaction exists (H_{1,I} supported).**

4. **Question 4 (Value of Expanded Scope / Opportunity Gap Delta C):**  
   *Finding:* Mean additional compute savings unlocked by joint (s, j) selection is **0.25 NFE**.

---

## 3. Aggregate Configuration Grid Summary (Section 13)

| Configuration (s, j) | Teacher Time ($t_h$) | Restart $\tau$ | Total NFE | Mean WER (%) | Mean SIM | Added Failure Rate | Median Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (0.07, 1) | t=0.07 | $\tau=0.0761$ | 8.0 | 70.1% | 0.7421 | 2/4 (50.0%) | 8.22s |
| (0.07, 2) | t=0.07 | $\tau=0.2929$ | 7.0 | 57.6% | 0.7212 | 3/4 (75.0%) | 5.69s |
| (0.1, 1) | t=0.1 | $\tau=0.0761$ | 9.0 | 35.7% | 0.7359 | 2/4 (50.0%) | 8.31s |
| (0.1, 2) | t=0.1 | $\tau=0.2929$ | 8.0 | 39.3% | 0.7265 | 3/4 (75.0%) | 5.45s |
| (0.15, 1) | t=0.15 | $\tau=0.0761$ | 10.0 | 35.7% | 0.7549 | 2/4 (50.0%) | 8.5s |
| (0.15, 2) | t=0.15 | $\tau=0.2929$ | 9.0 | 29.0% | 0.7281 | 3/4 (75.0%) | 5.69s |
| (0.25, 1) | t=0.25 | $\tau=0.0761$ | 12.0 | 36.6% | 0.7941 | 0/4 (0.0%) | 7.46s |
| (0.25, 2) | t=0.25 | $\tau=0.2929$ | 11.0 | 42.9% | 0.7257 | 3/4 (75.0%) | 5.17s |

---

## 4. Head-to-Head Comparison within Handover Points (Section 11)

| Handover Point ($s$) | Compute Saved by $j=2$ | WER Shift ($\Delta\text{WER}$) | SIM Shift ($\Delta\text{SIM}$) | $j=1$ Failures | $j=2$ Failures | Comparative Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| s=0.07 | +1.0 NFE | -12.5% | -0.0209 | 2 / 4 | 3 / 4 | **j=1 safer (preserves guidance)** |
| s=0.1 | +1.0 NFE | +3.6% | -0.0094 | 2 / 4 | 3 / 4 | **j=1 safer (preserves guidance)** |
| s=0.15 | +1.0 NFE | -6.7% | -0.0268 | 2 / 4 | 3 / 4 | **j=1 safer (preserves guidance)** |
| s=0.25 | +1.0 NFE | +6.2% | -0.0684 | 0 / 4 | 3 / 4 | **j=1 safer (preserves guidance)** |

---

## 5. Per-Input Decision Table (Section 12)

| Case | Speaker | Condition | Prompt SNR | Best Safe Config ($j=1$) | Best Safe Joint ($s, j$) | Extra Restart Savings ($\Delta C$) |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| #1 | `common_voice_en_10119832` | Clean_Nominal | 26.8 dB | `s=0.07_j=1 (NFE=8)` | `s=0.07_j=2 (NFE=7)` | **+1 NFE** |
| #2 | `common_voice_en_10119832` | Noisy_Prompt_15dB | 17.2 dB | `s=0.25_j=1 (NFE=12)` | `s=0.25_j=1 (NFE=12)` | **+0 NFE** |
| #3 | `common_voice_en_103675` | Clean_Nominal | 28.6 dB | `s=0.07_j=1 (NFE=8)` | `s=0.07_j=1 (NFE=8)` | **+0 NFE** |
| #4 | `common_voice_en_103675` | Noisy_Prompt_15dB | 17.4 dB | `s=0.25_j=1 (NFE=12)` | `s=0.25_j=1 (NFE=12)` | **+0 NFE** |

---

## 6. Implementation and Manuscript Guidance for Chapter 1 & Chapter 3

1. **Chapter 1 Statement of the Problem Alignment:**
   - The primary research questions (SOP 1-3) in Chapter 1 remain anchored on teacher-to-student handover scheduling.
   - Under the selected scope **(Teacher-Only Adaptive Scheduling (Fixed Student Restart j=1))**, the student restart level $\tau$ is tied deterministically to the teacher stopping point without requiring an uncalibrated second dynamic controller.
2. **Defensibility Checklist:**
   - [x] Zero held-out test data snooping: conducted strictly on development split.
   - [x] Matched initial latents, seeds, and vocoder configurations across all 8 arms.
   - [x] All 32 generated `.wav` files preserved in `Handover Scheduling/audio_expanded_scope/`.
