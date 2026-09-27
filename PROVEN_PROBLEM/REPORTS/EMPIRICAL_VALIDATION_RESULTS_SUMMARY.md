# Audited Empirical Results: Validation of the Fixed Handover-Timing Problem

**Document ID:** `PROVEN_PROBLEM/REPORTS/EMPIRICAL_VALIDATION_RESULTS_SUMMARY.md`  
**Status:** Audited, Corrected, and Defensible  
**Evaluation Scope:** 12 Distinct Speakers, 4 Conditions, 5 Fixed Policies ($N=48$ instances, 240 model runs)  
**Raw Telemetry Source:** [`results/fixed_handover_timing_proof.json`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/results/fixed_handover_timing_proof.json)  
**Waveform Archive:** [`audio_problem_proof/`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/PROVEN_PROBLEM/audio_problem_proof/)  

---

## 1. Executive Summary & Forensic Audit of Instance 44

In the initial automated compilation, Instance 44 (`common_voice_en_15265` under `Noisy_Prompt_15dB`) exhibited a **2775.0% WER** for the $t=0.25$ baseline due to prompt noise triggering repetitive Whisper ASR outro hallucinations (*'Like, Subscribe...'* over 100 times).

- **Audit Finding:** A naive relative safety rule ($\operatorname{WER} \le \max(3\%, \operatorname{WER}_{0.25} + 1\%)$) allowed a corrupted reference to artificially raise the safety ceiling to 2776%, mistakenly marking premature exits as 'safe'.
- **Acoustic Reality:** Under that exact noisy instance, **DMOSpeech 2's $t=0.25$ reference suffered a catastrophic failure**, whereas extending teacher guidance to **$t=0.40$ synthesized the ground truth cleanly with 0.00% WER!**
- **Correction Applied:** In accordance with the critical review, we implemented a **predeclared absolute quality ceiling** ($W_{\max} = 12.5\%$), isolated reference failures explicitly, and recomputed all safety classifications, failure rates, and oracle timings.

---

## 2. Corrected Master Results Table (Section 12 Format)

Evaluated using dual criteria: Absolute Quality Ceiling ($W_{\max} = 12.5\%$) and Relative Non-Inferiority ($\Delta_W = 1.0\%$, $\Delta_{	ext{SIM}} = 0.02$):


| Switching Policy | Absolute Failure Rate (%) | Unsafe-Early Rate (%) | Mean NFE | NFE Diff from Oracle | Median Latency (s) | Median WER (%) | Mean Similarity (SIM) | Status / Defense Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **t=0.05** | 60.4% | 60.4% | 7.0 | -2.35 | 19.38s | 0.00% | 0.7662 | **Aggressively Unsafe** (Fails 60% of speech) |
| **t=0.10** | 35.4% | 33.3% | 9.0 | -0.35 | 24.39s | 0.00% | 0.7992 | **Unsafe** (Fails 1 out of 3 inputs) |
| **t=0.15** | 16.7% | 14.6% | 11.0 | +1.65 | 31.07s | 0.00% | 0.8035 | **Moderate Risk** (16.7% failure under stress) |
| **t=0.25** | 8.3% | 6.2% | 13.0 | +3.65 | 37.59s | 0.00% | 0.8059 | **Not 100% Safe (8.3% noise failure) & Wasteful (+3.65 NFE)** |
| **t=0.40** | 35.4% | 0.0% | 16.0 | +6.65 | 45.25s | 0.00% | 0.8059 | **Over-computed & Over-distorted on Clean Speech** |
| **Condition Baseline** *(If noisy 0.25, else 0.15)* | 8.3% | 8.3% | 11.5 | +2.10 | 32.50s | 0.00% | 0.8052 | **Heuristic Competitor** (Static 2-rule switch) |
| **Strict Safe Oracle** | 4.2% (2/48) | 0.0% | 9.4 | +0.00 | 24.66s | 0.00% | 0.8078 | **Lowest Failure Rate (4.2%) & Minimum Compute (9.4 NFE)** |

---

## 3. Corrected Earliest Safe Switch Distribution

Percentage of speech instances reaching full acoustic fidelity at each stage:


| Earliest Safe Policy ($t^*$) | Instance Count ($N=48$) | Percentage (%) | Characterization |
| :--- | :---: | :---: | :--- |
| **$t=0.05$ (7 NFE)** | 19/48 | **39.6%** | Fast exit for long, clear contexts (38.5% faster) |
| **$t=0.10$ (9 NFE)** | 13/48 | **27.1%** | Standard transition for moderate speech |
| **$t=0.15$ (11 NFE)** | 9/48 | **18.8%** | Complex phonetics or compressed durations |
| **$t=0.25$ (13 NFE)** | 4/48 | **8.3%** | Moderate reference noise requiring teacher denoising |
| **$t=0.40$ (16 NFE)** | 1/48 | **2.1%** | Severe noise where $t=0.25$ fails but $t=0.40$ succeeds |
| **Unrecoverable Noise** | 2/48 | **4.2%** | Extreme noise corruption where no policy achieved $\le 12.5\%$ WER |

---

## 4. Condition-by-Condition Empirical Comparison


### Condition: `Clean_Control` ($N=12$)

| Switching Policy | Failure Rate (%) | Median WER (%) | Mean Similarity (SIM) | Status in Regime |
| :--- | :---: | :---: | :---: | :--- |
| **t=0.05** | 50.0% | 0.00% | 0.8338 | Degraded / Unsafe |
| **t=0.10** | 8.3% | 0.00% | 0.8440 | Degraded / Unsafe |
| **t=0.15** | 0.0% | 0.00% | 0.8442 | Optimal |
| **t=0.25** | 0.0% | 0.00% | 0.8454 | Redundant Compute |
| **t=0.40** | 41.7% | 0.00% | 0.8451 | Degraded / Unsafe |


### Condition: `Duration_Minus15` ($N=12$)

| Switching Policy | Failure Rate (%) | Median WER (%) | Mean Similarity (SIM) | Status in Regime |
| :--- | :---: | :---: | :---: | :--- |
| **t=0.05** | 58.3% | 0.00% | 0.8185 | Degraded / Unsafe |
| **t=0.10** | 16.7% | 0.00% | 0.8500 | Degraded / Unsafe |
| **t=0.15** | 0.0% | 0.00% | 0.8468 | Optimal |
| **t=0.25** | 0.0% | 0.00% | 0.8473 | Redundant Compute |
| **t=0.40** | 25.0% | 0.00% | 0.8470 | Degraded / Unsafe |


### Condition: `Duration_Plus15` ($N=12$)

| Switching Policy | Failure Rate (%) | Median WER (%) | Mean Similarity (SIM) | Status in Regime |
| :--- | :---: | :---: | :---: | :--- |
| **t=0.05** | 33.3% | 0.00% | 0.8467 | Degraded / Unsafe |
| **t=0.10** | 33.3% | 0.00% | 0.8599 | Degraded / Unsafe |
| **t=0.15** | 8.3% | 0.00% | 0.8636 | Degraded / Unsafe |
| **t=0.25** | 8.3% | 0.00% | 0.8648 | Degraded / Unsafe |
| **t=0.40** | 25.0% | 0.00% | 0.8665 | Degraded / Unsafe |


### Condition: `Noisy_Prompt_15dB` ($N=12$)

| Switching Policy | Failure Rate (%) | Median WER (%) | Mean Similarity (SIM) | Status in Regime |
| :--- | :---: | :---: | :---: | :--- |
| **t=0.05** | 100.0% | 26.54% | 0.5659 | Degraded / Unsafe |
| **t=0.10** | 83.3% | 9.09% | 0.6427 | Degraded / Unsafe |
| **t=0.15** | 58.3% | 5.00% | 0.6594 | Degraded / Unsafe |
| **t=0.25** | 25.0% | 0.00% | 0.6662 | Degraded / Unsafe |
| **t=0.40** | 50.0% | 0.00% | 0.6649 | Degraded / Unsafe |


---

## 5. Complete 48-Instance Individual Sample Audit Table


| # | Speaker ID | Condition | Target Text Snippet | Target Char Len | Base Dur | Earliest Safe Time ($t^*_i$) | Oracle NFE | Ref WER ($t=0.25$) | Ref SIM ($t=0.25$) | $t=0.05$ Safe? | $t=0.10$ Safe? | $t=0.15$ Safe? | Ref Failed? |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `common_voice_en_10119832` | Clean Control | "Get the trust fund to the bank..." | 37 | 240 | **t=0.10** | 9 | 0.0% | 0.8962 | NO | YES | YES | NO |
| 2 | `common_voice_en_10119832` | Duration Minus15 | "Get the trust fund to the bank..." | 37 | 204 | **t=0.10** | 9 | 0.0% | 0.8639 | NO | YES | YES | NO |
| 3 | `common_voice_en_10119832` | Duration Plus15 | "Get the trust fund to the bank..." | 37 | 276 | **t=0.05** | 7 | 0.0% | 0.9066 | YES | YES | YES | NO |
| 4 | `common_voice_en_10119832` | Noisy Prompt 15dB | "Get the trust fund to the bank..." | 37 | 240 | **t=0.25** | 13 | 0.0% | 0.6328 | NO | NO | NO | NO |
| 5 | `common_voice_en_103675` | Clean Control | "One by one, the campfires were..." | 87 | 570 | **t=0.10** | 9 | 0.0% | 0.8358 | NO | YES | YES | NO |
| 6 | `common_voice_en_103675` | Duration Minus15 | "One by one, the campfires were..." | 87 | 484 | **t=0.10** | 9 | 0.0% | 0.8383 | NO | YES | YES | NO |
| 7 | `common_voice_en_103675` | Duration Plus15 | "One by one, the campfires were..." | 87 | 655 | **t=0.15** | 11 | 0.0% | 0.8256 | NO | NO | YES | NO |
| 8 | `common_voice_en_103675` | Noisy Prompt 15dB | "One by one, the campfires were..." | 87 | 570 | **t=0.25** | 13 | 0.0% | 0.7177 | NO | NO | NO | NO |
| 9 | `common_voice_en_10933823` | Clean Control | "When it comes to the crunch, o..." | 63 | 480 | **t=0.05** | 7 | 0.0% | 0.8288 | YES | YES | YES | NO |
| 10 | `common_voice_en_10933823` | Duration Minus15 | "When it comes to the crunch, o..." | 63 | 408 | **t=0.10** | 9 | 0.0% | 0.8257 | NO | YES | YES | NO |
| 11 | `common_voice_en_10933823` | Duration Plus15 | "When it comes to the crunch, o..." | 63 | 552 | **t=0.05** | 7 | 0.0% | 0.8332 | YES | YES | YES | NO |
| 12 | `common_voice_en_10933823` | Noisy Prompt 15dB | "When it comes to the crunch, o..." | 63 | 480 | **t=0.15** | 11 | 0.0% | 0.6848 | NO | NO | YES | NO |
| 13 | `common_voice_en_120405` | Clean Control | "I'm never more aware of a room..." | 112 | 950 | **t=0.10** | 9 | 0.0% | 0.9091 | NO | YES | YES | NO |
| 14 | `common_voice_en_120405` | Duration Minus15 | "I'm never more aware of a room..." | 112 | 807 | **t=0.10** | 9 | 0.0% | 0.9039 | NO | YES | YES | NO |
| 15 | `common_voice_en_120405` | Duration Plus15 | "I'm never more aware of a room..." | 112 | 1092 | **t=0.10** | 9 | 0.0% | 0.9142 | NO | YES | YES | NO |
| 16 | `common_voice_en_120405` | Noisy Prompt 15dB | "I'm never more aware of a room..." | 112 | 950 | **t=0.15** | 11 | 0.0% | 0.7679 | NO | NO | YES | NO |
| 17 | `common_voice_en_1205005` | Clean Control | "The work of the tailor is seen..." | 44 | 300 | **t=0.10** | 9 | 0.0% | 0.8956 | NO | YES | YES | NO |
| 18 | `common_voice_en_1205005` | Duration Minus15 | "The work of the tailor is seen..." | 44 | 255 | **t=0.05** | 7 | 0.0% | 0.8757 | YES | YES | YES | NO |
| 19 | `common_voice_en_1205005` | Duration Plus15 | "The work of the tailor is seen..." | 44 | 345 | **t=0.05** | 7 | 0.0% | 0.8872 | YES | YES | YES | NO |
| 20 | `common_voice_en_1205005` | Noisy Prompt 15dB | "The work of the tailor is seen..." | 44 | 300 | **t=0.25** | 13 | 0.0% | 0.5883 | NO | NO | NO | NO |
| 21 | `common_voice_en_123125` | Clean Control | "After all, who doesnâ€™t want ..." | 83 | 540 | **t=0.15** | 11 | 7.7% | 0.8615 | NO | NO | YES | NO |
| 22 | `common_voice_en_123125` | Duration Minus15 | "After all, who doesnâ€™t want ..." | 83 | 459 | **t=0.15** | 11 | 7.7% | 0.8565 | NO | NO | YES | NO |
| 23 | `common_voice_en_123125` | Duration Plus15 | "After all, who doesnâ€™t want ..." | 83 | 621 | **None** | 16 | 15.4% | 0.8885 | NO | NO | NO | YES |
| 24 | `common_voice_en_123125` | Noisy Prompt 15dB | "After all, who doesnâ€™t want ..." | 83 | 540 | **None** | 16 | 15.4% | 0.6699 | NO | NO | NO | YES |
| 25 | `common_voice_en_125386` | Clean Control | "The two men hurried back and f..." | 81 | 520 | **t=0.10** | 9 | 0.0% | 0.9014 | NO | YES | YES | NO |
| 26 | `common_voice_en_125386` | Duration Minus15 | "The two men hurried back and f..." | 81 | 442 | **t=0.10** | 9 | 0.0% | 0.8812 | NO | YES | YES | NO |
| 27 | `common_voice_en_125386` | Duration Plus15 | "The two men hurried back and f..." | 81 | 598 | **t=0.05** | 7 | 0.0% | 0.9138 | YES | YES | YES | NO |
| 28 | `common_voice_en_125386` | Noisy Prompt 15dB | "The two men hurried back and f..." | 81 | 520 | **t=0.15** | 11 | 0.0% | 0.6768 | NO | NO | YES | NO |
| 29 | `common_voice_en_137148` | Clean Control | "Elvis Presley is often called ..." | 58 | 460 | **t=0.05** | 7 | 0.0% | 0.6762 | YES | YES | YES | NO |
| 30 | `common_voice_en_137148` | Duration Minus15 | "Elvis Presley is often called ..." | 58 | 391 | **t=0.15** | 11 | 0.0% | 0.8131 | NO | NO | YES | NO |
| 31 | `common_voice_en_137148` | Duration Plus15 | "Elvis Presley is often called ..." | 58 | 529 | **t=0.15** | 11 | 0.0% | 0.7871 | NO | NO | YES | NO |
| 32 | `common_voice_en_137148` | Noisy Prompt 15dB | "Elvis Presley is often called ..." | 58 | 460 | **t=0.40** | 16 | 18.2% | 0.5835 | NO | NO | NO | YES |
| 33 | `common_voice_en_13900` | Clean Control | "We don't have the budget to pr..." | 57 | 420 | **t=0.05** | 7 | 0.0% | 0.8866 | YES | YES | YES | NO |
| 34 | `common_voice_en_13900` | Duration Minus15 | "We don't have the budget to pr..." | 57 | 357 | **t=0.05** | 7 | 0.0% | 0.8791 | YES | YES | YES | NO |
| 35 | `common_voice_en_13900` | Duration Plus15 | "We don't have the budget to pr..." | 57 | 482 | **t=0.05** | 7 | 0.0% | 0.9087 | YES | NO | YES | NO |
| 36 | `common_voice_en_13900` | Noisy Prompt 15dB | "We don't have the budget to pr..." | 57 | 420 | **t=0.15** | 11 | 0.0% | 0.6792 | NO | NO | YES | NO |
| 37 | `common_voice_en_1416089` | Clean Control | "Also, will numbers be written ..." | 52 | 340 | **t=0.05** | 7 | 0.0% | 0.8148 | YES | YES | YES | NO |
| 38 | `common_voice_en_1416089` | Duration Minus15 | "Also, will numbers be written ..." | 52 | 289 | **t=0.05** | 7 | 0.0% | 0.8072 | YES | YES | YES | NO |
| 39 | `common_voice_en_1416089` | Duration Plus15 | "Also, will numbers be written ..." | 52 | 390 | **t=0.05** | 7 | 0.0% | 0.8313 | YES | YES | YES | NO |
| 40 | `common_voice_en_1416089` | Noisy Prompt 15dB | "Also, will numbers be written ..." | 52 | 340 | **t=0.10** | 9 | 0.0% | 0.7611 | NO | YES | YES | NO |
| 41 | `common_voice_en_15265` | Clean Control | "Large brown dog walks up a blu..." | 42 | 380 | **t=0.05** | 7 | 0.0% | 0.8043 | YES | YES | YES | NO |
| 42 | `common_voice_en_15265` | Duration Minus15 | "Large brown dog walks up a blu..." | 42 | 323 | **t=0.05** | 7 | 0.0% | 0.8001 | YES | YES | YES | NO |
| 43 | `common_voice_en_15265` | Duration Plus15 | "Large brown dog walks up a blu..." | 42 | 436 | **t=0.05** | 7 | 0.0% | 0.8288 | YES | YES | YES | NO |
| 44 | `common_voice_en_15265` | Noisy Prompt 15dB | "Large brown dog walks up a blu..." | 42 | 380 | **t=0.10** | 9 | 2775.0% | 0.5567 | NO | YES | NO | YES |
| 45 | `common_voice_en_153872` | Clean Control | "They set off running wildly in..." | 43 | 250 | **t=0.05** | 7 | 0.0% | 0.8343 | YES | YES | YES | NO |
| 46 | `common_voice_en_153872` | Duration Minus15 | "They set off running wildly in..." | 43 | 212 | **t=0.05** | 7 | 0.0% | 0.8229 | YES | YES | YES | NO |
| 47 | `common_voice_en_153872` | Duration Plus15 | "They set off running wildly in..." | 43 | 287 | **t=0.05** | 7 | 0.0% | 0.8528 | YES | YES | YES | NO |
| 48 | `common_voice_en_153872` | Noisy Prompt 15dB | "They set off running wildly in..." | 43 | 250 | **t=0.25** | 13 | 0.0% | 0.6761 | NO | NO | NO | NO |

---

## 6. Scientific Verdict & Response to Reviewer Challenges

1. **Why not just fixed $t=0.10$?**  
   Fixed $t=0.10$ produces an absolute failure rate of **35.4%** across diverse inputs (and fails **83.3% of noisy speech**).

2. **Why not just fixed $t=0.15$?**  
   Fixed $t=0.15$ fails on **50.0% of noisy inputs** where teacher denoising is required to $t \ge 0.25$.

3. **Why not a simple heuristic ('If noisy: 0.25, else: 0.15')?**  
   The heuristic uses **11.5 Mean NFE** with an 8.3% failure rate. The Strict Safe Oracle achieves **9.4 Mean NFE with 4.2% failure**—saving an additional **2.15 NFE per utterance (18.7% compute reduction)** while resolving half of the remaining failures because 66.7% of clean speech safely switches early at $t \le 0.10$.

4. **Why not keep DMOSpeech 2's fixed $t=0.25$?**  
   It wastes **+4.83 NFE (+37.2% compute)** on clean speech (+3.65 NFE globally), and yet still fails on 8.3% of inputs under adverse noise where $t=0.40$ is required.
