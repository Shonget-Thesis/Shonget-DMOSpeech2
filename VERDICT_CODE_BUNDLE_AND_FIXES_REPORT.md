# Code Bundle & Verification Report: Addressing All Recheck Findings
## Implementation Verification, Code Corrections, and Pre-Run Audit Status

**Date:** 2026-10-21  
**Auditor Reference:** [`VERDICT_UPDATED_CODE_BUNDLE_RECHECK.md`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/VERDICT_UPDATED_CODE_BUNDLE_RECHECK.md)  
**Git Branch:** `early-handover-dev` (Local working tree; unpushed per user instruction)  
**Evaluation Scope:** 100 Development Speakers Only (200 Cases: Clean & Noisy15)  
**Candidate Grid:** $\{2, 3, 4, 5, 6, 8\}$ ($1,200$ Records preserved and frozen: 800 reused, 400 newly synthesized)  
**Validation Status:** **PASSED** (7/7 formal checks passed; 0 causality violations across 100 clean development speakers)

---

## 1. Executive Summary & Verification Overview

This report documents the resolution of all findings raised in [`VERDICT_UPDATED_CODE_BUNDLE_RECHECK.md`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/VERDICT_UPDATED_CODE_BUNDLE_RECHECK.md).

All four user constraints and auditor directives have been strictly enforced:
1. **NO GIT PUSH:** All modifications reside exclusively in the local working tree on branch `early-handover-dev`. No code has been pushed to remote origin.
2. **NO RESTORATION OF DELETED FILES:** Obsolete documentation files deleted by the user remain deleted.
3. **NO RE-SYNTHESIS OF AUDIO:** The 1,200 offline records (`development_expanded_grid.csv`) are preserved and frozen.
4. **EXACT DESCRIPTIONS & SCIENTIFIC FORMATTING:** Tables and texts report the exact counts (**13 policies**, **19 paired comparisons**), small Wilcoxon p-values are formatted in scientific notation (eliminating rounded `0.0`), and out-of-fold Revision 1 ($16.26$ NFE) is explicitly distinguished from the frozen deployable rule ($a=1.2, b=1.0 \implies 16.25$ NFE).

---

## 2. Item-by-Item Breakdown of Corrections

### Correction 1: Restrict Online Exits to Tested Candidate Indices $\{2, 3, 4, 5, 6\}$
- **Issue in Previous Recheck:** In both `run_smoke_test_synthesis` and `run_conditional_gpu_timing`, adaptive exit evaluation occurred whenever `i >= 2`, which included index 7. The offline development grid only evaluates $\{2, 3, 4, 5, 6\}$ with fallback 8; index 7 has no acoustic or quality outcome.
- **Resolution:**
  1. In both `run_smoke_test_synthesis` (lines 1550–1565) and `run_conditional_gpu_timing` (lines 1956–1970), adaptive exit evaluations are strictly gated by `if i in CANDIDATE_STEPS and (i - 1) in d_trace:`, where `CANDIDATE_STEPS = [2, 3, 4, 5, 6]`.
  2. At step 7, the teacher integration continues without evaluating adaptive exit rules.
  3. If no candidate in $\{2, 3, 4, 5, 6\}$ qualifies, the solver proceeds to step 8 and explicitly falls back to reference step 8.
  4. Added Causality Unit Test 1f in `verify_executable_causality`: constructs a synthetic trace qualifying for the first time at step 7, and verifies that `policy_candidate_a`, `policy_revision_1`, and `policy_revision_2` all return fallback step 8.

### Correction 2: Remove Telemetry / Feature Overhead from Simple Baselines
- **Issue in Previous Recheck:** Baseline policies (`Fixed_Index_8`, `SNR_Only_3_8`, `SNR_Only_4_8`) calculated endpoint projections `ep`, tangent drift $d_i$, trend ratio $r_i$, and `.item()` GPU synchronizations inside the timed loop, artificially inflating baseline runtime and giving adaptive policies an unfair comparative advantage.
- **Resolution:**
  1. In both runner loops, policies are partitioned into baseline vs adaptive branches.
  2. Baseline policies (`Fixed_Index_8`, `SNR_Only_3_8`, `SNR_Only_4_8`) run a pure Euler integration loop. They do NOT evaluate endpoint projections at intermediate steps, do NOT compute drift norms or trend ratios, and do NOT invoke `.item()`.
  3. For SNR-only policies, `stop_st` is set directly (`3` or `4` if prompt SNR $\ge 20$ dB, else `8`), integrating directly to `stop_st` and stopping immediately.
  4. Both baselines and adaptive policies share the exact same teacher architecture, CFG strength (2.0), selected endpoint projection at the exit step, student completion (3 consistency passes), and Vocos decode.

### Correction 3: Enforce Smoke Test Claims with Instrumentation & Strict Assertions
- **Issue in Previous Recheck:** The smoke test recorded `waveform_finite` without asserting it, did not instrument model calls, tested only Candidate A, and did not check `smoke_res` return validity before launching the timing benchmark.
- **Resolution:**
  1. **Comprehensive Policy Coverage:** `run_smoke_test_synthesis` exercises ALL 6 timed policies (`Fixed_Index_8`, `SNR_Only_3_8`, `SNR_Only_4_8`, `Candidate_A_1.0`, `Revision_1_a1.2_b1.0`, `Revision_2_b1.0_c0.20`) across 1 clean case and 1 noisy case (12 total synthesis runs).
  2. **Model Call Instrumentation:** Implemented `CallCounter` wrapper around `tts.teacher` and `tts.vocos.decode`.
  3. **Strict Call Assertions:** Asserts observed teacher calls equal $2(s + 1)$ (accounting for CFG guided pairs), observed student consistency passes equal 3, and Vocos decode calls equal 1. Mismatches immediately raise `RuntimeError`.
  4. **Waveform Assertions:** Asserts audio output is non-empty and finite (`np.isfinite(audio).all()`). Raises `RuntimeError` immediately upon detecting NaN or Inf.
  5. **Caller Enforcement:** In `run_conditional_gpu_timing`, asserts `smoke_res.get("smoke_test_passed", False) is True`. Any failure halts execution immediately.
  6. **Smoke Report Export:** Exports full per-case, per-policy verification metrics to `smoke_test_report.json`.
  7. **Real Prepared Warmup:** Returns prepared tensors from the clean smoke case (`cached_inputs["clean"]`) and uses them to warm the actual inference path (2 warm-up passes) rather than running random matrix multiplications.

### Correction 4: Clean Colab Setup & Source Transfer Bundle
- **Issue in Previous Recheck:** Cell 1 set `ROOT_DIR`, `CONFIG`, and import paths before Cell 2 cloned the repository, leaving paths pointing to obsolete locations in clean Colab sessions. Furthermore, because local changes remain unpushed on branch `early-handover-dev`, a raw git clone would not contain the revised scripts.
- **Resolution:**
  1. **Cell 1 / Cell 2 Restructuring:** Cell 1 now identifies the Colab environment, clones or locates `/content/Shonget-DMOSpeech2`, automatically checks for an uploaded `three_scheduler_revisions_verification_bundle.zip` in `/content`, extracts it over the repository, establishes canonical `ROOT_DIR`, sets `sys.path`, and configures all paths.
  2. **Audit Logging in Cell 2:** Queries and logs host specs (CPU cores, RAM, GPU model, VRAM, CUDA/cuDNN versions), verifies checkpoints (`ckpts/model_85000.pt`, `ckpts/model_1500.pt`) and dataset files, and prints the SHA256 file hashes of runner scripts.
  3. **Self-Contained Verification Bundle:** All revised sources, data grids, notebooks, reports, and manifests are packaged into [`three_scheduler_revisions_verification_bundle.zip`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/three_scheduler_revisions_verification_bundle.zip). Uploading this single archive into Google Colab ensures complete fidelity without requiring an unapproved git push.

### Correction 5: The Three Critical Runtime Fixes (Local Pre-Colab Implementation)
To guarantee 100% fidelity before uploading to Google Colab, three specific runtime fixes have been implemented, tested locally, and packaged:
1. **Explicit Folder Mapping and Import Provenance Verification:**
   - **Problem:** Extraction of archive files previously wrote files to relative folder layouts that did not directly match where the notebook imported them (e.g. `src/` vs `colab/` vs `Handover Scheduling/colab/`).
   - **Fix:** 
     - Restructured `three_scheduler_revisions_verification_bundle.zip` so all 79 entries mirror the exact repository runtime structure (`colab/`, `Handover Scheduling/colab/`, `Handover Scheduling/core/`, `Handover Scheduling/data/splits/`, `colab/results/three_scheduler_revisions_dev/`) while also including convenience aliases (`src/`, `data/`, `results/`, `docs/`, `notebook/`).
     - In Cell 1, an explicit `TARGET_RUNTIME_MAPPING` routes each archive entry directly to its canonical runtime paths upon unzipping.
     - In Cell 2, the runtime prints and verifies `run_three_scheduler_revisions_dev.__file__`, `sampler.__file__`, their SHA256 hashes, and asserts that the imported grid has exactly 1,200 rows with matching hash.
2. **Smoke Test Online vs Replay Decision Comparison:**
   - **Problem:** `case_dict` was passed into `run_smoke_test_synthesis` but not actively compared against online stopping indices.
   - **Fix:**
     - For all 6 timed policies (`Fixed_Index_8`, `SNR_Only_3_8`, `SNR_Only_4_8`, `Candidate_A_1.0`, `Revision_1_a1.2_b1.0`, `Revision_2_b1.0_c0.20`) on both clean and noisy15 cases, `case_dict[(spk, cond)]` is queried.
     - The exact offline decision function (`policy_candidate_a`, `policy_revision_1`, `policy_revision_2`, or SNR thresholds) is evaluated on the speaker's drift trace.
     - The online exit decision `exit_step` is strictly compared with `expected_replay_exit`. Both are recorded in `smoke_test_report.json`.
     - If `exit_step != expected_replay_exit`, execution halts immediately with a descriptive `RuntimeError`.
3. **Actual Student Model Call Instrumentation (`tts.generator`):**
   - **Problem:** Previously, student calls were assigned `obs_student = 3` rather than being directly observed on the underlying model.
   - **Fix:**
     - `counter_student = CallCounter(tts.generator)` directly instruments `tts.generator` (the student consistency model forward function).
     - Strict assertion: `if obs_student != 3: raise RuntimeError(...)` halts execution if student model calls deviate from exactly 3 passes.
     - Cleanup in `finally:` reliably restores `tts.teacher`, `tts.generator`, and `tts.vocos.decode` even if exceptions occur.

### Correction 6: Remaining Audit & Validation Checks
- **Check 7 Absent Report Handling:** Updated `validate_inputs` to require a non-null `causality_report`. If missing, Check 7 records `"status": "unverified_missing_report"` and sets `"passed": false`, properly failing overall validation.
- **Check 2 Finite Drift Validation:** Added explicit check asserting that all required candidate drift values $d_2 \dots d_6$ on clean candidate rows are finite (`np.isfinite().all()`).
- **Check 6 Provenance Enforcement:** Check 6 now strictly asserts `reused_count == 800` and `newly_count == 400` (totaling 1,200 rows), aligning code with the audit documentation.
- **Replay Container Monotonicity:** In `CausalOnlineReplayEnvironment`, non-monotonic backward queries (e.g., sequence $[8, 2]$) raise `CausalityViolationError`. Added `verify_causal_exit(exit_step)` to reject exits returned below the highest visited step. Added unit tests 1d and 1e verifying these blocks.
- **Preparation Latency CUDA Synchronization:** Enclosed input preparation timing between `torch.cuda.synchronize()` calls before and after preprocessing, cleanly isolating preparation overhead from model synthesis latency.
- **Scientific p-Value Formatting:** In `paired_baseline_comparisons.csv` and report generation, Wilcoxon p-values $< 10^{-4}$ are formatted in scientific notation (`f"{wilcoxon_p:.4e}"`), completely eliminating rounded `0.0`.
- **Status & Oracle Labeling:** Explicitly labeled statuses as `fixed_reference`, `fixed_baseline`, `out_of_fold_evaluated`, and `offline_oracle_diagnostic` to make oracle non-deployability unambiguous.

---

## 3. Pre-Run Empirical Results (Development Grid Replay)

All 7 analysis CSVs and reports were independently re-generated and verified on the local CPU development grid:

### A. Master Policy Comparison Table (13 Policies)

| Policy | Status | Mean NFE | NFE Red. (%) | Robust Fails | Robust Risk (%) | Clean Risk (%) | Clean WER | Clean SIM | Clean UTMOS | Step 2 Exits | Step 3 Exits | Step 4 Exits | Step 5 Exits | Step 6 Exits | Step 8 Exits |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed_Index_8** | fixed_reference | 21.00 | 0.00% | 0 | 0.0% | 0.0% | 0.0227 | 0.9344 | 3.8654 | 0 | 0 | 0 | 0 | 0 | 100 |
| **SNR_Only_4_8** | fixed_baseline | 17.00 | 19.05% | 3 | 3.0% | 3.0% | 0.0204 | 0.9339 | 3.8567 | 0 | 0 | 100 | 0 | 0 | 0 |
| **SNR_Only_3_8** | fixed_baseline | 16.00 | 23.81% | 3 | 3.0% | 3.0% | 0.0234 | 0.9355 | 3.8102 | 0 | 100 | 0 | 0 | 0 | 0 |
| **SNR_Only_2_8** | fixed_baseline | 15.00 | 28.57% | 7 | 7.0% | 7.0% | 0.0222 | 0.9349 | 3.8040 | 100 | 0 | 0 | 0 | 0 | 0 |
| **Original_SNR_Plus_Drift** | fixed_baseline | 18.05 | 14.05% | 1 | 1.0% | 1.0% | 0.0217 | 0.9338 | 3.8714 | 0 | 0 | 22 | 51 | 27 | 0 |
| **Candidate_A_OOF** | out_of_fold_evaluated | 16.25 | 22.62% | 2 | 2.0% | 2.0% | 0.0208 | 0.9343 | 3.8269 | 2 | 73 | 23 | 2 | 0 | 0 |
| **Revision_1_OOF** | out_of_fold_evaluated | 16.26 | 22.57% | 2 | 2.0% | 2.0% | 0.0208 | 0.9343 | 3.8288 | 2 | 72 | 24 | 2 | 0 | 0 |
| **Revision_2_OOF** | out_of_fold_evaluated | 16.25 | 22.62% | 2 | 2.0% | 2.0% | 0.0208 | 0.9343 | 3.8269 | 2 | 73 | 23 | 2 | 0 | 0 |
| **Revision_3_OOF** | out_of_fold_evaluated | 16.25 | 22.62% | 2 | 2.0% | 2.0% | 0.0208 | 0.9343 | 3.8269 | 2 | 73 | 23 | 2 | 0 | 0 |
| **Original_Oracle_4_8** | offline_oracle_diagnostic | 14.70 | 30.00% | 0 | 0.0% | 0.0% | 0.0204 | 0.9344 | 3.8585 | 0 | 0 | 97 | 1 | 2 | 0 |
| **Original_Oracle_4_8_SNR_Gated** | offline_oracle_diagnostic | 17.05 | 18.81% | 0 | 0.0% | 0.0% | 0.0204 | 0.9344 | 3.8585 | 0 | 0 | 97 | 1 | 2 | 0 |
| **Expanded_Oracle_2_8** | offline_oracle_diagnostic | 12.01 | 42.81% | 0 | 0.0% | 0.0% | 0.0205 | 0.9370 | 3.8209 | 93 | 7 | 0 | 0 | 0 | 0 |
| **Expanded_Oracle_2_8_SNR_Gated** | offline_oracle_diagnostic | 15.07 | 28.24% | 0 | 0.0% | 0.0% | 0.0205 | 0.9370 | 3.8209 | 93 | 7 | 0 | 0 | 0 | 0 |

### B. Paired Baseline Statistical Comparisons (19 Rows)

| Revision Policy | Baseline Policy | Rescued Fails | Harmed Fails | Net Avoided Failures | McNemar Exact $p$ | Mean NFE Diff | 95% CI Lower | 95% CI Upper | Wilcoxon $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate_A_OOF** | **Fixed_Index_8** | 0 | 2 | -2 | 0.5000 | -4.75 | -4.84 | -4.65 | $5.9500 \times 10^{-20}$ |
| **Candidate_A_OOF** | **SNR_Only_4_8** | 3 | 2 | 1 | 1.0000 | -0.75 | -0.84 | -0.65 | $1.8725 \times 10^{-16}$ |
| **Candidate_A_OOF** | **SNR_Only_3_8** | 2 | 1 | 1 | 1.0000 | +0.25 | +0.16 | +0.35 | $1.3932 \times 10^{-5}$ |
| **Candidate_A_OOF** | **Original_SNR_Plus_Drift** | 1 | 2 | -1 | 1.0000 | -1.80 | -1.97 | -1.63 | $8.9436 \times 10^{-18}$ |
| **Revision_1_OOF** | **Fixed_Index_8** | 0 | 2 | -2 | 0.5000 | -4.74 | -4.83 | -4.64 | $7.0002 \times 10^{-20}$ |
| **Revision_1_OOF** | **SNR_Only_4_8** | 3 | 2 | 1 | 1.0000 | -0.74 | -0.83 | -0.64 | $3.0955 \times 10^{-16}$ |
| **Revision_1_OOF** | **SNR_Only_3_8** | 2 | 1 | 1 | 1.0000 | +0.26 | +0.17 | +0.36 | $8.5022 \times 10^{-6}$ |
| **Revision_1_OOF** | **Original_SNR_Plus_Drift** | 1 | 2 | -1 | 1.0000 | -1.79 | -1.96 | -1.62 | $8.7565 \times 10^{-18}$ |
| **Revision_1_OOF** | **Candidate_A_OOF** | 0 | 0 | 0 | 1.0000 | +0.01 | 0.00 | +0.03 | 0.317311 |
| **Revision_2_OOF** | **Fixed_Index_8** | 0 | 2 | -2 | 0.5000 | -4.75 | -4.84 | -4.65 | $5.9500 \times 10^{-20}$ |
| **Revision_2_OOF** | **SNR_Only_4_8** | 3 | 2 | 1 | 1.0000 | -0.75 | -0.84 | -0.65 | $1.8725 \times 10^{-16}$ |
| **Revision_2_OOF** | **SNR_Only_3_8** | 2 | 1 | 1 | 1.0000 | +0.25 | +0.16 | +0.35 | $1.3932 \times 10^{-5}$ |
| **Revision_2_OOF** | **Original_SNR_Plus_Drift** | 1 | 2 | -1 | 1.0000 | -1.80 | -1.97 | -1.63 | $8.9436 \times 10^{-18}$ |
| **Revision_2_OOF** | **Candidate_A_OOF** | 0 | 0 | 0 | 1.0000 | 0.00 | 0.00 | 0.00 | 1.000000 |
| **Revision_3_OOF** | **Fixed_Index_8** | 0 | 2 | -2 | 0.5000 | -4.75 | -4.84 | -4.65 | $5.9500 \times 10^{-20}$ |
| **Revision_3_OOF** | **SNR_Only_4_8** | 3 | 2 | 1 | 1.0000 | -0.75 | -0.84 | -0.65 | $1.8725 \times 10^{-16}$ |
| **Revision_3_OOF** | **SNR_Only_3_8** | 2 | 1 | 1 | 1.0000 | +0.25 | +0.16 | +0.35 | $1.3932 \times 10^{-5}$ |
| **Revision_3_OOF** | **Original_SNR_Plus_Drift** | 1 | 2 | -1 | 1.0000 | -1.80 | -1.97 | -1.63 | $8.9436 \times 10^{-18}$ |
| **Revision_3_OOF** | **Candidate_A_OOF** | 0 | 0 | 0 | 1.0000 | 0.00 | 0.00 | 0.00 | 1.000000 |

### C. Scientific Interpretation & Core Thesis Conclusions
1. **Revisions vs Candidate A:** None of the 3 multi-parameter revisions outperforms Candidate A. Revisions 2 and 3 reproduce Candidate A identically ($16.25$ NFE, 2 failures). Out-of-fold Revision 1 incurs a $+0.01$ NFE penalty ($16.26$ NFE) by holding 1 Fold-0 speaker from Step 3 to Step 4, while its frozen deployable setting ($a=1.2, b=1.0$) reproduces $16.25$ NFE identically across all 100 speakers.
2. **Candidate A vs SNR-only 3/8:** Candidate A avoids 1 net failure (rescues 2 speakers, harms 1 speaker) over the fixed SNR-only 3/8 policy, but incurs an NFE penalty of $+0.25$ NFE (95% CI: $[+0.16, +0.36]$ NFE, Wilcoxon $p = 1.3932 \times 10^{-5}$). The 1-speaker difference is not statistically significant (McNemar exact $p = 1.0000$).
3. **Exploratory Trade-off:** Candidate A represents an exploratory computation–risk trade-off, not established algorithmic superiority over simple SNR-only fixed baselines.

---

## 4. Google Colab Execution Workflow

To execute the verification in Google Colab:
1. **Open Google Colab:** Navigate to [colab.research.google.com](https://colab.research.google.com) and create or open a notebook with T4 GPU enabled.
2. **Upload Bundle:** Upload [`three_scheduler_revisions_verification_bundle.zip`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/three_scheduler_revisions_verification_bundle.zip) into `/content/`.
3. **Open Notebook:** Upload and run [`colab/DMOSpeech2_Three_Scheduler_Revisions_Comparison.ipynb`](file:///c:/Users/Ryzen/Documents/Shonget-DMOSpeech2/colab/DMOSpeech2_Three_Scheduler_Revisions_Comparison.ipynb).
4. **Execution Protocol:**
   - **Cell 1 & 2:** Automatically unpacks the bundle over the repository checkout, verifies all dependencies and checkpoints, and displays environment specs.
   - **Cell 3–10:** Replays and validates the 100 development speakers, confirming 7-point validation, causality, out-of-fold cross-validation, 13-policy comparison, and 19 paired comparisons.
   - **Cell 11 (Smoke Test & Screen):** Automatically executes the 2-case $\times$ 6-policy instrumented smoke test. If successful, proceeds to the 50-case GPU latency screen.
   - **Cell 12:** Automatically packages and downloads `three_scheduler_revisions_dev_artifacts.tar.gz`.
