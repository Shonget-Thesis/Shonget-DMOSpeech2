"""
Objective 1 Analysis Script: Candidate Handover Characterization & Audit
========================================================================
Comprehensive analysis of the development split grid results (N=100 speakers / 800 evaluations):
1. Test Description and Protocol Specification.
2. Hardware Environment and Execution Limitations.
3. Candidate Failure Rates with exact Clopper-Pearson 95% CIs (Step 8 marked as 0% by definition).
4. Combined descriptive results with cluster bootstrap by utterance (1,000 resamples).
5. Failure mode decomposition: WER-only, SIM-only, Both, plus Median WER, IQR, and Paired Deltas.
6. Per-utterance-condition Oracle step distribution (Definition 3.1).
7. Condition-specific correlation audit:
   - Pooled vs. Clean-only vs. Noisy15-only Spearman correlations.
   - Comprehensive explanation of Simpson's paradox / condition confounding on normalized drift.
8. Benchmarking of 8 Candidate Schedulers on the Development Split:
   - Fixed Steps 4, 5, 6, 8, SNR-Only, Drift-Only, Proposed (SNR+Drift), and Oracle.
"""

import os
import sys
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GRID_CSV_PATH = os.path.join(RESULTS_DIR, "grid.csv")
GRID_PARQUET_PATH = os.path.join(RESULTS_DIR, "grid.parquet")
REPORT_PATH = os.path.join(RESULTS_DIR, "OBJ1_ANALYSIS_REPORT.md")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "obj1_summary.json")

STEPS = [4, 5, 6, 8]
CONDITIONS = ["clean", "noisy15"]
STEP_NFE_MAP = {4: 13, 5: 15, 6: 17, 8: 21}
STEP_TIME_MAP = {4: "0.0761", 5: "0.1181", 6: "0.1685", 8: "0.2929"}


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    res = binomtest(k, n).proportion_ci(confidence_level=1.0 - alpha, method="exact")
    return float(res.low), float(res.high)


def cluster_bootstrap_failure_rate(df: pd.DataFrame, step: int, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes cluster-bootstrap 95% confidence interval for a candidate step
    by resampling utterances (clusters of clean + noisy15) rather than independent rows.
    """
    df_step = df[df["teacher_step"] == step]
    unique_utts = df_step["utt_id"].unique()
    n_utts = len(unique_utts)
    if n_utts == 0:
        return 0.0, 0.0, 0.0

    grouped = df_step.groupby("utt_id")["fail"].agg(["sum", "count"]).reset_index()
    k_arr = grouped["sum"].values
    n_arr = grouped["count"].values

    rng = np.random.RandomState(seed)
    boot_rates = []
    for _ in range(n_boot):
        sample_indices = rng.choice(len(k_arr), size=len(k_arr), replace=True)
        boot_k = k_arr[sample_indices].sum()
        boot_n = n_arr[sample_indices].sum()
        boot_rates.append(boot_k / boot_n if boot_n > 0 else 0.0)

    point_estimate = float(k_arr.sum() / n_arr.sum())
    ci_low = float(np.percentile(boot_rates, 2.5))
    ci_high = float(np.percentile(boot_rates, 97.5))
    return point_estimate, ci_low, ci_high


def determine_oracle_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the per-utterance-condition oracle step:
    The cheapest non-failing candidate step among {4, 5, 6, 8}.
    """
    oracle_records = []
    for (utt_id, cond), group in df.groupby(["utt_id", "condition"]):
        non_failing = group[group["fail"] == 0].sort_values("nfe")
        if len(non_failing) > 0:
            oracle_row = non_failing.iloc[0]
            oracle_step = int(oracle_row["teacher_step"])
            oracle_nfe = int(oracle_row["nfe"])
        else:
            oracle_step = 8
            oracle_nfe = 21

        ref_row = group[group["teacher_step"] == 8].iloc[0]
        oracle_records.append({
            "utt_id": utt_id,
            "condition": cond,
            "oracle_step": oracle_step,
            "oracle_nfe": oracle_nfe,
            "prompt_snr": float(ref_row["prompt_snr"]),
            "d_4": float(ref_row["d_4"]),
            "d_5": float(ref_row["d_5"]),
            "d_6": float(ref_row["d_6"]),
        })
    return pd.DataFrame(oracle_records)


def run_analysis():
    print("=" * 80)
    print("OBJECTIVE 1: CANDIDATE HANDOVER TIMING ANALYSIS & AUDIT")
    print("=" * 80)

    if os.path.exists(GRID_PARQUET_PATH):
        try:
            df = pd.read_parquet(GRID_PARQUET_PATH)
            print(f"Loaded {len(df)} rows from {GRID_PARQUET_PATH}")
        except Exception:
            df = pd.read_csv(GRID_CSV_PATH)
            print(f"Loaded {len(df)} rows from {GRID_CSV_PATH}")
    elif os.path.exists(GRID_CSV_PATH):
        df = pd.read_csv(GRID_CSV_PATH)
        print(f"Loaded {len(df)} rows from {GRID_CSV_PATH}")
    else:
        raise FileNotFoundError(f"Neither {GRID_PARQUET_PATH} nor {GRID_CSV_PATH} exists! Run run_grid.py first.")

    # 1. Verification of Step 8 Reference (Check from Task 4)
    step8_fails = df[df["teacher_step"] == 8]["fail"].sum()
    assert step8_fails == 0, f"Integrity check failed: Step 8 has {step8_fails} failures, expected 0 by construction!"
    print("[Pass] Integrity Check: Step-8 rows have fail = 0 by construction.")

    # 2. Per-Step Failure Rates & Clopper-Pearson CIs
    failure_table = []
    for cond in CONDITIONS:
        df_cond = df[df["condition"] == cond]
        for step in STEPS:
            sub = df_cond[df_cond["teacher_step"] == step]
            k = int(sub["fail"].sum())
            n = len(sub)
            rate = k / n if n > 0 else 0.0
            ci_low, ci_high = clopper_pearson_ci(k, n)
            meta_nfe = int(sub["nfe"].iloc[0]) if n > 0 else 0
            failure_table.append({
                "condition": cond,
                "teacher_step": step,
                "nfe": meta_nfe,
                "fail_count": k,
                "total_count": n,
                "fail_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    # Combined cluster bootstrap
    combined_failure_table = []
    for step in STEPS:
        point, ci_low, ci_high = cluster_bootstrap_failure_rate(df, step=step)
        sub = df[df["teacher_step"] == step]
        combined_failure_table.append({
            "teacher_step": step,
            "nfe": int(sub["nfe"].iloc[0]),
            "fail_count": int(sub["fail"].sum()),
            "total_count": len(sub),
            "cluster_point_rate": point,
            "cluster_ci_low": ci_low,
            "cluster_ci_high": ci_high,
        })

    # 3. Oracle Steps and Distribution
    oracle_df = determine_oracle_steps(df)
    oracle_dist = {}
    for cond in CONDITIONS:
        cond_oracles = oracle_df[oracle_df["condition"] == cond]["oracle_step"]
        counts = cond_oracles.value_counts().to_dict()
        total = len(cond_oracles)
        oracle_dist[cond] = {
            step: {
                "count": counts.get(step, 0),
                "fraction": counts.get(step, 0) / total if total > 0 else 0.0,
            }
            for step in STEPS
        }

    # 4. Correlation Audit: Pooled, Clean-Only, and Noisy-Only
    correlations = {"pooled": {}, "clean": {}, "noisy15": {}}
    features = ["d_4", "d_5", "d_6", "prompt_snr"]
    
    # Pooled
    for feat in features:
        corr, p_val = spearmanr(oracle_df[feat], oracle_df["oracle_step"])
        correlations["pooled"][feat] = {"spearman_rho": float(corr), "p_value": float(p_val)}
        
    # Clean only
    clean_sub = oracle_df[oracle_df["condition"] == "clean"]
    for feat in features:
        corr, p_val = spearmanr(clean_sub[feat], clean_sub["oracle_step"])
        correlations["clean"][feat] = {"spearman_rho": float(corr), "p_value": float(p_val)}
        
    # Noisy only
    noisy_sub = oracle_df[oracle_df["condition"] == "noisy15"]
    for feat in features:
        corr, p_val = spearmanr(noisy_sub[feat], noisy_sub["oracle_step"])
        correlations["noisy15"][feat] = {"spearman_rho": float(corr), "p_value": float(p_val)}

    # 5. Failure Mode Decomposition & Paired Deltas
    piv_wer = df.pivot(index=["utt_id", "condition"], columns="teacher_step", values="wer")
    piv_sim = df.pivot(index=["utt_id", "condition"], columns="teacher_step", values="sim")
    
    decomp_table = []
    for cond in CONDITIONS:
        sub_c = df[df["condition"] == cond]
        w_cond = piv_wer.xs(cond, level="condition")
        s_cond = piv_sim.xs(cond, level="condition")
        for step in STEPS:
            st_sub = sub_c[sub_c["teacher_step"] == step]
            k_wer = int(((st_sub["wer_failed"] == 1) & (st_sub["sim_failed"] == 0)).sum())
            k_sim = int(((st_sub["wer_failed"] == 0) & (st_sub["sim_failed"] == 1)).sum())
            k_both = int(((st_sub["wer_failed"] == 1) & (st_sub["sim_failed"] == 1)).sum())
            k_tot = int((st_sub["fail"] == 1).sum())
            
            med_wer = float(st_sub["wer"].median())
            q25_wer = float(st_sub["wer"].quantile(0.25))
            q75_wer = float(st_sub["wer"].quantile(0.75))
            iqr_wer = q75_wer - q25_wer
            
            if step != 8:
                d_wer = (w_cond[step] - w_cond[8])
                d_sim = (s_cond[step] - s_cond[8])
                mean_dwer = float(d_wer.mean())
                med_dwer = float(d_wer.median())
                mean_dsim = float(d_sim.mean())
                med_dsim = float(d_sim.median())
            else:
                mean_dwer = 0.0
                med_dwer = 0.0
                mean_dsim = 0.0
                med_dsim = 0.0
                
            decomp_table.append({
                "condition": cond,
                "step": step,
                "nfe": STEP_NFE_MAP[step],
                "total_fail": k_tot,
                "wer_only": k_wer,
                "sim_only": k_sim,
                "both": k_both,
                "median_wer": med_wer,
                "iqr_wer": iqr_wer,
                "mean_dwer": mean_dwer,
                "median_dwer": med_dwer,
                "mean_dsim": mean_dsim,
                "median_dsim": med_dsim,
            })

    # 6. Quality summary
    quality_summary = []
    for cond in CONDITIONS:
        for step in STEPS:
            sub = df[(df["condition"] == cond) & (df["teacher_step"] == step)]
            meta_nfe = STEP_NFE_MAP[step]
            savings = (21 - meta_nfe) / 21 * 100
            quality_summary.append({
                "condition": cond,
                "step": step,
                "nfe": meta_nfe,
                "savings_vs_ref": savings,
                "mean_wer": float(sub["wer"].mean()),
                "median_wer": float(sub["wer"].median()),
                "mean_sim": float(sub["sim"].mean()),
                "median_sim": float(sub["sim"].median()),
                "mean_utmos": float(sub["utmos"].mean()),
                "fail_rate": float(sub["fail"].mean()),
            })

    # 7. Comparative Evaluation of Candidate Schedulers
    scheduler_results = []
    # Build per-utterance records
    utt_cond_records = []
    for (utt_id, cond), g in df.groupby(["utt_id", "condition"]):
        ref_row = g[g["teacher_step"] == 8].iloc[0]
        snr = float(ref_row["prompt_snr"])
        d4 = float(ref_row["d_4"])
        step_map = {int(r["teacher_step"]): r for _, r in g.iterrows()}
        
        non_f = g[g["fail"] == 0].sort_values("nfe")
        s_oracle = int(non_f.iloc[0]["teacher_step"]) if len(non_f) > 0 else 8
        
        # Sequential decision function: checks steps 4, 5, 6 sequentially
        def get_sequential_step(l_thresh: float) -> int:
            if snr >= 20.0:
                for st in [4, 5, 6]:
                    d_val = float(ref_row[f"d_{st}"])
                    if d_val <= l_thresh:
                        return st
            return 8

        p_choices = {
            "Fixed Step 4": 4,
            "Fixed Step 5": 5,
            "Fixed Step 6": 6,
            "Fixed Step 8": 8,
            "SNR-Only (theta=20.0 dB)": 4 if snr >= 20.0 else 8,
            "Drift-Only (lambda=0.60)": 4 if d4 <= 0.60 else 8,
            "Binary SNR + d_4 diagnostic policy (lambda=0.60)": 4 if (snr >= 20.0 and d4 <= 0.60) else 8,
            "Sequential Policy (lambda=0.60, theta=20.0 dB)": get_sequential_step(0.60),
            "Sequential Policy (lambda=0.70, theta=20.0 dB)": get_sequential_step(0.70),
            "Oracle Policy": s_oracle,
        }
        utt_cond_records.append({"utt_id": utt_id, "condition": cond, "choices": p_choices, "data": step_map})

    policy_names = [
        "Fixed Step 4", "Fixed Step 5", "Fixed Step 6", "Fixed Step 8",
        "SNR-Only (theta=20.0 dB)", "Drift-Only (lambda=0.60)",
        "Binary SNR + d_4 diagnostic policy (lambda=0.60)",
        "Sequential Policy (lambda=0.60, theta=20.0 dB)",
        "Sequential Policy (lambda=0.70, theta=20.0 dB)",
        "Oracle Policy"
    ]

    for p_name in policy_names:
        nfes = []
        clean_fails = []
        noisy_fails = []
        spk_robust = {}
        step_counts = {4: 0, 5: 0, 6: 0, 8: 0}
        for rec in utt_cond_records:
            st = rec["choices"][p_name]
            step_counts[st] += 1
            row = rec["data"][st]
            nfe = int(row["nfe"])
            fail = int(row["fail"])
            nfes.append(nfe)
            if rec["condition"] == "clean":
                clean_fails.append(fail)
            else:
                noisy_fails.append(fail)
            spk_robust[rec["utt_id"]] = max(spk_robust.get(rec["utt_id"], 0), fail)

        mean_nfe = float(np.mean(nfes))
        savings = (21.0 - mean_nfe) / 21.0 * 100.0
        rob_rate = float(np.mean(list(spk_robust.values())))
        scheduler_results.append({
            "policy": p_name,
            "mean_nfe": mean_nfe,
            "savings_pct": savings,
            "robust_fail_rate": rob_rate,
            "clean_fail_rate": float(np.mean(clean_fails)),
            "noisy_fail_rate": float(np.mean(noisy_fails)),
            "step_counts": step_counts,
        })

    summary_data = {
        "failure_table_condition": failure_table,
        "combined_cluster_bootstrap": combined_failure_table,
        "oracle_distribution": oracle_dist,
        "correlations": correlations,
        "failure_decomposition": decomp_table,
        "quality_summary": quality_summary,
        "scheduler_comparison": scheduler_results,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Runtime accounting calculations
    total_pipe_min = 716.20
    n_spk = len(df["speaker_id"].unique())
    n_cases = len(df.groupby(["utt_id", "condition"]))
    n_evals = len(df)
    pipe_min_case = total_pipe_min / n_cases
    pipe_sec_case = pipe_min_case * 60.0
    pipe_min_spk = total_pipe_min / n_spk
    pipe_sec_eval = (total_pipe_min * 60.0) / n_evals
    model_mean_sec = float(df["runtime_sec"].mean())
    model_median_sec = float(df["runtime_sec"].median())

    # 8. Format Markdown Report
    report_lines = [
        "# Objective 1 Analysis & Audit Report: Characterization of Teacher Handover Timing",
        "",
        "**Document Status:** Development-Set Problem Characterization & Decision-Rule Audit (N=100 Speakers / 800 Runs)  ",
        "**Governing Thesis Artifact:** Chapter 3 Research Objective 1  ",
        "",
        "---",
        "",
        "## Test Description & Protocol Specification",
        "",
        "- **Research Objective:** Objective 1 — Characterize the empirical effect of teacher handover timing on quality, failure probability, and compute on the Development Split.",
        "- **Dataset & Split Scope:** 100 unique, speaker-disjoint Common Voice speakers drawn from the Seed-TTS English evaluation dataset (`development_manifest.json`).",
        "- **Acoustic Regimes:** Dual-condition evaluation for every speaker: `clean` and `noisy15` (Additive White Gaussian Noise injected at 15.0 +/- 0.1 dB relative to prompt power before feature extraction and RMS normalization).",
        "- **Candidate Handover Grid:** Teacher steps 4, 5, 6, and 8, corresponding to actual swayed Euler integration times $t \\in [0.0761, 0.1181, 0.1685, 0.2929]$.",
        "- **Trajectory Probe (Zero Additional NFE):** Normalized endpoint drift evaluated on generated frames only with zero additional neural-network forward passes:",
        "  $$d_i = \\frac{\\| \\hat{x}_1(t_i) - \\hat{x}_1(t_{i-1}) \\|_2}{\\| \\hat{x}_1(t_i) \\|_2 + \\epsilon}$$",
        "- **Student Consistency Completion:** Fixed restart level $j=1$ (times $\\tau \\in [0.0761, 0.2929, 0.6173]$, exactly 3 student forward passes).",
        "- **Cost Accounting (Primary Unit = Forward NFE):**",
        "  - Step 4: 5 teacher steps ($2\\times$ CFG) + 3 student steps = **13 NFE** (38.1% savings vs reference)",
        "  - Step 5: 6 teacher steps ($2\\times$ CFG) + 3 student steps = **15 NFE** (28.6% savings vs reference)",
        "  - Step 6: 7 teacher steps ($2\\times$ CFG) + 3 student steps = **17 NFE** (19.0% savings vs reference)",
        "  - Step 8 (Reference): 9 teacher steps ($2\\times$ CFG) + 3 student steps = **21 NFE** (0.0% savings)",
        "- **Chapter 3 Definition 3.1 Failure Criterion (Reference-Aware Degradation):**",
        "  $$\\text{fail}(x, k) = \\text{WER}(x, k) > \\max(0.125, \\text{WER}(x, \\text{ref}) + 0.01) \\lor \\text{SIM}(x, k) < \\text{SIM}(x, \\text{ref}) - 0.02$$",
        "  Evaluated strictly against the matched Step-8 reference synthesized with identical input and student noise seeds.",
        "- **Evaluation Metrics:** Whisper large-v3 for Word Error Rate (WER), WavLM-large x-vector cosine similarity against ground-truth target audio (SIM), and SpeechMOS UTMOS for naturalness.",
        "",
        "---",
        "",
        "## Hardware Environment & Execution Limitations",
        "",
        "- **Host Processor (CPU):** AMD Ryzen 7 5700U with Radeon Graphics (8 physical cores, 16 logical threads, base clock 1.8 GHz, boost up to 4.3 GHz).",
        "- **GPU Accelerator Availability:** Integrated AMD Radeon Graphics without native NVIDIA CUDA execution support (`torch.cuda.is_available() == False`).",
        "- **Execution Target:** Host CPU execution using evaluation mode (`torch.inference_mode()`), batch size 1, full 32-bit floating point precision (`torch.float32` / FP32), and isolated local `torch.Generator` instances for bitwise reproducibility.",
        "- **Computational Limitations & Precise Latency Accounting:**",
        f"  Total sequential execution time across all 800 evaluations on host CPU was **716.20 minutes (~11.94 hours)**. The exact time accounting breaks down as:",
        f"  - **{pipe_min_case:.3f} minutes ({pipe_sec_case:.2f} seconds)** per utterance-condition case (200 cases across 100 speakers × 2 conditions).",
        f"  - **{pipe_min_spk:.3f} minutes** per speaker (including both clean and noisy15 conditions).",
        f"  - **{pipe_sec_eval:.3f} seconds** per candidate evaluation (800 candidate evaluations).",
        f"  - **Bare model-generation runtime** accounted for a mean of **{model_mean_sec:.2f} seconds** (median **{model_median_sec:.2f} seconds**) per candidate evaluation (121.02 minutes cumulative across 800 evaluations). The remainder (~{pipe_sec_eval - model_mean_sec:.2f} seconds per candidate evaluation) represents evaluation-pipeline overhead (Whisper large-v3 transcription, WavLM-large x-vector extraction and target cosine comparison, UTMOS scoring, tensor operations, and file I/O).",
        "- **Data Isolation:** Calibration and held-out test splits were strictly isolated and never accessed by this runner.",
        "",
        "---",
        "",
        "## 1. Candidate Failure Rates (Definition 3.1)",
        "",
        "> [!IMPORTANT]",
        "> **Reference-Relative Interpretation:** Failure denotes unacceptable *additional degradation* relative to the matched Step 8 reference. Passing the failure rule does not by itself establish high absolute synthesis quality, especially when the reference is degraded by an adverse prompt condition. Step 8 failure is 0% by definition; confidence intervals are not applicable.",
        "",
        "### 1.1 Condition-Specific Failure Rates (Exact Clopper-Pearson 95% Confidence Intervals)",
        "",
        "| Condition | Teacher Step | Actual Time | Total NFE | Failures / Total | Failure Rate (%) | Exact 95% Clopper-Pearson CI |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for row in failure_table:
        time_val = STEP_TIME_MAP[row["teacher_step"]]
        if row["teacher_step"] == 8:
            ci_str = "0% by definition (CI not applicable)"
        else:
            ci_str = f"[{row['ci_low']*100:.2f}%, {row['ci_high']*100:.2f}%]"
        report_lines.append(
            f"| `{row['condition']}` | Step {row['teacher_step']} | {time_val} | {row['nfe']} | "
            f"{row['fail_count']} / {row['total_count']} | {row['fail_rate']*100:.2f}% | {ci_str} |"
        )

    report_lines.extend([
        "",
        "### 1.2 Combined Descriptive Results (Cluster Bootstrap by Utterance, 1,000 resamples)",
        "",
        "| Teacher Step | Total NFE | Failures / Total | Cluster Mean Rate (%) | Cluster Bootstrap 95% CI |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ])
    for row in combined_failure_table:
        if row["teacher_step"] == 8:
            ci_str = "0% by definition (CI not applicable)"
        else:
            ci_str = f"[{row['cluster_ci_low']*100:.2f}%, {row['cluster_ci_high']*100:.2f}%]"
        report_lines.append(
            f"| Step {row['teacher_step']} | {row['nfe']} | {row['fail_count']} / {row['total_count']} | "
            f"{row['cluster_point_rate']*100:.2f}% | {ci_str} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Failure Mode Decomposition & Distributional Statistics",
        "",
        "To explain discrepancies between mean WER and failure rate (such as extreme Whisper repetition loops in noisy audio), we report failure breakdown by modality along with Median WER and Interquartile Range (IQR):",
        "",
        "| Condition | Step | Total Failures | WER-Only Fail | SIM-Only Fail | Both Failed | Median WER (%) | WER IQR (%) | Paired Mean $\\Delta$WER (%) | Paired Median $\\Delta$WER (%) | Paired Median $\\Delta$SIM |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for d in decomp_table:
        report_lines.append(
            f"| `{d['condition']}` | Step {d['step']} | **{d['total_fail']}** | {d['wer_only']} | {d['sim_only']} | {d['both']} | "
            f"{d['median_wer']*100:.2f}% | {d['iqr_wer']*100:.2f}% | {d['mean_dwer']*100:.2f}% | {d['median_dwer']*100:.2f}% | {d['median_dsim']:+.4f} |"
        )

    report_lines.extend([
        "",
        "> [!NOTE]",
        "> **Explanation of Noisy WER Outliers & Modality Contributions:**",
        "> 1. For noisy prompts, Mean WER at Step 5 (70.40%) and Step 6 (64.17%) was heavily distorted by extreme ASR repetition loops: one utterance (`common_voice_en_31959919-common_voice_en_31959922`) exhibited WER = 2,690.9% at Step 5 and 2,600.0% at Step 6, while two additional separate utterances experienced repetition loops at Step 5 (WER = 2,220.0%) and Step 6 (WER = 1,730.8%). The **Median WER** is strictly monotonic: Step 4 (17.42%) -> Step 5 (11.81%) -> Step 6 (10.00%) = Step 8 (10.00%).",
        "> 2. Both intelligibility and speaker-similarity degradation contributed substantially to noisy-condition failures. At Step 4, SIM was involved in 42 failures (27 SIM-only, 15 both) and WER was involved in 36 failures (21 WER-only, 15 both).",
        "",
        "---",
        "",
        "## 3. Oracle Step Distribution by Condition",
        "",
        r"The **oracle step** is defined as the cheapest non-failing candidate ($\min \{k \in \{4, 5, 6, 8\} \mid \text{fail}(x, k) = 0\}$).",
        "",
        "| Condition | Oracle Step 4 (NFE 13) | Oracle Step 5 (NFE 15) | Oracle Step 6 (NFE 17) | Oracle Step 8 (NFE 21) |",
        "|---|:---:|:---:|:---:|:---:|",
    ])
    for cond in CONDITIONS:
        dist = oracle_dist[cond]
        report_lines.append(
            f"| `{cond}` | {dist[4]['fraction']*100:.1f}% ({dist[4]['count']}) | "
            f"{dist[5]['fraction']*100:.1f}% ({dist[5]['count']}) | "
            f"{dist[6]['fraction']*100:.1f}% ({dist[6]['count']}) | "
            f"{dist[8]['fraction']*100:.1f}% ({dist[8]['count']}) |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Signal Correlation Audit: Condition Confounding and Drift Diagnosis",
        "",
        "### 4.1 Condition-Specific vs. Pooled Spearman Rank Correlations with Oracle Step",
        "",
        "> [!NOTE]",
        "> Pooled Spearman correlations treat paired clean and noisy observations as independent and are reported descriptively. Condition-specific analyses contain exactly one observation per speaker and are statistically independent.",
        "",
        "| Feature | Pooled $\\rho$ (N=200, Descriptive) | Pooled $p$-value | Clean-Only $\\rho$ (N=100) | Clean $p$-value | Noisy-Only $\\rho$ (N=100) | Noisy $p$-value |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for feat in features:
        p_res = correlations["pooled"][feat]
        c_res = correlations["clean"][feat]
        n_res = correlations["noisy15"][feat]
        report_lines.append(
            f"| `{feat}` | {p_res['spearman_rho']:.4f} | {p_res['p_value']:.2e} | "
            f"{c_res['spearman_rho']:.4f} | {c_res['p_value']:.2e} | "
            f"{n_res['spearman_rho']:.4f} | {n_res['p_value']:.2e} |"
        )

    report_lines.extend([
        "",
        "### 4.2 Diagnostic Resolution of Condition Confounding on Drift",
        "",
        "1. **Condition Confounding Across Regimes:**",
        "   - In `clean` audio, normalized drift is large (mean $d_4 = 0.595$, median $0.605$) and oracle step is uniformly low (98% at Step 4).",
        "   - In `noisy15` audio, normalized drift exhibited substantially lower values (mean $d_4 = 0.179$, median $0.175$). The present experiment did not isolate the cause of this reduction; it may reflect changes in predicted velocity, endpoint magnitude, normalization behavior, or their interaction. At the same time, noisy audio requires higher oracle steps (32% at Step 8).",
        "   - Pooling clean and noisy samples artificially pairs high drift with low oracle steps, producing an unadjusted negative correlation ($\\rho = -0.6139$).",
        "2. **Within-Condition Predictive Power:**",
        "   - Within clean speech, variance is minimal because 98% of inputs succeed at Step 4.",
        "   - Within noisy speech, passing samples have slightly *higher* drift ($0.186 \\pm 0.030$) than failing samples ($0.175 \\pm 0.037$).",
        "   - Consequently, the simple univariate rule $d_i \\le \\lambda$ alone cannot differentiate passing from failing noisy utterances.",
        "",
        "---",
        "",
        "## 5. Comparative Evaluation of Candidate Schedulers (Development Split)",
        "",
        "To test whether trajectory drift provides additional risk-compute value beyond prompt SNR alone, we benchmark 10 candidate policies across all 100 development speakers (200 condition-pairs):",
        "",
        "| Scheduler Policy | Decision Rule | Mean NFE | NFE Savings (%) | Robust Failure Rate (%) | Clean Fail Rate (%) | Noisy15 Fail Rate (%) | Step Distribution {4, 5, 6, 8} |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for s in scheduler_results:
        rule_desc = {
            "Fixed Step 4": "Always Step 4",
            "Fixed Step 5": "Always Step 5",
            "Fixed Step 6": "Always Step 6",
            "Fixed Step 8": "Always Step 8 (Reference)",
            "SNR-Only (theta=20.0 dB)": "If $\\text{SNR} \\ge 20\\text{ dB} \\to 4$, else $8$",
            "Drift-Only (lambda=0.60)": "If $d_4 \\le 0.60 \\to 4$, else $8$",
            "Binary SNR + d_4 diagnostic policy (lambda=0.60)": "If $\\text{SNR} \\ge 20\\text{ dB}$ AND $d_4 \\le 0.60 \\to 4$, else $8$",
            "Sequential Policy (lambda=0.60, theta=20.0 dB)": "Sequential $4\\to 5\\to 6$ if $d_i \\le 0.60$ and $\\text{SNR} \\ge 20$, else $8$",
            "Sequential Policy (lambda=0.70, theta=20.0 dB)": "Sequential $4\\to 5\\to 6$ if $d_i \\le 0.70$ and $\\text{SNR} \\ge 20$, else $8$",
            "Oracle Policy": "Cheapest non-failing candidate per utterance",
        }[s["policy"]]
        sc = s["step_counts"]
        sc_str = f"{{{sc[4]}, {sc[5]}, {sc[6]}, {sc[8]}}}"
        report_lines.append(
            f"| **{s['policy']}** | {rule_desc} | {s['mean_nfe']:.2f} | **{s['savings_pct']:.1f}%** | "
            f"**{s['robust_fail_rate']*100:.1f}%** | {s['clean_fail_rate']*100:.1f}% | {s['noisy_fail_rate']*100:.1f}% | {sc_str} |"
        )

    report_lines.extend([
        "",
        "### Key Takeaways from the Policy Comparison:",
        "- **Drift-Only Fails Disastrously under Noise (63.0% Robust Failure Rate):** Because noisy prompts collapse $d_4$ to $\\sim 0.18$ (which is $\\le 0.60$), a drift-only rule triggers early exit on 100% of noisy inputs, resulting in severe degradation.",
        "- **SNR-Only Baseline Performance:** On the development split, the SNR-only policy achieved 19.0% mean NFE reduction with a 2.0% robust failure rate. Independent calibration and held-out testing are required before making a reliability claim.",
        "- **Binary SNR + $d_4$ Diagnostic Filter:** Adding the $d_4$ filter in the binary diagnostic policy reduced clean development failures from two cases to one while reducing mean NFE savings from 19.0% to 9.3%. A one-case difference is insufficient to establish a statistically reliable improvement.",
        "- **Complete Sequential Scheduler Recovers Compute Efficiency:** Evaluating the complete sequential $\\{4, 5, 6, 8\\}$ policy restores substantial compute savings (**16.1% savings at $\\lambda=0.60$** and **18.4% savings at $\\lambda=0.70$**) with a **1.0% robust failure rate**. This occurs because boundary clean cases that do not pass $d_4 \\le \\lambda$ can safely exit at Step 5 (15 NFE) or Step 6 (17 NFE) instead of immediately defaulting to Step 8 (21 NFE).",
        "",
        "---",
        "",
        "## 6. Preregistration & Policy Freezing Before Task 5 Calibration",
        "",
        "To maintain rigorous Learn-Then-Test guarantees and prevent model-family selection from leaking into calibration data, we freeze the following specifications using development-split insights only:",
        "",
        "1. **Primary Policy Family:** Sequential Input-Adaptive Scheduler with prompt-SNR gate and normalized trajectory drift probe.",
        "2. **Candidate Decision Sequence:** Evaluated in order $4 \\to 5 \\to 6 \\to 8$. At candidate $i \\in \\{4, 5, 6\\}$, the scheduler selects step $i$ if:",
        "   $$\\text{Prompt SNR} \\ge \\theta \\; (20.0\\text{ dB}) \\quad \\text{AND} \\quad d_i \\le \\lambda$$",
        "   If no candidate qualifies, the policy falls back to Reference Step 8.",
        "3. **Preregistered Threshold Grid:** $\\Lambda = [0.20, 0.25, 0.30, 0.35, \\dots, 1.20]$ tested strictly in conservative-to-aggressive order.",
        "4. **Statistical Target:** Primary Learn-Then-Test guarantee controlling robust speaker-level failure risk at level $\\alpha = 0.05$ with confidence $1 - \\delta = 0.90$ ($P(R(\\pi_{\\hat{\\lambda}}) > \\alpha) \\le \\delta$).",
        "5. **Independent Calibration Unit:** Paired robust Bernoulli loss $\\text{robust\\_fail}_i(\\lambda) = \\max(\\text{fail}_{i,\\text{clean}}, \\text{fail}_{i,\\text{noisy15}})$ over $n = 200$ independent calibration speakers.",
        "6. **Prespecified Comparators for Task 6:** Fixed steps {4, 5, 6, 8}, SNR-Only ($\\theta = 20.0\\text{ dB}$), and Drift-Only ($\\lambda = 0.60$) are retained as prespecified baseline comparators on the held-out test split, but will not be selected post-hoc during calibration.",
        "",
        "---",
        "",
        "## 7. Quality & Computational Cost Trade-Off Summary",
        "",
        "| Condition | Candidate Step | NFE | NFE Savings (%) | Mean WER (%) | Median WER (%) | Mean SIM | Median SIM | Mean UTMOS | Failure Rate (%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for q in quality_summary:
        report_lines.append(
            f"| `{q['condition']}` | Step {q['step']} | {q['nfe']} | {q['savings_vs_ref']:.1f}% | "
            f"{q['mean_wer']*100:.2f}% | {q['median_wer']*100:.2f}% | {q['mean_sim']:.4f} | {q['median_sim']:.4f} | "
            f"{q['mean_utmos']:.4f} | {q['fail_rate']*100:.1f}% |"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines[:50]))
    print(f"\nFull analysis report written to: {REPORT_PATH}")
    print(f"Summary data saved to: {SUMMARY_JSON_PATH}")

    return summary_data


if __name__ == "__main__":
    run_analysis()
