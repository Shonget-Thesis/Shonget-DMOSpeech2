"""
Objective 3: Learn-Then-Test (LTT) Risk Calibration
===================================================
Implements the exact Chapter 3 Learn-Then-Test calibration procedure:
1. Frozen configuration (candidate steps, lambda grid, SNR gate, alpha, delta).
2. Conservative-to-aggressive ordered hypothesis testing.
3. Paired robust Bernoulli loss: robust_fail_i(lambda) = max(fail_clean, fail_noisy15).
4. Binomial p-value calculation: p = scipy.stats.binom.cdf(k, n, alpha).
5. Fixed-sequence stopping rule: accept while p <= delta; stop at first p > delta.
6. Returns certified lambda_hat, or 'reference_only' if the first lambda fails.
7. Saves full audit log with configuration hash and step decisions.
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import binom

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_calibration_config(path: str = CONFIG_PATH) -> Tuple[Dict[str, Any], str]:
    """Loads frozen calibration configuration and returns its SHA-256 hash."""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cfg = json.loads(raw_text)
    cfg_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return cfg, cfg_hash


def select_handover_step(
    drift_dict: Dict[int, float],
    prompt_snr: float,
    lambda_val: float,
    snr_threshold: float,
    candidate_steps: List[int] = [4, 5, 6],
    reference_step: int = 8,
) -> int:
    """
    Simulates the input-adaptive decision rule for a single drift trace and prompt SNR:
    Steps through ordered candidates. Selects step i if:
      (prompt_snr >= snr_threshold) AND (drift_i <= lambda_val).
    Otherwise falls back to reference_step (Step 8).
    """
    for st in candidate_steps:
        d_i = drift_dict.get(st, None)
        if d_i is not None and prompt_snr >= snr_threshold and d_i <= lambda_val:
            return st
    return reference_step


def compute_robust_losses_for_lambda(
    df: pd.DataFrame,
    lambda_val: float,
    snr_threshold: float = 20.0,
    candidate_steps: List[int] = [4, 5, 6],
    reference_step: int = 8,
) -> Dict[str, Any]:
    """
    Computes paired robust Bernoulli losses for an entire dataset split under lambda_val:
    robust_fail_i(lambda) = max(fail_clean, fail_noisy15).
    Sample size n is the number of distinct speakers/utterances.
    """
    # Pre-index lookup table by (utt_id, condition, teacher_step) -> row dict
    step_nfe_map = {4: 13, 5: 15, 6: 17, 8: 21}
    
    unique_utts = df["utt_id"].unique()
    n_speakers = len(unique_utts)
    
    robust_failures: List[int] = []
    clean_failures: List[int] = []
    noisy_failures: List[int] = []
    step_choices: List[int] = []
    nfe_values: List[int] = []
    
    # Group by utterance
    utt_groups = df.groupby("utt_id")
    
    for utt_id, ugroup in utt_groups:
        cond_rows = {r["condition"]: r for _, r in ugroup.iterrows()}
        clean_row = ugroup[ugroup["condition"] == "clean"]
        noisy_row = ugroup[ugroup["condition"] == "noisy15"]
        
        # Build drift dictionary for clean
        clean_ref = clean_row[clean_row["teacher_step"] == reference_step].iloc[0]
        clean_drifts = {st: float(clean_ref.get(f"d_{st}", 999.0)) for st in candidate_steps}
        clean_snr = float(clean_ref["prompt_snr"])
        clean_sel_step = select_handover_step(
            clean_drifts, clean_snr, lambda_val, snr_threshold, candidate_steps, reference_step
        )
        
        # Build drift dictionary for noisy15
        noisy_ref = noisy_row[noisy_row["teacher_step"] == reference_step].iloc[0]
        noisy_drifts = {st: float(noisy_ref.get(f"d_{st}", 999.0)) for st in candidate_steps}
        noisy_snr = float(noisy_ref["prompt_snr"])
        noisy_sel_step = select_handover_step(
            noisy_drifts, noisy_snr, lambda_val, snr_threshold, candidate_steps, reference_step
        )
        
        # Lookup failure indicators
        clean_cand_row = clean_row[clean_row["teacher_step"] == clean_sel_step].iloc[0]
        noisy_cand_row = noisy_row[noisy_row["teacher_step"] == noisy_sel_step].iloc[0]
        
        fail_clean = int(clean_cand_row["fail"])
        fail_noisy = int(noisy_cand_row["fail"])
        
        # Chapter 3 Equation (3.11): robust failure is max of paired conditions
        robust_fail = max(fail_clean, fail_noisy)
        
        robust_failures.append(robust_fail)
        clean_failures.append(fail_clean)
        noisy_failures.append(fail_noisy)
        
        step_choices.extend([clean_sel_step, noisy_sel_step])
        nfe_values.extend([step_nfe_map[clean_sel_step], step_nfe_map[noisy_sel_step]])
    
    k_robust = int(np.sum(robust_failures))
    k_clean = int(np.sum(clean_failures))
    k_noisy = int(np.sum(noisy_failures))
    
    # Step selection counts
    step_counts = {st: step_choices.count(st) for st in candidate_steps + [reference_step]}
    total_evals = len(step_choices)
    step_dist = {st: step_counts[st] / total_evals if total_evals > 0 else 0.0 for st in step_counts}
    
    return {
        "lambda": lambda_val,
        "n_speakers": n_speakers,
        "k_robust_failures": k_robust,
        "empirical_risk": k_robust / n_speakers if n_speakers > 0 else 0.0,
        "k_clean_failures": k_clean,
        "clean_risk": k_clean / n_speakers if n_speakers > 0 else 0.0,
        "k_noisy_failures": k_noisy,
        "noisy_risk": k_noisy / n_speakers if n_speakers > 0 else 0.0,
        "mean_nfe": float(np.mean(nfe_values)) if nfe_values else 21.0,
        "step_distribution": step_dist,
        "robust_failures_array": robust_failures,
    }


def compute_binomial_p_value(k: int, n: int, alpha: float) -> float:
    """
    Computes exact binomial p-value under null hypothesis H_g: R(pi_lambda) > alpha.
    p = P(Bin(n, alpha) <= k) = binom.cdf(k, n, alpha).
    Note: k must be an integer (number of robust failures).
    """
    assert isinstance(k, (int, np.integer)), f"k must be an integer, got {type(k)}"
    assert isinstance(n, (int, np.integer)), f"n must be an integer, got {type(n)}"
    return float(binom.cdf(k, n, alpha))


def run_fixed_sequence_ltt(
    df_calib: pd.DataFrame,
    lambda_grid: List[float],
    alpha: float = 0.05,
    delta: float = 0.10,
    snr_threshold: float = 20.0,
    candidate_steps: List[int] = [4, 5, 6],
    reference_step: int = 8,
) -> Dict[str, Any]:
    """
    Executes the exact fixed-sequence testing procedure across the ordered lambda grid:
    1. Tests lambdas in preregistered conservative -> aggressive order.
    2. Accepts while p_value <= delta.
    3. Stops at the first p_value > delta.
    4. Returns last accepted lambda as lambda_hat, or 'reference_only' if none accepted.
    """
    n_speakers = df_calib["utt_id"].nunique()
    testing_table: List[Dict[str, Any]] = []
    certified_lambda: Optional[float] = None
    stopped = False

    for l_val in lambda_grid:
        eval_dict = compute_robust_losses_for_lambda(
            df=df_calib,
            lambda_val=l_val,
            snr_threshold=snr_threshold,
            candidate_steps=candidate_steps,
            reference_step=reference_step,
        )
        k = eval_dict["k_robust_failures"]
        p_val = compute_binomial_p_value(k=k, n=n_speakers, alpha=alpha)

        if not stopped:
            if p_val <= delta:
                status = "ACCEPTED"
                certified_lambda = l_val
            else:
                status = "REJECTED_STOP"
                stopped = True
        else:
            status = "UNTESTED_STOPPED"

        testing_table.append({
            "lambda": l_val,
            "k_failures": k,
            "n_speakers": n_speakers,
            "empirical_risk": eval_dict["empirical_risk"],
            "p_value": p_val,
            "status": status,
            "mean_nfe": eval_dict["mean_nfe"],
            "clean_risk": eval_dict["clean_risk"],
            "noisy_risk": eval_dict["noisy_risk"],
            "step_distribution": eval_dict["step_distribution"],
        })

    return {
        "alpha": alpha,
        "delta": delta,
        "n_speakers": n_speakers,
        "lambda_hat": certified_lambda if certified_lambda is not None else "reference_only",
        "testing_table": testing_table,
    }


def format_calibration_report(
    primary_res: Dict[str, Any],
    secondary_res: Dict[str, Any],
    config_hash: str,
) -> str:
    """Generates markdown report of Learn-Then-Test calibration results."""
    lines = [
        "# Learn-Then-Test (LTT) Risk Calibration Report",
        "",
        f"**Configuration SHA-256:** `{config_hash}`  ",
        f"**Calibration Unit:** Independent Speakers ($n = {primary_res['n_speakers']}$)  ",
        f"**Primary Certification Level:** $\\alpha = {primary_res['alpha']}$, $\\delta = {primary_res['delta']}$  ",
        f"**Primary Selected Threshold ($\\hat{{\\lambda}}$):** `{primary_res['lambda_hat']}`  ",
        f"**Secondary Selected Threshold ($\\alpha = {secondary_res['alpha']}$):** `{secondary_res['lambda_hat']}`  ",
        "",
        "---",
        "",
        "## 1. Primary Fixed-Sequence Hypothesis Testing Table ($\\alpha = 0.05$, $\\delta = 0.10$)",
        "",
        "| $\\lambda$ | Robust Failures ($k$) | Total Speakers ($n$) | Empirical Risk ($\\hat{R}$) | Binomial $p$-value | Decision Status | Mean NFE | Clean Risk | Noisy15 Risk |",
        "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]

    for row in primary_res["testing_table"]:
        lines.append(
            f"| {row['lambda']:.2f} | {row['k_failures']} | {row['n_speakers']} | "
            f"{row['empirical_risk']*100:.2f}% | {row['p_value']:.4e} | **{row['status']}** | "
            f"{row['mean_nfe']:.2f} | {row['clean_risk']*100:.2f}% | {row['noisy_risk']*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Secondary Fixed-Sequence Hypothesis Testing Table ($\\alpha = 0.10$, $\\delta = 0.10$)",
        "",
        "| $\\lambda$ | Robust Failures ($k$) | Total Speakers ($n$) | Empirical Risk ($\\hat{R}$) | Binomial $p$-value | Decision Status | Mean NFE |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ])

    for row in secondary_res["testing_table"]:
        lines.append(
            f"| {row['lambda']:.2f} | {row['k_failures']} | {row['n_speakers']} | "
            f"{row['empirical_risk']*100:.2f}% | {row['p_value']:.4e} | **{row['status']}** | "
            f"{row['mean_nfe']:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Mathematical Interpretation",
        "",
        "- The fixed-sequence testing controls family-wise error rate without Bonferroni conservatism.",
        "- Stopping occurs at the first threshold where the null $H_g: R > \\alpha$ cannot be rejected at level $\\delta$.",
        f"- Selected threshold $\\hat{{\\lambda}} = {primary_res['lambda_hat']}$ guarantees that the true robust failure risk satisfies $P(R(\\pi_{{\\hat{{\\lambda}}}}) > \\alpha) \\le \\delta$ under i.i.d. sampling.",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 80)
    print("OBJECTIVE 3: LEARN-THEN-TEST CALIBRATION MODULE")
    print("=" * 80)
    cfg, cfg_hash = load_calibration_config()
    print(f"Loaded frozen config with SHA-256 hash: {cfg_hash}")
    print(f"Lambda grid: {cfg['lambda_grid']}")
    print(f"Primary bound: alpha={cfg['primary_guarantee']['alpha']}, delta={cfg['primary_guarantee']['delta']}")

    calib_parquet = os.path.join(RESULTS_DIR, "calib_grid.parquet")
    calib_csv = os.path.join(RESULTS_DIR, "calib_grid.csv")

    if os.path.exists(calib_parquet):
        df_calib = pd.read_parquet(calib_parquet)
        print(f"\nLoaded {len(df_calib)} candidate rows from {calib_parquet}")
    elif os.path.exists(calib_csv):
        df_calib = pd.read_csv(calib_csv)
        print(f"\nLoaded {len(df_calib)} candidate rows from {calib_csv}")
    else:
        print("\nNote: Calibration grid cache (calib_grid.parquet / calib_grid.csv) not found.")
        print("Run 'python phase2_calib_ltt_200spk/run_calib_grid.py' first to generate candidate outcomes on the calibration split.")
        sys.exit(0)

    # Execute Primary Calibration (alpha=0.05, delta=0.10)
    print("\nRunning Primary Fixed-Sequence LTT (alpha=0.05, delta=0.10)...")
    primary_res = run_fixed_sequence_ltt(
        df_calib=df_calib,
        lambda_grid=cfg["lambda_grid"],
        alpha=cfg["primary_guarantee"]["alpha"],
        delta=cfg["primary_guarantee"]["delta"],
        snr_threshold=cfg["snr_gate_threshold_db"],
        candidate_steps=cfg["candidate_steps"],
        reference_step=cfg["reference_step"],
    )
    print(f"Primary Certified Threshold (lambda_hat): {primary_res['lambda_hat']}")

    # Execute Secondary Calibration (alpha=0.10, delta=0.10)
    print("Running Secondary Fixed-Sequence LTT (alpha=0.10, delta=0.10)...")
    secondary_res = run_fixed_sequence_ltt(
        df_calib=df_calib,
        lambda_grid=cfg["lambda_grid"],
        alpha=cfg["secondary_analysis"]["alpha"],
        delta=cfg["secondary_analysis"]["delta"],
        snr_threshold=cfg["snr_gate_threshold_db"],
        candidate_steps=cfg["candidate_steps"],
        reference_step=cfg["reference_step"],
    )
    print(f"Secondary Certified Threshold (lambda_hat): {secondary_res['lambda_hat']}")

    # Save summary json
    summary_path = os.path.join(RESULTS_DIR, "ltt_summary.json")
    summary_payload = {
        "config_sha256": cfg_hash,
        "config": cfg,
        "primary_results": primary_res,
        "secondary_results": secondary_res,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nCalibration summary saved to: {summary_path}")

    # Write report
    report_md = format_calibration_report(primary_res, secondary_res, cfg_hash)
    report_path = os.path.join(RESULTS_DIR, "LTT_CALIBRATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Calibration markdown report saved to: {report_path}")

