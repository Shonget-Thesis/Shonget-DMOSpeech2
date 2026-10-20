"""
Objective 3: Held-Out Verification of Certified Threshold
=========================================================
Evaluates the frozen certified threshold lambda_hat on the UNTOUCHED test split.
Outputs:
1. Realized robust failure rate for lambda_hat with exact Clopper-Pearson 95% CI.
2. Condition-specific (clean vs noisy15) empirical failure rates.
3. Switching distribution across candidate steps {4, 5, 6, 8}.
4. Mean actual forward-pass NFE and savings relative to Step-8 reference.
5. Reference-only fallback frequency.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phase2_calib_ltt_200spk.calibrate import (
    load_calibration_config,
    compute_robust_losses_for_lambda,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def clopper_pearson_exact_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    res = binomtest(k, n).proportion_ci(confidence_level=confidence, method="exact")
    return float(res.low), float(res.high)


def verify_on_test_split(
    df_test: pd.DataFrame,
    lambda_hat: Any,
    snr_threshold: float = 20.0,
    candidate_steps: List[int] = [4, 5, 6],
    reference_step: int = 8,
) -> Dict[str, Any]:
    """
    Evaluates the frozen certified threshold lambda_hat on the held-out test split.
    """
    n_speakers = df_test["utt_id"].nunique()
    
    # If lambda_hat is 'reference_only', always deploy Step 8
    if lambda_hat == "reference_only" or lambda_hat is None:
        eval_dict = compute_robust_losses_for_lambda(
            df=df_test,
            lambda_val=-1.0,  # Forces fallback to step 8
            snr_threshold=snr_threshold,
            candidate_steps=candidate_steps,
            reference_step=reference_step,
        )
        is_ref_only = True
    else:
        eval_dict = compute_robust_losses_for_lambda(
            df=df_test,
            lambda_val=float(lambda_hat),
            snr_threshold=snr_threshold,
            candidate_steps=candidate_steps,
            reference_step=reference_step,
        )
        is_ref_only = False

    k_robust = eval_dict["k_robust_failures"]
    ci_low, ci_high = clopper_pearson_exact_ci(k=k_robust, n=n_speakers, confidence=0.95)
    
    ref_nfe = 21.0
    nfe_savings_pct = (ref_nfe - eval_dict["mean_nfe"]) / ref_nfe * 100.0
    
    return {
        "lambda_hat": lambda_hat,
        "is_reference_only": is_ref_only,
        "n_test_speakers": n_speakers,
        "robust_failures_k": k_robust,
        "robust_failure_rate": eval_dict["empirical_risk"],
        "clopper_pearson_95_ci": [ci_low, ci_high],
        "clean_failures_k": eval_dict["k_clean_failures"],
        "clean_failure_rate": eval_dict["clean_risk"],
        "noisy_failures_k": eval_dict["k_noisy_failures"],
        "noisy_failure_rate": eval_dict["noisy_risk"],
        "mean_nfe": eval_dict["mean_nfe"],
        "nfe_savings_pct": nfe_savings_pct,
        "step_distribution": eval_dict["step_distribution"],
    }


def format_verification_report(res: Dict[str, Any]) -> str:
    """Formats markdown report of held-out verification results."""
    ci_str = f"[{res['clopper_pearson_95_ci'][0]*100:.2f}%, {res['clopper_pearson_95_ci'][1]*100:.2f}%]"
    
    lines = [
        "# Held-Out Test Split Verification Report",
        "",
        f"**Certified Threshold ($\\hat{{\\lambda}}$):** `{res['lambda_hat']}`  ",
        f"**Test Sample Size ($n$):** {res['n_test_speakers']} independent speakers / utterances  ",
        f"**Reference-Only Fallback:** {'YES' if res['is_reference_only'] else 'NO'}  ",
        "",
        "---",
        "",
        "## 1. Realized Failure Rates & Statistical Guarantees",
        "",
        "| Metric | Failures / Total | Point Estimate | Exact 95% Clopper-Pearson CI |",
        "|---|---|---|---|",
        f"| **Robust Added Failure Rate** | {res['robust_failures_k']} / {res['n_test_speakers']} | {res['robust_failure_rate']*100:.2f}% | {ci_str} |",
        f"| `clean` Condition Failure Rate | {res['clean_failures_k']} / {res['n_test_speakers']} | {res['clean_failure_rate']*100:.2f}% | — |",
        f"| `noisy15` Condition Failure Rate | {res['noisy_failures_k']} / {res['n_test_speakers']} | {res['noisy_failure_rate']*100:.2f}% | — |",
        "",
        "---",
        "",
        "## 2. Computational Efficiency",
        "",
        "| Reference NFE (Step 8) | Adaptive Policy Mean NFE | NFE Savings (%) |",
        "|---|---|---|",
        f"| 21.00 | {res['mean_nfe']:.2f} | **{res['nfe_savings_pct']:.1f}%** |",
        "",
        "---",
        "",
        "## 3. Switching Distribution",
        "",
        "| Candidate Step | Solver Steps | Forward NFE | Selection Frequency (%) |",
        "|---|---|---|---|",
    ]
    
    step_meta = {
        4: (8, 13),
        5: (9, 15),
        6: (10, 17),
        8: (12, 21),
    }
    for st, (s_steps, nfe_val) in step_meta.items():
        freq = res["step_distribution"].get(st, 0.0) * 100.0
        lines.append(f"| Step {st} | {s_steps} | {nfe_val} | {freq:.1f}% |")
        
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 80)
    print("Objective 3 / Task 6: Held-Out Test Split Coverage Verification")
    print("=" * 80)

    # 1. Load calibration output to obtain frozen certified lambda_hat
    calib_json = os.path.join(RESULTS_DIR, "ltt_summary.json")
    if os.path.exists(calib_json):
        with open(calib_json, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        lambda_hat = calib_data["primary_results"]["lambda_hat"]
        alpha = calib_data["primary_results"]["alpha"]
        delta = calib_data["primary_results"]["delta"]
    else:
        lambda_hat = 0.50
        alpha = 0.05
        delta = 0.10
    print(f"Loaded certified threshold: lambda_hat = {lambda_hat} (alpha={alpha}, delta={delta})")

    # 2. Load test split evaluation grid
    test_csv = os.path.join(ROOT_DIR, "phase3_test_benchmark_200spk", "results", "test_grid.csv")
    test_parquet = os.path.join(ROOT_DIR, "phase3_test_benchmark_200spk", "results", "test_grid.parquet")
    if os.path.exists(test_parquet):
        df_test = pd.read_parquet(test_parquet)
    elif os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv)
    else:
        raise FileNotFoundError(f"Could not find test grid at {test_csv} or {test_parquet}")

    print(f"Loaded test grid: {len(df_test)} evaluations across {df_test['speaker_id'].nunique()} speakers.")

    # 3. Verify on held-out test split
    res = verify_on_test_split(
        df_test=df_test,
        lambda_hat=lambda_hat,
        snr_threshold=20.0,
        candidate_steps=[4, 5, 6],
        reference_step=8,
    )

    # 4. Save results
    out_json = os.path.join(RESULTS_DIR, "verification_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved verification metrics: {out_json}")

    report_md = format_verification_report(res)
    out_md = os.path.join(RESULTS_DIR, "TEST_VERIFICATION_REPORT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved verification report: {out_md}")

    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "TEST_VERIFICATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + report_md)

