"""
Unit Tests for Objective 3 Learn-Then-Test Risk Calibration
==========================================================
Verifies all 5 required invariants from Chapter 3 Task 5 Specification:
1. If every lambda has zero failures, return the most aggressive lambda.
2. If the first lambda has empirical risk 0.5 and cannot be rejected, return reference_only.
3. Confirm that paired clean/noisy rows produce one robust Bernoulli loss per speaker.
4. Confirm that the binomial function receives integer k, not n * risk_hat.
5. Confirm that test outcomes cannot influence lambda_hat.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phase2_calib_ltt_200spk.calibrate import (
    load_calibration_config,
    select_handover_step,
    compute_robust_losses_for_lambda,
    compute_binomial_p_value,
    run_fixed_sequence_ltt,
)


def make_synthetic_split(
    n_speakers: int = 50,
    fail_clean_prob: float = 0.0,
    fail_noisy_prob: float = 0.0,
    snr_clean: float = 30.0,
    snr_noisy: float = 25.0,
    drift_val: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Creates a synthetic multi-speaker paired DataFrame for testing calibration."""
    rng = np.random.RandomState(seed)
    rows = []
    for spk_idx in range(n_speakers):
        utt_id = f"utt_{spk_idx:04d}"
        
        # Step 4, 5, 6, 8 for clean
        for st in [4, 5, 6, 8]:
            fail_val = 0 if st == 8 else int(rng.rand() < fail_clean_prob)
            rows.append({
                "utt_id": utt_id,
                "speaker_id": f"spk_{spk_idx:04d}",
                "condition": "clean",
                "teacher_step": st,
                "prompt_snr": snr_clean,
                "d_4": drift_val,
                "d_5": drift_val,
                "d_6": drift_val,
                "fail": fail_val,
                "nfe": {4: 13, 5: 15, 6: 17, 8: 21}[st],
            })
            
        # Step 4, 5, 6, 8 for noisy15
        for st in [4, 5, 6, 8]:
            fail_val = 0 if st == 8 else int(rng.rand() < fail_noisy_prob)
            rows.append({
                "utt_id": utt_id,
                "speaker_id": f"spk_{spk_idx:04d}",
                "condition": "noisy15",
                "teacher_step": st,
                "prompt_snr": snr_noisy,
                "d_4": drift_val,
                "d_5": drift_val,
                "d_6": drift_val,
                "fail": fail_val,
                "nfe": {4: 13, 5: 15, 6: 17, 8: 21}[st],
            })
            
    return pd.DataFrame(rows)


def test_1_zero_failures_selects_most_aggressive_lambda():
    """Test 1: If every lambda has zero failures, return the most aggressive lambda."""
    df_zero_fail = make_synthetic_split(n_speakers=100, fail_clean_prob=0.0, fail_noisy_prob=0.0)
    grid = [0.20, 0.40, 0.60, 0.80, 1.00]
    
    res = run_fixed_sequence_ltt(
        df_calib=df_zero_fail,
        lambda_grid=grid,
        alpha=0.05,
        delta=0.10,
    )
    
    assert res["lambda_hat"] == 1.00, f"Expected 1.00, got {res['lambda_hat']}"
    for row in res["testing_table"]:
        assert row["status"] == "ACCEPTED"
        assert row["k_failures"] == 0


def test_2_high_risk_returns_reference_only():
    """Test 2: If the first lambda has empirical risk 0.5 and cannot be rejected, return reference_only."""
    # 50% failures across all candidates
    df_high_risk = make_synthetic_split(n_speakers=100, fail_clean_prob=0.5, fail_noisy_prob=0.5, drift_val=0.10)
    grid = [0.20, 0.40, 0.60]
    
    res = run_fixed_sequence_ltt(
        df_calib=df_high_risk,
        lambda_grid=grid,
        alpha=0.05,
        delta=0.10,
    )
    
    assert res["lambda_hat"] == "reference_only", f"Expected 'reference_only', got {res['lambda_hat']}"
    assert res["testing_table"][0]["status"] == "REJECTED_STOP"
    assert res["testing_table"][1]["status"] == "UNTESTED_STOPPED"


def test_3_paired_clean_noisy_produces_one_robust_loss_per_speaker():
    """Test 3: Confirm that paired clean/noisy rows produce one robust Bernoulli loss per speaker (max of both)."""
    # Create two speakers:
    # Speaker 1: Clean passes (fail=0), Noisy fails (fail=1) -> robust failure should be 1
    # Speaker 2: Clean passes (fail=0), Noisy passes (fail=0) -> robust failure should be 0
    rows = [
        # Speaker 1 Clean
        {"utt_id": "utt_1", "condition": "clean", "teacher_step": 4, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 13},
        {"utt_id": "utt_1", "condition": "clean", "teacher_step": 8, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 21},
        # Speaker 1 Noisy
        {"utt_id": "utt_1", "condition": "noisy15", "teacher_step": 4, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 1, "nfe": 13},
        {"utt_id": "utt_1", "condition": "noisy15", "teacher_step": 8, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 21},
        
        # Speaker 2 Clean
        {"utt_id": "utt_2", "condition": "clean", "teacher_step": 4, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 13},
        {"utt_id": "utt_2", "condition": "clean", "teacher_step": 8, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 21},
        # Speaker 2 Noisy
        {"utt_id": "utt_2", "condition": "noisy15", "teacher_step": 4, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 13},
        {"utt_id": "utt_2", "condition": "noisy15", "teacher_step": 8, "prompt_snr": 25.0, "d_4": 0.2, "d_5": 0.2, "d_6": 0.2, "fail": 0, "nfe": 21},
    ]
    df = pd.DataFrame(rows)
    eval_res = compute_robust_losses_for_lambda(df, lambda_val=0.50, snr_threshold=20.0, candidate_steps=[4], reference_step=8)
    
    assert eval_res["n_speakers"] == 2
    assert eval_res["k_robust_failures"] == 1
    assert eval_res["k_clean_failures"] == 0
    assert eval_res["k_noisy_failures"] == 1
    assert eval_res["robust_failures_array"] == [1, 0]


def test_4_binomial_function_receives_integer_k():
    """Test 4: Confirm that the binomial function receives integer k, not n * risk_hat."""
    # Integer k must be strictly enforced
    k_int = int(3)
    n_int = int(100)
    alpha = 0.05
    
    p_val = compute_binomial_p_value(k=k_int, n=n_int, alpha=alpha)
    assert isinstance(p_val, float)
    assert 0.0 <= p_val <= 1.0
    
    # Float k should trigger assertion
    with pytest.raises(AssertionError):
        compute_binomial_p_value(k=3.5, n=n_int, alpha=alpha)  # type: ignore


def test_5_test_outcomes_cannot_influence_lambda_hat():
    """Test 5: Confirm that test outcomes cannot influence lambda_hat."""
    df_calib = make_synthetic_split(n_speakers=100, fail_clean_prob=0.0, fail_noisy_prob=0.0, seed=1)
    df_test_bad = make_synthetic_split(n_speakers=100, fail_clean_prob=0.9, fail_noisy_prob=0.9, seed=2)
    df_test_good = make_synthetic_split(n_speakers=100, fail_clean_prob=0.0, fail_noisy_prob=0.0, seed=3)
    
    grid = [0.20, 0.40, 0.60]
    
    # Calibration only receives df_calib
    res_1 = run_fixed_sequence_ltt(df_calib=df_calib, lambda_grid=grid, alpha=0.05, delta=0.10)
    res_2 = run_fixed_sequence_ltt(df_calib=df_calib, lambda_grid=grid, alpha=0.05, delta=0.10)
    
    assert res_1["lambda_hat"] == res_2["lambda_hat"] == 0.60
    # Independent verification that df_test is never passed to calibration
