"""
Unit Tests for Task 2: Evaluation Metrics and Failure Module
============================================================
Tests:
1. Seed-TTS / F5-TTS text normalization (punctuation removal, whitespace collapsing, lowercasing).
2. Fractional WER calculation and edit operation counters.
3. Definition 3.1 Failure Function:
   fail(x, k) = WER(x,k) > max(0.125, WER(x,ref) + 0.01) OR SIM(x,k) < SIM(x,ref) - 0.02
4. Edge cases: zero reference WER, large degradation, boundary margins.
"""

import pytest
from core.metrics import (
    normalize_text,
    compute_wer_details,
    check_failure,
    load_failure_config,
)


def test_text_normalization():
    raw_1 = "Hello, World! This is a test... Right?"
    assert normalize_text(raw_1) == "hello world this is a test right"

    raw_2 = "   Multiple   spaces   and\ttabs\nnewlines! "
    assert normalize_text(raw_2) == "multiple spaces and tabs newlines"

    raw_3 = "One by one, the campfires were extinguished."
    assert normalize_text(raw_3) == "one by one the campfires were extinguished"


def test_wer_details_computation():
    truth = "get the trust fund to the bank early"
    hypo_exact = "Get the trust fund to the bank early."
    res_exact = compute_wer_details(truth, hypo_exact)
    assert res_exact["wer"] == 0.0
    assert res_exact["substitutions"] == 0
    assert res_exact["deletions"] == 0
    assert res_exact["insertions"] == 0
    assert res_exact["hits"] == 8

    hypo_sub = "get the trust money to the bank early"
    res_sub = compute_wer_details(truth, hypo_sub)
    assert res_sub["wer"] == 1 / 8  # 0.125
    assert res_sub["substitutions"] == 1
    assert res_sub["hits"] == 7

    hypo_del = "get the trust to the bank early"
    res_del = compute_wer_details(truth, hypo_del)
    assert res_del["wer"] == 1 / 8
    assert res_del["deletions"] == 1

    hypo_ins = "please get the trust fund to the bank early"
    res_ins = compute_wer_details(truth, hypo_ins)
    assert res_ins["wer"] == 1 / 8
    assert res_ins["insertions"] == 1


def test_failure_rule_definition_3_1():
    """
    fail(x, k) =
        WER(x,k) > max(0.125, WER(x,ref) + 0.01)
        OR
        SIM(x,k) < SIM(x,ref) - 0.02
    """
    cfg = {"absolute_wer_ceiling": 0.125, "relative_wer_margin": 0.01, "similarity_margin": 0.02}

    # Case 1: Identical metrics to reference -> Pass (fail=False)
    fail, diag = check_failure(wer_cand=0.02, wer_ref=0.02, sim_cand=0.85, sim_ref=0.85, config=cfg)
    assert not fail
    assert not diag["wer_failed"]
    assert not diag["sim_failed"]

    # Case 2: Candidate WER exceeds absolute ceiling (0.125) even when ref WER is 0.00
    # wer_threshold = max(0.125, 0.00 + 0.01) = 0.125
    fail, diag = check_failure(wer_cand=0.13, wer_ref=0.00, sim_cand=0.85, sim_ref=0.85, config=cfg)
    assert fail
    assert diag["wer_failed"]
    assert not diag["sim_failed"]

    # Case 3: Ref WER is already high (0.15). Candidate is 0.155 -> Pass because threshold is 0.16 (ref + 0.01)
    fail, diag = check_failure(wer_cand=0.155, wer_ref=0.15, sim_cand=0.85, sim_ref=0.85, config=cfg)
    assert not fail

    # Case 4: Ref WER is 0.15. Candidate is 0.165 -> Fail (exceeds ref + 0.01)
    fail, diag = check_failure(wer_cand=0.165, wer_ref=0.15, sim_cand=0.85, sim_ref=0.85, config=cfg)
    assert fail
    assert diag["wer_failed"]

    # Case 5: Similarity drop exceeds 0.02 -> Fail
    fail, diag = check_failure(wer_cand=0.02, wer_ref=0.02, sim_cand=0.829, sim_ref=0.850, config=cfg)
    assert fail
    assert not diag["wer_failed"]
    assert diag["sim_failed"]

    # Case 6: Similarity drop within 0.02 margin -> Pass
    fail, diag = check_failure(wer_cand=0.02, wer_ref=0.02, sim_cand=0.835, sim_ref=0.850, config=cfg)
    assert not fail
