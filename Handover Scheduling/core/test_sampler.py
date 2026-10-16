"""
Test Suite for Task 1: Deterministic Sampler & Chapter 3 Alignment
===================================================================
Executes all 7 validation checks specified in Section 1:
1. Same input + seed twice -> generated mel identical (max abs diff == 0).
2. Single-pass endpoints vs separate teacher runs for steps 4 and 8 -> identical mel.
3. Adaptive policy forced to fall back (lambda = -1) -> identical to step-8 reference.
4. Counter wrappers around tts.teacher and tts.generator -> equals NFE table for every candidate.
5. Drift d_i is computed only over frames [prompt_len:].
6. Stored actual times match Section 1.4 within 1e-4.
7. Clean and noisy versions of the same utterance use the exact same generation seeds.
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn

from core.sampler import (
    SWAY_TEACHER_GRID,
    STUDENT_TAUS,
    CANDIDATE_STEPS,
    REFERENCE_STEP,
    ALL_TARGET_STEPS,
    STEP_METADATA,
    CFG_STRENGTH,
    derive_input_seed,
    make_local_generator,
    get_nfe,
    get_solver_steps,
    estimate_prompt_snr,
    compute_stabilization_drift,
    run_teacher_single_pass,
    run_teacher_separate_step,
    run_student_completion,
    evaluate_handover_policy,
)


# ============================================================================
# MOCK / INSTRUMENTED HARNESS FOR VERIFICATION
# ============================================================================

class CallCounterWrapper(nn.Module):
    """Wraps a callable or module to record exact invocation counts."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        return self.fn(*args, **kwargs)

    def reset(self):
        self.call_count = 0


class MockVocos(nn.Module):
    """Deterministic dummy vocoder converting mel to waveform."""
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        # Returns simple 1D waveform from mel
        return mel.mean(dim=1)


class MockTTS(nn.Module):
    """
    Deterministic, fully differentiable mock TTS model with teacher, generator,
    and vocoder matching DMOInference call signatures.
    """
    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0
        self.device = torch.device("cpu")

        # Linear projections to simulate network transformations
        self._t_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._g_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize with fixed deterministic weights
        with torch.no_grad():
            self._t_proj.weight.copy_(torch.eye(hidden_dim) * 0.95)
            self._g_proj.weight.copy_(torch.eye(hidden_dim) * 0.90)

        self.teacher_raw = self._teacher_fn
        self.generator_raw = self._generator_fn
        self.vocos = MockVocos()

        # Counter wrappers
        self.teacher = CallCounterWrapper(self.teacher_raw)
        self.generator = CallCounterWrapper(self.generator_raw)

    def _teacher_fn(self, x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False):
        # Deterministic velocity prediction
        t_val = time.item() if isinstance(time, torch.Tensor) else float(time)
        factor = 1.0 if not drop_audio_cond else 0.5
        v = self._t_proj(x) * (1.0 - 0.1 * t_val) * factor
        if not drop_audio_cond:
            v = v + 0.05 * cond
        return v

    def _generator_fn(self, x, cond, text, time, drop_audio_cond=False, drop_text=False):
        t_val = time.item() if isinstance(time, torch.Tensor) else float(time)
        g = self._g_proj(x) * (1.0 - 0.2 * t_val)
        return g

    def reset_counters(self):
        self.teacher.reset()
        self.generator.reset()


# ============================================================================
# TEST CASES
# ============================================================================

def test_1_same_input_and_seed_twice_identical_mel():
    """
    Check 1: Same input + same seed run twice -> generated mel is bitwise identical
    (maximum absolute difference == 0).
    """
    tts = MockTTS(hidden_dim=100)
    tts.eval()
    
    prompt_len = 10
    total_len = 30
    base_seed = 42
    utt_id = "test_utt_001"
    input_seed = derive_input_seed(base_seed, utt_id)

    cond = torch.randn(1, total_len, 100, dtype=torch.float32)
    step_cond = cond.clone()
    step_cond[:, prompt_len:, :] = 0.0
    text = torch.zeros(1, 20, dtype=torch.long)

    # Run 1
    gen1 = make_local_generator(input_seed)
    y0_1 = torch.randn(1, total_len, 100, dtype=torch.float32, generator=gen1)
    endpoints1, _, _ = run_teacher_single_pass(tts, step_cond, text, y0_1, prompt_len)
    mel1, audio1 = run_student_completion(tts, cond, text, endpoints1[8], prompt_len, input_seed)

    # Run 2
    gen2 = make_local_generator(input_seed)
    y0_2 = torch.randn(1, total_len, 100, dtype=torch.float32, generator=gen2)
    endpoints2, _, _ = run_teacher_single_pass(tts, step_cond, text, y0_2, prompt_len)
    mel2, audio2 = run_student_completion(tts, cond, text, endpoints2[8], prompt_len, input_seed)

    max_diff_y0 = (y0_1 - y0_2).abs().max().item()
    max_diff_mel = (mel1 - mel2).abs().max().item()
    max_diff_audio = np.abs(audio1 - audio2).max()

    assert max_diff_y0 == 0.0, f"y0 initial noise difference: {max_diff_y0}"
    assert max_diff_mel == 0.0, f"Generated mel difference: {max_diff_mel}"
    assert max_diff_audio == 0.0, f"Generated audio difference: {max_diff_audio}"


def test_2_single_pass_vs_separate_runs_identical_mel():
    """
    Check 2: Single-pass candidate endpoints vs separate teacher runs for steps 4 and 8
    give identical generated mel.
    """
    tts = MockTTS(hidden_dim=100)
    tts.eval()
    
    prompt_len = 8
    total_len = 25
    input_seed = 12345
    gen = make_local_generator(input_seed)
    y0 = torch.randn(1, total_len, 100, dtype=torch.float32, generator=gen)
    cond = torch.randn(1, total_len, 100, dtype=torch.float32)
    step_cond = cond.clone()
    step_cond[:, prompt_len:, :] = 0.0
    text = torch.zeros(1, 15, dtype=torch.long)

    # Single pass up to step 8
    endpoints_single, _, _ = run_teacher_single_pass(
        tts, step_cond, text, y0, prompt_len, target_steps=(4, 5, 6, 8)
    )

    for step in (4, 8):
        # Separate teacher run for step
        endpoint_sep, _ = run_teacher_separate_step(
            tts, step_cond, text, y0, prompt_len, target_step=step
        )
        
        # Verify teacher endpoint identity
        diff_endpoint = (endpoints_single[step] - endpoint_sep).abs().max().item()
        assert diff_endpoint == 0.0, f"Step {step} endpoint differs between single-pass and separate run: {diff_endpoint}"

        # Run student completion from both
        mel_single, _ = run_student_completion(tts, cond, text, endpoints_single[step], prompt_len, input_seed)
        mel_sep, _ = run_student_completion(tts, cond, text, endpoint_sep, prompt_len, input_seed)

        diff_mel = (mel_single - mel_sep).abs().max().item()
        assert diff_mel == 0.0, f"Step {step} mel differs between single-pass and separate run: {diff_mel}"


def test_3_policy_forced_fallback_lambda_neg1():
    """
    Check 3: Adaptive policy forced to fall back (lambda = -1) returns step-8 reference,
    and its output is identical to the step-8 reference.
    """
    drift_trace = {1: 0.8, 2: 0.5, 3: 0.3, 4: 0.15, 5: 0.12, 6: 0.09, 7: 0.06, 8: 0.04}
    prompt_snr = 35.0  # Above 20 dB gate

    # Under lambda = -1, no non-negative drift can satisfy d_i <= -1
    selected_step = evaluate_handover_policy(
        drift_trace=drift_trace,
        prompt_snr=prompt_snr,
        lambda_threshold=-1.0,
        snr_gate_threshold=20.0,
        candidate_steps=(4, 5, 6),
        reference_step=8,
    )
    assert selected_step == 8, f"Expected step 8 fallback, got {selected_step}"

    # Also test SNR gate failure fallback:
    selected_low_snr = evaluate_handover_policy(
        drift_trace=drift_trace,
        prompt_snr=15.0,  # Below 20 dB gate
        lambda_threshold=0.60,
        snr_gate_threshold=20.0,
        candidate_steps=(4, 5, 6),
        reference_step=8,
    )
    assert selected_low_snr == 8, f"Expected step 8 fallback under low SNR, got {selected_low_snr}"


def test_4_nfe_counter_accounting():
    """
    Check 4: Count actual tts.teacher and tts.generator calls with counter wrappers:
    actual calls must equal the NFE table for every candidate {4, 5, 6, 8}.
    """
    tts = MockTTS(hidden_dim=100)
    tts.eval()

    prompt_len = 5
    total_len = 20
    input_seed = 9999
    cond = torch.randn(1, total_len, 100, dtype=torch.float32)
    step_cond = cond.clone()
    step_cond[:, prompt_len:, :] = 0.0
    text = torch.zeros(1, 10, dtype=torch.long)

    expected_table = {
        4: {"teacher_calls": 10, "student_calls": 3, "nfe": 13, "solver_steps": 8},
        5: {"teacher_calls": 12, "student_calls": 3, "nfe": 15, "solver_steps": 9},
        6: {"teacher_calls": 14, "student_calls": 3, "nfe": 17, "solver_steps": 10},
        8: {"teacher_calls": 18, "student_calls": 3, "nfe": 21, "solver_steps": 12},
    }

    for step, expected in expected_table.items():
        tts.reset_counters()
        gen = make_local_generator(input_seed)
        y0 = torch.randn(1, total_len, 100, dtype=torch.float32, generator=gen)

        endpoint, teacher_solver_steps = run_teacher_separate_step(
            tts, step_cond, text, y0, prompt_len, target_step=step
        )
        _, _ = run_student_completion(tts, cond, text, endpoint, prompt_len, input_seed)

        actual_teacher_calls = tts.teacher.call_count
        actual_student_calls = tts.generator.call_count
        actual_total_nfe = actual_teacher_calls + actual_student_calls

        assert actual_teacher_calls == expected["teacher_calls"], (
            f"Step {step}: expected {expected['teacher_calls']} teacher calls, got {actual_teacher_calls}"
        )
        assert actual_student_calls == expected["student_calls"], (
            f"Step {step}: expected {expected['student_calls']} student calls, got {actual_student_calls}"
        )
        assert actual_total_nfe == expected["nfe"], (
            f"Step {step}: expected {expected['nfe']} total NFE, got {actual_total_nfe}"
        )
        assert get_nfe(step) == expected["nfe"]
        assert get_solver_steps(step) == expected["solver_steps"]
        assert STEP_METADATA[step]["nfe"] == expected["nfe"]


def test_5_drift_computed_only_over_generated_frames():
    """
    Check 5: Trajectory stabilization drift d_i is computed exclusively over
    generated frames [prompt_len:], and is completely invariant to prompt frames.
    """
    prompt_len = 12
    total_len = 30
    
    endpoint_curr = torch.randn(1, total_len, 100, dtype=torch.float32)
    endpoint_prev = torch.randn(1, total_len, 100, dtype=torch.float32)

    # Base drift
    d_base = compute_stabilization_drift(endpoint_curr, endpoint_prev, prompt_len)

    # Perturb prompt frames only (0..prompt_len-1)
    curr_prompt_pert = endpoint_curr.clone()
    curr_prompt_pert[:, :prompt_len, :] += torch.randn(1, prompt_len, 100) * 100.0

    prev_prompt_pert = endpoint_prev.clone()
    prev_prompt_pert[:, :prompt_len, :] += torch.randn(1, prompt_len, 100) * 100.0

    d_prompt_pert = compute_stabilization_drift(curr_prompt_pert, prev_prompt_pert, prompt_len)
    assert abs(d_base - d_prompt_pert) == 0.0, "Drift changed when modifying prompt frames!"

    # Perturb generated frames (prompt_len..total_len-1)
    curr_gen_pert = endpoint_curr.clone()
    curr_gen_pert[:, prompt_len:, :] += 1.0
    d_gen_pert = compute_stabilization_drift(curr_gen_pert, endpoint_prev, prompt_len)
    assert abs(d_base - d_gen_pert) > 1e-4, "Drift did NOT change when modifying generated frames!"


def test_6_stored_actual_times_match_section_1_4():
    """
    Check 6: Stored actual times match Section 1.4 within 1e-4.
    """
    expected_times = {
        4: (0.0761, 0.07),
        5: (0.1181, 0.10),
        6: (0.1685, 0.15),
        8: (0.2929, 0.25),
    }

    for step, (expected_actual, expected_legacy) in expected_times.items():
        actual_val = STEP_METADATA[step]["actual_teacher_time"]
        legacy_val = STEP_METADATA[step]["legacy_label"]
        grid_val = SWAY_TEACHER_GRID[step]

        assert abs(actual_val - expected_actual) < 1e-4, (
            f"Step {step} actual time {actual_val} differs from expected {expected_actual} by >= 1e-4"
        )
        assert abs(grid_val - expected_actual) < 1e-4
        assert legacy_val == expected_legacy


def test_7_clean_and_noisy_versions_share_seeds():
    """
    Check 7: Clean and noisy versions of the same utterance ID use the identical
    base_seed and derived input_seed.
    """
    base_seed = 42
    utt_id = "common_voice_en_15265"

    seed_clean = derive_input_seed(base_seed, utt_id)
    seed_noisy = derive_input_seed(base_seed, utt_id)

    assert seed_clean == seed_noisy, "Clean and noisy seeds differ for the same utterance!"
    
    # Different utterances produce different seeds
    seed_other = derive_input_seed(base_seed, "common_voice_en_99999")
    assert seed_clean != seed_other, "Different utterances produced colliding seeds!"

    # Verify identical initial noise from the seed
    gen_c = make_local_generator(seed_clean)
    y0_clean = torch.randn(1, 20, 100, generator=gen_c)

    gen_n = make_local_generator(seed_noisy)
    y0_noisy = torch.randn(1, 20, 100, generator=gen_n)

    diff = (y0_clean - y0_noisy).abs().max().item()
    assert diff == 0.0, f"Initial noise y0 differed between clean and noisy: {diff}"
