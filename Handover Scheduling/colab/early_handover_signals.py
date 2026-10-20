"""
Early Handover Signal Formulations for DMOSpeech 2
===================================================
Implements the signal formulations specified in VERDICT_COLAB_EARLY_HANDOVER_DEVELOPMENT_TEST:
- Baseline Tangent Drift: d_i = ||E_i - E_{i-1}|| / (||E_i|| + eps)
- Candidate A (Drift Trend): q_A(i) = d_i / (d_{i-1} + eps)
- Candidate B (Local Endpoint Stability): q_B(i) = 90th percentile of local window endpoint relative changes
- Candidate C (Conditioning Disagreement): q_C(i) = ||v_cond - v_uncond|| / (||v_cond|| + ||v_uncond|| + eps)
All calculations are strictly restricted to generated speech frames [prompt_len:].
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


def compute_tangent_drift(
    endpoint_curr: torch.Tensor,
    endpoint_prev: torch.Tensor,
    prompt_len: int,
    eps: float = 1e-8,
) -> float:
    """
    Computes standard normalized tangent drift d_i over generated frames [prompt_len:].
    """
    diff = (endpoint_curr - endpoint_prev)[:, prompt_len:, :]
    norm_ref = torch.norm(endpoint_curr[:, prompt_len:, :]) + eps
    return float((torch.norm(diff) / norm_ref).item())


def compute_candidate_a_drift_trend(
    d_curr: float,
    d_prev: float,
    eps: float = 1e-8,
) -> float:
    """
    Candidate A: Drift Trend
    q_A(i) = d_i / (d_{i-1} + eps)
    Hypothesis: The rate at which endpoint changes shrink may be more informative
    than a single absolute change.
    """
    return float(d_curr / (d_prev + eps))


def compute_candidate_b_local_stability(
    endpoint_curr: torch.Tensor,
    endpoint_prev: torch.Tensor,
    prompt_len: int,
    window_size: int = 16,
    percentile: float = 90.0,
    eps: float = 1e-8,
) -> float:
    """
    Candidate B: Local Endpoint Stability
    Divides generated frames into non-overlapping windows of length window_size.
    For each window k:
        r_{i,k} = ||E_{i,k} - E_{i-1,k}|| / (||E_{i,k}|| + eps)
    Returns q_B(i) = percentile_90({r_{i,k}}).
    """
    gen_curr = endpoint_curr[:, prompt_len:, :]
    gen_prev = endpoint_prev[:, prompt_len:, :]
    total_frames = gen_curr.shape[1]

    if total_frames == 0:
        return 0.0

    window_ratios = []
    # Non-overlapping windows; include remaining final window
    for start_idx in range(0, total_frames, window_size):
        end_idx = min(start_idx + window_size, total_frames)
        w_curr = gen_curr[:, start_idx:end_idx, :]
        w_prev = gen_prev[:, start_idx:end_idx, :]

        diff_w = torch.norm(w_curr - w_prev)
        norm_w = torch.norm(w_curr) + eps
        ratio = float((diff_w / norm_w).item())
        window_ratios.append(ratio)

    if not window_ratios:
        return 0.0

    return float(np.percentile(window_ratios, percentile))


def compute_candidate_c_conditioning_disagreement(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    prompt_len: int,
    eps: float = 1e-8,
) -> float:
    """
    Candidate C: Conditioning Disagreement
    Using pre-guidance conditional and unconditional predictions over generated frames:
        q_C(i) = ||v_cond(i) - v_uncond(i)||_gen / (||v_cond(i)||_gen + ||v_uncond(i)||_gen + eps)
    """
    v_c_gen = v_cond[:, prompt_len:, :]
    v_u_gen = v_uncond[:, prompt_len:, :]

    diff = torch.norm(v_c_gen - v_u_gen)
    norm_c = torch.norm(v_c_gen)
    norm_u = torch.norm(v_u_gen)

    return float((diff / (norm_c + norm_u + eps)).item())


def extract_all_trajectory_signals(
    tts: Any,
    step_cond: torch.Tensor,
    text: torch.Tensor,
    y0: torch.Tensor,
    prompt_len: int,
    target_steps: Tuple[int, ...] = (2, 3, 4, 5, 6, 8),
    cfg_strength: float = 2.0,
    window_size: int = 16,
) -> Tuple[
    Dict[int, torch.Tensor],     # endpoints at target steps
    Dict[int, float],            # d_i (standard drift) for i in 1..max_step
    Dict[int, float],            # q_A (drift trend) for i in 2..max_step
    Dict[int, float],            # q_B (local stability) for i in 1..max_step
    Dict[int, float],            # q_C (conditioning disagreement) for i in 0..max_step
    int                          # teacher steps executed
]:
    """
    Single-pass Euler trajectory execution up to max(target_steps) extracting:
    - Endpoints E_i at target steps
    - Standard drift d_i
    - Candidate A: q_A(i) = d_i / (d_{i-1} + eps)
    - Candidate B: q_B(i) = percentile_90({r_{i,k}})
    - Candidate C: q_C(i) = relative disagreement norm between v_cond and v_uncond
    Requires zero additional teacher forward passes (all computed algebraically from Euler updates).
    """
    from core.sampler import SWAY_TEACHER_GRID

    max_step = max(target_steps)
    device = step_cond.device
    state = y0.clone()

    endpoints: Dict[int, torch.Tensor] = {}
    d_trace: Dict[int, float] = {}
    q_A_trace: Dict[int, float] = {}
    q_B_trace: Dict[int, float] = {}
    q_C_trace: Dict[int, float] = {}

    prev_endpoint: Optional[torch.Tensor] = None
    teacher_steps_executed = 0

    with torch.inference_mode():
        for i in range(max_step + 1):
            t_curr_val = SWAY_TEACHER_GRID[i]
            t_next_val = SWAY_TEACHER_GRID[i + 1]
            t_curr = torch.tensor(t_curr_val, device=device, dtype=torch.float32)

            # Guided teacher forward pass: 2 evaluations for CFG (pred=v_cond & null_pred=v_uncond)
            v_cond = tts.teacher(
                x=state,
                cond=step_cond,
                text=text,
                time=t_curr,
                mask=None,
                drop_audio_cond=False,
                drop_text=False,
            )
            v_uncond = tts.teacher(
                x=state,
                cond=step_cond,
                text=text,
                time=t_curr,
                mask=None,
                drop_audio_cond=True,
                drop_text=True,
            )

            # Guided velocity matching implementation:
            # v_guided = v_cond + cfg_strength * (v_cond - v_uncond)
            v_curr = v_cond + (v_cond - v_uncond) * cfg_strength
            teacher_steps_executed += 1

            # Candidate C: Conditioning disagreement at step i
            q_C_trace[i] = compute_candidate_c_conditioning_disagreement(v_cond, v_uncond, prompt_len)

            # Projected endpoint estimate x_hat_1(t_i)
            endpoint_curr = state + (1.0 - t_curr_val) * v_curr

            if prev_endpoint is not None:
                # Standard drift d_i
                d_i = compute_tangent_drift(endpoint_curr, prev_endpoint, prompt_len)
                d_trace[i] = d_i

                # Candidate B: Local endpoint stability
                q_B_i = compute_candidate_b_local_stability(
                    endpoint_curr, prev_endpoint, prompt_len, window_size=window_size
                )
                q_B_trace[i] = q_B_i

                # Candidate A: Drift trend (requires d_{i-1})
                if (i - 1) in d_trace:
                    d_prev = d_trace[i - 1]
                    q_A_i = compute_candidate_a_drift_trend(d_i, d_prev)
                    q_A_trace[i] = q_A_i

            # Save endpoint if this step is a target candidate
            if i in target_steps:
                endpoints[i] = endpoint_curr.clone()

            # Euler state update: x <- x + (t_{i+1} - t_i) * v
            if i < max_step:
                dt = t_next_val - t_curr_val
                state = state + dt * v_curr
                prev_endpoint = endpoint_curr

    return endpoints, d_trace, q_A_trace, q_B_trace, q_C_trace, teacher_steps_executed
