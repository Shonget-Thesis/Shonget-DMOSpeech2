"""
Deterministic Handover Sampler for DMOSpeech 2
==============================================
Implements Task 1 of the Chapter 3 Handover-Scheduling Specification:
- Deterministic FP32 Euler trajectory for the teacher model (16-interval sway grid)
- Zero-overhead projected endpoint calculation: x_hat_1(t_i) = x_ti + (1 - t_i) * v
- Trajectory stabilization drift d_i computed exclusively over generated frames [prompt_len:]
- Single-pass teacher execution (steps 0..8) serving candidate steps {4, 5, 6} and reference {8}
- Consistent student restart completion (j = 1, tau in [0.0761, 0.2929, 0.6173])
- Seed isolation using local torch.Generator objects derived from SHA-256 of utterance ID
- Strict forward-pass (NFE) accounting: teacher_nfe = 2*(step+1), student_nfe = 3
"""

import math
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

# ============================================================================
# 1. CANONICAL CONSTANTS AND GRIDS (Task 1.3 - 1.5, 1.7)
# ============================================================================

CFG_STRENGTH: float = 2.0
CANDIDATE_STEPS: Tuple[int, ...] = (4, 5, 6)
REFERENCE_STEP: int = 8
ALL_TARGET_STEPS: Tuple[int, ...] = (4, 5, 6, 8)
EXPANDED_CANDIDATE_STEPS: Tuple[int, ...] = (2, 3, 4, 5, 6)
EXPANDED_TARGET_STEPS: Tuple[int, ...] = (2, 3, 4, 5, 6, 8)
STUDENT_RESTART_J: int = 1

# Teacher Sway Grid (16 intervals swayed with coefficient s = -1):
# t_i = 1 - cos(pi/2 * i/16), i = 0...16
SWAY_TEACHER_GRID: List[float] = [
    float(1.0 - math.cos(math.pi / 2.0 * i / 16.0)) for i in range(17)
]

# Student Sway Grid (4 intervals swayed, j=1 restart -> indices 1, 2, 3):
# s_base = [0.0, 0.25, 0.50, 0.75]
# tau = 1 - cos(pi/2 * s_base)
STUDENT_TAUS: List[float] = [
    float(1.0 - math.cos(math.pi / 2.0 * s)) for s in (0.25, 0.50, 0.75)
]  # [0.07612046748871326, 0.2928932188134524, 0.6173165676349102]

# Canonical step metadata table (Section 1.4, expanded with steps 2 & 3)
STEP_METADATA: Dict[int, Dict[str, Any]] = {
    2: {
        "teacher_step": 2,
        "legacy_label": 0.02,
        "actual_teacher_time": SWAY_TEACHER_GRID[2],
        "switch_time": SWAY_TEACHER_GRID[2],
        "teacher_solver_steps": 3,
        "student_solver_steps": 3,
        "solver_steps": 6,
        "teacher_nfe": 6,
        "student_nfe": 3,
        "nfe": 9,
    },
    3: {
        "teacher_step": 3,
        "legacy_label": 0.04,
        "actual_teacher_time": SWAY_TEACHER_GRID[3],
        "switch_time": SWAY_TEACHER_GRID[3],
        "teacher_solver_steps": 4,
        "student_solver_steps": 3,
        "solver_steps": 7,
        "teacher_nfe": 8,
        "student_nfe": 3,
        "nfe": 11,
    },
    4: {
        "teacher_step": 4,
        "legacy_label": 0.07,
        "actual_teacher_time": SWAY_TEACHER_GRID[4],
        "switch_time": SWAY_TEACHER_GRID[4],
        "teacher_solver_steps": 5,
        "student_solver_steps": 3,
        "solver_steps": 8,
        "teacher_nfe": 10,
        "student_nfe": 3,
        "nfe": 13,
    },
    5: {
        "teacher_step": 5,
        "legacy_label": 0.10,
        "actual_teacher_time": SWAY_TEACHER_GRID[5],
        "switch_time": SWAY_TEACHER_GRID[5],
        "teacher_solver_steps": 6,
        "student_solver_steps": 3,
        "solver_steps": 9,
        "teacher_nfe": 12,
        "student_nfe": 3,
        "nfe": 15,
    },
    6: {
        "teacher_step": 6,
        "legacy_label": 0.15,
        "actual_teacher_time": SWAY_TEACHER_GRID[6],
        "switch_time": SWAY_TEACHER_GRID[6],
        "teacher_solver_steps": 7,
        "student_solver_steps": 3,
        "solver_steps": 10,
        "teacher_nfe": 14,
        "student_nfe": 3,
        "nfe": 17,
    },
    8: {
        "teacher_step": 8,
        "legacy_label": 0.25,
        "actual_teacher_time": SWAY_TEACHER_GRID[8],
        "switch_time": SWAY_TEACHER_GRID[8],
        "teacher_solver_steps": 9,
        "student_solver_steps": 3,
        "solver_steps": 12,
        "teacher_nfe": 18,
        "student_nfe": 3,
        "nfe": 21,
    },
}


# ============================================================================
# 2. STABLE SEED GENERATION (Task 1.2)
# ============================================================================

def derive_input_seed(base_seed: int, utterance_id: str) -> int:
    """
    Derives an integer input seed from base_seed and a stable SHA-256 hash of utterance_id.
    Guarantees cross-platform and cross-run consistency (unlike Python's built-in hash()).
    """
    digest = hashlib.sha256(utterance_id.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return (int(base_seed) + offset) % (2**31 - 1)


def make_local_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """
    Creates an isolated local torch.Generator instance to prevent global RNG mutations.
    """
    dev_str = str(device) if device is not None else "cpu"
    if "cuda" in dev_str and torch.cuda.is_available():
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


# ============================================================================
# 3. COST ACCOUNTING FUNCTIONS (Task 1.7)
# ============================================================================

def get_nfe(step: int) -> int:
    """
    Computes total neural network forward evaluations:
        teacher_nfe(step) = 2 * (step + 1) [due to 2x CFG factor]
        student_nfe = 3
        total_nfe = 2 * (step + 1) + 3
    """
    return 2 * (int(step) + 1) + 3


def get_solver_steps(step: int) -> int:
    """
    Computes ODE/consistency solver steps:
        teacher_solver_steps = step + 1
        student_solver_steps = 3
        total = step + 4
    """
    return (int(step) + 1) + 3


# ============================================================================
# 4. PROMPT SNR AND DRIFT METRICS (Task 1.3, Task 2.1)
# ============================================================================

def estimate_prompt_snr(waveform: np.ndarray, sample_rate: int = 24000, eps: float = 1e-8) -> float:
    """
    Prompt-SNR Estimator (Energy-percentile VAD) matching Section 2.1.
    Computes prompt SNR by contrasting active speech frames (upper 50%)
    against the noise floor (lowest 10%).
    
    Fail-safe: Returns 0.0 dB if fewer than 2 frames exist.
    """
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    frame_length = round(0.025 * sample_rate)  # 25 ms
    hop_length = round(0.010 * sample_rate)    # 10 ms
    num_frames = 1 + (len(waveform) - frame_length) // hop_length
    if num_frames < 2:
        return 0.0

    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(num_frames, frame_length),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0]),
    )
    energy = np.mean(frames**2, axis=-1)
    sorted_energy = np.sort(energy)

    noise_count = max(1, round(0.10 * len(sorted_energy)))
    upper_start = len(sorted_energy) // 2

    noise_energy = np.mean(sorted_energy[:noise_count])
    signal_energy = np.mean(sorted_energy[upper_start:])

    snr = 10.0 * np.log10((signal_energy + eps) / (noise_energy + eps))
    return float(np.clip(snr, 0.0, 50.0))


def compute_stabilization_drift(
    endpoint_curr: torch.Tensor,
    endpoint_prev: torch.Tensor,
    prompt_len: int,
    eps: float = 1e-8,
) -> float:
    """
    Computes trajectory stabilization drift score d_i:
        d_i = || x_hat_1(t_i) - x_hat_1(t_{i-1}) ||_M / ( || x_hat_1(t_i) ||_M + eps )
    Computed STRICTLY over generated frames [prompt_len:] (Task 1.3, Check 5).
    """
    diff = (endpoint_curr - endpoint_prev)[:, prompt_len:, :]
    norm_ref = torch.norm(endpoint_curr[:, prompt_len:, :]) + eps
    return float((torch.norm(diff) / norm_ref).item())


# ============================================================================
# 5. CORE DETERMINISTIC SAMPLER IMPLEMENTATION (Task 1.1, 1.3, 1.5, 1.6)
# ============================================================================

def run_teacher_single_pass(
    tts: Any,
    step_cond: torch.Tensor,
    text: torch.Tensor,
    y0: torch.Tensor,
    prompt_len: int,
    target_steps: Tuple[int, ...] = ALL_TARGET_STEPS,
    cfg_strength: float = CFG_STRENGTH,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float], int]:
    """
    Runs one single teacher Euler integration pass up to max(target_steps) (Step 8).
    
    Zero neural overhead: computes guided velocity v once per step, reusing it for both
    Euler state evolution and projected endpoint calculation x_hat_1(t_i).
    
    Returns:
        endpoints: Dict mapping step index (e.g. 4, 5, 6, 8) to projected endpoint tensor x_hat_1(t_i).
        drift_trace: Dict mapping step index (1..8) to stabilization score d_i.
        teacher_steps_executed: Total teacher Euler steps performed.
    """
    max_step = max(target_steps)
    device = step_cond.device
    state = y0.clone()
    endpoints: Dict[int, torch.Tensor] = {}
    drift_trace: Dict[int, float] = {}
    prev_endpoint: Optional[torch.Tensor] = None
    teacher_steps_executed = 0

    with torch.inference_mode():
        for i in range(max_step + 1):
            t_curr_val = SWAY_TEACHER_GRID[i]
            t_next_val = SWAY_TEACHER_GRID[i + 1]
            t_curr = torch.tensor(t_curr_val, device=device, dtype=torch.float32)

            # Guided teacher forward pass: 2 evaluations for CFG (pred & null_pred)
            pred = tts.teacher(
                x=state,
                cond=step_cond,
                text=text,
                time=t_curr,
                mask=None,
                drop_audio_cond=False,
                drop_text=False,
            )
            null_pred = tts.teacher(
                x=state,
                cond=step_cond,
                text=text,
                time=t_curr,
                mask=None,
                drop_audio_cond=True,
                drop_text=True,
            )
            v_curr = pred + (pred - null_pred) * cfg_strength
            teacher_steps_executed += 1

            # Projected endpoint estimate x_hat_1(t_i) (zero additional forward passes)
            endpoint_curr = state + (1.0 - t_curr_val) * v_curr

            # Stabilization drift d_i (steps 1..8) computed exclusively on [prompt_len:]
            if prev_endpoint is not None:
                d_i = compute_stabilization_drift(endpoint_curr, prev_endpoint, prompt_len)
                drift_trace[i] = d_i

            # Save endpoint if this step is a target candidate
            if i in target_steps:
                endpoints[i] = endpoint_curr.clone()

            # Euler state update: x <- x + (t_{i+1} - t_i) * v
            if i < max_step:
                dt = t_next_val - t_curr_val
                state = state + dt * v_curr
                prev_endpoint = endpoint_curr

    return endpoints, drift_trace, teacher_steps_executed


def run_teacher_separate_step(
    tts: Any,
    step_cond: torch.Tensor,
    text: torch.Tensor,
    y0: torch.Tensor,
    prompt_len: int,
    target_step: int,
    cfg_strength: float = CFG_STRENGTH,
) -> Tuple[torch.Tensor, int]:
    """
    Executes a separate standalone teacher integration specifically up to target_step.
    Used by test_sampler.py to verify that single-pass endpoints are bitwise identical
    to separate isolated runs.
    """
    endpoints, _, steps = run_teacher_single_pass(
        tts=tts,
        step_cond=step_cond,
        text=text,
        y0=y0,
        prompt_len=prompt_len,
        target_steps=(target_step,),
        cfg_strength=cfg_strength,
    )
    return endpoints[target_step], steps


def run_student_completion(
    tts: Any,
    cond: torch.Tensor,
    text: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    prompt_len: int,
    input_seed: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Executes student consistency completion from teacher endpoint x_hat_1(t_k).
    Restart level j = 1 with 3 consistency steps: [0.0761, 0.2929, 0.6173].
    
    Student noise eps is derived using local generator seeded with:
        student_seed = input_seed + int(tau * 100)
    
    Returns:
        mel: Generated mel spectrogram (batch, n_mels, time)
        audio: Decoded 24 kHz waveform as 1D numpy array
    """
    device = cond.device
    x_curr = teacher_endpoint.clone()

    with torch.inference_mode():
        for tau in STUDENT_TAUS:
            student_seed = (int(input_seed) + int(tau * 100)) % (2**31 - 1)
            gen_s = make_local_generator(student_seed, device=device)

            # Draw local seeded Gaussian noise
            eps = torch.randn(
                x_curr.shape,
                dtype=torch.float32,
                device=device,
                generator=gen_s,
            )

            time_tensor = torch.tensor([tau], device=device, dtype=torch.float32)
            t_scalar = time_tensor.unsqueeze(-1).unsqueeze(-1)

            # Denoising interpolation
            phi = (1.0 - t_scalar) * eps + t_scalar * x_curr

            # Student neural forward pass
            pred_s = tts.generator(
                x=phi,
                cond=cond,
                text=text,
                time=time_tensor,
                drop_audio_cond=False,
                drop_text=False,
            )
            x_curr = phi + (1.0 - t_scalar) * pred_s

        # Vocos vocoder synthesis
        scale = getattr(tts, "scale", 1.0)
        mel = x_curr.permute(0, 2, 1) * scale
        generated_wave = tts.vocos.decode(mel[..., prompt_len:])
        audio = generated_wave.cpu().numpy().squeeze()

    return mel, audio


# ============================================================================
# 6. ADAPTIVE DECISION RULE (Objective 2 / Figure 1)
# ============================================================================

def evaluate_handover_policy(
    drift_trace: Dict[int, float],
    prompt_snr: float,
    lambda_threshold: float,
    snr_gate_threshold: float = 20.0,
    candidate_steps: Tuple[int, ...] = CANDIDATE_STEPS,
    reference_step: int = REFERENCE_STEP,
) -> int:
    """
    Evaluates the input-adaptive handover decision rule:
    Returns the earliest candidate step k in candidate_steps where:
        d_k <= lambda_threshold AND prompt_snr >= snr_gate_threshold.
    If no candidate qualifies (or SNR gate fails), returns reference_step (8).
    
    Note: Setting lambda_threshold = -1 forces reference fallback (Check 3).
    """
    if prompt_snr < snr_gate_threshold:
        return reference_step

    for step in candidate_steps:
        d_i = drift_trace.get(step)
        if d_i is not None and d_i <= lambda_threshold:
            return step

    return reference_step
