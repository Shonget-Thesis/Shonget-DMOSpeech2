"""
Adaptive Teacher-to-Student Handover Scheduler for DMOSpeech 2
==============================================================
Implementation of the Thesis Figure 1 Architecture:
- Trajectory Stabilization Drift Metric (d_i) with zero probe overhead
- Energy-Percentile Prompt-SNR Safety Gate (theta = 20.0 dB) with fail-safe short-prompt fallback
- Risk-Calibrated Drift Threshold (lambda = 0.60) loaded from frozen config
- Fixed Student Consistency Restart (j = 1, tau = 0.0761, 3 consistency steps)
- Dual Metric Accounting: Solver Steps vs. Actual Forward-Pass NFE (2x CFG factor)
- Conservative Reference Fallback Ceiling (Step 8, s = 0.25)

Directly connects to DMOSpeech 2 (`DMOInference` / `src/infer.py`).
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio

# Add DMOSpeech 2 base directory to path if not present
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from f5_tts.infer.utils_infer import convert_char_to_pinyin, preprocess_ref_audio_text

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spec", "calibrated_handover_config.json")
)


# ============================================================================
# 1. CORE ALGORITHM FUNCTIONS (Figure 1 Mathematical Specification)
# ============================================================================

def estimate_prompt_snr(waveform: np.ndarray, sample_rate: int = 24000, eps: float = 1e-8) -> float:
    """
    Prompt-SNR Estimator (Energy-percentile VAD).
    
    Computes prompt SNR by contrasting the energy of active speech frames
    (upper 50% percentile) against the background noise floor (lowest 10% percentile).
    
    Fail-safe handling: If fewer than 2 frames are available, returns 0.0 dB
    so that ambiguous short prompts safely fail the 20 dB gate and default
    to conservative reference handover.
    
    Returns:
        Estimated SNR in dB (clipped to [0.0, 50.0]).
    """
    frame_length = round(0.025 * sample_rate)  # 25 ms
    hop_length = round(0.010 * sample_rate)    # 10 ms
    num_frames = 1 + (len(waveform) - frame_length) // hop_length
    if num_frames < 2:
        # Conservative fail-safe: insufficient audio frames to verify SNR
        return 0.0

    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(num_frames, frame_length),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0])
    )
    energy = np.mean(frames ** 2, axis=-1)
    sorted_energy = np.sort(energy)
    
    noise_count = max(1, round(0.10 * len(sorted_energy)))
    upper_start = len(sorted_energy) // 2
    
    noise_energy = np.mean(sorted_energy[:noise_count])
    signal_energy = np.mean(sorted_energy[upper_start:])
    
    snr = 10.0 * np.log10((signal_energy + eps) / (noise_energy + eps))
    return float(np.clip(snr, 0.0, 50.0))


def compute_stabilization_drift(endpoint_curr: torch.Tensor, 
                                endpoint_prev: torch.Tensor, 
                                prompt_len: int, 
                                eps: float = 1e-8) -> float:
    """
    Computes the trajectory stabilization score d_i:
        d_i = || x_hat_1(t_i) - x_hat_1(t_{i-1}) ||_M / ( || x_hat_1(t_i) ||_M + eps )
    
    Calculated exclusively over generated (non-prompt) frames selected by mask M.
    """
    diff = (endpoint_curr - endpoint_prev)[:, prompt_len:, :]
    norm_ref = torch.norm(endpoint_curr[:, prompt_len:, :]) + eps
    return float((torch.norm(diff) / norm_ref).item())


# ============================================================================
# 2. ADAPTIVE SCHEDULING CONTROLLER (Figure 1 Decision Rule)
# ============================================================================

class AdaptiveHandoverScheduler:
    """
    Risk-Calibrated Input-Adaptive Handover Controller for DMOSpeech 2.
    Loads calibrated parameters from the frozen cross-validation artifact.
    """
    def __init__(
        self,
        config_path: str = None,
        lambda_threshold: float = None,
        snr_gate_threshold: float = None,
        candidate_steps: dict = None,
        reference_step: int = None,
        student_restart_j: int = None,
        cfg_strength: float = None,
        sway_coefficient: float = None
    ):
        # 1. Attempt to load from frozen calibration config artifact
        cfg = {}
        cfg_file = config_path or DEFAULT_CONFIG_PATH
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file, "r") as f:
                    data = json.load(f)
                    cfg = data.get("parameters", {})
            except Exception as e:
                print(f"[AdaptiveScheduler] Warning: Failed to load config artifact {cfg_file}: {e}")

        # 2. Assign parameters (explicit args > config artifact > canonical defaults)
        self.lambda_threshold = (
            lambda_threshold if lambda_threshold is not None 
            else cfg.get("lambda_threshold", 0.60)
        )
        self.snr_gate_threshold = (
            snr_gate_threshold if snr_gate_threshold is not None 
            else cfg.get("snr_gate_threshold_db", 20.0)
        )
        self.reference_step = (
            reference_step if reference_step is not None 
            else cfg.get("reference_fallback_step", 8)
        )
        self.student_restart_j = (
            student_restart_j if student_restart_j is not None 
            else cfg.get("student_restart_j", 1)
        )
        self.cfg_strength = (
            cfg_strength if cfg_strength is not None 
            else cfg.get("cfg_strength", 2.0)
        )
        self.sway_coefficient = (
            sway_coefficient if sway_coefficient is not None 
            else cfg.get("sway_coefficient", -1.0)
        )

        if candidate_steps is not None:
            self.candidate_steps = candidate_steps
        elif "candidate_steps" in cfg:
            self.candidate_steps = {int(k): float(v) for k, v in cfg["candidate_steps"].items()}
        else:
            self.candidate_steps = {
                4: 0.07,   # Step 4 (t = 0.0761, 5 teacher steps -> 13 actual NFE)
                5: 0.10,   # Step 5 (t = 0.1181, 6 teacher steps -> 15 actual NFE)
                6: 0.15    # Step 6 (t = 0.1685, 7 teacher steps -> 17 actual NFE)
            }

        # Swayed Euler time grid for teacher (16 intervals swayed)
        t_base = torch.linspace(0, 1, 16 + 1)
        self.t_vec = t_base + self.sway_coefficient * (torch.cos(torch.pi / 2 * t_base) - 1 + t_base)

        # Swayed consistency time grid for student (4 intervals swayed)
        s_base = torch.Tensor([0.0, 0.25, 0.50, 0.75])
        self.student_steps = s_base + self.sway_coefficient * (torch.cos(torch.pi / 2 * s_base) - 1 + s_base)


    def run_teacher_euler_loop(self, tts, step_cond, text_ids, y0, prompt_len, prompt_snr, device):
        """
        Executes teacher sampling under adaptive early stopping.
        Zero probe overhead: reuses guided velocity v_curr to estimate endpoint.
        
        Accounting:
            For each Euler step, 2 forward evaluations are executed for CFG:
            (pred: conditional, null_pred: unconditional).
        """
        state = y0
        prev_endpoint = None
        selected_step = self.reference_step
        selected_s = 0.25
        selected_score = None
        selected_endpoint = None
        n_teacher_steps = 0
        trace = []

        # Step through teacher Euler trajectory
        for i in range(len(self.t_vec) - 1):
            t_curr = self.t_vec[i].to(device)
            t_next = self.t_vec[i + 1].to(device)

            # Guided teacher forward pass: 2 evaluations for CFG
            with torch.inference_mode():
                pred = tts.teacher(x=state, cond=step_cond, text=text_ids, time=t_curr, mask=None, drop_audio_cond=False, drop_text=False)
                null_pred = tts.teacher(x=state, cond=step_cond, text=text_ids, time=t_curr, mask=None, drop_audio_cond=True, drop_text=True)
                v_curr = pred + (pred - null_pred) * self.cfg_strength
            n_teacher_steps += 1

            # Deterministic predicted endpoint estimate
            endpoint_curr = state + (1.0 - t_curr) * v_curr

            # Trajectory stabilization drift
            d_i = None
            if prev_endpoint is not None:
                d_i = compute_stabilization_drift(endpoint_curr, prev_endpoint, prompt_len)

            # Decision Rule: checks candidate step, drift <= lambda, and prompt SNR >= theta
            qualifies = (
                i in self.candidate_steps
                and i != self.reference_step
                and d_i is not None
                and d_i <= self.lambda_threshold
                and prompt_snr >= self.snr_gate_threshold
            )

            trace.append({
                "step": i,
                "t": round(float(t_curr), 4),
                "d_i": round(d_i, 4) if d_i is not None else None,
                "qualifies": bool(qualifies)
            })

            # Handover Trigger or Fallback Ceiling
            if qualifies or i == self.reference_step:
                selected_step = i
                selected_s = self.candidate_steps.get(i, 0.25)
                selected_score = d_i
                selected_endpoint = endpoint_curr
                break

            # Euler state transition
            dt = t_next - t_curr
            state = state + dt * v_curr
            prev_endpoint = endpoint_curr.detach()

        return selected_endpoint, selected_step, selected_s, selected_score, n_teacher_steps, trace


    def run_student_completion(self, tts, cond_tensor, text_ids, teacher_endpoint, prompt_len, seed, device):
        """
        Executes student consistency completion from teacher endpoint T.
        Fixed restart level j=1 (executes exactly 3 steps to t=1.0).
        Directly matches official DMOSpeech 2 `_student_sampling` convention (src/infer.py).
        """
        active_student_steps = self.student_steps[self.student_restart_j:]  # [0.0761, 0.2929, 0.6173]
        x_curr = teacher_endpoint

        for st_val in active_student_steps:
            time_tensor = torch.Tensor([st_val]).to(device)
            # Match native DMOSpeech 2 stochastic drawing convention
            x0_s = torch.randn_like(x_curr)
            t_scalar = time_tensor.unsqueeze(-1).unsqueeze(-1)

            # Baseline stochastic re-noising convention at tau
            phi = (1 - t_scalar) * x0_s + t_scalar * x_curr

            with torch.no_grad():
                pred_s = tts.generator(x=phi, cond=cond_tensor, text=text_ids, time=time_tensor, drop_audio_cond=False, drop_text=False)
                output = phi + (1 - t_scalar) * pred_s
            x_curr = output

        # Vocos neural vocoder decode
        mel = x_curr.permute(0, 2, 1) * tts.scale
        with torch.inference_mode():
            gen_audio_torch = tts.vocos.decode(mel[..., prompt_len:])
        
        gen_audio = gen_audio_torch.cpu().numpy().squeeze()
        return gen_audio, len(active_student_steps)


# ============================================================================
# 3. DIRECT DMOSPEECH 2 INFERENCE CONNECTOR
# ============================================================================

def generate_adaptive(
    tts,
    gen_text: str,
    audio_path: str,
    prompt_text: str = None,
    duration: int = None,
    lambda_threshold: float = None,
    snr_gate_threshold: float = None,
    seed: int = 42
):
    """
    Main Entry Point: Generates speech using DMOSpeech 2 with Input-Adaptive Handover.
    
    Args:
        tts: Initialized DMOInference instance.
        gen_text: Text string to synthesize.
        audio_path: File path to reference prompt WAV.
        prompt_text: Transcript of prompt audio (if None, predicted via ASR).
        duration: Target mel length in frames (if None, predicted by duration model).
        lambda_threshold: Calibrated drift threshold (if None, loaded from spec config).
        snr_gate_threshold: Prompt SNR safety gate (if None, loaded from spec config).
        seed: Random seed for reproducibility.
        
    Returns:
        audio (np.ndarray): 24 kHz synthesized waveform.
        telemetry (dict): Decision metadata (selected s, dual NFE/step accounting, drift, SNR, trace).
    """
    t_start = time.time()
    device = tts.device
    
    # 1. Load and preprocess prompt audio
    audio_tensor, sr = torchaudio.load(audio_path)
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    if sr != tts.target_sample_rate:
        audio_tensor = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio_tensor)
    
    # 2. Prompt-SNR Gate Estimate (with fail-safe)
    raw_audio_np = audio_tensor.squeeze().cpu().numpy()
    prompt_snr = estimate_prompt_snr(raw_audio_np, tts.target_sample_rate)

    # 3. Text and Duration Processing
    ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(audio_path, prompt_text)
    if duration is None:
        duration = tts.predict_duration(audio_path, gen_text, ref_text_clean)

    text_list = [ref_text_clean + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)
    ref_audio_len = audio_tensor.shape[-1] // tts.hop_length
    total_duration = ref_audio_len + duration

    cond_tensor, text_ids, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
        audio_tensor.to(device), final_text_list, total_duration
    )
    prompt_len = cond_mask.sum().item()
    # Verified alignment with src/infer.py: cond_mask matches ref_audio_len exactly
    assert prompt_len == ref_audio_len, f"Prompt mask frame mismatch: {prompt_len} vs {ref_audio_len}"

    # 4. Stochastic Initial Noise y0
    torch.manual_seed(seed)
    y0 = [torch.randn(dur, 100, device=device, dtype=step_cond.dtype) for dur in dur_tensor]
    y0 = torch.nn.utils.rnn.pad_sequence(y0, padding_value=0, batch_first=True)

    # 5. Initialize Adaptive Scheduler Controller (loads frozen config)
    scheduler = AdaptiveHandoverScheduler(
        lambda_threshold=lambda_threshold,
        snr_gate_threshold=snr_gate_threshold
    )

    # 6. Execute Teacher Euler Loop with Adaptive Handover
    endpoint, sel_step, sel_s, score, teacher_steps, trace = scheduler.run_teacher_euler_loop(
        tts, step_cond, text_ids, y0, prompt_len, prompt_snr, device
    )

    # 7. Execute Student Consistency Completion (fixed j=1, 3 consistency steps)
    gen_audio, student_steps = scheduler.run_student_completion(
        tts, cond_tensor, text_ids, endpoint, prompt_len, seed, device
    )

    # Rigorous NFE Accounting (Teacher CFG requires 2 evaluations per step)
    total_solver_steps = teacher_steps + student_steps
    actual_teacher_nfe = 2 * teacher_steps
    actual_student_nfe = 1 * student_steps
    actual_total_nfe = actual_teacher_nfe + actual_student_nfe
    elapsed_time = time.time() - t_start

    telemetry = {
        "selected_step": sel_step,
        "selected_s": sel_s,
        "stabilization_score_d_i": score,
        "prompt_snr_db": round(prompt_snr, 2),
        "solver_steps": {
            "teacher_steps": teacher_steps,
            "student_steps": student_steps,
            "total_steps": total_solver_steps
        },
        "actual_nfe": {
            "teacher_nfe": actual_teacher_nfe,
            "student_nfe": actual_student_nfe,
            "total_nfe": actual_total_nfe
        },
        # Direct convenience keys
        "teacher_solver_steps": teacher_steps,
        "student_solver_steps": student_steps,
        "total_solver_steps": total_solver_steps,
        "teacher_nfe": actual_teacher_nfe,
        "student_nfe": actual_student_nfe,
        "total_nfe": actual_total_nfe,
        "latency_sec": round(elapsed_time, 2),
        "trace": trace
    }

    return gen_audio, telemetry


def patch_dmospeech_inference(tts):
    """
    Monkey-patches the DMOSpeech 2 DMOInference instance to add `.generate_adaptive()`.
    
    Usage:
        tts = DMOInference(...)
        patch_dmospeech_inference(tts)
        audio, telemetry = tts.generate_adaptive(gen_text="Hello", audio_path="ref.wav")
    """
    import types
    tts.generate_adaptive = types.MethodType(
        lambda self, gen_text, audio_path, **kwargs: generate_adaptive(self, gen_text, audio_path, **kwargs),
        tts
    )
    return tts
