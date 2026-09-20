"""
SLR Bounded Handover: DMOSpeech 2 Accelerated Inference Engine
Implements the formal mathematical recommendations from:
'Teacher-Student Handover in Accelerated TTS Inference: A structured evidence review
and mathematical synthesis of switching, blending, and bounded endpoint correction'
(SLR_for_Teacher_Student_Inferencing.pdf, Section 9)

Supports 5 distinct Handover Modes for rigorous thesis ablation:
1. 'slr_cap'    : SLR Section 9 Norm-Bounded Euclidean Projection (lambda = min(1, rho / RMS(D)))
2. 'tanh'       : Bounded Hyperbolic Tangent Cushion (w = 1.0 - 0.26 * tanh(||D|| / ||T||))
3. 'fixed'      : Fixed Handover Relaxation (w = 0.85)
4. 'baseline'   : Default DMOSpeech 2 Hard Cutoff (w = 1.0)
5. 'cosine'     : Static Sway Cosine Ablation (w = 0.0142)
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from infer import DMOInference


class SLRBoundedDMOSpeechInference(DMOInference):
    """
    DMOSpeech 2 with SLR-Recommended Bounded Endpoint Relaxation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Runtime diagnostics cache
        self.last_diagnostics = {}
        self.current_mode = "slr_cap"
        self.current_rho = 1.5
        self.current_noise_seed = None

    def _student_sampling(
        self,
        x1,
        cond,
        text,
        student_start_step,
        verbose,
        sway_coeff=-1,
        mode=None,
        rho=None,
        noise_seed=None
    ):
        """
        Student sampling with configurable handover mechanism and optional noise seeding
        for strict paired comparisons (SLR Section 9.7 & Proposition 4).
        """
        if mode is None:
            mode = getattr(self, "current_mode", "slr_cap")
        if rho is None:
            rho = getattr(self, "current_rho", 1.5)
        if noise_seed is None:
            noise_seed = getattr(self, "current_noise_seed", None)

        steps = torch.Tensor([0, 0.25, 0.5, 0.75])
        steps = steps + sway_coeff * (torch.cos(torch.pi / 2 * steps) - 1 + steps)
        steps = steps[student_start_step:]

        # Preserve the Teacher's anchor state T
        T = x1.clone()

        # Deterministic generator if noise_seed provided
        rng = None
        if noise_seed is not None:
            rng = torch.Generator(device=x1.device)
            rng.manual_seed(noise_seed)

        for idx, step in enumerate(steps):
            time = torch.Tensor([step]).to(x1.device)

            if rng is not None:
                x0 = torch.randn(x1.shape, generator=rng, device=x1.device, dtype=x1.dtype)
            else:
                x0 = torch.randn_like(x1)

            t = time.unsqueeze(-1).unsqueeze(-1)
            phi = (1 - t) * x0 + t * x1

            if verbose:
                print(f"current step: {step}")
            with torch.no_grad():
                pred = self.generator(
                    x=phi,
                    cond=cond,
                    text=text,
                    time=time,
                    drop_audio_cond=False,
                    drop_text=False
                )

                # S is the raw student endpoint proposal
                S = phi + (1 - t) * pred

            if idx == 0:
                # Proposed displacement vector D = S - T
                D = S - T
                
                # Coordinate count & Euclidean L2 norm
                d = D.numel()
                norm_D = torch.norm(D).item()
                norm_T = (torch.norm(T) + 1e-8).item()
                
                # RMS per coordinate: r = ||D|| / sqrt(d) (SLR Eq. 27)
                rms_r = norm_D / np.sqrt(d)
                ratio_shock = norm_D / norm_T

                # Compute handover weight according to mode
                if mode == "slr_cap":
                    # SLR Section 9: Euclidean Ball Projection / Displacement Cap
                    # lambda_rho = 1.0 if r <= rho else rho / r (SLR Eq. 28)
                    lam = 1.0 if rms_r <= rho else float(rho / rms_r)
                    accepted_rms = min(rms_r, rho)
                    withheld_rms = max(0.0, rms_r - rho)
                    method_name = f"SLR Bounded Cap (rho={rho:.2f})"

                elif mode == "tanh":
                    # Adaptive TAGS-Flow: Dynamic tanh cushion
                    lam = float(np.clip(1.0 - 0.26 * np.tanh(ratio_shock), 0.74, 1.00))
                    accepted_rms = lam * rms_r
                    withheld_rms = (1.0 - lam) * rms_r
                    method_name = "Adaptive Bounded tanh"

                elif mode == "fixed":
                    # Fixed first relaxation (SLR Section 8.6, P2)
                    lam = 0.85
                    accepted_rms = lam * rms_r
                    withheld_rms = (1.0 - lam) * rms_r
                    method_name = "Fixed Relaxation (w=0.85)"

                elif mode == "cosine":
                    # Raw sway-time cosine ablation (SLR Section 6.1)
                    # (1 - cos(pi * 0.07612)) / 2 approx 0.01423
                    lam = float((1.0 - np.cos(np.pi * step.item())) / 2.0)
                    accepted_rms = lam * rms_r
                    withheld_rms = (1.0 - lam) * rms_r
                    method_name = "Sway Cosine Ablation"

                elif mode == "baseline":
                    # Default DMOSpeech 2 hard replacement
                    lam = 1.00
                    accepted_rms = rms_r
                    withheld_rms = 0.0
                    method_name = "DMOSpeech 2 Baseline (Hard Switch)"

                else:
                    raise ValueError(f"Unknown handover mode: {mode}")

                # Apply handover: E = T + lambda * (S - T)
                x1 = T + lam * D

                # Cache comprehensive diagnostics
                self.last_diagnostics = {
                    "mode": mode,
                    "method_name": method_name,
                    "norm_teacher": norm_T,
                    "norm_delta": norm_D,
                    "rms_displacement": rms_r,
                    "collision_ratio": ratio_shock,
                    "effective_lambda": lam,
                    "accepted_rms": accepted_rms,
                    "withheld_rms": withheld_rms,
                    "percent_transmitted": lam * 100.0,
                    "percent_damped": (1.0 - lam) * 100.0
                }

                if verbose:
                    print(f"   [{method_name} Handover Applied]")
                    print(f"      Teacher Norm ||T||   : {norm_T:.4f}")
                    print(f"      Delta Norm ||D||     : {norm_D:.4f}")
                    print(f"      RMS Displacement r   : {rms_r:.4f}")
                    print(f"      Effective Weight lam : {lam:.4f} ({lam*100.0:.1f}% accepted)")
                    print(f"      Accepted RMS         : {accepted_rms:.4f}")
                    print(f"      Withheld RMS         : {withheld_rms:.4f}")
            else:
                x1 = S

        return x1

    def generate(
        self,
        gen_text,
        audio_path,
        prompt_text=None,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        student_start_step=1,
        duration=None,
        dp_softmax_range=0.7,
        temperature=0,
        eta=1.0,
        cfg_strength=2.0,
        sway_coefficient=-1.0,
        verbose=False,
        mode="slr_cap",
        rho=1.5,
        noise_seed=None
    ):
        """
        Synthesize speech with specified handover mode and noise seed.
        """
        self.current_mode = mode
        self.current_rho = rho
        self.current_noise_seed = noise_seed

        return super().generate(
            gen_text=gen_text,
            audio_path=audio_path,
            prompt_text=prompt_text,
            teacher_steps=teacher_steps,
            teacher_stopping_time=teacher_stopping_time,
            student_start_step=student_start_step,
            duration=duration,
            dp_softmax_range=dp_softmax_range,
            temperature=temperature,
            eta=eta,
            cfg_strength=cfg_strength,
            sway_coefficient=sway_coefficient,
            verbose=verbose
        )
