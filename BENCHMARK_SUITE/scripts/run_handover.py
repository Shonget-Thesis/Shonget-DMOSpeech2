"""
Unified Modular Inference Script for DMOSpeech 2 Handover Benchmarks
Supports running any handover method cleanly on the canonical 96-sample dataset:
  --method baseline        : Hard Handover (w=1.0)
  --method tags            : Original Adaptive TAGS (w = 1 - 0.26 tanh)
  --method ortho           : Unclamped Directional Orthogonal Relay (w_parallel=1.0, w_perp=0.60)
  --method clamped_ortho   : Selective Energy-Clamped Orthogonal Relay (SEC-Ortho, Proposed)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR   = os.path.dirname(CURRENT_DIR)
BASE_DIR    = os.path.dirname(SUITE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from infer import DMOInference
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text, convert_char_to_pinyin,
    target_rms, target_sample_rate, hop_length
)

DATA_DIR      = os.path.join(BASE_DIR, "BENCHMARK_100_SAMPLES", "data")
MANIFEST_PATH = os.path.join(SUITE_DIR, "data", "manifest_96.json")


class UnifiedHandoverInference(DMOInference):
    def synthesize_handover(
        self,
        ref_audio_path,
        prompt_text,
        gen_text,
        method="clamped_ortho",
        sample_seed=42,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        sway_coeff=-1.0
    ):
        device = self.device

        # 1. Predict duration
        duration_pred = self.predict_duration(ref_audio_path, gen_text, prompt_text)

        # 2. Preprocess audio & text
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, prompt_text)
        audio, sr = torchaudio.load(ref_audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(device)

        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // self.hop_length
        duration_total = ref_audio_len + duration_pred

        with torch.inference_mode():
            cond, text, step_cond, cond_mask, max_duration, duration_tensor = self._prepare_inputs(
                audio, final_text_list, duration_total
            )

        # 3. Deterministic Teacher Sampling
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        with torch.inference_mode():
            T = self._teacher_sampling(
                step_cond=step_cond,
                text=text,
                cond_mask=cond_mask,
                max_duration=max_duration,
                duration=duration_tensor,
                teacher_steps=teacher_steps,
                teacher_stopping_time=teacher_stopping_time,
                eta=1.0,
                cfg_strength=2.0,
                verbose=False,
                sway_sampling_coef=sway_coeff
            )

        # 4. Student Step 1 setup
        steps = torch.Tensor([0, 0.25, 0.5, 0.75])
        steps = steps + sway_coeff * (torch.cos(torch.pi / 2 * steps) - 1 + steps)
        steps = steps[1:].to(device)

        rng = torch.Generator(device=device)
        rng.manual_seed(sample_seed)

        t1 = steps[0].view(1, 1, 1)
        time1 = steps[0].view(1)
        x0_1 = torch.randn(T.shape, generator=rng, device=device, dtype=T.dtype)
        phi_1 = (1.0 - t1) * x0_1 + t1 * T

        with torch.no_grad():
            pred_1 = self.generator(
                x=phi_1, cond=cond, text=text, time=time1,
                drop_audio_cond=False, drop_text=False
            )
            S = phi_1 + (1.0 - t1) * pred_1

        D = S - T
        b_idx = int(cond_mask.sum().item())

        # Remaining noise for steps 2 & 3
        remaining_noise = [
            torch.randn(T.shape, generator=rng, device=device, dtype=T.dtype)
            for _ in range(len(steps) - 1)
        ]

        # 5. Handover Assembly based on method
        if method == "baseline":
            x1 = S.clone()
        elif method == "tags":
            ratio_shock = (torch.norm(D) / (torch.norm(T) + 1e-8)).item()
            w_orig = float(np.clip(1.0 - 0.26 * np.tanh(ratio_shock), 0.74, 1.00))
            x1 = T + w_orig * D
        elif method == "ortho":
            u_S = S / (torch.norm(S) + 1e-8)
            proj_scalar = torch.sum(D * u_S)
            D_parallel = proj_scalar * u_S
            D_ortho = D - D_parallel
            x1 = T + D_parallel + 0.60 * D_ortho
        elif method == "clamped_ortho":
            u_S = S / (torch.norm(S) + 1e-8)
            proj_scalar = torch.sum(D * u_S)
            D_parallel = proj_scalar * u_S
            D_ortho = D - D_parallel
            x1_ortho = T + D_parallel + 0.60 * D_ortho
            norm_S_frame = torch.norm(S, dim=-1, keepdim=True)
            norm_x1_frame = torch.norm(x1_ortho, dim=-1, keepdim=True) + 1e-8
            scale_frame = torch.clamp(norm_S_frame / norm_x1_frame, max=1.0)
            x1 = x1_ortho * scale_frame
        else:
            raise ValueError(f"Unknown handover method: {method}")

        # 6. Execute remaining steps 2 & 3
        curr_x1 = x1.clone()
        for idx, step_t in enumerate(steps[1:]):
            time_t = step_t.view(1)
            t_val = step_t.view(1, 1, 1)
            x0_t = remaining_noise[idx]
            phi_t = (1.0 - t_val) * x0_t + t_val * curr_x1

            with torch.no_grad():
                pred_t = self.generator(
                    x=phi_t, cond=cond, text=text, time=time_t,
                    drop_audio_cond=False, drop_text=False
                )
                curr_x1 = phi_t + (1.0 - t_val) * pred_t

        with torch.inference_mode():
            mel = curr_x1.permute(0, 2, 1) * self.scale
            wave = self.vocos.decode(mel[..., b_idx:]).cpu().numpy().squeeze()

        return wave
