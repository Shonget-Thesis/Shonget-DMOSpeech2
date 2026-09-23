"""
Benchmark Approach 4: Pure Directional Tangent Projection (Parameter-Free)
x1 = T + D_parallel
where D_parallel = ((D . S) / ||S||^2) * S
Zero hyperparameters, zero tuned weights, zero heuristic ranges.
Evaluates on all 96 LibriSpeech test-clean samples without touching existing branches.
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
import jiwer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from transformers import pipeline

CURRENT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "BENCHMARK_100_SAMPLES", "data")
MANIFEST_PATH     = os.path.join(DATA_DIR, "manifest_100.json")
OUTPUTS_DIR       = os.path.join(CURRENT_DIR, "outputs", "app4_pure_tangent")
RESULTS_DIR       = os.path.join(CURRENT_DIR, "results")
METRICS_JSON      = os.path.join(RESULTS_DIR, "novel_approaches_metrics.json")
ROOT_METRICS_JSON = os.path.join(CURRENT_DIR, "novel_approaches_metrics.json")


class PureTangentInference(DMOInference):
    def synthesize_pure_tangent(
        self,
        ref_audio_path,
        prompt_text,
        gen_text,
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

        # Displacement metrics
        D = S - T
        d = D.numel()
        norm_D = torch.norm(D).item()
        norm_T = (torch.norm(T) + 1e-8).item()
        rms_r = norm_D / np.sqrt(d)
        ratio_shock = norm_D / norm_T

        # Remaining noise for steps 2 & 3
        remaining_noise = [
            torch.randn(T.shape, generator=rng, device=device, dtype=T.dtype)
            for _ in range(len(steps) - 1)
        ]

        # Boundary frame index (where reference prompt ends)
        b_idx = int(cond_mask.sum().item())

        # -------------------------------------------------------------
        # Approach 4: Pure Directional Tangent Projection (Parameter-Free)
        # x1 = T + D_parallel
        # D_parallel = ((D . S) / ||S||^2) * S
        # -------------------------------------------------------------
        norm_S = torch.norm(S) + 1e-8
        u_S = S / norm_S
        proj_scalar = torch.sum(D * u_S)
        D_parallel = proj_scalar * u_S
        curr_x1 = (T + D_parallel).clone()

        # Branch remaining student steps (steps[1:])
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

        diag = {
            "norm_teacher": round(norm_T, 4),
            "norm_delta": round(norm_D, 4),
            "rms_displacement": round(rms_r, 4),
            "ratio_shock": round(ratio_shock, 4),
            "boundary_frame": b_idx,
            "proj_scalar": round(proj_scalar.item(), 4),
            "norm_S": round(norm_S.item(), 4)
        }

        return wave, diag


def eval_spectral_flux(y, sr=24000):
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    t_frames = max(1, int(len(flux) * 0.15))
    t_flux = flux[:t_frames]
    return {
        "mean_flux": round(float(np.mean(t_flux)), 2),
        "peak_flux": round(float(np.max(t_flux)), 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Pure Directional Tangent Benchmark")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to evaluate")
    args = parser.parse_args()

    print("=" * 80)
    print("  APPROACH 4: PURE DIRECTIONAL TANGENT PROJECTION BENCHMARK (x1 = T + D_parallel)")
    print("  Zero Parameters | Parameter-Free Formulation | Strict Causal Isolation")
    print(f"  Target Sample Count: {args.max_samples}")
    print("=" * 80)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(METRICS_JSON):
        print(f"[ERROR] Existing metrics file not found: {METRICS_JSON}")
        sys.exit(1)

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        master_results = json.load(f)

    # Index existing results by sample_id
    results_map = {item["sample_id"]: item for item in master_results}
    print(f"[SETUP] Loaded {len(master_results)} existing samples from novel_approaches_metrics.json.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)[:args.max_samples]

    print("\n[1/3] Initializing Speech Model in strict EVAL() mode...")
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    tts = PureTangentInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    tts.generator.eval()
    tts.teacher.eval()
    tts.model.eval()

    print("\n[2/3] Loading Whisper-Base for transcription & WER...")
    try:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")
    except Exception as e:
        print(f"Warning: Whisper failed to load ({e}). Continuing without ASR.")
        asr_pipe = None

    print(f"\n[3/3] Running Pure Tangent Benchmark on eligible samples...\n")
    start_time = time.time()
    evaluated_count = 0

    for item in manifest:
        sid = item["sample_id"]
        if sid not in results_map:
            # Skip if not part of the active 96-sample evaluated dataset
            continue

        sample_record = results_map[sid]
        if "app4_pure_tangent" in sample_record["evaluations"]:
            # Already evaluated
            continue

        spk_id = item["speaker_id"]
        ref_text = item["ref_text"]
        gen_text = item["gen_text"]
        rel_flac = item["rel_flac_path"]
        ref_audio_path = os.path.join(DATA_DIR, rel_flac)

        if not os.path.exists(ref_audio_path):
            continue

        t0 = time.time()
        temp_wav = os.path.join(DATA_DIR, f"temp_pure_tangent_{sid}.wav")
        audio_np, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
        if len(audio_np) > int(4.0 * sr):
            audio_np = audio_np[:int(4.0 * sr)]
        sf.write(temp_wav, audio_np, 24000)

        try:
            wave, diag = tts.synthesize_pure_tangent(
                ref_audio_path=temp_wav,
                prompt_text=ref_text,
                gen_text=gen_text,
                sample_seed=42 + sid
            )
        except Exception as e:
            print(f"  [ERROR] Sample {sid} failed: {e}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            continue

        out_file = os.path.join(OUTPUTS_DIR, f"sample_{sid}_app4_tangent.wav")
        sf.write(out_file, wave, 24000)

        flux = eval_spectral_flux(wave)
        transcript = ""
        wer_score = None
        if asr_pipe is not None:
            try:
                res = asr_pipe(out_file)
                transcript = res.get("text", "").strip()
                if transcript:
                    wer_score = round(float(jiwer.wer(gen_text.lower(), transcript.lower())), 4)
            except Exception:
                transcript = "N/A"

        sample_record["evaluations"]["app4_pure_tangent"] = {
            "name": "Approach 4 (Pure Tangent)",
            "effective_w": 1.0,
            "peak_flux": flux["peak_flux"],
            "mean_flux": flux["mean_flux"],
            "transcript": transcript,
            "wer": wer_score
        }

        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        evaluated_count += 1
        elapsed = time.time() - t0

        b_p = sample_record["evaluations"]["baseline"]["peak_flux"]
        o_p = sample_record["evaluations"]["app1_ortho"]["peak_flux"]
        t_p = flux["peak_flux"]
        print(f"  Sample {sid:2d}/96 ({spk_id}) in {elapsed:.1f}s | Tangent={t_p:.1f} | Ortho(0.60)={o_p:.1f} | Base={b_p:.1f} | WER={wer_score}")

        # Incremental save after each sample
        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(master_results, f, indent=2)
        with open(ROOT_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(master_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n[COMPLETE] Evaluated {evaluated_count} samples for Approach 4 in {total_time/60:.1f} min.")


if __name__ == "__main__":
    main()
