"""
Benchmark of 3 Novel Handover Approaches vs Baseline and Original TAGS:
1. Approach 1: Directional Orthogonal Decomposition TAGS (damps lateral drift, preserves forward velocity)
2. Approach 2: Boundary-Localized Temporal Window TAGS (cushions the prompt seam B = cond_mask.sum())
3. Approach 3: Calibrated Dynamic-Range TAGS (expanded w in [0.65, 1.00] based on z-score)
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

CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "BENCHMARK_100_SAMPLES", "data")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest_100.json")
OUTPUTS_DIR   = os.path.join(CURRENT_DIR, "outputs")
RESULTS_DIR   = os.path.join(CURRENT_DIR, "results")
METRICS_JSON  = os.path.join(RESULTS_DIR, "novel_approaches_metrics.json")
REPORT_MD     = os.path.join(RESULTS_DIR, "NOVEL_APPROACHES_REPORT.md")


class NovelHandoverInference(DMOInference):
    """
    Evaluates 5 comparative handover modes:
    1. baseline: Hard Handover (w=1.0)
    2. orig_tags: Original TAGS (w = 1 - 0.26 tanh(rho))
    3. approach1_ortho: Orthogonal Directional Decomposition (w_ortho = 0.60)
    4. approach2_boundary: Boundary-Localized Gaussian Window (lambda = 0.40, sigma = 15 frames)
    5. approach3_calibrated: Population Z-score Calibrated Gate (w in [0.65, 1.00])
    """

    def synthesize_novel_branches(
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
        total_time_frames = T.shape[1]

        # -------------------------------------------------------------
        # Define the Initial Handover Tensor x1 for each approach
        # -------------------------------------------------------------
        branches_x1 = {}

        # Mode 1: Baseline (Hard)
        branches_x1["baseline"] = {"x1": T + D, "effective_w": 1.0000, "name": "Baseline (Hard)"}

        # Mode 2: Original TAGS
        w_orig = float(np.clip(1.0 - 0.26 * np.tanh(ratio_shock), 0.74, 1.00))
        branches_x1["orig_tags"] = {"x1": T + w_orig * D, "effective_w": round(w_orig, 4), "name": "Original TAGS"}

        # Mode 3: Approach 1 — Directional Orthogonal Decomposition
        # Decompose D into parallel component (along S direction) and orthogonal component
        norm_S = torch.norm(S) + 1e-8
        u_S = S / norm_S
        proj_scalar = torch.sum(D * u_S)
        D_parallel = proj_scalar * u_S
        D_ortho = D - D_parallel
        w_ortho = 0.60  # Damp lateral drift by 40%, keep 100% of trajectory progression
        x1_ortho = T + D_parallel + w_ortho * D_ortho
        branches_x1["app1_ortho"] = {"x1": x1_ortho, "effective_w": round(w_ortho, 4), "name": "Approach 1 (Orthogonal)"}

        # Mode 4: Approach 2 — Boundary-Localized Temporal TAGS
        # Gaussian window centered at the seam between prompt and generation
        t_coords = torch.arange(total_time_frames, device=device, dtype=T.dtype).view(1, -1, 1)
        sigma = 15.0  # ~160ms transition window
        lambda_seam = 0.40  # 40% damping at the exact seam
        gaussian_window = lambda_seam * torch.exp(-0.5 * ((t_coords - b_idx) / sigma) ** 2)
        w_temporal = 1.0 - gaussian_window  # Shape: (1, T, 1)
        x1_boundary = T + w_temporal * D
        branches_x1["app2_boundary"] = {"x1": x1_boundary, "effective_w": round(float(1.0 - lambda_seam), 4), "name": "Approach 2 (Boundary-Window)"}

        # Mode 5: Approach 3 — Calibrated Dynamic-Range TAGS
        # Z-score normalization of shock ratio: mu=0.60, sigma=0.06
        z_rho = (ratio_shock - 0.60) / 0.06
        w_calibrated = float(0.65 + 0.35 / (1.0 + np.exp(z_rho)))
        branches_x1["app3_calibrated"] = {"x1": T + w_calibrated * D, "effective_w": round(w_calibrated, 4), "name": "Approach 3 (Calibrated-Z)"}

        # -------------------------------------------------------------
        # Branch remaining student steps (steps[1:])
        # -------------------------------------------------------------
        generated_waves = {}
        modes_diag = {}

        for m_key, m_info in branches_x1.items():
            curr_x1 = m_info["x1"].clone()

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

            generated_waves[m_key] = wave
            modes_diag[m_key] = {
                "name": m_info["name"],
                "effective_w": m_info["effective_w"]
            }

        diag = {
            "norm_teacher": round(norm_T, 4),
            "norm_delta": round(norm_D, 4),
            "rms_displacement": round(rms_r, 4),
            "ratio_shock": round(ratio_shock, 4),
            "boundary_frame": b_idx,
            "modes": modes_diag
        }

        return generated_waves, diag


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


def write_novel_report(results):
    total = len(results)
    if total == 0:
        return

    keys = ["baseline", "orig_tags", "app1_ortho", "app2_boundary", "app3_calibrated"]
    labels = {
        "baseline":          "Baseline (Hard)",
        "orig_tags":         "Original TAGS",
        "app1_ortho":        "Approach 1 (Orthogonal Projection)",
        "app2_boundary":     "Approach 2 (Boundary-Localized Window)",
        "app3_calibrated":   "Approach 3 (Calibrated Dynamic Range)"
    }

    avg_peak = {k: float(np.mean([item["evaluations"][k]["peak_flux"] for item in results])) for k in keys}
    avg_mean = {k: float(np.mean([item["evaluations"][k]["mean_flux"] for item in results])) for k in keys}
    avg_wer  = {k: float(np.mean([item["evaluations"][k].get("wer", 0.0) if item["evaluations"][k].get("wer") is not None else 0.0 for item in results])) for k in keys}
    avg_w    = {k: float(np.mean([item["evaluations"][k]["effective_w"] for item in results])) for k in keys}

    wins = {k: 0 for k in keys}
    for r in results:
        best_k = min(keys, key=lambda k: r["evaluations"][k]["peak_flux"])
        wins[best_k] += 1

    md = f"""# Benchmark of 3 Novel Handover Approaches vs. Baseline & Original TAGS

**Dataset:** {total} Cross-Sentence Pairs from LibriSpeech `test-clean`  
**Causal Isolation:** Enforced with `generator.eval()` & `teacher.eval()`  
**ASR Engine:** `openai/whisper-base` + `jiwer`  

---

## 1. Executive Summary Table

| Method | Mean Weight $\\bar{{w}}$ | Peak Flux ↓ | Mean Flux ↓ | WER ↓ | Peak Wins | Δ Peak vs. Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Hard Handover)** | `{avg_w['baseline']:.4f}` | **{avg_peak['baseline']:.2f}** | **{avg_mean['baseline']:.2f}** | **{avg_wer['baseline']*100:.2f}%** | {wins['baseline']}/{total} | — |
| **Original TAGS ($w=1-0.26\\tanh$)** | `{avg_w['orig_tags']:.4f}` | **{avg_peak['orig_tags']:.2f}** | **{avg_mean['orig_tags']:.2f}** | **{avg_wer['orig_tags']*100:.2f}%** | {wins['orig_tags']}/{total} | `{avg_peak['orig_tags'] - avg_peak['baseline']:+.2f}` |
| **Approach 1: Orthogonal Projection** | `{avg_w['app1_ortho']:.4f}` | **{avg_peak['app1_ortho']:.2f}** | **{avg_mean['app1_ortho']:.2f}** | **{avg_wer['app1_ortho']*100:.2f}%** | {wins['app1_ortho']}/{total} | `{avg_peak['app1_ortho'] - avg_peak['baseline']:+.2f}` |
| **Approach 2: Boundary Windowing** | `{avg_w['app2_boundary']:.4f}` | **{avg_peak['app2_boundary']:.2f}** | **{avg_mean['app2_boundary']:.2f}** | **{avg_wer['app2_boundary']*100:.2f}%** | {wins['app2_boundary']}/{total} | `{avg_peak['app2_boundary'] - avg_peak['baseline']:+.2f}` |
| **Approach 3: Calibrated Gate** | `{avg_w['app3_calibrated']:.4f}` | **{avg_peak['app3_calibrated']:.2f}** | **{avg_mean['app3_calibrated']:.2f}** | **{avg_wer['app3_calibrated']*100:.2f}%** | {wins['app3_calibrated']}/{total} | `{avg_peak['app3_calibrated'] - avg_peak['baseline']:+.2f}` |

---

## 2. Per-Sample Detailed Comparison (First 15)

| ID | Shock ρ | Base Peak | Orig TAGS | App 1 (Ortho) | App 2 (Boundary) | App 3 (Calibrated) | Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for r in results[:15]:
        sid = r["sample_id"]
        rho = r["ratio_shock"]
        b_p = r["evaluations"]["baseline"]["peak_flux"]
        o_p = r["evaluations"]["orig_tags"]["peak_flux"]
        a1_p = r["evaluations"]["app1_ortho"]["peak_flux"]
        a2_p = r["evaluations"]["app2_boundary"]["peak_flux"]
        a3_p = r["evaluations"]["app3_calibrated"]["peak_flux"]
        winner = min(keys, key=lambda k: r["evaluations"][k]["peak_flux"])
        win_label = {"baseline": "BASE", "orig_tags": "ORIG", "app1_ortho": "ORTHO", "app2_boundary": "BOUND", "app3_calibrated": "CALIB"}.get(winner, winner)
        md += f"| {sid} | `{rho:.3f}` | `{b_p:.1f}` | `{o_p:.1f}` | `{a1_p:.1f}` | `{a2_p:.1f}` | `{a3_p:.1f}` | **{win_label}** |\n"

    md += f"""
---

## 3. Analysis & Thesis Discussion
- **Approach 1 (Orthogonal Projection):** Damps only the lateral drift while preserving 100% of forward trajectory momentum.
- **Approach 2 (Boundary Windowing):** Selectively targets the reference-to-generation seam ($t_{{boundary}} = cond_mask.sum()$) rather than dampening the whole sequence uniformly.
- **Approach 3 (Calibrated Dynamic Range):** Expands the dynamic adaptation range from $\\pm 0.02$ to $\\pm 0.17$, giving the model real room to react dynamically to high vs. low shock sentences.
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] Written to {REPORT_MD}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3 Novel Handover Approaches.")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing metrics")
    args = parser.parse_args()

    print("=" * 75)
    print("  3 NOVEL HANDOVER APPROACHES BENCHMARK (Separate Directory)       ")
    print(f"  Target Samples: {args.max_samples}")
    print("=" * 75)

    for k in ["baseline", "orig_tags", "app1_ortho", "app2_boundary", "app3_calibrated"]:
        os.makedirs(os.path.join(OUTPUTS_DIR, k), exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)[:args.max_samples]

    all_results = []
    evaluated_ids = set()
    if args.resume and os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)
            evaluated_ids = {r["sample_id"] for r in all_results}
        print(f"[RESUME] Found {len(all_results)} existing samples. Resuming from sample {len(all_results)+1}.")

    print("\n[1/3] Initializing Speech Model in strict EVAL() mode...")
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    tts = NovelHandoverInference(
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

    print(f"\n[3/3] Running Novel Approaches Benchmark ({len(manifest)} samples)...\n")
    start_time = time.time()

    for item in manifest:
        sid = item["sample_id"]
        if sid in evaluated_ids:
            continue

        spk_id = item["speaker_id"]
        ref_text = item["ref_text"]
        gen_text = item["gen_text"]
        rel_flac = item["rel_flac_path"]
        ref_audio_path = os.path.join(DATA_DIR, rel_flac)

        if not os.path.exists(ref_audio_path):
            continue

        t0 = time.time()
        print(f">>> Sample {sid}/{len(manifest)} (spk={spk_id})")

        temp_wav = os.path.join(DATA_DIR, f"temp_novel_{sid}.wav")
        audio_np, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
        if len(audio_np) > int(4.0 * sr):
            audio_np = audio_np[:int(4.0 * sr)]
        sf.write(temp_wav, audio_np, 24000)

        try:
            waves, diag = tts.synthesize_novel_branches(
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

        sample_eval = {
            "sample_id": sid,
            "speaker_id": spk_id,
            "ref_text": ref_text,
            "gen_text": gen_text,
            "norm_teacher": diag["norm_teacher"],
            "norm_delta": diag["norm_delta"],
            "rms_displacement": diag["rms_displacement"],
            "ratio_shock": diag["ratio_shock"],
            "boundary_frame": diag["boundary_frame"],
            "evaluations": {}
        }

        for m_key, wav in waves.items():
            mode_dir = os.path.join(OUTPUTS_DIR, m_key)
            out_file = os.path.join(mode_dir, f"sample_{sid}_{m_key}.wav")
            sf.write(out_file, wav, 24000)

            flux = eval_spectral_flux(wav)
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

            sample_eval["evaluations"][m_key] = {
                "name": diag["modes"][m_key]["name"],
                "effective_w": diag["modes"][m_key]["effective_w"],
                "peak_flux": flux["peak_flux"],
                "mean_flux": flux["mean_flux"],
                "transcript": transcript,
                "wer": wer_score
            }

        all_results.append(sample_eval)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        elapsed = time.time() - t0
        b_p = sample_eval["evaluations"]["baseline"]["peak_flux"]
        o_p = sample_eval["evaluations"]["orig_tags"]["peak_flux"]
        a1_p = sample_eval["evaluations"]["app1_ortho"]["peak_flux"]
        a2_p = sample_eval["evaluations"]["app2_boundary"]["peak_flux"]
        a3_p = sample_eval["evaluations"]["app3_calibrated"]["peak_flux"]

        print(f"    Done {elapsed:.1f}s | Base={b_p:.1f} | Orig={o_p:.1f} | Ortho={a1_p:.1f} | Bound={a2_p:.1f} | Calib={a3_p:.1f}")

        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        write_novel_report(all_results)

    total_time = time.time() - start_time
    print(f"\n[COMPLETE] {len(all_results)} samples evaluated in {total_time/60:.1f} min.")


if __name__ == "__main__":
    main()
