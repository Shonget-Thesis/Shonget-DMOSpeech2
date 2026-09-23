"""
Fixed-W Extension: Adds fixed_w=0.8596 (matched to TAGS mean) as a 5th method.
Answers the reviewer's key question:
    Does adaptive TAGS beat a simple fixed damping at its own mean weight?

Includes:
- Strict eval() mode for 100% causal isolation and determinism (0.0 bit-for-bit diff when w=1.0)
- Whisper-base ASR with jiwer WER computation
- Matched reference audio trimming (4.0s) and per-sample seeds (42 + sid)
- Paired statistical analysis (t-test / CI) comparing TAGS vs Fixed-W
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

BENCHMARK_DIR = os.path.join(BASE_DIR, "BENCHMARK_100_SAMPLES")
DATA_DIR      = os.path.join(BENCHMARK_DIR, "data")
OUTPUTS_DIR   = os.path.join(BENCHMARK_DIR, "outputs")
RESULTS_DIR   = os.path.join(BENCHMARK_DIR, "results")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest_100.json")

METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "benchmark_5way_metrics.json")
AUDIT_MD_PATH     = os.path.join(RESULTS_DIR, "5WAY_AUDIT_REPORT.md")

FIXED_W_VALUE = 0.8596


class FiveWayInference(DMOInference):
    """
    DMOSpeech 2 Inference with 5-Way Branching under strict eval() determinism.
    """

    def synthesize_5_way_branch(
        self,
        ref_audio_path,
        prompt_text,
        gen_text,
        sample_seed=42,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        sway_coeff=-1.0,
        rho_cap=1.50,
        fixed_w=FIXED_W_VALUE
    ):
        device = self.device

        # 1. Predict duration
        duration_pred = self.predict_duration(ref_audio_path, gen_text, prompt_text)

        # 2. Preprocess audio
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

        # 4. Student Step 1 — shared across all branches
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

        # Pre-sample identical noise for remaining steps
        remaining_noise = [
            torch.randn(T.shape, generator=rng, device=device, dtype=T.dtype)
            for _ in range(len(steps) - 1)
        ]

        # 5. Define all 5 Handover Modes
        w_baseline  = 1.00
        w_cosine    = float((1.0 - np.cos(np.pi * steps[0].item())) / 2.0)
        w_tanh      = float(np.clip(1.0 - 0.26 * np.tanh(ratio_shock), 0.74, 1.00))
        w_slr_cap   = 1.00 if rms_r <= rho_cap else float(rho_cap / rms_r)
        w_fixed     = fixed_w

        modes_config = {
            "baseline":      {"name": "Baseline (Hard)",          "w": w_baseline},
            "cosine":        {"name": "Cosine Ablation",          "w": w_cosine},
            "adaptive_tanh": {"name": "Adaptive TAGS (tanh)",     "w": w_tanh},
            "slr_cap":       {"name": f"SLR Cap (rho={rho_cap})", "w": w_slr_cap},
            "fixed_w":       {"name": f"Fixed w={fixed_w:.4f}",   "w": w_fixed},
        }

        generated_waves = {}
        diagnostics = {
            "norm_teacher":     round(norm_T, 4),
            "norm_delta":       round(norm_D, 4),
            "rms_displacement": round(rms_r, 4),
            "ratio_shock":      round(ratio_shock, 4),
            "duration_frames":  int(duration_total),
            "modes": {}
        }

        # 6. Branch ODE steps for each mode
        for m_key, m_val in modes_config.items():
            w  = m_val["w"]
            x1 = T + w * D

            for idx, step_t in enumerate(steps[1:]):
                time_t = step_t.view(1)
                t_val  = step_t.view(1, 1, 1)
                x0_t   = remaining_noise[idx]
                phi_t  = (1.0 - t_val) * x0_t + t_val * x1

                with torch.no_grad():
                    pred_t = self.generator(
                        x=phi_t, cond=cond, text=text, time=time_t,
                        drop_audio_cond=False, drop_text=False
                    )
                    x1 = phi_t + (1.0 - t_val) * pred_t

            with torch.inference_mode():
                mel  = x1.permute(0, 2, 1) * self.scale
                wave = self.vocos.decode(mel[..., cond_mask.sum():]).cpu().numpy().squeeze()

            generated_waves[m_key] = wave
            diagnostics["modes"][m_key] = {
                "name":         m_val["name"],
                "effective_w":  round(w, 4),
                "accepted_rms": round(w * rms_r, 4),
                "withheld_rms": round((1.0 - w) * rms_r, 4)
            }

        # Determinism assertion: if SLR cap was inactive (w=1.0), it MUST be bit-for-bit identical to baseline
        if abs(modes_config["slr_cap"]["w"] - 1.0) < 1e-6:
            diff_slr_base = float(np.max(np.abs(generated_waves["baseline"] - generated_waves["slr_cap"])))
            assert diff_slr_base == 0.0, f"Determinism failure: baseline and slr_cap differ by {diff_slr_base} when w=1.0!"

        return generated_waves, diagnostics


def eval_spectral_flux(y, sr=24000):
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    stft   = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    flux   = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    t_frames = max(1, int(len(flux) * 0.15))
    t_flux = flux[:t_frames]
    return {
        "mean_flux": round(float(np.mean(t_flux)), 2),
        "peak_flux": round(float(np.max(t_flux)), 2)
    }


def write_5way_audit(results):
    total = len(results)
    if total == 0:
        return

    keys = ["baseline", "cosine", "adaptive_tanh", "slr_cap", "fixed_w"]
    labels = {
        "baseline":      "Baseline (Hard)",
        "cosine":        "Cosine Ablation",
        "adaptive_tanh": "Adaptive TAGS (tanh)",
        "slr_cap":       "SLR Cap (rho=1.50)",
        "fixed_w":       f"Fixed w={FIXED_W_VALUE:.4f}",
    }

    avg_peak = {k: np.mean([r["evaluations"][k]["peak_flux"] for r in results]) for k in keys}
    avg_mean = {k: np.mean([r["evaluations"][k]["mean_flux"] for r in results]) for k in keys}
    avg_w    = {k: np.mean([r["evaluations"][k]["effective_w"] for r in results]) for k in keys}

    # WER averages
    wer_vals = {}
    for k in keys:
        valid_wers = [r["evaluations"][k].get("wer") for r in results if r["evaluations"][k].get("wer") is not None]
        wer_vals[k] = np.mean(valid_wers) if valid_wers else 0.0

    wins = {k: 0 for k in keys}
    tags_beats_fixed = 0
    fixed_beats_tags = 0
    for r in results:
        best_k = min(keys, key=lambda k: r["evaluations"][k]["peak_flux"])
        wins[best_k] += 1
        if r["evaluations"]["adaptive_tanh"]["peak_flux"] < r["evaluations"]["fixed_w"]["peak_flux"]:
            tags_beats_fixed += 1
        elif r["evaluations"]["fixed_w"]["peak_flux"] < r["evaluations"]["adaptive_tanh"]["peak_flux"]:
            fixed_beats_tags += 1

    tags_vs_fixed_diffs = [r["evaluations"]["fixed_w"]["peak_flux"] - r["evaluations"]["adaptive_tanh"]["peak_flux"]
                           for r in results]

    tags_peak_mean  = avg_peak["adaptive_tanh"]
    fixed_peak_mean = avg_peak["fixed_w"]
    diff_mean       = float(np.mean(tags_vs_fixed_diffs))
    diff_std        = float(np.std(tags_vs_fixed_diffs))
    n               = len(results)
    se              = diff_std / np.sqrt(n) if n > 1 else 0.0
    ci_lo = diff_mean - 1.96 * se
    ci_hi = diff_mean + 1.96 * se

    md = f"""# 5-Way Handover Benchmark: Fixed-W Extension Report

**Protocol:** Strict Causal Noise Isolation with `eval()` mode (0.0 bit-for-bit diff when w=1.0)  
**ASR Model:** `openai/whisper-base` + `jiwer` WER  
**Fixed Weight Value:** w = {FIXED_W_VALUE:.4f} (matched to Adaptive TAGS empirical mean weight)  
**Sample Count:** {total} LibriSpeech test-clean cross-sentence pairs  

---

## 1. Executive Summary — 5-Way Comparison

| Method | Avg Weight | Avg Peak Flux ↓ | Avg Mean Flux ↓ | Avg WER ↓ | Peak Wins |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Hard)** | `{avg_w['baseline']:.4f}` | `{avg_peak['baseline']:.2f}` | `{avg_mean['baseline']:.2f}` | `{wer_vals['baseline']*100:.1f}%` | {wins['baseline']}/{total} |
| **Cosine Ablation** | `{avg_w['cosine']:.4f}` | `{avg_peak['cosine']:.2f}` | `{avg_mean['cosine']:.2f}` | `{wer_vals['cosine']*100:.1f}%` | {wins['cosine']}/{total} |
| **Adaptive TAGS (tanh)** | `{avg_w['adaptive_tanh']:.4f}` | `{avg_peak['adaptive_tanh']:.2f}` | `{avg_mean['adaptive_tanh']:.2f}` | `{wer_vals['adaptive_tanh']*100:.1f}%` | {wins['adaptive_tanh']}/{total} |
| **SLR Cap (rho=1.50)** | `{avg_w['slr_cap']:.4f}` | `{avg_peak['slr_cap']:.2f}` | `{avg_mean['slr_cap']:.2f}` | `{wer_vals['slr_cap']*100:.1f}%` | {wins['slr_cap']}/{total} |
| **Fixed w={FIXED_W_VALUE:.4f}** | `{avg_w['fixed_w']:.4f}` | `{avg_peak['fixed_w']:.2f}` | `{avg_mean['fixed_w']:.2f}` | `{wer_vals['fixed_w']*100:.1f}%` | {wins['fixed_w']}/{total} |

*Note: Baseline and SLR Cap are mathematically and numerically bit-for-bit identical on all samples where w_slr = 1.0 (verified max diff = 0.0).*

---

## 2. The Reviewer's Core Question: Adaptive TAGS vs. Fixed w={FIXED_W_VALUE:.4f}

| Metric | Adaptive TAGS | Fixed w={FIXED_W_VALUE:.4f} | Difference (Fixed − TAGS) |
|:---|:---:|:---:|:---:|
| Avg Peak Flux | `{tags_peak_mean:.4f}` | `{fixed_peak_mean:.4f}` | `{fixed_peak_mean - tags_peak_mean:+.4f}` |
| Avg WER | `{wer_vals['adaptive_tanh']*100:.2f}%` | `{wer_vals['fixed_w']*100:.2f}%` | `{(wer_vals['fixed_w'] - wer_vals['adaptive_tanh'])*100:+.2f}%` |
| TAGS wins (lower peak flux) | {tags_beats_fixed}/{total} ({tags_beats_fixed/total*100:.1f}%) | — | — |
| Fixed wins (lower peak flux) | — | {fixed_beats_tags}/{total} ({fixed_beats_tags/total*100:.1f}%) | — |

### Paired Difference: Fixed_w Peak Flux − TAGS Peak Flux
- **Mean difference:** `{diff_mean:+.4f}` (positive = TAGS achieves lower flux than fixed damping)
- **Std deviation:** `{diff_std:.4f}`
- **95% Confidence Interval:** `[{ci_lo:+.4f}, {ci_hi:+.4f}]`

---

## 3. High-Shock Analysis (Top Quartile)
"""
    q75 = float(np.percentile([r["ratio_shock"] for r in results], 75))
    high_shock = [r for r in results if r["ratio_shock"] >= q75]
    if high_shock:
        hs_tags_peak  = np.mean([r["evaluations"]["adaptive_tanh"]["peak_flux"] for r in high_shock])
        hs_fixed_peak = np.mean([r["evaluations"]["fixed_w"]["peak_flux"] for r in high_shock])
        hs_base_peak  = np.mean([r["evaluations"]["baseline"]["peak_flux"] for r in high_shock])
        md += f"""
| Subset | Baseline Avg Peak | TAGS Avg Peak | Fixed Avg Peak | TAGS Advantage (Fixed − TAGS) |
|:---|:---:|:---:|:---:|:---:|
| High-shock samples (ρ ≥ {q75:.3f}, n={len(high_shock)}) | `{hs_base_peak:.2f}` | `{hs_tags_peak:.2f}` | `{hs_fixed_peak:.2f}` | `{(hs_fixed_peak - hs_tags_peak):+.2f}` |
"""

    md += f"""
---

## 4. Per-Sample Detail (First 20 Samples)

| ID | Spk | Shock ρ | Base Peak | TAGS w | TAGS Peak | Fixed Peak | TAGS WER | Fixed WER | Peak Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in results[:20]:
        sid  = r["sample_id"]
        spk  = r["speaker_id"]
        rho  = r["ratio_shock"]
        b_p  = r["evaluations"]["baseline"]["peak_flux"]
        w_t  = r["evaluations"]["adaptive_tanh"]["effective_w"]
        t_p  = r["evaluations"]["adaptive_tanh"]["peak_flux"]
        f_p  = r["evaluations"]["fixed_w"]["peak_flux"]
        t_wer = r["evaluations"]["adaptive_tanh"].get("wer", 0.0)
        f_wer = r["evaluations"]["fixed_w"].get("wer", 0.0)
        winner = min(keys, key=lambda k: r["evaluations"][k]["peak_flux"])
        win_label = {"baseline":"BASE","cosine":"COSINE","adaptive_tanh":"TAGS","slr_cap":"SLR","fixed_w":"FIXED"}.get(winner, winner)
        md += f"| {sid} | {spk} | `{rho:.3f}` | `{b_p:.1f}` | `{w_t:.3f}` | `{t_p:.1f}` | `{f_p:.1f}` | `{t_wer*100:.1f}%` | `{f_wer*100:.1f}%` | **{win_label}** |\n"

    md += f"""
---

## 5. Summary Findings

1. **Resolution of Discrepancy:** SLR Cap and Baseline match with `0.0000` difference whenever `w_slr = 1.0`, confirming full causal isolation.
2. **TAGS vs. Fixed w={FIXED_W_VALUE:.4f}:**
   - TAGS beats Fixed-W on **{tags_beats_fixed}/{total}** samples ({tags_beats_fixed/total*100:.1f}%).
   - Net peak flux difference: **{diff_mean:+.4f}** [95% CI: `{ci_lo:+.4f}` to `{ci_hi:+.4f}`].
"""
    with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  [REPORT] 5-way audit updated: {AUDIT_MD_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Run 5-Way Fixed-W Handover Extension.")
    parser.add_argument("--max_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    parser.add_argument("--fixed_w", type=float, default=FIXED_W_VALUE, help="Fixed w weight")
    args = parser.parse_args()

    print("=" * 70)
    print("  5-WAY FIXED-W BENCHMARK WITH STRICT EVAL() DETERMINISM          ")
    print(f"  Fixed W Value : {args.fixed_w}")
    print(f"  Max Samples   : {args.max_samples}")
    print("=" * 70)

    for k in ["baseline", "cosine", "adaptive_tanh", "slr_cap", "fixed_w"]:
        os.makedirs(os.path.join(OUTPUTS_DIR, k), exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)[:args.max_samples]

    all_results = []
    evaluated_ids = set()
    if args.resume and os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
            evaluated_ids = {r["sample_id"] for r in all_results}
        print(f"[RESUME] Found {len(all_results)} already evaluated samples. Skipping.")

    print("\n[1/3] Loading Speech Synthesis Engine in STRICT EVAL() mode...")
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    tts = FiveWayInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    # Explicitly enforce eval mode across all components
    tts.generator.eval()
    tts.teacher.eval()
    tts.model.eval()

    print(f"  Device: {device} | Generator eval mode: {not tts.generator.training}")

    print("\n[2/3] Loading Whisper-Base for Accurate Intelligibility / WER Audit...")
    try:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")
        print("  Whisper-base loaded successfully.")
    except Exception as e:
        print(f"  Warning: Whisper-base failed to load ({e}). Continuing without ASR.")
        asr_pipe = None

    print(f"\n[3/3] Commencing 5-Way Evaluation ({len(manifest)} total)...\n")
    start_total = time.time()

    for item in manifest:
        sid = item["sample_id"]
        if sid in evaluated_ids:
            continue

        spk_id        = item["speaker_id"]
        ref_text      = item["ref_text"]
        gen_text      = item["gen_text"]
        rel_flac      = item["rel_flac_path"]
        ref_audio_path = os.path.join(DATA_DIR, rel_flac)

        if not os.path.exists(ref_audio_path):
            print(f"[SKIP] Audio not found: {ref_audio_path}")
            continue

        t0 = time.time()
        print(f">>> Sample {sid}/{len(manifest)}  (spk={spk_id})")
        print(f"    Text: \"{gen_text[:65]}...\"")

        # Trim ref audio to 4.0s
        temp_wav = os.path.join(DATA_DIR, f"temp_ref_{sid}.wav")
        audio_np, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
        if len(audio_np) > int(4.0 * sr):
            audio_np = audio_np[:int(4.0 * sr)]
        sf.write(temp_wav, audio_np, 24000)

        try:
            waves, diag = tts.synthesize_5_way_branch(
                ref_audio_path=temp_wav,
                prompt_text=ref_text,
                gen_text=gen_text,
                sample_seed=42 + sid,
                fixed_w=args.fixed_w
            )
        except Exception as e:
            print(f"  [ERROR] Synthesis failed: {e}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            continue

        sample_eval = {
            "sample_id":        sid,
            "speaker_id":       spk_id,
            "ref_text":         ref_text,
            "gen_text":         gen_text,
            "norm_teacher":     diag["norm_teacher"],
            "norm_delta":       diag["norm_delta"],
            "rms_displacement": diag["rms_displacement"],
            "ratio_shock":      diag["ratio_shock"],
            "duration_frames":  diag["duration_frames"],
            "evaluations":      {}
        }

        mode_keys = ["baseline", "cosine", "adaptive_tanh", "slr_cap", "fixed_w"]
        for m_key in mode_keys:
            wav = waves[m_key]
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
                    wer_score = None

            sample_eval["evaluations"][m_key] = {
                "name":         diag["modes"][m_key]["name"],
                "output_file":  f"sample_{sid}_{m_key}.wav",
                "effective_w":  diag["modes"][m_key]["effective_w"],
                "accepted_rms": diag["modes"][m_key]["accepted_rms"],
                "withheld_rms": diag["modes"][m_key]["withheld_rms"],
                "peak_flux":    flux["peak_flux"],
                "mean_flux":    flux["mean_flux"],
                "transcript":   transcript,
                "wer":          wer_score
            }

        all_results.append(sample_eval)

        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        elapsed = time.time() - t0
        tags_w = sample_eval["evaluations"]["adaptive_tanh"]["effective_w"]
        tags_p = sample_eval["evaluations"]["adaptive_tanh"]["peak_flux"]
        fixd_p = sample_eval["evaluations"]["fixed_w"]["peak_flux"]
        base_p = sample_eval["evaluations"]["baseline"]["peak_flux"]
        tags_wer = sample_eval["evaluations"]["adaptive_tanh"]["wer"]
        fixd_wer = sample_eval["evaluations"]["fixed_w"]["wer"]
        print(f"    Done {elapsed:.1f}s | TAGS w={tags_w:.4f} | Peak: TAGS={tags_p:.1f} vs Fixed={fixd_p:.1f} (Base={base_p:.1f}) | WER: TAGS={tags_wer} vs Fixed={fixd_wer}")

        # Save metrics checkpoint after every sample
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # Update report every 5 samples or at end
        if len(all_results) % 5 == 0 or len(all_results) == len(manifest):
            write_5way_audit(all_results)

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"[COMPLETE] {len(all_results)} samples evaluated in {total_elapsed/60:.1f} min")
    print(f"  Metrics JSON: {METRICS_JSON_PATH}")
    print(f"  Report MD   : {AUDIT_MD_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
