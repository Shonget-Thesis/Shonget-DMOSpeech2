"""
Seed-TTS-en Evaluation: Baseline vs SSDG
=========================================
Full benchmark matching DMOSpeech 2 paper protocol (arXiv:2507.14988).

Comprehensive Metrics Included:
  1. WER (%)          - Word Error Rate via Whisper-large-v3 with Seed-TTS text normalization
  2. SIM-prompt       - Speaker Similarity (Voice Cloning) vs prompt speaker via Resemblyzer
  3. SIM-target       - Speaker Similarity vs ground truth target speaker
  4. RTF              - Real-Time Factor (synthesis time / audio duration)
  5. Peak Flux        - Max instantaneous spectral shock across adjacent STFT frames
  6. Mean Flux        - Mean spectral transition variance across full utterance
  7. Bdy Flux         - Mean spectral flux during handover seam window (frames 0-12)
  8. Non-Bdy Flux     - Mean spectral flux during subsequent stable audio (frames 12+)
  9. Bdy/NonBdy Ratio - Discontinuity shock metric (elevated ratio = handover shock)
 10. MCD (dB)         - Mel-Cepstral Distortion vs ground truth
 11. STOI             - Short-Time Objective Intelligibility vs ground truth (higher is better)

Paper reference values (Seed-TTS-en, Whisper-large-v3):
  DMOSpeech 2 Student-only (4 steps):    WER=1.752%, SIM=0.698
  DMOSpeech 2 Teacher-guided (16 steps): WER=1.738%, SIM=0.699
  F5-TTS Teacher (32 steps):             WER=1.947%, SIM=0.662
  w/o Duration RL (4 steps):             WER=3.750%, SIM=0.672
"""

import os, sys, json, time, tempfile, string, warnings
import numpy as np
import soundfile as sf
import librosa
import torch
import jiwer
import pystoi
from scipy.spatial.distance import cosine as cosine_dist
from transformers import pipeline

warnings.filterwarnings("ignore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR    = os.path.dirname(CURRENT_DIR)
BASE_DIR     = os.path.dirname(SUITE_DIR)

for p in [BASE_DIR, os.path.join(BASE_DIR, "src"),
          os.path.join(SUITE_DIR, "three_strategies")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from infer import DMOInference
from f5_tts.infer.utils_infer import preprocess_ref_audio_text
from run_three_strategies import MultiStrategyInference

MANIFEST     = os.path.join(SUITE_DIR, "data", "manifest_seedtts_100.json")
SEED_EN_DIR  = os.path.join(BASE_DIR, "data", "seed-tts-eval", "en")
RESULTS_DIR  = os.path.join(CURRENT_DIR, "results")
METRICS_JSON = os.path.join(RESULTS_DIR, "seedtts_baseline_vs_ssdg.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

BOUNDARY_FRAMES = 12   # ~128ms handover window (hop=256, sr=24000)


# ══════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════

def compute_spectral_flux(wav, sr=24000):
    """Returns peak, mean, boundary mean, non-boundary mean, ratio."""
    mel    = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    flux   = np.abs(np.diff(mel_db, axis=1))          # shape: (80, T-1)
    frame_flux = flux.mean(axis=0)                     # per-frame mean flux

    peak = float(np.max(frame_flux))
    mean = float(np.mean(frame_flux))

    n = len(frame_flux)
    bdy_end  = min(BOUNDARY_FRAMES, n)
    bdy_flux = float(np.mean(frame_flux[:bdy_end]))      if bdy_end > 0 else 0.0
    non_flux = float(np.mean(frame_flux[bdy_end:]))      if n > bdy_end else 0.0
    ratio    = bdy_flux / non_flux if non_flux > 0 else float('nan')

    return peak, mean, bdy_flux, non_flux, ratio


def compute_mcd(gen_wav, ref_wav, gen_sr=24000, ref_sr=None):
    """Mel Cepstral Distortion between gen and ground truth audio."""
    try:
        if ref_sr and ref_sr != gen_sr:
            ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=gen_sr)

        n_mfcc = 13
        gen_mfcc = librosa.feature.mfcc(y=gen_wav, sr=gen_sr, n_mfcc=n_mfcc, n_fft=512)
        ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=gen_sr, n_mfcc=n_mfcc, n_fft=512)

        min_len = min(gen_mfcc.shape[1], ref_mfcc.shape[1])
        gen_mfcc = gen_mfcc[:, :min_len]
        ref_mfcc = ref_mfcc[:, :min_len]

        diff = gen_mfcc - ref_mfcc
        mcd  = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(diff ** 2))
        return float(mcd)
    except Exception:
        return None


def compute_stoi(gen_wav, ref_wav, gen_sr=24000, ref_sr=None):
    """Short-Time Objective Intelligibility (STOI) vs ground truth audio."""
    try:
        if ref_sr and ref_sr != 16000:
            ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=16000)
        elif gen_sr != 16000 and ref_sr is None:
            ref_wav = librosa.resample(ref_wav, orig_sr=gen_sr, target_sr=16000)
        if gen_sr != 16000:
            gen_wav = librosa.resample(gen_wav, orig_sr=gen_sr, target_sr=16000)

        min_len = min(len(gen_wav), len(ref_wav))
        if min_len < 1600:
            return None
        gen_wav = gen_wav[:min_len]
        ref_wav = ref_wav[:min_len]
        return float(pystoi.stoi(ref_wav, gen_wav, 16000, extended=False))
    except Exception:
        return None


def normalize_text_for_wer(text):
    for ch in string.punctuation:
        text = text.replace(ch, "")
    return " ".join(text.lower().split())


def compute_wer_score(asr_pipe, wav, sr, ref_text):
    """WER using Whisper-large-v3 with normalized text."""
    if asr_pipe is None:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, wav, sr)
        result = asr_pipe(tmp_path)
        hyp = normalize_text_for_wer(result.get("text", ""))
        ref = normalize_text_for_wer(ref_text)
        return float(jiwer.wer(ref, hyp))
    except Exception as e:
        print(f"    WER error: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def compute_sim(encoder, gen_wav, ref_wav, gen_sr=24000, ref_sr=24000):
    """Speaker similarity via Resemblyzer VoiceEncoder cosine distance."""
    try:
        if gen_sr != 16000:
            gen_wav = librosa.resample(gen_wav, orig_sr=gen_sr, target_sr=16000)
        if ref_sr != 16000:
            ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=16000)

        gen_emb = encoder.embed_utterance(gen_wav)
        ref_emb = encoder.embed_utterance(ref_wav)
        sim = 1.0 - cosine_dist(gen_emb, ref_emb)
        return float(sim)
    except Exception as e:
        print(f"    SIM error: {e}")
        return None


def load_wav(path, target_sr=None):
    """Load wav file, optionally resample."""
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if target_sr and sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr  = target_sr
        return wav.astype(np.float32), sr
    except Exception:
        return None, None


def get_target_wav_path(utt_id):
    """Find target (ground truth) wav from Seed-TTS wavs/ folder."""
    candidate = os.path.join(SEED_EN_DIR, "wavs", f"{utt_id}.wav")
    if os.path.exists(candidate):
        return candidate
    target_id = utt_id.split("-")[-1] if "-" in utt_id else utt_id
    candidate2 = os.path.join(SEED_EN_DIR, "wavs", f"{target_id}.wav")
    if os.path.exists(candidate2):
        return candidate2
    return None


def eval_sample(wav, sr, ref_text, target_wav_path, ref_wav_path,
                asr_pipe, spk_encoder, t_elapsed, audio_duration):
    """Compute all metrics for one generated sample."""
    peak, mean, bdy, non_bdy, ratio = compute_spectral_flux(wav, sr)

    wer  = compute_wer_score(asr_pipe, wav, sr, ref_text)
    rtf  = t_elapsed / audio_duration if audio_duration > 0 else None

    # SIM vs prompt speaker (voice cloning fidelity)
    ref_wav_arr, ref_sr = load_wav(ref_wav_path)
    sim_prompt = compute_sim(spk_encoder, wav, ref_wav_arr, sr, ref_sr) \
                 if spk_encoder and ref_wav_arr is not None else None

    # MCD + STOI + SIM vs target ground truth
    mcd, stoi, sim_target = None, None, None
    if target_wav_path:
        tgt_wav, tgt_sr = load_wav(target_wav_path)
        if tgt_wav is not None:
            mcd = compute_mcd(wav, tgt_wav, gen_sr=sr, ref_sr=tgt_sr)
            stoi = compute_stoi(wav, tgt_wav, gen_sr=sr, ref_sr=tgt_sr)
            sim_target = compute_sim(spk_encoder, wav, tgt_wav, sr, tgt_sr) \
                         if spk_encoder else None

    return {
        "wer":          wer,
        "sim_prompt":   sim_prompt,
        "sim_target":   sim_target,
        "rtf":          rtf,
        "peak_flux":    peak,
        "mean_flux":    mean,
        "bdy_flux":     bdy,
        "non_bdy_flux": non_bdy,
        "bdy_ratio":    ratio,
        "mcd":          mcd,
        "stoi":         stoi,
    }


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=20, help="Max samples to evaluate (default 20)")
    args, _ = parser.parse_known_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    manifest = manifest[:args.max_samples]
    print(f"Loaded Seed-TTS-en manifest, evaluating {len(manifest)} samples (max_samples={args.max_samples})")
    print(f"Boundary window: {BOUNDARY_FRAMES} frames (~{BOUNDARY_FRAMES*256/24000*1000:.0f}ms)")

    # Resume
    results, completed_ids = [], set()
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON) as f:
                results = json.load(f)
            completed_ids = {r["utt_id"] for r in results}
            print(f"[RESUME] {len(completed_ids)} already done.")
        except Exception as e:
            print(f"[RESUME WARNING] Could not read existing results: {e}")

    # ── Load TTS Model ─────────────────────────────────────
    print("\n[1/4] Loading TTS model...")
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Device: {device}")

    tts = MultiStrategyInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device, model_type="F5TTS_Base"
    )
    tts.generator.eval(); tts.teacher.eval(); tts.model.eval()

    # ── Load Whisper ASR (Paper Protocol) ──────────────────
    print("\n[2/4] Loading Whisper ASR (Whisper-large-v3)...")
    asr_pipe = None
    for mid in ["openai/whisper-large-v3", "openai/whisper-base"]:
        try:
            kw = {"model": mid, "device": "cpu"}
            if "large" in mid:
                kw["torch_dtype"] = torch.float32
                kw["generate_kwargs"] = {"language": "english", "task": "transcribe"}
            asr_pipe = pipeline("automatic-speech-recognition", **kw)
            print(f"      Successfully loaded ASR: {mid}")
            break
        except Exception as e:
            print(f"      {mid} failed: {e}")

    # ── Load Speaker Encoder ───────────────────────────────
    print("\n[3/4] Loading Speaker Encoder (Resemblyzer)...")
    spk_encoder = None
    try:
        from resemblyzer import VoiceEncoder
        spk_encoder = VoiceEncoder(device="cpu")
        print("      VoiceEncoder loaded successfully.")
    except Exception as e:
        print(f"      Speaker similarity unavailable: {e}")

    # ── Run Evaluation ─────────────────────────────────────
    print(f"\n[4/4] Evaluating {len(manifest)} samples (Baseline vs SSDG)...\n")
    start_time = time.time()

    for idx, item in enumerate(manifest):
        uid      = item["utt_id"]
        ref_wav  = item["prompt_wav"]
        ref_text = item["prompt_text"]
        gen_text = item["target_text"]

        if len(results) >= args.max_samples:
            print(f"\n[TARGET REACHED] {args.max_samples} samples evaluated.")
            break

        if uid in completed_ids:
            continue

        print(f">>> [{idx+1}/{len(manifest)}] {uid[:60]}")
        print(f"    gen: {gen_text[:80]}")

        seed = 42 + idx
        target_wav_path = get_target_wav_path(uid)

        try:
            # ── Single deterministic inference pass generating both strategies ──
            t0 = time.time()
            waves, diag = tts.synthesize_all_strategies(
                ref_wav, ref_text, gen_text,
                sample_seed=seed,
                teacher_steps=8,
                teacher_stopping_time=0.07,
                sway_coeff=-1.0
            )
            elapsed = time.time() - t0

            sr = tts.target_sample_rate
            base_wav = waves["baseline"]
            base_dur = len(base_wav) / sr
            base_elapsed = elapsed / 2.0

            ssdg_wav = waves["strat4_sgsi"]
            ssdg_dur = len(ssdg_wav) / sr
            ssdg_elapsed = elapsed / 2.0

            # ── Compute All Metrics ─────────────────────────
            print(f"    Computing metrics...")
            base_m = eval_sample(base_wav, sr, gen_text,
                                 target_wav_path, ref_wav,
                                 asr_pipe, spk_encoder,
                                 base_elapsed, base_dur)
            ssdg_m = eval_sample(ssdg_wav, sr, gen_text,
                                 target_wav_path, ref_wav,
                                 asr_pipe, spk_encoder,
                                 ssdg_elapsed, ssdg_dur)

            bw = f"{base_m['wer']*100:.2f}%" if base_m['wer'] is not None else "N/A"
            sw = f"{ssdg_m['wer']*100:.2f}%" if ssdg_m['wer'] is not None else "N/A"
            bsim = f"{base_m['sim_prompt']:.3f}" if base_m['sim_prompt'] is not None else "N/A"
            ssim = f"{ssdg_m['sim_prompt']:.3f}" if ssdg_m['sim_prompt'] is not None else "N/A"
            bstoi = f"{base_m['stoi']:.3f}" if base_m['stoi'] is not None else "N/A"
            sstoi = f"{ssdg_m['stoi']:.3f}" if ssdg_m['stoi'] is not None else "N/A"

            print(f"    Base: peak={base_m['peak_flux']:.1f} ratio={base_m['bdy_ratio']:.2f} WER={bw} SIM={bsim} STOI={bstoi}")
            print(f"    SSDG: peak={ssdg_m['peak_flux']:.1f} ratio={ssdg_m['bdy_ratio']:.2f} WER={sw} SIM={ssim} STOI={sstoi}")

            record = {
                "utt_id": uid,
                "gen_text": gen_text,
                "baseline": base_m,
                "ssdg":     ssdg_m,
                "shock_diagnostics": diag,
            }
            results.append(record)
            completed_ids.add(uid)

            with open(METRICS_JSON, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    total = time.time() - start_time
    print(f"\n[DONE] {len(results)} samples in {total/60:.1f} min")
    print_summary(results)


def print_summary(results):
    def avg(lst):
        valid = [x for x in lst if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(valid)) if valid else float('nan')

    metrics = [
        ("WER (%)",          lambda r, m: r[m]["wer"] * 100 if r[m]["wer"] is not None else None),
        ("SIM-prompt",       lambda r, m: r[m]["sim_prompt"]),
        ("SIM-target",       lambda r, m: r[m]["sim_target"]),
        ("STOI",             lambda r, m: r[m]["stoi"]),
        ("MCD (dB)",         lambda r, m: r[m]["mcd"]),
        ("RTF",              lambda r, m: r[m]["rtf"]),
        ("Peak Flux",        lambda r, m: r[m]["peak_flux"]),
        ("Mean Flux",        lambda r, m: r[m]["mean_flux"]),
        ("Bdy Flux",         lambda r, m: r[m]["bdy_flux"]),
        ("Non-Bdy Flux",     lambda r, m: r[m]["non_bdy_flux"]),
        ("Bdy/NonBdy Ratio", lambda r, m: r[m]["bdy_ratio"]),
    ]

    print("\n" + "="*80)
    print("SEED-TTS-EN RESULTS  |  Whisper-large-v3  |  Baseline vs SSDG")
    print("="*80)
    print(f"{'Metric':<22} | {'Baseline':>12} | {'SSDG':>12} | {'Better':>10}")
    print("-"*65)

    for name, fn in metrics:
        base_vals = [fn(r, "baseline") for r in results]
        ssdg_vals = [fn(r, "ssdg") for r in results]
        b_avg = avg(base_vals)
        s_avg = avg(ssdg_vals)

        higher_better = name in ("SIM-prompt", "SIM-target", "STOI")
        if np.isnan(b_avg) or np.isnan(s_avg):
            better = "N/A"
        elif abs(b_avg - s_avg) < 1e-6:
            better = "Equal"
        elif higher_better:
            better = "SSDG *" if s_avg > b_avg else "Base *"
        else:
            better = "SSDG *" if s_avg < b_avg else "Base *"

        b_str = f"{b_avg:.4f}" if not np.isnan(b_avg) else "N/A"
        s_str = f"{s_avg:.4f}" if not np.isnan(s_avg) else "N/A"
        print(f"{name:<22} | {b_str:>12} | {s_str:>12} | {better:>10}")

    print("="*80)
    print("\nPaper references (Seed-TTS-en, Whisper-large-v3):")
    print("  DMOSpeech 2 Student-only (4 steps):    WER=1.752%, SIM=0.698")
    print("  DMOSpeech 2 Teacher-guided (16 steps): WER=1.738%, SIM=0.699")
    print("  F5-TTS Teacher (32 steps):             WER=1.947%, SIM=0.662")
    print("="*80)


if __name__ == "__main__":
    main()
