import os, sys, time, json, warnings, string
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

BASE_DIR = r"c:\Users\Rhenel Jhon Sajol\Desktop\TTS\DMOSpeech 2"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from infer import DMOInference
from resemblyzer import VoiceEncoder, preprocess_wav
import jiwer
from transformers import pipeline

OUTPUT_DIR = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "both_problems_diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_spectral_flux(audio_path, sr=24000):
    y, _ = librosa.load(audio_path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    diff = np.diff(S, axis=1)
    pos_diff = np.maximum(0, diff)
    flux = np.sqrt(np.mean(pos_diff**2, axis=0))
    return float(np.max(flux)), float(np.mean(flux))

def compute_mcd(synth_path, ref_path, sr=24000):
    try:
        y_synth, _ = librosa.load(synth_path, sr=sr)
        y_ref, _ = librosa.load(ref_path, sr=sr)
        mfcc_synth = librosa.feature.mfcc(y=y_synth, sr=sr, n_mfcc=13)[1:]
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)[1:]
        min_len = min(mfcc_synth.shape[1], mfcc_ref.shape[1])
        diff = mfcc_synth[:, :min_len] - mfcc_ref[:, :min_len]
        return float(np.mean(np.sqrt(2 * np.sum(diff**2, axis=0))))
    except Exception:
        return 999.0

def main():
    print("=" * 75)
    print("EMPIRICAL DIAGNOSTIC: TESTING BOTH FAILURE MODES IN DMOSPEECH 2")
    print("=" * 75)

    manifest_path = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample = manifest[0]
    prompt_wav = os.path.join(BASE_DIR, sample["prompt_wav"])
    prompt_text = sample["prompt_text"]
    target_text = sample["target_text"]

    print(f"Target Text: \"{target_text}\"")
    print(f"Prompt Wav:  {prompt_wav}")

    # Load Model
    print("\nLoading DMOSpeech 2 Model...")
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    tts = DMOInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )

    # Load ASR (Whisper-base for fast CPU transcription)
    print("Loading Whisper ASR (whisper-base for fast CPU benchmarking)...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        torch_dtype=torch.float32,
        device="cpu"
    )

    print("Loading VoiceEncoder...")
    voice_enc = VoiceEncoder()
    prompt_wav_pre = preprocess_wav(prompt_wav)
    prompt_emb = voice_enc.embed_utterance(prompt_wav_pre)

    nominal_dur = tts.predict_duration(prompt_wav, target_text, prompt_text)
    print(f"Nominal Predicted Duration: {nominal_dur} frames (~{nominal_dur * 256 / 24000:.2f}s)")

    # ═════════════════════════════════════════════════════════════════════════
    # TEST 1: THE DURATION / CADENCE SENSITIVITY VULNERABILITY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("EXPERIMENT 1: DURATION / CADENCE SENSITIVITY IN DMOSPEECH 2")
    print("Testing perturbations: -30%, -15%, 0% (Nominal), +15%, +30%")
    print("=" * 75)

    duration_multipliers = [
        ("rushed_30pct", 0.70, "Rushed (-30% duration)"),
        ("tight_15pct",  0.85, "Tight (-15% duration)"),
        ("nominal_0pct", 1.00, "Nominal (0% duration)"),
        ("loose_15pct",  1.15, "Loose (+15% duration)"),
        ("dragged_30pct",1.30, "Dragged (+30% duration)")
    ]

    exp1_results = []
    for tag, mult, desc in duration_multipliers:
        test_dur = int(round(nominal_dur * mult))
        print(f"\nEvaluating {desc}: {test_dur} frames...")
        
        t0 = time.time()
        wav = tts.generate(
            gen_text=target_text,
            audio_path=prompt_wav,
            prompt_text=prompt_text,
            teacher_steps=0,
            student_start_step=0,
            duration=test_dur
        )
        elapsed = time.time() - t0
        dur_sec = len(wav) / 24000.0

        wav_path = os.path.join(OUTPUT_DIR, f"exp1_{tag}.wav")
        sf.write(wav_path, wav, 24000)

        # Metrics
        peak_f, mean_f = compute_spectral_flux(wav_path)
        mcd_val = compute_mcd(wav_path, prompt_wav)

        # ASR
        trans = asr(wav_path, generate_kwargs={"task": "transcribe", "language": "en"})["text"].strip()
        ref_clean = target_text.lower().translate(str.maketrans("", "", string.punctuation))
        hyp_clean = trans.lower().translate(str.maketrans("", "", string.punctuation))
        wer = jiwer.wer(ref_clean, hyp_clean) * 100.0

        # SIM
        synth_pre = preprocess_wav(wav_path)
        synth_emb = voice_enc.embed_utterance(synth_pre)
        from scipy.spatial.distance import cosine as cosine_dist
        sim = 1.0 - cosine_dist(prompt_emb, synth_emb)

        res = {
            "tag": tag,
            "desc": desc,
            "multiplier": mult,
            "duration_frames": test_dur,
            "audio_sec": round(dur_sec, 2),
            "wer": round(wer, 2),
            "transcription": trans,
            "sim": round(float(sim), 4),
            "peak_flux": round(peak_f, 4),
            "mean_flux": round(mean_f, 4),
            "mcd": round(mcd_val, 2)
        }
        exp1_results.append(res)
        print(f"  ASR: \"{trans}\"")
        print(f"  WER: {wer:.1f}% | SIM: {sim:.4f} | Audio Len: {dur_sec:.2f}s | Peak Flux: {peak_f:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # TEST 2: THE FEW-STEP DISCRETIZATION / TRUNCATION COLLAPSE
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("EXPERIMENT 2: FEW-STEP TRUNCATION COLLAPSE IN DMOSPEECH 2")
    print("Testing: 1-Step, 2-Step, 4-Step Student vs 32-Step Teacher")
    print("=" * 75)

    step_regimes = [
        ("step_1", "1-Step Student", 0, 3, "student"),
        ("step_2", "2-Step Student", 0, 2, "student"),
        ("step_4", "4-Step Student (Standard)", 0, 0, "student"),
        ("teacher_32", "32-Step Teacher (Gold Standard)", 32, 0, "teacher")
    ]

    exp2_results = []
    for tag, desc, t_steps, s_start, mode in step_regimes:
        print(f"\nEvaluating {desc}...")
        t0 = time.time()
        if mode == "teacher":
            wav = tts.generate_teacher_only(
                gen_text=target_text,
                audio_path=prompt_wav,
                prompt_text=prompt_text,
                teacher_steps=t_steps,
                duration=nominal_dur
            )
        else:
            wav = tts.generate(
                gen_text=target_text,
                audio_path=prompt_wav,
                prompt_text=prompt_text,
                teacher_steps=t_steps,
                student_start_step=s_start,
                duration=nominal_dur
            )
        elapsed = time.time() - t0
        dur_sec = len(wav) / 24000.0
        rtf = elapsed / dur_sec

        wav_path = os.path.join(OUTPUT_DIR, f"exp2_{tag}.wav")
        sf.write(wav_path, wav, 24000)

        peak_f, mean_f = compute_spectral_flux(wav_path)
        mcd_val = compute_mcd(wav_path, prompt_wav)

        trans = asr(wav_path, generate_kwargs={"task": "transcribe", "language": "en"})["text"].strip()
        ref_clean = target_text.lower().translate(str.maketrans("", "", string.punctuation))
        hyp_clean = trans.lower().translate(str.maketrans("", "", string.punctuation))
        wer = jiwer.wer(ref_clean, hyp_clean) * 100.0

        synth_pre = preprocess_wav(wav_path)
        synth_emb = voice_enc.embed_utterance(synth_pre)
        from scipy.spatial.distance import cosine as cosine_dist
        sim = 1.0 - cosine_dist(prompt_emb, synth_emb)

        res = {
            "tag": tag,
            "desc": desc,
            "elapsed_sec": round(elapsed, 2),
            "rtf": round(rtf, 2),
            "wer": round(wer, 2),
            "transcription": trans,
            "sim": round(float(sim), 4),
            "peak_flux": round(peak_f, 4),
            "mean_flux": round(mean_f, 4),
            "mcd": round(mcd_val, 2)
        }
        exp2_results.append(res)
        print(f"  ASR: \"{trans}\"")
        print(f"  WER: {wer:.1f}% | SIM: {sim:.4f} | RTF: {rtf:.2f}x | Peak Flux: {peak_f:.4f}")

    # Save summary json
    summary_path = os.path.join(OUTPUT_DIR, "both_problems_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"target_text": target_text, "experiment_1_duration": exp1_results, "experiment_2_steps": exp2_results}, f, indent=2)

    print("\n" + "=" * 75)
    print(f"ALL EXPERIMENTS COMPLETE. Summary saved to {summary_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
