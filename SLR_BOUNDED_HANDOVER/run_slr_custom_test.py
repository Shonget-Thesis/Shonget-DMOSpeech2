"""
Single-Utterance Custom Test Runner for SLR Bounded Handover
Supports all handover modes (slr_cap, tanh, fixed, baseline, cosine)
and accepts any audio format (.wav, .flac, .mp3, .m4a) from ADAPTIVE_TAGS/input or custom.
"""

import os
import sys
import argparse
import torch
import librosa
import soundfile as sf
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from SLR_BOUNDED_HANDOVER.slr_bounded_inference import SLRBoundedDMOSpeechInference


def eval_spectral_flux(audio_path):
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    t_frames = int(len(flux) * 0.15)
    t_flux = flux[:t_frames]
    return {
        "duration_sec": round(len(y) / sr, 2),
        "mean_flux": round(float(np.mean(t_flux)), 2),
        "max_spike": round(float(np.max(t_flux)), 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Run SLR Bounded Handover Inference")
    parser.add_argument(
        "--audio",
        type=str,
        default="ADAPTIVE_TAGS/input/input_1_female_nature.wav",
        help="Path to reference audio file (default: ADAPTIVE_TAGS/input/input_1_female_nature.wav)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of the SLR bounded handover speech synthesis system.",
        help="Target text to synthesize"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="Some call me nature. Others call me Mother Nature.",
        help="Transcript of reference audio prompt"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="slr_cap",
        choices=["slr_cap", "tanh", "fixed", "baseline", "cosine"],
        help="Handover mode: slr_cap (recommended), tanh, fixed, baseline, cosine"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.5,
        help="Displacement radius cap for slr_cap mode (default: 1.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random noise seed for student sampling (default: 42)"
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=0.0,
        help="Target duration in seconds (0 = auto-predict with duration predictor)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: SLR_BOUNDED_HANDOVER/output/output_{mode}.wav)"
    )
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(BASE_DIR, audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")

    # Set default output path
    out_file = args.output
    if out_file is None:
        out_file = os.path.join(BASE_DIR, "SLR_BOUNDED_HANDOVER", "output", f"custom_output_{args.mode}.wav")
    elif not os.path.isabs(out_file):
        out_file = os.path.join(BASE_DIR, out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print("==================================================================")
    print("      SLR BOUNDED HANDOVER: SINGLE-UTTERANCE INFERENCE            ")
    print("==================================================================")
    print(f"Reference Audio : {audio_path}")
    print(f"Target Text    : \"{args.text}\"")
    print(f"Reference Text : \"{args.ref_text}\"")
    print(f"Handover Mode  : {args.mode}")
    if args.mode == "slr_cap":
        print(f"Radius Cap rho : {args.rho}")
    print(f"Noise Seed     : {args.seed}")

    # Pre-process audio to clean 24kHz if needed
    temp_wav = os.path.join(BASE_DIR, "SLR_BOUNDED_HANDOVER", "temp_custom_ref.wav")
    os.makedirs(os.path.dirname(temp_wav), exist_ok=True)
    y, sr = librosa.load(audio_path, sr=24000)
    if len(y) > int(4.0 * sr):
        y = y[:int(4.0 * sr)]
    sf.write(temp_wav, y, 24000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")

    tts = SLRBoundedDMOSpeechInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )

    target_frames = int(args.duration_sec * 93.75) if args.duration_sec > 0 else None

    wav = tts.generate(
        gen_text=args.text,
        audio_path=temp_wav,
        prompt_text=args.ref_text,
        duration=target_frames,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        student_start_step=1,
        verbose=True,
        mode=args.mode,
        rho=args.rho,
        noise_seed=args.seed
    )

    sf.write(out_file, wav, 24000)
    flux_info = eval_spectral_flux(out_file)

    print("\n------------------------------------------------------------------")
    print("[SUCCESS] Audio generation completed!")
    print(f"Output saved to : {out_file}")
    diag = tts.last_diagnostics
    print(f"Method Name     : {diag.get('method_name')}")
    print(f"RMS Displ. r    : {diag.get('rms_displacement', 0.0):.4f}")
    print(f"Acceptance lam  : {diag.get('effective_lambda', 1.0):.4f} ({diag.get('percent_transmitted', 100.0):.1f}% accepted)")
    print(f"Accepted RMS    : {diag.get('accepted_rms', 0.0):.4f}")
    print(f"Withheld RMS    : {diag.get('withheld_rms', 0.0):.4f}")
    print(f"Peak Spec. Flux : {flux_info['max_spike']:.2f}")
    print(f"Mean Spec. Flux : {flux_info['mean_flux']:.2f}")
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
