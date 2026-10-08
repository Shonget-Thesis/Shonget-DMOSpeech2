"""
Custom User Inference Script for Adaptive TAGS-Flow
Supports any audio format (.m4a, .mp3, .wav, .flac) including humming/speech.
"""

import os
import sys
import argparse
import torch
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ADAPTIVE_TAGS.adaptive_smooth_dmospeech import AdaptiveDMOSpeechInference


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive TAGS-Flow with custom audio and text")
    parser.add_argument("--audio", type=str, default="1.m4a", help="Path to reference audio file")
    parser.add_argument("--text", type=str, default="Hello, adaptive TAGS-Flow is currently on air.", help="Text to synthesize")
    parser.add_argument("--ref_text", type=str, default="Hmm.", help="Transcript of reference audio (use 'Hmm.' for humming)")
    parser.add_argument("--duration_sec", type=float, default=3.5, help="Target duration in seconds (for unvoiced/humming audio)")
    parser.add_argument("--output", type=str, default="ADAPTIVE_TAGS/output/output_custom_adaptive.wav", help="Output path")
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(BASE_DIR, audio_path)

    print(f"=== Adaptive TAGS-Flow Custom Inference ===")
    print(f"Reference Audio: {audio_path}")
    print(f"Target Text:     \"{args.text}\"")
    print(f"Reference Text:  \"{args.ref_text}\"")

    # If audio is not a standard 24kHz wav or is m4a, convert & trim cleanly
    temp_wav = os.path.join(BASE_DIR, "ADAPTIVE_TAGS", "input", "temp_custom_ref.wav")
    os.makedirs(os.path.dirname(temp_wav), exist_ok=True)
    
    y, sr = librosa.load(audio_path, sr=24000)
    # Clip reference to 3.5s if longer to keep prompt compact
    if len(y) > int(3.5 * sr):
        y = y[:int(3.5 * sr)]
    sf.write(temp_wav, y, 24000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")

    tts = AdaptiveDMOSpeechInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )

    # 93.75 mel frames per second at 24000 Hz / hop 256
    target_frames = int(args.duration_sec * 93.75) if args.duration_sec > 0 else None

    wav = tts.generate(
        gen_text=args.text,
        audio_path=temp_wav,
        prompt_text=args.ref_text,
        duration=target_frames,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        student_start_step=1,
        verbose=True
    )

    out_file = args.output
    if not os.path.isabs(out_file):
        out_file = os.path.join(BASE_DIR, out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    sf.write(out_file, wav, 24000)

    print(f"\n[SUCCESS] Audio generated successfully!")
    print(f"Saved to: {out_file}")
    print(f"Collision Ratio r : {tts.last_collision_ratio:.4f}")
    print(f"Dynamic Cushion w : {tts.last_dynamic_w:.4f} (Transmitted {tts.last_dynamic_w*100:.1f}%)")
    print(f"Absorbed Shock    : {tts.last_absorbed_shock:.2f} units (Damped {(1.0-tts.last_dynamic_w)*100:.1f}%)")


if __name__ == "__main__":
    main()
