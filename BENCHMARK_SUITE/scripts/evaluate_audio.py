"""
Unified Audio Evaluation Script for DMOSpeech 2
Computes:
1. Librosa STFT Spectral Flux (Mean and Peak over first 15% transition frames)
2. Whisper-Base ASR Word Error Rate (WER) using jiwer
"""

import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
import jiwer
from transformers import pipeline


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


def eval_wav_file(wav_path, reference_text=None, asr_pipe=None):
    if not os.path.exists(wav_path):
        return None
    y, sr = sf.read(wav_path)
    res = eval_spectral_flux(y, sr)
    if asr_pipe is not None and reference_text:
        try:
            asr_res = asr_pipe(wav_path)
            hyp = asr_res.get("text", "").strip()
            res["transcript"] = hyp
            res["wer"] = round(float(jiwer.wer(reference_text.lower(), hyp.lower())), 4)
        except Exception as e:
            res["wer"] = None
            res["transcript"] = ""
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file")
    parser.add_argument("--text", type=str, default=None, help="Reference text for WER")
    args = parser.parse_args()

    asr = None
    if args.text:
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")

    result = eval_wav_file(args.wav, args.text, asr)
    print(json.dumps(result, indent=2))
