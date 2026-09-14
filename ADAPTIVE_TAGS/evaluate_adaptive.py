"""
Evaluation Benchmark for Adaptive TAGS-Flow (Option 1)
Strictly contained within ADAPTIVE_TAGS directory.
Compares:
1. Baseline DMOSpeech 2 (output.wav)
2. Constant Cushion (output_smooth.wav)
3. Adaptive TAGS-Flow Option 1 (ADAPTIVE_TAGS/output_adaptive.wav)
"""

import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTIVE_DIR = os.path.join(BASE_DIR, "ADAPTIVE_TAGS")


def compute_spectral_metrics(audio_path, label):
    if not os.path.exists(audio_path):
        return None
        
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
        
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    
    # First 15% transition zone
    t_frames = int(len(flux) * 0.15)
    t_flux = flux[:t_frames]
    
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=256)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)))
    
    return {
        "label": label,
        "audio_path": audio_path,
        "sample_rate": sr,
        "duration_sec": round(len(y) / sr, 3),
        "transition_mean_flux": round(float(np.mean(t_flux)), 4),
        "transition_max_spike": round(float(np.max(t_flux)), 4),
        "overall_mean_flux": round(float(np.mean(flux)), 4),
        "spectral_flatness": round(flatness, 6),
        "spectral_centroid_hz": round(centroid, 1)
    }


def main():
    print("==================================================================")
    print("      ADAPTIVE TAGS-FLOW (OPTION 1) BENCHMARK COMPARISON         ")
    print("==================================================================")
    
    baseline_wav = os.path.join(BASE_DIR, "output.wav")
    constant_wav = os.path.join(BASE_DIR, "output_smooth.wav")
    adaptive_wav = os.path.join(ADAPTIVE_DIR, "output_adaptive.wav")
    
    m_base = compute_spectral_metrics(baseline_wav, "Baseline DMOSpeech 2 (w=1.0)")
    m_const = compute_spectral_metrics(constant_wav, "Constant Cushion (w=0.85)")
    m_adapt = compute_spectral_metrics(adaptive_wav, "Adaptive Option 1 (Dynamic w)")
    
    results = {
        "baseline": m_base,
        "constant_cushion": m_const,
        "adaptive_option_1": m_adapt
    }
    
    print("\n--- SPECTRAL BENCHMARK RESULTS ---")
    if m_base:
        print(f"1. BASELINE:         Transition Mean Flux = {m_base['transition_mean_flux']:.4f} | Peak Spike = {m_base['transition_max_spike']:.4f}")
    if m_const:
        pct_c = ((m_base['transition_mean_flux'] - m_const['transition_mean_flux']) / m_base['transition_mean_flux']) * 100
        print(f"2. CONSTANT CUSHION: Transition Mean Flux = {m_const['transition_mean_flux']:.4f} | Peak Spike = {m_const['transition_max_spike']:.4f} (-{pct_c:.1f}%)")
    if m_adapt:
        pct_a = ((m_base['transition_mean_flux'] - m_adapt['transition_mean_flux']) / m_base['transition_mean_flux']) * 100
        print(f"3. ADAPTIVE OPTION1: Transition Mean Flux = {m_adapt['transition_mean_flux']:.4f} | Peak Spike = {m_adapt['transition_max_spike']:.4f} (-{pct_a:.1f}%)")
        
    metrics_dir = os.path.join(ADAPTIVE_DIR, "metricsresults")
    os.makedirs(metrics_dir, exist_ok=True)
    out_json = os.path.join(metrics_dir, "adaptive_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Benchmark results saved to: {out_json}")


if __name__ == "__main__":
    main()
