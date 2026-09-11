import os
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_audio(audio_path, output_dir="eval_results", label="baseline"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    print(f"=== Evaluating Audio Quality for: {label} ===")
    print(f"File: {audio_path}")
    
    # Load audio
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
        
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    
    # Spectral Features using librosa
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    
    # 1. Spectral Flux (frame-to-frame change: abrupt changes create high flux spikes)
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    mean_flux = float(np.mean(flux))
    max_flux_spike = float(np.max(flux))
    flux_std = float(np.std(flux))
    
    # 2. Spectral Flatness (higher flatness = more noise/harshness, lower = harmonic)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=256)[0]
    mean_flatness = float(np.mean(flatness))
    
    # 3. Spectral Centroid (timbre brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0]
    mean_centroid = float(np.mean(centroid))
    
    # 4. High-frequency energy ratio (> 4kHz)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf_mask = freq_bins >= 4000
    total_energy = np.sum(stft**2) + 1e-8
    hf_energy = np.sum(stft[hf_mask, :]**2)
    hf_ratio = float(hf_energy / total_energy)
    
    # 5. Transition Zone Analysis (first 10-15% of frames where teacher hands off to student)
    num_frames = stft.shape[1]
    trans_end_frame = int(num_frames * 0.15)
    trans_flux = flux[:trans_end_frame]
    trans_flux_mean = float(np.mean(trans_flux)) if len(trans_flux) > 0 else 0.0
    trans_flux_max = float(np.max(trans_flux)) if len(trans_flux) > 0 else 0.0
    
    metrics = {
        "label": label,
        "audio_file": audio_path,
        "sample_rate_hz": sr,
        "duration_sec": round(duration, 3),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
        "mean_spectral_flux": round(mean_flux, 4),
        "max_spectral_flux_spike": round(max_flux_spike, 4),
        "flux_std_deviation": round(flux_std, 4),
        "transition_zone_mean_flux": round(trans_flux_mean, 4),
        "transition_zone_max_flux_spike": round(trans_flux_max, 4),
        "mean_spectral_flatness": round(mean_flatness, 6),
        "mean_spectral_centroid_hz": round(mean_centroid, 1),
        "high_frequency_energy_ratio": round(hf_ratio, 4)
    }
    
    # Save Metrics JSON
    json_path = os.path.join(output_dir, f"{label}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {json_path}")
    
    # Plot Mel Spectrogram with Transition Marker
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=100)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    plt.figure(figsize=(12, 5), dpi=150)
    librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    
    # Handover marker line (~ 0.4s to 0.6s into speech)
    handover_time = duration * 0.076  # 7.6% point of flow schedule
    plt.axvline(x=handover_time, color='red', linestyle='--', linewidth=2, label=f'Teacher->Student Switch (~{handover_time:.2f}s)')
    
    plt.title(f"Mel-Spectrogram [{label.upper()} DMOSpeech 2] - Audio Quality & Handover Analysis")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Mel)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    spec_png_path = os.path.join(output_dir, f"{label}_spectrogram.png")
    plt.savefig(spec_png_path)
    plt.close()
    print(f"Spectrogram saved to: {spec_png_path}")
    
    print("\n--- Summary Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("==================================================\n")
    return metrics

if __name__ == "__main__":
    evaluate_audio("test_output.wav", output_dir="eval_results", label="baseline_dmospeech2")
