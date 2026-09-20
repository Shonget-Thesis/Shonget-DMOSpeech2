"""
Spectrogram Generation & Visual Acoustic Audit: 4-Way Comparison
Generates publication-quality 4-way comparative spectrogram plots for all 5 voice samples:
1. Baseline (DMOSpeech 2 Hard Cutoff, lam=1.0)
2. Cosine (Sway-Time Cosine Schedule Ablation, lam=0.0142)
3. Adaptive TAGS-Flow (Dynamic Bounded tanh Cushion)
4. SLR Bounded Cap (Section 9 Euclidean Projection, rho=1.5)
"""

import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLR_DIR = os.path.join(BASE_DIR, "SLR_BOUNDED_HANDOVER")
OUTPUT_DIR = os.path.join(SLR_DIR, "output")
METRICS_DIR = os.path.join(SLR_DIR, "metricsresults")
SPEC_DIR = os.path.join(METRICS_DIR, "spectrograms")
os.makedirs(SPEC_DIR, exist_ok=True)

SAMPLES = [
    {"id": 1, "name": "Female Nature", "desc": "Soft, calm, serene feminine British/American"},
    {"id": 2, "name": "Male Country", "desc": "Rustic, folksy, gritty Southern male storytelling"},
    {"id": 3, "name": "Male Broadcast Narrator", "desc": "Authoritative, crisp, professional documentary narration"},
    {"id": 4, "name": "Male Academic Lecture", "desc": "Fast-paced, intellectual analytical scientific lecture"},
    {"id": 5, "name": "Deep Dramatic Cinematic", "desc": "Dark, gravelly, intense theatrical male voice"}
]

MODES = ["baseline", "cosine", "tanh", "slr_cap"]
TITLES = {
    "baseline": "1. Baseline DMOSpeech 2 (Hard Cutoff, lam=1.00)",
    "cosine": "2. Cosine Schedule Ablation (lam=0.0142)",
    "tanh": "3. Adaptive TAGS-Flow (Dynamic Bounded tanh)",
    "slr_cap": "4. SLR Bounded Displacement Cap (rho=1.50)"
}


def compute_spectral_flux(y, hop_length=256):
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))
    t_frames = int(len(flux) * 0.15)
    t_flux = flux[:t_frames]
    return float(np.max(t_flux)), stft


def plot_sample(sample):
    s_id = sample["id"]
    s_name = sample["name"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Acoustic Audit: Sample {s_id} - {s_name} ({sample['desc']})\n"
        f"4-Way Comparison: Baseline vs. Cosine vs. Adaptive TAGS vs. SLR Bounded Cap",
        fontsize=13, fontweight='bold', y=0.98
    )

    for i, mode in enumerate(MODES):
        wav_file = os.path.join(OUTPUT_DIR, f"sample_{s_id}_{mode}.wav")
        ax = axes[i]

        if not os.path.exists(wav_file):
            ax.text(0.5, 0.5, f"Audio file not found: {os.path.basename(wav_file)}\n(Run batch_compare_5_samples.py first)",
                    ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_title(TITLES[mode], fontsize=11, fontweight='bold', loc='left')
            continue

        y, sr = librosa.load(wav_file, sr=24000)
        peak_flux, stft = compute_spectral_flux(y)
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)

        img = librosa.display.specshow(
            stft_db,
            sr=sr,
            hop_length=256,
            x_axis='time',
            y_axis='hz',
            ax=ax,
            cmap='magma',
            vmin=-80,
            vmax=0
        )

        # Highlight handover boundary window (first 15% frames)
        duration = len(y) / sr
        boundary_time = duration * 0.15
        ax.axvspan(0, boundary_time, color='cyan', alpha=0.12, label='Handover Transition Window')
        ax.axvline(x=boundary_time, color='cyan', linestyle='--', linewidth=1.2)

        ax.set_title(
            f"{TITLES[mode]}  |  Peak Boundary Flux Spike: {peak_flux:.2f}",
            fontsize=11, fontweight='bold', loc='left'
        )
        ax.set_ylabel("Freq (Hz)")
        if i == len(MODES) - 1:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xlabel("")

        # Add colorbar
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(SPEC_DIR, f"audit_sample_{s_id}_{s_name.lower().replace(' ', '_')}.png")
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[SAVED] Spectrogram audit plot saved: {out_png}")


def main():
    print("==================================================================")
    print("      SLR BOUNDED HANDOVER: 4-WAY SPECTROGRAM PLOT GENERATOR      ")
    print("==================================================================")
    for sample in SAMPLES:
        plot_sample(sample)
    print("\nAll spectrogram comparison plots generated in:")
    print(f"--> {SPEC_DIR}")


if __name__ == "__main__":
    main()
