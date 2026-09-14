"""
Spectrogram Generation & Visual Acoustic Audit
Generates publication-quality side-by-side spectrogram comparison plots for all 5 voice samples
across Baseline, Constant Cushion (0.85), and Adaptive TAGS-Flow.
No re-synthesis needed; reads existing audio files from disk.
"""

import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTIVE_DIR = os.path.join(BASE_DIR, "ADAPTIVE_TAGS")
BASE_OUT = os.path.join(ADAPTIVE_DIR, "output_baseline")
CONST_OUT = os.path.join(ADAPTIVE_DIR, "output_constant")
ADAPT_OUT = os.path.join(ADAPTIVE_DIR, "output")
METRICS_DIR = os.path.join(ADAPTIVE_DIR, "metricsresults")
SPEC_DIR = os.path.join(METRICS_DIR, "spectrograms")
os.makedirs(SPEC_DIR, exist_ok=True)

SAMPLES = [
    {
        "id": 1,
        "name": "Female Nature",
        "desc": "Soft, calm, serene feminine British/American",
        "file_base": "sample_1_female_nature.wav",
        "file_const": "sample_1_female_nature.wav",
        "file_adapt": "sample_1_female_nature.wav",
        "flux_base": 24.72,
        "flux_const": 30.15,
        "flux_adapt": 20.22,
        "w_adapt": 0.8498
    },
    {
        "id": 2,
        "name": "Male Country",
        "desc": "Rustic, folksy, gritty Southern male storytelling",
        "file_base": "sample_2_male_country.wav",
        "file_const": "sample_2_male_country.wav",
        "file_adapt": "sample_2_male_country.wav",
        "flux_base": 16.45,
        "flux_const": 0.55,
        "flux_adapt": 15.81,
        "w_adapt": 0.8410
    },
    {
        "id": 3,
        "name": "Male Broadcast Narrator",
        "desc": "Authoritative, crisp, professional documentary narration",
        "file_base": "sample_3_male_broadcast_narrator.wav",
        "file_const": "sample_3_male_broadcast_narrator.wav",
        "file_adapt": "sample_3_male_broadcast_narrator.wav",
        "flux_base": 13.61,
        "flux_const": 38.19,
        "flux_adapt": 18.63,
        "w_adapt": 0.8469
    },
    {
        "id": 4,
        "name": "Male Academic Lecture",
        "desc": "Fast-paced, intellectual, analytical scientific lecture",
        "file_base": "sample_4_male_academic_lecture.wav",
        "file_const": "sample_4_male_academic_lecture.wav",
        "file_adapt": "sample_4_male_academic_lecture.wav",
        "flux_base": 9.50,
        "flux_const": 30.95,
        "flux_adapt": 14.33,
        "w_adapt": 0.8291
    },
    {
        "id": 5,
        "name": "Deep Dramatic Cinematic Voice",
        "desc": "Dark, gravelly, intense theatrical male voice",
        "file_base": "sample_5_deep_dramatic_cinematic_voice.wav",
        "file_const": "sample_5_deep_dramatic_cinematic_voice.wav",
        "file_adapt": "sample_5_deep_dramatic_cinematic_voice.wav",
        "flux_base": 18.60,
        "flux_const": 0.01,
        "flux_adapt": 1.62,
        "w_adapt": 0.8793
    }
]


def load_mel(path):
    y, sr = librosa.load(path, sr=24000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=100)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    duration = len(y) / sr
    return mel_db, sr, duration


def plot_sample(s):
    p_base = os.path.join(BASE_OUT, s["file_base"])
    p_const = os.path.join(CONST_OUT, s["file_const"])
    p_adapt = os.path.join(ADAPT_OUT, s["file_adapt"])

    mel_b, sr, dur_b = load_mel(p_base)
    mel_c, sr, dur_c = load_mel(p_const)
    mel_a, sr, dur_a = load_mel(p_adapt)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.28)

    # Subplot 1: Baseline
    img0 = librosa.display.specshow(mel_b, sr=sr, hop_length=256, x_axis='time', y_axis='mel',
                                    ax=axes[0], cmap='magma', vmin=-70, vmax=0)
    axes[0].set_title(f"🔴 Baseline DMOSpeech 2 (w = 1.0) | Boundary Turbulence Flux: {s['flux_base']:.2f}",
                      fontsize=12, fontweight='bold', color='#b30000', pad=6)
    axes[0].axvline(x=0.07, color='cyan', linestyle='--', linewidth=1.5, label='Handover Boundary (t=0.07s)')
    axes[0].set_ylabel("Frequency (Hz)", fontsize=10)
    axes[0].legend(loc='upper right', framealpha=0.8, fontsize=9)

    # Subplot 2: Constant Cushion
    img1 = librosa.display.specshow(mel_c, sr=sr, hop_length=256, x_axis='time', y_axis='mel',
                                    ax=axes[1], cmap='magma', vmin=-70, vmax=0)
    axes[1].set_title(f"🟡 Constant Cushion (w = 0.85) | Boundary Turbulence Flux: {s['flux_const']:.2f}",
                      fontsize=12, fontweight='bold', color='#b8860b', pad=6)
    axes[1].axvline(x=0.07, color='cyan', linestyle='--', linewidth=1.5)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=10)

    # Subplot 3: Adaptive TAGS-Flow
    img2 = librosa.display.specshow(mel_a, sr=sr, hop_length=256, x_axis='time', y_axis='mel',
                                    ax=axes[2], cmap='magma', vmin=-70, vmax=0)
    axes[2].set_title(f"🟢 Adaptive TAGS-Flow (Dynamic w = {s['w_adapt']:.4f}) | Boundary Turbulence Flux: {s['flux_adapt']:.2f}",
                      fontsize=12, fontweight='bold', color='#006622', pad=6)
    axes[2].axvline(x=0.07, color='cyan', linestyle='--', linewidth=1.5)
    axes[2].set_ylabel("Frequency (Hz)", fontsize=10)
    axes[2].set_xlabel("Time (seconds)", fontsize=10)

    # Shared Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img2, cax=cbar_ax, format="%+2.0f dB")

    fig.suptitle(f"Sample {s['id']}: {s['name']} ({s['desc']})\nTri-Way Mel-Spectrogram & Handover Acoustic Boundary Audit",
                 fontsize=14, fontweight='bold', y=0.98)

    out_png = os.path.join(SPEC_DIR, f"spectrogram_sample_{s['id']}_{s['name'].lower().replace(' ', '_')}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] -> {out_png}")
    return out_png


def main():
    print("==================================================================")
    print("    GENERATING TRI-WAY SPECTROGRAM COMPARISONS ACROSS 5 SAMPLES   ")
    print("==================================================================")

    generated_paths = []
    for s in SAMPLES:
        print(f"Generating Spectrogram for Sample #{s['id']}: {s['name']}...")
        out_file = plot_sample(s)
        generated_paths.append((s, out_file))

    # Generate Markdown Gallery
    md_path = os.path.join(METRICS_DIR, "SPECTROGRAM_VISUAL_AUDIT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🎨 Tri-Way Mel-Spectrogram Visual Audit: Baseline vs. Constant vs. Adaptive\n\n")
        f.write("**High-Resolution Frequency-Domain Analysis Across 5 Diverse Voice Profiles**\n\n")
        f.write("Each plot illustrates the speech frequency trajectory (0–12 kHz) with a highlighted dashed line indicating the exact **Teacher-to-Student handover boundary ($t = 0.07\\text{s}$)**.\n\n")
        f.write("---\n\n")

        for s, p in generated_paths:
            fname = os.path.basename(p)
            f.write(f"## Sample {s['id']}: {s['name']}\n")
            f.write(f"**Acoustic Profile:** *{s['desc']}*\n\n")
            f.write(f"* 🔴 **Baseline Flux ($w=1.0$):** `{s['flux_base']:.2f}`\n")
            f.write(f"* 🟡 **Constant Flux ($w=0.85$):** `{s['flux_const']:.2f}`\n")
            f.write(f"* 🟢 **Adaptive Flux ($w={s['w_adapt']:.4f}$):** **`{s['flux_adapt']:.2f}`**\n\n")
            f.write(f"![Sample {s['id']} Spectrogram](spectrograms/{fname})\n\n")
            f.write("---\n\n")

    print(f"\n[SUCCESS] All 5 spectrogram plots and audit gallery generated in:")
    print(f"  Folder: {SPEC_DIR}")
    print(f"  Audit:  {md_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
