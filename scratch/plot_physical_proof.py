import os
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

src_dir = r"c:\Users\Rhenel Jhon Sajol\Desktop\TTS\DMOSpeech 2\BENCHMARK_SUITE\both_problems_diagnostic"
dst_dir = r"c:\Users\Rhenel Jhon Sajol\Desktop\TTS\DMOSpeech 2\THESIS_EXPERIMENT\NEW_PATH"
audio_dst = os.path.join(dst_dir, "audio")
os.makedirs(audio_dst, exist_ok=True)

# Copy files
for f in ["exp2_step_4.wav", "exp2_step_2.wav", "exp2_step_1.wav"]:
    src_file = os.path.join(src_dir, f)
    dst_file = os.path.join(audio_dst, f)
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)

# Load audios
y4, sr = librosa.load(os.path.join(src_dir, "exp2_step_4.wav"), sr=24000)
y2, _  = librosa.load(os.path.join(src_dir, "exp2_step_2.wav"), sr=24000)
y1, _  = librosa.load(os.path.join(src_dir, "exp2_step_1.wav"), sr=24000)

fig, axes = plt.subplots(3, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1, 2]})

cases = [
    ("4-Step Baseline (0.0% WER) - 'Get the trust fund to the bank early.'", y4, "forestgreen"),
    ("2-Step Collapse (62.5% WER) - 'They have to fund for the thing. Early.'", y2, "darkorange"),
    ("1-Step Failure (100.0% WER) - '. . . . . . . . . . .' (Silence)", y1, "crimson")
]

for idx, (title, y, color) in enumerate(cases):
    # Waveform
    axes[idx, 0].plot(np.linspace(0, len(y)/sr, len(y)), y, color=color, alpha=0.8)
    axes[idx, 0].set_title(f"Waveform: {title.split(' - ')[0]}", fontsize=10, fontweight="bold")
    axes[idx, 0].set_ylabel("Amplitude")
    axes[idx, 0].set_ylim(-0.8, 0.8)
    axes[idx, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="linear", ax=axes[idx, 1], cmap="inferno")
    axes[idx, 1].set_title(f"Spectrogram: {title}", fontsize=10)
    axes[idx, 1].set_ylim(0, 8000)

plt.tight_layout()
plot_path = os.path.join(dst_dir, "physical_proof_spectrogram_collapse.png")
plt.savefig(plot_path, dpi=200)
print("Spectrogram proof saved successfully to:", plot_path)
