"""
Batch 5-Sample Evaluation for BASELINE DMOSpeech 2
Runs pure baseline DMOSpeech 2 on the 5 inputs in ADAPTIVE_TAGS/input/
and saves outputs into ADAPTIVE_TAGS/output_baseline/
"""

import os
import sys
import json
import torch
import soundfile as sf
import numpy as np
import librosa
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ADAPTIVE_DIR = os.path.join(BASE_DIR, "ADAPTIVE_TAGS")
INPUT_DIR = os.path.join(ADAPTIVE_DIR, "input")
OUT_BASE_DIR = os.path.join(ADAPTIVE_DIR, "output_baseline")
os.makedirs(OUT_BASE_DIR, exist_ok=True)

from infer import DMOInference


TEST_CONFIGS = [
    {
        "id": 1,
        "name": "Female Nature",
        "tone": "Soft, calm, serene feminine British/American English",
        "audio_path": os.path.join(INPUT_DIR, "input_1_female_nature.wav"),
        "ref_text": "Some call me nature. Others call me Mother Nature.",
        "gen_text": "Hello! This is a test of the adaptive smooth speech synthesis system.",
        "out_filename": "sample_1_female_nature.wav"
    },
    {
        "id": 2,
        "name": "Male Country",
        "tone": "Rustic, folksy, gritty Southern male storytelling dialect",
        "audio_path": os.path.join(INPUT_DIR, "input_2_male_country.flac"),
        "ref_text": "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "gen_text": "Well howdy there! The quick brown fox jumps right over that lazy dog.",
        "out_filename": "sample_2_male_country.wav"
    },
    {
        "id": 3,
        "name": "Male Broadcast Narrator",
        "tone": "Authoritative, crisp, professional documentary narration",
        "audio_path": os.path.join(INPUT_DIR, "input_3_male_broadcast_narrator.flac"),
        "ref_text": "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "gen_text": "Artificial intelligence and speech technologies are transforming human communication worldwide.",
        "out_filename": "sample_3_male_broadcast_narrator.wav"
    },
    {
        "id": 4,
        "name": "Male Academic Lecture",
        "tone": "Fast-paced, intellectual, analytical scientific lecture",
        "audio_path": os.path.join(INPUT_DIR, "input_4_male_academic_lecture.flac"),
        "ref_text": "The difference in the rainbow depends considerably upon the size of the drops and the width of the colored band increases as the size of the drops increases.",
        "gen_text": "Spectral analysis demonstrates a significant reduction in boundary transition turbulence.",
        "out_filename": "sample_4_male_academic_lecture.wav"
    },
    {
        "id": 5,
        "name": "Deep Dramatic Cinematic Voice",
        "tone": "Dark, gravelly, intense theatrical male voice",
        "audio_path": os.path.join(INPUT_DIR, "input_5_deep_dramatic_cinematic.mp3"),
        "ref_text": "If you're good at something, never do it for free.",
        "gen_text": "Why so serious? We have eliminated the boundary collision completely.",
        "out_filename": "sample_5_deep_dramatic_cinematic_voice.wav"
    }
]


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
    print("==================================================================")
    print("      BASELINE DMOSPEECH 2: 5 DIVERSE VOICES & TONES BENCHMARK    ")
    print("==================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    
    print("[1/3] Initializing Pure Baseline DMOSpeech 2 Engine...")
    tts = DMOInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    
    print("[2/3] Initializing Whisper ASR for Verification...")
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
    
    results = []
    
    print("\n[3/3] Running 5-Voice Baseline Generation...")
    for item in TEST_CONFIGS:
        sample_id = item["id"]
        print(f"\n>>> Running Baseline Sample {sample_id}/5: {item['name']} ({item['tone']})")
        print(f"    Input Audio: {item['audio_path']}")
        print(f"    Target Text: \"{item['gen_text']}\"")
        
        # Pure Baseline Inference (w=1.0)
        wav_data = tts.generate(
            gen_text=item["gen_text"],
            audio_path=item["audio_path"],
            prompt_text=item["ref_text"],
            teacher_steps=8,
            teacher_stopping_time=0.07,
            student_start_step=1,
            verbose=False
        )
        
        out_wav = os.path.join(OUT_BASE_DIR, item["out_filename"])
        sf.write(out_wav, wav_data, 24000)
        print(f"    [Saved Baseline Audio] -> {out_wav}")
        
        flux_info = eval_spectral_flux(out_wav)
        asr_result = pipe(out_wav)["text"].strip()
        
        entry = {
            "id": sample_id,
            "name": item["name"],
            "tone": item["tone"],
            "output_file": out_wav,
            "target_text": item["gen_text"],
            "whisper_transcript": asr_result,
            "transition_mean_flux": flux_info["mean_flux"],
            "transition_max_spike": flux_info["max_spike"],
            "duration_sec": flux_info["duration_sec"]
        }
        results.append(entry)
        
        print(f"    Baseline Transition Mean Flux: {entry['transition_mean_flux']}")
        print(f"    Baseline Peak Spike:           {entry['transition_max_spike']}")
        print(f"    Whisper Transcription:         \"{entry['whisper_transcript']}\"")

    # Save Baseline JSON into metricsresults
    metrics_dir = os.path.join(ADAPTIVE_DIR, "metricsresults")
    os.makedirs(metrics_dir, exist_ok=True)
    base_json = os.path.join(metrics_dir, "five_samples_baseline_results.json")
    with open(base_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved Baseline JSON Report] -> {base_json}")
    
    # Load Adaptive Results for Side-by-Side Comparison
    adapt_json = os.path.join(metrics_dir, "five_samples_benchmark_results.json")
    if not os.path.exists(adapt_json):
        adapt_json = os.path.join(ADAPTIVE_DIR, "five_samples_benchmark_results.json")
        
    if os.path.exists(adapt_json):
        with open(adapt_json, "r") as f:
            adapt_results = json.load(f)
            
        comp_metrics_path = os.path.join(metrics_dir, "BASELINE_VS_ADAPTIVE_5_SAMPLES_REPORT.md")
        with open(comp_metrics_path, "w", encoding="utf-8") as f:
            f.write("# 🔬 5-Voice Head-to-Head Comparison: Baseline DMOSpeech 2 vs. Adaptive TAGS-Flow\n\n")
            f.write("**Direct Side-by-Side Acoustic Turbulence & Intelligibility Benchmark Across 5 Diverse Voices**\n\n")
            f.write("| # | Speaker Profile | Baseline Flux (🔴) | Adaptive Flux (🟢) | Dynamic $w$ | Shock Absorbed | Turbulence Reduction | Audio Links |\n")
            f.write("| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |\n")
            
            for b, a in zip(results, adapt_results):
                b_flux = b["transition_mean_flux"]
                a_flux = a["transition_mean_flux"]
                red_pct = round(((b_flux - a_flux) / b_flux) * 100, 1) if b_flux > 0 else 0.0
                sym = "🏆" if red_pct > 0 else "≈"
                
                f.write(f"| **{b['id']}** | **{b['name']}**<br>*{b['tone']}* | `{b_flux}` | **`{a_flux}`** | **`{a['dynamic_w']}`** | `{a['shock_pct_absorbed']}%` | {sym} **`{red_pct}%`** | [🔴 Baseline](../output_baseline/{b['name'].lower().replace(' ', '_')}.wav)<br>[🟢 Adaptive](../output/sample_{a['id']}_{a['name'].lower().replace(' ', '_')}.wav) |\n")
            
            f.write("\n\n---\n\n## 📋 Intelligibility Comparison (Whisper ASR):\n\n")
            for b, a in zip(results, adapt_results):
                f.write(f"### Sample {b['id']}: {b['name']}\n")
                f.write(f"* **Target Text:** \"{b['target_text']}\"\n")
                f.write(f"* **🔴 Baseline Transcript:** \"{b['whisper_transcript']}\"\n")
                f.write(f"* **🟢 Adaptive Transcript:** \"{a['whisper_transcript']}\"\n\n")
        print(f"[Saved Side-by-Side Comparison Report to metricsresults] -> {comp_metrics_path}")

    print("\n==================================================================")
    print("      5-VOICE BASELINE EVALUATION COMPLETED SUCCESSFULLY!         ")
    print("==================================================================")


if __name__ == "__main__":
    main()
