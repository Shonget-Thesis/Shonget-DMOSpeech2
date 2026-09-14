"""
Batch 5-Sample Evaluation for Constant Cushioning (w = 0.85)
Tests the 5 audio prompts in ADAPTIVE_TAGS/input/
Outputs to ADAPTIVE_TAGS/output_constant/
Updates metrics in ADAPTIVE_TAGS/metricsresults/
"""

import os
import sys
import json
import time
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
OUT_CONST_DIR = os.path.join(ADAPTIVE_DIR, "output_constant")
METRICS_DIR = os.path.join(ADAPTIVE_DIR, "metricsresults")
os.makedirs(OUT_CONST_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

from infer import DMOInference


class ConstantCushionDMOSpeechInference(DMOInference):
    """
    DMOSpeech 2 architecture with Constant Cushioning (w = 0.85).
    Applies fixed 15% shock absorption at the teacher-student handover boundary.
    """

    def _student_sampling(self, x1, cond, text, student_start_step, verbose, sway_coeff=-1):
        steps = torch.Tensor([0, 0.25, 0.5, 0.75])
        steps = steps + sway_coeff * (torch.cos(torch.pi / 2 * steps) - 1 + steps)
        steps = steps[student_start_step:]

        x1_teacher = x1.clone()

        for idx, step in enumerate(steps):
            t_val = torch.Tensor([step]).to(x1.device)
            x0 = torch.randn_like(x1)
            t = t_val.unsqueeze(-1).unsqueeze(-1)
            phi = (1 - t) * x0 + t * x1

            if verbose:
                print(f"current step: {step}")
            with torch.no_grad():
                pred = self.generator(
                    x=phi,
                    cond=cond,
                    text=text,
                    time=t_val,
                    drop_audio_cond=False,
                    drop_text=False
                )
                output = phi + (1 - t) * pred

            if idx == 0:
                delta = output - x1_teacher
                w_blend = 0.85  # Constant cushion w = 0.85
                x1 = x1_teacher + w_blend * delta
                if verbose:
                    print(f"   [Constant Cushion Applied] w_blend={w_blend:.2f}")
            else:
                x1 = output

        return x1


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
    boundary_idx = int(len(flux) * 0.10)
    window = flux[max(0, boundary_idx - 10): min(len(flux), boundary_idx + 10)]
    return float(np.mean(window)), float(np.max(window))


def main():
    print("==================================================================")
    print("  5-VOICE CONSTANT CUSHION (w = 0.85) BENCHMARK EVALUATION        ")
    print("==================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Utilizing compute device: {device}")
    
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    
    print("[1/3] Initializing Constant Cushion (w = 0.85) Engine...")
    tts = ConstantCushionDMOSpeechInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    
    print("[2/3] Loading OpenAI Whisper ASR for Intelligibility Validation...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
    
    results = []
    print("\n[3/3] Commencing Synthesis & Acoustic Analysis across 5 Voices...")
    
    for cfg in TEST_CONFIGS:
        print(f"\n---> Processing Sample #{cfg['id']}: [{cfg['name']}] ({cfg['tone']})")
        out_wav = os.path.join(OUT_CONST_DIR, cfg["out_filename"])
        
        t0 = time.time()
        audio = tts.generate(
            gen_text=cfg["gen_text"],
            audio_path=cfg["audio_path"],
            prompt_text=cfg["ref_text"],
            teacher_steps=8,
            teacher_stopping_time=0.07,
            student_start_step=1,
            verbose=False
        )
        t_gen = time.time() - t0
        sf.write(out_wav, audio, 24000)
        
        mean_flux, max_spike = eval_spectral_flux(out_wav)
        asr_res = asr(out_wav)
        transcription = asr_res["text"].strip()
        
        entry = {
            "id": cfg["id"],
            "name": cfg["name"],
            "tone": cfg["tone"],
            "input_path": cfg["audio_path"],
            "output_path": out_wav,
            "target_text": cfg["gen_text"],
            "whisper_transcript": transcription,
            "cushion_weight_w": 0.85,
            "formant_pct_passed": 85.0,
            "shock_pct_absorbed": 15.0,
            "transition_mean_flux": round(mean_flux, 4),
            "transition_max_spike": round(max_spike, 4),
            "generation_time_sec": round(t_gen, 2)
        }
        results.append(entry)
        
        print(f"    Constant Cushion w:            0.85 (15.0% Shock Damped, 85.0% Formants Passed)")
        print(f"    Transition Mean Flux:          {entry['transition_mean_flux']}")
        print(f"    Peak Spike:                    {entry['transition_max_spike']}")
        print(f"    Whisper Transcription:         \"{entry['whisper_transcript']}\"")

    # Save Constant JSON
    const_json = os.path.join(METRICS_DIR, "five_samples_constant_results.json")
    with open(const_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved Constant JSON Report] -> {const_json}")
    
    # Load Baseline and Adaptive Results for Tri-Way Head-to-Head Comparison
    base_json = os.path.join(METRICS_DIR, "five_samples_baseline_results.json")
    adapt_json = os.path.join(METRICS_DIR, "five_samples_benchmark_results.json")
    
    base_results = []
    adapt_results = []
    if os.path.exists(base_json):
        with open(base_json, "r") as f:
            base_results = json.load(f)
    if os.path.exists(adapt_json):
        with open(adapt_json, "r") as f:
            adapt_results = json.load(f)
            
    if base_results and adapt_results:
        tri_report_path = os.path.join(METRICS_DIR, "BASELINE_VS_CONSTANT_VS_ADAPTIVE_REPORT.md")
        with open(tri_report_path, "w", encoding="utf-8") as f:
            f.write("# 🔬 Complete Tri-Way Benchmark: Baseline vs. Constant Cushion (0.85) vs. Adaptive TAGS-Flow\n\n")
            f.write("**Direct Acoustic Turbulence, Dynamic Cushioning, and Intelligibility Audit Across 5 Diverse Voices**\n\n")
            f.write("| # | Speaker Profile | Baseline Flux ($w=1.0$) | Constant Flux ($w=0.85$) | Adaptive Flux ($w_{\\text{adapt}}$) | Dynamic $w$ | Constant Red. | Adaptive Red. | Audio Links |\n")
            f.write("| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |\n")
            
            for b, c, a in zip(base_results, results, adapt_results):
                b_flux = b["transition_mean_flux"]
                c_flux = c["transition_mean_flux"]
                a_flux = a["transition_mean_flux"]
                
                c_red = round(((b_flux - c_flux) / b_flux) * 100, 1) if b_flux > 0 else 0.0
                a_red = round(((b_flux - a_flux) / b_flux) * 100, 1) if b_flux > 0 else 0.0
                
                c_sym = "🏆" if c_red > 0 else "≈"
                a_sym = "🏆" if a_red > 0 else "≈"
                
                fname = b["out_filename"] if "out_filename" in b else f"sample_{b['id']}_{b['name'].lower().replace(' ', '_')}.wav"
                
                f.write(f"| **{b['id']}** | **{b['name']}**<br>*{b['tone']}* | `{b_flux}` | **`{c_flux}`** | **`{a_flux}`** | **`{a['dynamic_w']}`** | {c_sym} **`{c_red}%`** | {a_sym} **`{a_red}%`** | [🔴 Baseline](../output_baseline/{fname})<br>[🟡 Constant](../output_constant/{fname})<br>[🟢 Adaptive](../output/{fname}) |\n")
            
            f.write("\n\n---\n\n## 📋 Intelligibility Comparison (OpenAI Whisper ASR):\n\n")
            for b, c, a in zip(base_results, results, adapt_results):
                f.write(f"### Sample {b['id']}: {b['name']} ({b['tone']})\n")
                f.write(f"* **Target Text:** \"{b['target_text']}\"\n")
                f.write(f"* **🔴 Baseline Transcript:** \"{b['whisper_transcript']}\"\n")
                f.write(f"* **🟡 Constant (0.85) Transcript:** \"{c['whisper_transcript']}\"\n")
                f.write(f"* **🟢 Adaptive Transcript:** \"{a['whisper_transcript']}\"\n\n")
                
        print(f"[Saved Tri-Way Comparative Report] -> {tri_report_path}")

    print("\n==================================================================")
    print("   5-VOICE CONSTANT CUSHION EVALUATION COMPLETED SUCCESSFULLY!    ")
    print("==================================================================")


if __name__ == "__main__":
    main()
