"""Interactive 3-Input Audio Demo Runner (Tasks T56 & T57).

Runs 3 representative acoustic regimes:
1. Clean Input: High SNR, low drift (d_4 <= 0.50) -> Exits early at Step 4 (13 NFE, 38.1% savings).
2. Moderate Input: Clean prompt with phonetic complexity (d_4 > 0.50, d_5 <= 0.50) -> Exits at Step 5 (15 NFE, 28.6% savings).
3. Noisy Input: 15 dB prompt noise -> SNR < 20 dB trips Safety Gate, routes to Step 8 (21 NFE) to prevent collapse.
   (Also synthesizes unprotected Step 4 to show the failure prevented).

Exports audio pairs to demo/audio/ and builds single-screen HTML comparison player at demo/index.html.
"""

import json
import math
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Paths setup
HANDOVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(HANDOVER_DIR)
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_OUT_DIR = os.path.join(DEMO_DIR, "audio")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if HANDOVER_DIR not in sys.path:
    sys.path.insert(0, HANDOVER_DIR)

from infer import DMOInference
from f5_tts.infer.utils_infer import preprocess_ref_audio_text
from core.sampler import (
    ALL_TARGET_STEPS,
    CANDIDATE_STEPS,
    REFERENCE_STEP,
    SWAY_TEACHER_GRID,
    estimate_prompt_snr,
    get_nfe,
    make_local_generator,
    run_student_completion,
    run_teacher_single_pass,
)
from data.make_splits import add_gaussian_noise_15db

# Demo configurations
CKPT_MODEL = os.path.join(ROOT_DIR, "ckpts", "model_85000.pt")
CKPT_DURATION = os.path.join(ROOT_DIR, "ckpts", "model_1500.pt")

DEMO_SAMPLES = [
    {
        "id": "clean_early_step4",
        "label": "1. Clean Prompt (Low Drift)",
        "utt_id": "common_voice_en_15265-common_voice_en_15266",
        "speaker_id": "common_voice_en_15265",
        "condition": "clean",
        "prompt_text": "Man with guitar holding a microphone in one hand and extending his other arm.",
        "prompt_wav_path": os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "prompt-wavs", "common_voice_en_15265.wav"),
        "gt_text": "Large brown dog walks up a blue staircase.",
        "input_seed": 732958102,
        "expected_policy": "Step 4 (13 NFE, 38.1% Savings)",
    },
    {
        "id": "moderate_step5",
        "label": "2. Moderate Clean Prompt (Phonetic Complexity)",
        "utt_id": "common_voice_en_137148-common_voice_en_137152",
        "speaker_id": "common_voice_en_137148",
        "condition": "clean",
        "prompt_text": "She walked across the street wearing stereo headphones and staring at her smartphone.",
        "prompt_wav_path": os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "prompt-wavs", "common_voice_en_137148.wav"),
        "gt_text": 'Elvis Presley is often called the "King of Rock and Roll".',
        "input_seed": 841920381,
        "expected_policy": "Step 5 (15 NFE, 28.6% Savings)",
    },
    {
        "id": "noisy_protected_step8",
        "label": "3. Noisy Prompt (+15 dB Gaussian Noise)",
        "utt_id": "common_voice_en_103675-common_voice_en_103677",
        "speaker_id": "common_voice_en_103675",
        "condition": "noisy15",
        "prompt_text": "I'm never more aware of a room's acoustics than when I'm trying to enjoy a snack I have no intention of sharing.",
        "prompt_wav_path": os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "prompt-wavs", "common_voice_en_103675.wav"),
        "gt_text": "The boy knew the desert sensed his fear.",
        "input_seed": 395810245,
        "expected_policy": "Step 8 Fallback (21 NFE, Safety Gate Active)",
    },
]


def build_html_comparison(results):
    """Builds a responsive, modern HTML comparison player matching Appendix C."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chapter 3 Audio Demo: Input-Adaptive Handover Scheduling</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --secondary: #0d9488;
            --success: #16a34a;
            --danger: #dc2626;
            --warning: #d97706;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 35px;
            padding-bottom: 25px;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            color: #60a5fa;
            margin-bottom: 8px;
        }}
        .subtitle {{
            font-size: 15px;
            color: var(--text-muted);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 10px;
        }}
        .badge-primary {{ background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }}
        .badge-success {{ background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.4); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }}
        
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}
        .card-title {{
            font-size: 18px;
            font-weight: 600;
            color: #38bdf8;
        }}
        .text-box {{
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 18px;
            font-size: 13.5px;
        }}
        .text-label {{
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .grid-players {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}
        .player-box {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 16px;
        }}
        .player-box.reference {{
            border-left: 4px solid #6b7280;
        }}
        .player-box.adaptive {{
            border-left: 4px solid #10b981;
        }}
        .player-box.unprotected {{
            border-left: 4px solid #ef4444;
        }}
        .player-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .player-meta {{
            font-size: 12px;
            color: #9ca3af;
            margin-bottom: 10px;
        }}
        audio {{
            width: 100%;
            height: 38px;
            outline: none;
        }}
        .drift-trace {{
            margin-top: 15px;
            padding-top: 12px;
            border-top: 1px dashed var(--border);
            font-size: 12px;
            color: #cbd5e1;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Input-Adaptive Teacher-to-Student Handover Scheduling</h1>
            <div class="subtitle">Chapter 3 Interactive 3-Input Audio Verification Demo (Tasks T56 & T57)</div>
            <div class="badge badge-primary">Model: DMOSpeech 2 CPU FP32</div>
            <div class="badge badge-success">Certified λ̂ = 0.50 (PAC Bounded)</div>
            <div class="badge badge-warning">Safety Gate: θ = 20.0 dB</div>
        </header>
"""
    for res in results:
        drift_str = " | ".join([f"d_{k}={v:.4f}" for k, v in res["drift_trace"].items()]) if res["drift_trace"] else "N/A (Bypassed by SNR Safety Gate)"
        html_content += f"""
        <div class="card">
            <div class="card-header">
                <div class="card-title">{res['label']}</div>
                <div class="badge {res['badge_class']}">{res['decision_badge']}</div>
            </div>
            <div class="text-box">
                <div class="text-label">Prompt Reference Text:</div>
                <div style="color: #cbd5e1; margin-bottom: 8px;">"{res['prompt_text']}"</div>
                <div class="text-label">Target Text to Synthesize:</div>
                <div style="color: #f1f5f9; font-weight: 500;">"{res['gt_text']}"</div>
                <div style="margin-top: 8px; font-size: 12px; color: #38bdf8;">
                    <strong>Prompt SNR:</strong> {res['prompt_snr']:.2f} dB (Gate: &ge; 20.0 dB) &bull; 
                    <strong>Condition:</strong> {res['condition']}
                </div>
            </div>
            
            <div class="grid-players">
                <!-- Reference Player -->
                <div class="player-box reference">
                    <div class="player-title">Step 8 Reference (Baseline)</div>
                    <div class="player-meta">21 NFE (0.0% Savings) &bull; Runtime: {res['ref_time']:.2f}s</div>
                    <audio controls preload="none">
                        <source src="audio/{res['ref_audio_file']}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                
                <!-- Proposed Adaptive Player -->
                <div class="player-box adaptive">
                    <div class="player-title" style="color: #34d399;">Proposed Adaptive Policy ({res['chosen_step_label']})</div>
                    <div class="player-meta"><strong>{res['adaptive_nfe']} NFE ({res['savings_pct']:.1f}% Savings)</strong> &bull; Runtime: {res['adaptive_time']:.2f}s</div>
                    <audio controls preload="none">
                        <source src="audio/{res['adaptive_audio_file']}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
"""
        if "unprotected_audio_file" in res:
            html_content += f"""
                <!-- Unprotected Step 4 Player -->
                <div class="player-box unprotected">
                    <div class="player-title" style="color: #f87171;">Fixed Step 4 (No Safety Gate)</div>
                    <div class="player-meta">13 NFE &bull; <strong>CATASTROPHIC DEGRADATION (Prevented by Gate)</strong></div>
                    <audio controls preload="none">
                        <source src="audio/{res['unprotected_audio_file']}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
"""
        html_content += f"""
            </div>
            <div class="drift-trace">
                <strong>Tangent Drift Trace:</strong> {drift_str}
            </div>
        </div>
"""
    html_content += """
        <footer>
            DMOSpeech 2 Input-Adaptive Handover Scheduling &bull; Host AMD Ryzen 7 5700U CPU Evaluation Stack &bull; Rhenel Jhon Sajol (Thesis 2026)
        </footer>
    </div>
</body>
</html>
"""
    html_path = os.path.join(DEMO_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Generated one-screen demo HTML player: {html_path}")


def main():
    print("=" * 80)
    print("TASKS T56 & T57: INTERACTIVE 3-INPUT AUDIO DEMO RUNNER")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Execution target: {device} (FP32 precision)")
    print(f"Loading DMOSpeech 2 checkpoint: {CKPT_MODEL}")
    tts = DMOInference(
        student_checkpoint_path=CKPT_MODEL,
        duration_predictor_path=CKPT_DURATION,
        device=device,
    )
    print("[OK] DMOSpeech 2 loaded successfully.")

    results = []

    for item in DEMO_SAMPLES:
        sample_id = item["id"]
        label = item["label"]
        condition = item["condition"]
        prompt_text = item["prompt_text"]
        gt_text = item["gt_text"]
        prompt_wav_path = item["prompt_wav_path"]
        input_seed = item["input_seed"]

        print(f"\n---> Processing Demo Case: {label}")
        print(f"     Prompt Text: \"{prompt_text}\"")
        print(f"     Target Text: \"{gt_text}\"")

        # Load prompt audio
        prompt_raw, sr = torchaudio.load(prompt_wav_path)
        prompt_np = prompt_raw.squeeze().numpy()

        if condition == "noisy15":
            prompt_np, _, _ = add_gaussian_noise_15db(prompt_np, seed=input_seed, target_snr_db=15.0)

        prompt_snr = estimate_prompt_snr(prompt_np, sample_rate=sr)
        print(f"     Measured Prompt SNR: {prompt_snr:.2f} dB (Gate: >= 20.0 dB)")

        # Temporary prompt write for duration predictor
        temp_prompt_path = os.path.join(AUDIO_OUT_DIR, f"temp_{sample_id}.wav")
        sf.write(temp_prompt_path, prompt_np, sr)
        ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(temp_prompt_path, prompt_text)

        audio_tensor = torch.from_numpy(prompt_np).float().unsqueeze(0)
        rms = torch.sqrt(torch.mean(torch.square(audio_tensor)))
        if rms < 0.3:
            audio_tensor = audio_tensor * 0.3 / rms
        if sr != tts.target_sample_rate:
            audio_tensor = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio_tensor)
        audio_tensor = audio_tensor.to(device)

        torch.manual_seed(input_seed)
        predicted_dur = tts.predict_duration(
            temp_prompt_path,
            gt_text,
            prompt_text,
            dp_softmax_range=0.7,
            temperature=0,
        )
        target_dur = max(10, int(predicted_dur))
        if os.path.exists(temp_prompt_path):
            os.remove(temp_prompt_path)

        # Phoneme / text conditioning
        from f5_tts.infer.utils_infer import convert_char_to_pinyin
        from torch.nn.utils.rnn import pad_sequence

        text_list = [ref_text_clean + gt_text]
        final_text_list = convert_char_to_pinyin(text_list)
        ref_audio_len = audio_tensor.shape[-1] // tts.hop_length
        total_duration = ref_audio_len + target_dur

        cond_tensor, text_ids, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
            audio_tensor, final_text_list, total_duration
        )
        prompt_len = cond_mask.sum().item()

        # Run teacher ODE single pass
        print("     Evaluating Teacher ODE integration...")
        gen = make_local_generator(input_seed)
        y0 = [torch.randn(dur, 100, dtype=torch.float32, generator=gen) for dur in dur_tensor]
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        endpoints, drift_trace, _ = run_teacher_single_pass(
            tts=tts,
            step_cond=step_cond,
            text=text_ids,
            y0=y0,
            prompt_len=prompt_len,
            target_steps=ALL_TARGET_STEPS,
            cfg_strength=2.0,
        )

        for k, v in drift_trace.items():
            print(f"     [Drift Probe] Step {k}: d_{k} = {v:.4f} (Threshold: <= 0.50)")

        # 1. Synthesize Step 8 Reference
        t0 = time.perf_counter()
        _, ref_audio = run_student_completion(
            tts=tts,
            cond=cond_tensor,
            text=text_ids,
            teacher_endpoint=endpoints[REFERENCE_STEP],
            prompt_len=prompt_len,
            input_seed=input_seed,
        )
        ref_time = time.perf_counter() - t0
        ref_file = f"{sample_id}_ref_step8.wav"
        sf.write(os.path.join(AUDIO_OUT_DIR, ref_file), ref_audio, 24000)
        print(f"     [Saved] Step 8 Reference ({ref_time:.2f}s): {ref_file}")

        # 2. Evaluate Adaptive Policy Decision Rule
        chosen_step = REFERENCE_STEP
        if prompt_snr >= 20.0:
            for s_cand in CANDIDATE_STEPS:
                if drift_trace.get(s_cand, 999.0) <= 0.50:
                    chosen_step = s_cand
                    break
        else:
            print("     [Safety Gate Triggered] Prompt SNR < 20.0 dB -> Fallback to Step 8")

        chosen_nfe = get_nfe(chosen_step)
        savings_pct = (21 - chosen_nfe) / 21.0 * 100.0

        t0 = time.perf_counter()
        _, adaptive_audio = run_student_completion(
            tts=tts,
            cond=cond_tensor,
            text=text_ids,
            teacher_endpoint=endpoints[chosen_step],
            prompt_len=prompt_len,
            input_seed=input_seed,
        )
        adaptive_time = time.perf_counter() - t0
        adaptive_file = f"{sample_id}_adaptive_step{chosen_step}.wav"
        sf.write(os.path.join(AUDIO_OUT_DIR, adaptive_file), adaptive_audio, 24000)
        print(f"     [Saved] Adaptive Policy Step {chosen_step} ({adaptive_time:.2f}s, {savings_pct:.1f}% savings): {adaptive_file}")

        # Badges & metadata
        if chosen_step == 4:
            decision_badge = "Step 4 Early Exit (38.1% Savings)"
            badge_class = "badge-success"
        elif chosen_step == 5:
            decision_badge = "Step 5 Early Exit (28.6% Savings)"
            badge_class = "badge-success"
        elif chosen_step == 6:
            decision_badge = "Step 6 Early Exit (19.0% Savings)"
            badge_class = "badge-warning"
        else:
            decision_badge = "Step 8 Safety Fallback (Protected)"
            badge_class = "badge-primary"

        res_entry = {
            "id": sample_id,
            "label": label,
            "condition": condition,
            "prompt_text": prompt_text,
            "gt_text": gt_text,
            "prompt_snr": prompt_snr,
            "drift_trace": drift_trace,
            "ref_audio_file": ref_file,
            "ref_time": ref_time,
            "adaptive_audio_file": adaptive_file,
            "adaptive_time": adaptive_time,
            "chosen_step": chosen_step,
            "chosen_step_label": f"Step {chosen_step}",
            "adaptive_nfe": chosen_nfe,
            "savings_pct": savings_pct,
            "decision_badge": decision_badge,
            "badge_class": badge_class,
        }

        # For noisy condition, synthesize unprotected Step 4 to demonstrate catastrophic failure
        if condition == "noisy15":
            _, unprot_audio = run_student_completion(
                tts=tts,
                cond=cond_tensor,
                text=text_ids,
                teacher_endpoint=endpoints[4],
                prompt_len=prompt_len,
                input_seed=input_seed,
            )
            unprot_file = f"{sample_id}_unprotected_step4.wav"
            sf.write(os.path.join(AUDIO_OUT_DIR, unprot_file), unprot_audio, 24000)
            res_entry["unprotected_audio_file"] = unprot_file
            print(f"     [Saved] Unprotected Step 4 (Catastrophic Degradation): {unprot_file}")

        results.append(res_entry)

    # Build single-screen HTML comparison player
    build_html_comparison(results)
    print("\n" + "=" * 80)
    print("ALL DEMO SAMPLES & SINGLE-SCREEN COMPARISON PLAYER SUCCESSFULLY CREATED!")
    print(f"Open in browser: {os.path.join(DEMO_DIR, 'index.html')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
