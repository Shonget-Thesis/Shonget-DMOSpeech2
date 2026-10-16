"""
Task 2 Smoke Test: 5-Utterance Sanity Check on Step-8 Reference
===============================================================
Runs 5 predetermined Seed-TTS utterances through the step-8 reference model:
- Synthesis via core.sampler (Teacher 0..8 Euler pass + Student consistency restart)
- Evaluation via core.metrics (Whisper large-v3 WER, WavLM SIM vs GT recording, UTMOS)
- Wall-clock runtime timing and real-time factor (RTF)
- Output saved to core/sanity_check_results.json and displayed in markdown
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
import numpy as np
import torch
import soundfile as sf
import torchaudio
from torch.nn.utils.rnn import pad_sequence

# Add root and Handover Scheduling to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HANDOVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if HANDOVER_DIR not in sys.path:
    sys.path.insert(0, HANDOVER_DIR)

from infer import DMOInference
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
from core.sampler import (
    derive_input_seed,
    make_local_generator,
    estimate_prompt_snr,
    run_teacher_single_pass,
    run_student_completion,
    STEP_METADATA,
)
from core.metrics import MetricSuite, normalize_text

META_LST = os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "meta.lst")
CKPT_MODEL = os.path.join(ROOT_DIR, "ckpts", "model_85000.pt")
CKPT_DURATION = os.path.join(ROOT_DIR, "ckpts", "model_1500.pt")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "sanity_check_results.json")
OUTPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "sanity_check_audio")


def load_first_n_utterances(metalst_path: str, n: int = 5) -> List[Dict[str, str]]:
    """Loads first n utterances from seed-tts-eval/en/meta.lst."""
    items = []
    base_data_dir = os.path.dirname(metalst_path)
    with open(metalst_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 4:
                utt, prompt_text, prompt_wav_rel, gt_text = parts
                gt_wav = os.path.join(base_data_dir, "wavs", f"{utt}.wav")
            elif len(parts) >= 5:
                utt, prompt_text, prompt_wav_rel, gt_text, gt_wav_rel = parts[:5]
                gt_wav = os.path.join(base_data_dir, gt_wav_rel)
            else:
                continue

            prompt_wav = os.path.join(base_data_dir, prompt_wav_rel)
            items.append({
                "utt_id": utt,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                "gt_text": gt_text,
                "gt_wav": gt_wav,
            })
            if len(items) >= n:
                break
    return items


def main():
    print("=" * 80)
    print("TASK 2 SANITY CHECK: 5-UTTERANCE SMOKE TEST (STEP-8 REFERENCE)")
    print("=" * 80)

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    device = "cpu"
    base_seed = 42

    print(f"Loading Seed-TTS utterances from: {META_LST}")
    utterances = load_first_n_utterances(META_LST, n=5)
    print(f"Loaded {len(utterances)} candidate utterances.")

    print("\nInitializing DMOSpeech 2 model on CPU...")
    t_load_start = time.time()
    tts = DMOInference(
        student_checkpoint_path=CKPT_MODEL,
        duration_predictor_path=CKPT_DURATION,
        device=device,
    )
    print(f"DMOSpeech 2 loaded in {time.time() - t_load_start:.2f}s")

    print("\nInitializing MetricSuite (Whisper large-v3, WavLM-large, UTMOS)...")
    metric_suite = MetricSuite(device=device)
    print("MetricSuite initialized.")

    results = []

    for i, item in enumerate(utterances, start=1):
        utt_id = item["utt_id"]
        prompt_wav_path = item["prompt_wav"]
        prompt_text = item["prompt_text"]
        gt_text = item["gt_text"]
        gt_wav_path = item["gt_wav"]

        print(f"\n--- [{i}/5] Processing {utt_id} ---")
        input_seed = derive_input_seed(base_seed, utt_id)

        # 1. Load prompt audio
        prompt_audio_raw, sr = torchaudio.load(prompt_wav_path)
        prompt_np = prompt_audio_raw.squeeze().numpy()
        prompt_snr = estimate_prompt_snr(prompt_np, sample_rate=sr)

        # 2. Preprocess prompt audio and text
        ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(prompt_wav_path, prompt_text)
        audio_tensor = torch.from_numpy(prompt_np).float().unsqueeze(0)
        rms = torch.sqrt(torch.mean(torch.square(audio_tensor)))
        if rms < 0.3:
            audio_tensor = audio_tensor * 0.3 / rms
        if sr != tts.target_sample_rate:
            audio_tensor = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio_tensor)
        audio_tensor = audio_tensor.to(device)

        # 3. Predict Duration
        torch.manual_seed(input_seed)
        predicted_dur = tts.predict_duration(
            prompt_wav_path,
            gt_text,
            prompt_text,
            dp_softmax_range=0.7,
            temperature=0,
        )
        target_dur = max(10, int(predicted_dur))

        # 4. Input representations
        text_list = [ref_text_clean + gt_text]
        final_text_list = convert_char_to_pinyin(text_list)
        ref_audio_len = audio_tensor.shape[-1] // tts.hop_length
        total_duration = ref_audio_len + target_dur

        cond_tensor, text_ids, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
            audio_tensor, final_text_list, total_duration
        )
        prompt_len = cond_mask.sum().item()

        # 5. Synthesis with wall-clock timing (Step-8 Reference)
        gen = make_local_generator(input_seed)
        y0 = [torch.randn(dur, 100, dtype=torch.float32, generator=gen) for dur in dur_tensor]
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_synth_start = time.perf_counter()
        endpoints, drift_trace, steps_exec = run_teacher_single_pass(
            tts=tts,
            step_cond=step_cond,
            text=text_ids,
            y0=y0,
            prompt_len=prompt_len,
            target_steps=(8,),
            cfg_strength=2.0,
        )
        mel, gen_audio = run_student_completion(
            tts=tts,
            cond=cond_tensor,
            text=text_ids,
            teacher_endpoint=endpoints[8],
            prompt_len=prompt_len,
            input_seed=input_seed,
        )
        t_synth_end = time.perf_counter()
        runtime_sec = t_synth_end - t_synth_start

        audio_duration_sec = len(gen_audio) / 24000.0
        rtf = runtime_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0

        # Save audio
        gen_audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{utt_id}_step8_ref.wav")
        sf.write(gen_audio_path, gen_audio, 24000)

        # 6. Evaluation metrics
        print("  Evaluating metrics (Whisper large-v3, WavLM-large, UTMOS)...")
        eval_metrics = metric_suite.evaluate_utterance(
            cand_audio=gen_audio,
            gt_audio_path=gt_wav_path,
            gt_text=gt_text,
            ref_metrics=None,  # Step 8 is itself the reference
            sample_rate=24000,
        )

        row = {
            "index": i,
            "utt_id": utt_id,
            "base_seed": base_seed,
            "input_seed": input_seed,
            "gt_text": gt_text,
            "hypo_text": eval_metrics["unnormalized_hypo"],
            "normalized_truth": eval_metrics["normalized_truth"],
            "normalized_hypo": eval_metrics["normalized_hypo"],
            "wer": eval_metrics["wer"],
            "sim": eval_metrics["sim"],
            "utmos": eval_metrics["utmos"],
            "prompt_snr": prompt_snr,
            "runtime_sec": round(runtime_sec, 2),
            "rtf": round(rtf, 2),
            "audio_duration_sec": round(audio_duration_sec, 2),
            "nfe": STEP_METADATA[8]["nfe"],
        }
        results.append(row)

        print(f"  WER: {row['wer'] * 100:.2f}% | SIM: {row['sim']:.4f} | UTMOS: {row['utmos']:.4f}")
        print(f"  Runtime: {row['runtime_sec']:.2f}s (RTF: {row['rtf']:.2f}x)")
        print(f"  Hypo: \"{row['hypo_text']}\"")
        print(f"  Truth: \"{row['gt_text']}\"")

    # Save JSON results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    # Summary table
    mean_wer = np.mean([r["wer"] for r in results])
    mean_sim = np.mean([r["sim"] for r in results])
    mean_utmos = np.mean([r["utmos"] for r in results])
    mean_runtime = np.mean([r["runtime_sec"] for r in results])
    mean_rtf = np.mean([r["rtf"] for r in results])

    print("\n" + "=" * 80)
    print("5-UTTERANCE SMOKE TEST SUMMARY TABLE (STEP-8 REFERENCE)")
    print("=" * 80)
    print("| # | Utterance ID | WER (%) | SIM | UTMOS | Runtime (s) | RTF | NFE |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['index']} | `{r['utt_id']}` | {r['wer']*100:.2f}% | {r['sim']:.4f} | {r['utmos']:.4f} | {r['runtime_sec']:.2f}s | {r['rtf']:.2f}x | {r['nfe']} |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| **Mean** | **5-Sample Average** | **{mean_wer*100:.2f}%** | **{mean_sim:.4f}** | **{mean_utmos:.4f}** | **{mean_runtime:.2f}s** | **{mean_rtf:.2f}x** | **21** |")
    print("=" * 80)


if __name__ == "__main__":
    main()
