"""High-Speed GPU-Accelerated Expanded Benchmark Runner (for Google Colab / CUDA).

Supports:
- Running on NVIDIA CUDA (A100, V100, T4, RTX series) or CPU.
- Standard Held-Out Test Scope (200 speakers / 400 pairs).
- Expanded Scope: All speakers in Seed-TTS test-en (up to 666 speakers / 1,088 utterances).
- Evaluates Candidate Steps {4, 5, 6, 8}, Tangent Drift Probing, SNR Safety Gate,
  and computes WER (Whisper large-v3), SIM (WavLM-large), and UTMOS.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

# Paths setup
COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOVER_DIR = os.path.dirname(COLAB_DIR)
ROOT_DIR = os.path.dirname(HANDOVER_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
if HANDOVER_DIR not in sys.path:
    sys.path.insert(0, HANDOVER_DIR)

from infer import DMOInference
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
from core.sampler import (
    CANDIDATE_STEPS,
    REFERENCE_STEP,
    ALL_TARGET_STEPS,
    STEP_METADATA,
    CFG_STRENGTH,
    derive_input_seed,
    make_local_generator,
    estimate_prompt_snr,
    run_teacher_single_pass,
    run_student_completion,
)
from core.metrics import MetricSuite, check_failure, load_failure_config
from data.make_splits import add_gaussian_noise_15db


def load_expanded_manifest(scope: str = "test200", n_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads either the held-out test manifest or builds an expanded dataset."""
    splits_dir = os.path.join(HANDOVER_DIR, "data", "splits")
    test_manifest_path = os.path.join(splits_dir, "test_manifest.json")

    if scope == "test200":
        print(f"[Dataset] Loading standard held-out test manifest: {test_manifest_path}")
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    elif scope == "expanded":
        meta_lst = os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "meta.lst")
        print(f"[Dataset] Building expanded manifest from Seed-TTS meta.lst: {meta_lst}")
        if not os.path.exists(meta_lst):
            raise FileNotFoundError(f"meta.lst not found at {meta_lst}. Make sure Seed-TTS test-en is downloaded.")

        rows = []
        with open(meta_lst, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    utt_id = parts[0]
                    speaker_id = utt_id.split("-")[0] if "-" in utt_id else utt_id
                    prompt_rel = parts[1]
                    prompt_text = parts[2]
                    gt_rel = parts[3]
                    gt_text = parts[4]

                    p_path = os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", prompt_rel)
                    g_path = os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", gt_rel)

                    if os.path.exists(p_path) and os.path.exists(g_path):
                        in_seed = derive_input_seed(utt_id)
                        # Add clean
                        rows.append({
                            "utt_id": utt_id,
                            "speaker_id": speaker_id,
                            "condition": "clean",
                            "prompt_wav_path": p_path,
                            "prompt_text": prompt_text,
                            "gt_wav_path": g_path,
                            "gt_text": gt_text,
                            "base_seed": 42,
                            "input_seed": in_seed,
                        })
                        # Add noisy15
                        rows.append({
                            "utt_id": utt_id,
                            "speaker_id": speaker_id,
                            "condition": "noisy15",
                            "prompt_wav_path": p_path,
                            "prompt_text": prompt_text,
                            "gt_wav_path": g_path,
                            "gt_text": gt_text,
                            "base_seed": 42,
                            "input_seed": in_seed,
                        })
    else:
        # Custom subset
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

    if n_speakers is not None:
        selected_spks = list(dict.fromkeys([r["speaker_id"] for r in rows]))[:n_speakers]
        rows = [r for r in rows if r["speaker_id"] in selected_spks]

    print(f"[Dataset] Prepared {len(rows)} utterance-condition rows ({len(set(r['speaker_id'] for r in rows))} unique speakers).")
    return rows


def run_gpu_benchmark(
    scope: str = "test200",
    n_speakers: Optional[int] = None,
    device_name: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    print("=" * 80)
    print("HIGH-SPEED GPU-ACCELERATED ADAPTIVE HANDOVER BENCHMARK")
    print("=" * 80)

    # 1. Device selection
    if device_name is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_name

    device = torch.device(device_str)
    print(f"Target Compute Device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    # 2. Checkpoints
    ckpt_model = os.path.join(ROOT_DIR, "ckpts", "model_85000.pt")
    ckpt_dur = os.path.join(ROOT_DIR, "ckpts", "model_1500.pt")
    print(f"Loading Model Checkpoint: {ckpt_model}")
    tts = DMOInference(
        student_checkpoint_path=ckpt_model,
        duration_predictor_path=ckpt_dur,
        device=device,
    )
    print(f"Initializing MetricSuite on {device} (Whisper large-v3, WavLM-large, UTMOS)...")
    metric_suite = MetricSuite(device=device_str)
    failure_cfg = load_failure_config()
    print("[OK] All models initialized.")

    # 3. Data & outputs
    manifest_rows = load_expanded_manifest(scope, n_speakers)
    total_pairs = len(manifest_rows)

    out_dir = output_dir or os.path.join(COLAB_DIR, "results")
    audio_dir = os.path.join(out_dir, "sample_audio")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"gpu_grid_{scope}.csv")
    parquet_path = os.path.join(out_dir, f"gpu_grid_{scope}.parquet")

    rows_buffer: List[Dict[str, Any]] = []
    completed_keys = set()
    if os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            for _, r in df_exist.iterrows():
                completed_keys.add((str(r["utt_id"]), str(r["condition"]), int(r["teacher_step"])))
                rows_buffer.append(r.to_dict())
            print(f"[Resume] Found {len(completed_keys)} completed rows in {csv_path}.")
        except Exception as e:
            print(f"[Resume Warning] {e}")

    t_start = time.time()
    generated_count = 0

    for idx, item in enumerate(manifest_rows, start=1):
        utt_id = item["utt_id"]
        spk_id = item["speaker_id"]
        condition = item["condition"]
        prompt_wav_path = item["prompt_wav_path"]
        prompt_text = item["prompt_text"]
        gt_text = item["gt_text"]
        gt_wav_path = item["gt_wav_path"]
        input_seed = item["input_seed"]

        steps_needed = [s for s in ALL_TARGET_STEPS if (utt_id, condition, s) not in completed_keys]
        if not steps_needed:
            continue

        print(f"[{idx}/{total_pairs}] Processing {utt_id} ({condition}) on {device.type.upper()}...")

        # Load prompt audio
        prompt_raw, sr = torchaudio.load(prompt_wav_path)
        prompt_np = prompt_raw.squeeze().numpy()
        if condition == "noisy15":
            prompt_np, _, injected_snr = add_gaussian_noise_15db(prompt_np, seed=input_seed, target_snr_db=15.0)

        prompt_snr = estimate_prompt_snr(prompt_np, sample_rate=sr)

        # Temporary prompt write for duration predictor
        temp_prompt_path = os.path.join(out_dir, f"temp_{utt_id}_{condition}.wav")
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

        text_list = [ref_text_clean + gt_text]
        final_text_list = convert_char_to_pinyin(text_list)
        ref_audio_len = audio_tensor.shape[-1] // tts.hop_length
        total_duration = ref_audio_len + target_dur

        cond_tensor, text_ids, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
            audio_tensor, final_text_list, total_duration
        )
        prompt_len = cond_mask.sum().item()

        # Teacher single pass
        gen = make_local_generator(input_seed)
        y0 = [torch.randn(dur, 100, dtype=torch.float32, generator=gen) for dur in dur_tensor]
        y0 = pad_sequence(y0, padding_value=0, batch_first=True).to(device)

        endpoints, drift_trace, t_integration = run_teacher_single_pass(
            tts=tts,
            step_cond=step_cond,
            text=text_ids,
            y0=y0,
            prompt_len=prompt_len,
            target_steps=ALL_TARGET_STEPS,
            cfg_strength=CFG_STRENGTH,
        )

        candidate_audios = {}
        candidate_times = {}

        for step in ALL_TARGET_STEPS:
            t0 = time.perf_counter()
            _, syn_audio = run_student_completion(
                tts=tts,
                cond=cond_tensor,
                text=text_ids,
                teacher_endpoint=endpoints[step],
                prompt_len=prompt_len,
                input_seed=input_seed,
            )
            t_s = time.perf_counter() - t0
            candidate_audios[step] = syn_audio
            candidate_times[step] = t_s

        # Ground truth audio
        gt_raw, gt_sr = torchaudio.load(gt_wav_path)
        gt_np = gt_raw.squeeze().numpy()

        ref_audio = candidate_audios[REFERENCE_STEP]
        wer_ref = metric_suite.compute_wer(ref_audio, gt_text, sample_rate=24000)
        sim_ref = metric_suite.compute_similarity(ref_audio, gt_np, sample_rate=24000, gt_sr=gt_sr)
        utmos_ref = metric_suite.compute_utmos(ref_audio, sample_rate=24000)

        for step in ALL_TARGET_STEPS:
            audio_k = candidate_audios[step]
            dur_sec = len(audio_k) / 24000.0
            wall_time = candidate_times[step]
            rtf = wall_time / dur_sec if dur_sec > 0 else 0.0

            if step == REFERENCE_STEP:
                wer_k, sim_k, utmos_k = wer_ref, sim_ref, utmos_ref
                is_fail = False
                fail_diag = {"fail_wer": False, "fail_sim": False}
            else:
                wer_k = metric_suite.compute_wer(audio_k, gt_text, sample_rate=24000)
                sim_k = metric_suite.compute_similarity(audio_k, gt_np, sample_rate=24000, gt_sr=gt_sr)
                utmos_k = metric_suite.compute_utmos(audio_k, sample_rate=24000)
                is_fail, fail_diag = check_failure(wer_k, wer_ref, sim_k, sim_ref, config=failure_cfg)

            row = {
                "utt_id": utt_id,
                "speaker_id": spk_id,
                "condition": condition,
                "teacher_step": step,
                "nfe": STEP_METADATA[step]["nfe"],
                "switch_time": STEP_METADATA[step]["switch_time"],
                "prompt_snr": float(prompt_snr),
                "d_4": float(drift_trace.get(4, -1.0)),
                "d_5": float(drift_trace.get(5, -1.0)),
                "d_6": float(drift_trace.get(6, -1.0)),
                "wer": float(wer_k),
                "similarity": float(sim_k),
                "utmos": float(utmos_k),
                "wer_ref": float(wer_ref),
                "similarity_ref": float(sim_ref),
                "utmos_ref": float(utmos_ref),
                "is_failure": bool(is_fail),
                "fail_wer": bool(fail_diag["fail_wer"]),
                "fail_sim": bool(fail_diag["fail_sim"]),
                "synthesis_time": float(wall_time),
                "audio_duration": float(dur_sec),
                "rtf": float(rtf),
            }
            rows_buffer.append(row)
            completed_keys.add((utt_id, condition, step))
            generated_count += 1

        # Checkpoint every 10 pairs
        if idx % 10 == 0 or idx == total_pairs:
            df_cur = pd.DataFrame(rows_buffer)
            df_cur.to_csv(csv_path, index=False)
            df_cur.to_parquet(parquet_path, index=False)
            elapsed = time.time() - t_start
            print(f"[Checkpoint] Processed {idx}/{total_pairs} pairs | {len(rows_buffer)} rows saved | Elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - t_start
    df_final = pd.DataFrame(rows_buffer)
    df_final.to_csv(csv_path, index=False)
    df_final.to_parquet(parquet_path, index=False)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED ON {device.type.upper()}!")
    print(f"Total Rows: {len(df_final)} | Time: {elapsed_total:.2f}s ({elapsed_total/60:.1f} min)")
    print(f"Results saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Accelerated Adaptive Handover Benchmark")
    parser.add_argument("--scope", type=str, default="test200", choices=["test200", "expanded"], help="Scope: test200 or expanded (all)")
    parser.add_argument("--n_speakers", type=int, default=None, help="Limit to N speakers (for quick tests)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or cuda:0")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    run_gpu_benchmark(scope=args.scope, n_speakers=args.n_speakers, device_name=args.device, output_dir=args.output_dir)
