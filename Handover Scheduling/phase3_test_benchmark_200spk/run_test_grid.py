"""
Objective 4 Held-Out Test Split Grid Runner (Task 6 / Phase 3)
==============================================================
Runs the full grid of candidate handover steps {4, 5, 6, 8} on the
HELD-OUT TEST SPLIT ONLY under clean and noisy15 conditions.

Specifications (Chapter 3 / Task 6):
- Held-out test split only (200 speakers / 400 utterance-condition pairs)
- Data isolation: Calibrated threshold lambda_hat = 0.50 and SNR threshold 20.0 dB
  are permanently frozen and NEVER tuned or modified based on test outcomes.
- Single-pass teacher integration through Step 8 serving all candidates
- Student consistency restart (j=1, 3 steps) with matched local seeds
- Metric evaluation: Whisper large-v3 WER, WavLM-large SIM vs GT, UTMOS
- Definition 3.1 Failure detection strictly against matched Step-8 reference
- Resume-safe: checkpoints after every pair and skips completed (utt_id, condition, step) rows
- Saves audio for a deterministic 10% sample only
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

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

TEST_MANIFEST_PATH = os.path.join(HANDOVER_DIR, "data", "splits", "test_manifest.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
AUDIO_SAMPLE_DIR = os.path.join(RESULTS_DIR, "sample_audio")
GRID_CSV_PATH = os.path.join(RESULTS_DIR, "test_grid.csv")
GRID_PARQUET_PATH = os.path.join(RESULTS_DIR, "test_grid.parquet")

CKPT_MODEL = os.path.join(ROOT_DIR, "ckpts", "model_85000.pt")
CKPT_DURATION = os.path.join(ROOT_DIR, "ckpts", "model_1500.pt")


def is_sample_audio_target(utt_id: str, condition: str, sampling_rate: float = 0.10) -> bool:
    """Deterministically selects a 10% sample of utterances to save audio."""
    import hashlib
    key = f"{utt_id}_{condition}_test_audio_sample"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    val = int(digest[:6], 16) / 0xFFFFFF
    return val < sampling_rate


def run_test_grid(max_instances: Optional[int] = None):
    print("=" * 80)
    print("OBJECTIVE 4: HELD-OUT TEST SPLIT GRID RUNNER (TASK 6 / PHASE 3)")
    print("=" * 80)

    # 1. Verification of data isolation (Task 6 Security Check)
    assert os.path.basename(TEST_MANIFEST_PATH) == "test_manifest.json", (
        "Security check failed: Objective 4 runner must ONLY run on test_manifest.json!"
    )
    print(f"Loading test manifest: {TEST_MANIFEST_PATH}")
    with open(TEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = json.load(f)

    if max_instances is not None:
        manifest_rows = manifest_rows[:max_instances]

    total_pairs = len(manifest_rows)
    print(f"Loaded {total_pairs} utterance-condition pairs (200 speakers across clean and noisy15).")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(AUDIO_SAMPLE_DIR, exist_ok=True)

    # 2. Check for existing progress (Resume safety)
    rows_buffer: List[Dict[str, Any]] = []
    completed_keys = set()
    if os.path.exists(GRID_CSV_PATH):
        try:
            existing_df = pd.read_csv(GRID_CSV_PATH)
            for _, r in existing_df.iterrows():
                completed_keys.add((str(r["utt_id"]), str(r["condition"]), int(r["teacher_step"])))
                rows_buffer.append(r.to_dict())
            print(f"Resuming: found {len(completed_keys)} previously completed rows in {GRID_CSV_PATH}.")
        except Exception as e:
            print(f"Warning: Could not read existing grid CSV: {e}")

    # 3. Initialize models on host CPU
    device = "cpu"
    print("\nLoading DMOSpeech 2 model on CPU...")
    tts = DMOInference(
        student_checkpoint_path=CKPT_MODEL,
        duration_predictor_path=CKPT_DURATION,
        device=device,
    )
    print("Initializing Chapter 3 MetricSuite (Whisper large-v3, WavLM-large, UTMOS)...")
    metric_suite = MetricSuite(device=device)
    failure_cfg = load_failure_config()
    print("Models successfully initialized.")

    t_grid_start = time.time()

    for idx, item in enumerate(manifest_rows, start=1):
        utt_id = item["utt_id"]
        spk_id = item["speaker_id"]
        condition = item["condition"]
        prompt_wav_path = item["prompt_wav_path"]
        prompt_text = item["prompt_text"]
        gt_text = item["gt_text"]
        gt_wav_path = item["gt_wav_path"]
        base_seed = item["base_seed"]
        input_seed = item["input_seed"]

        # Check if all 4 candidate steps are already done for this pair
        steps_needed = [s for s in ALL_TARGET_STEPS if (utt_id, condition, s) not in completed_keys]
        if not steps_needed:
            continue

        print(f"\n[{idx}/{total_pairs}] Processing {utt_id} | Condition: {condition} (needed steps: {steps_needed})")

        # 4. Load & condition prompt audio
        prompt_raw, sr = torchaudio.load(prompt_wav_path)
        prompt_np = prompt_raw.squeeze().numpy()

        if condition == "noisy15":
            prompt_np, _, injected_snr = add_gaussian_noise_15db(
                prompt_np, seed=input_seed, target_snr_db=15.0
            )

        # Prompt SNR is estimated strictly on the prompt before RMS normalization
        prompt_snr = estimate_prompt_snr(prompt_np, sample_rate=sr)

        # 5. Preprocessing & Duration prediction
        temp_prompt_path = os.path.join(RESULTS_DIR, f"temp_prompt_test_{idx}.wav")
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

        # 6. Single teacher pass serving all steps {4, 5, 6, 8}
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
            cfg_strength=CFG_STRENGTH,
        )

        save_audio_this_pair = is_sample_audio_target(utt_id, condition)

        # 7. Synthesize and evaluate all candidate steps
        step_outputs: Dict[int, Dict[str, Any]] = {}
        for step in ALL_TARGET_STEPS:
            t_student_start = time.perf_counter()
            mel, gen_audio = run_student_completion(
                tts=tts,
                cond=cond_tensor,
                text=text_ids,
                teacher_endpoint=endpoints[step],
                prompt_len=prompt_len,
                input_seed=input_seed,
            )
            runtime_sec = time.perf_counter() - t_student_start

            if save_audio_this_pair:
                out_wav_path = os.path.join(AUDIO_SAMPLE_DIR, f"{utt_id}_{condition}_step{step}.wav")
                sf.write(out_wav_path, gen_audio, 24000)

            eval_res = metric_suite.evaluate_utterance(
                cand_audio=gen_audio,
                gt_audio_path=gt_wav_path,
                gt_text=gt_text,
                ref_metrics=None,
                sample_rate=24000,
            )
            eval_res["runtime_sec"] = runtime_sec
            step_outputs[step] = eval_res

        # 8. Compute failure against the matched Step-8 Reference (Definition 3.1)
        ref_wer = step_outputs[REFERENCE_STEP]["wer"]
        ref_sim = step_outputs[REFERENCE_STEP]["sim"]

        for step in ALL_TARGET_STEPS:
            out = step_outputs[step]
            is_fail, diag = check_failure(
                wer_cand=out["wer"],
                wer_ref=ref_wer,
                sim_cand=out["sim"],
                sim_ref=ref_sim,
                config=failure_cfg,
            )
            # Step 8 is ref -> fail = 0 by definition
            if step == REFERENCE_STEP:
                is_fail = False

            meta = STEP_METADATA[step]
            row_dict = {
                "utt_id": utt_id,
                "speaker_id": spk_id,
                "condition": condition,
                "teacher_step": step,
                "actual_teacher_time": meta["actual_teacher_time"],
                "legacy_label": meta["legacy_label"],
                "nfe": meta["nfe"],
                "solver_steps": meta["solver_steps"],
                "wer": out["wer"],
                "sim": out["sim"],
                "utmos": out["utmos"],
                "raw_transcript": out["unnormalized_hypo"],
                "normalized_transcript": out["normalized_hypo"],
                "normalized_target_text": out["normalized_truth"],
                "runtime_sec": round(out["runtime_sec"], 3),
                "prompt_snr": round(prompt_snr, 2),
                "base_seed": base_seed,
                "input_seed": input_seed,
                "fail": int(is_fail),
                "wer_failed": int(diag["wer_failed"]),
                "sim_failed": int(diag["sim_failed"]),
            }
            # Record drift trace d_1..d_8
            for d_idx in range(1, 9):
                row_dict[f"d_{d_idx}"] = round(drift_trace.get(d_idx, 0.0), 4)

            rows_buffer.append(row_dict)
            completed_keys.add((utt_id, condition, step))

        # Checkpoint save after every pair
        df_save = pd.DataFrame(rows_buffer)
        df_save.to_csv(GRID_CSV_PATH, index=False)
        df_save.to_parquet(GRID_PARQUET_PATH, index=False)

        ref_row = [r for r in rows_buffer if r["utt_id"] == utt_id and r["condition"] == condition and r["teacher_step"] == 8][-1]
        print(f"  [Ref Step 8] WER: {ref_row['wer']*100:.2f}% | SIM: {ref_row['sim']:.4f} | UTMOS: {ref_row['utmos']:.4f}")
        for st in (4, 5, 6):
            c_row = [r for r in rows_buffer if r["utt_id"] == utt_id and r["condition"] == condition and r["teacher_step"] == st][-1]
            status_str = "FAIL" if c_row["fail"] else "PASS"
            print(f"  [Step {st}] WER: {c_row['wer']*100:.2f}% | SIM: {c_row['sim']:.4f} | Fail: {status_str} (NFE {c_row['nfe']})")

    elapsed_total = time.time() - t_grid_start
    print("\n" + "=" * 80)
    print(f"OBJECTIVE 4 TEST SPLIT GRID RUN COMPLETED IN {elapsed_total / 60.0:.2f} MINUTES")
    print(f"Saved total {len(rows_buffer)} rows to:")
    print(f"  {GRID_CSV_PATH}")
    print(f"  {GRID_PARQUET_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    max_inst = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_test_grid(max_instances=max_inst)
