"""Full GPU Pipeline Timing Benchmark Runner (Chapter 3 Verification).

Measures core synthesis latency across 3 policies:
1. Fixed Step 8 Reference (Always index 8, 21 NFE)
2. SNR-Only Heuristic (SNR >= 20 dB: index 4 [13 NFE], else 8 [21 NFE])
3. Proposed Adaptive Scheduler (Frozen lambda=0.50, SNR gate=20.0 dB, early exit at 4, 5, 6, or 8)

Execution Boundary:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    [Prompt SNR calculation (if policy requires)]
    [Teacher Euler execution with genuine online early stopping]
    [Student consistency completion (j=1, 3 passes)]
    [Vocos neural vocoder decode]
    torch.cuda.synchronize()
    core_synthesis_time = time.perf_counter() - t0

Cohort: Full 500-Speaker Cohort (1,000 condition cases x 3 policies = 3,000 timed runs).
Preserves disjoint splits: 100 dev / 200 calib / 200 test.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

for p in [ROOT_DIR, os.path.join(ROOT_DIR, "src"), HANDOVER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tempfile

from infer import DMOInference
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
from core.sampler import (
    SWAY_TEACHER_GRID,
    STUDENT_TAUS,
    CFG_STRENGTH,
    derive_input_seed,
    make_local_generator,
    estimate_prompt_snr,
    compute_stabilization_drift,
)
from data.make_splits import add_gaussian_noise_15db


def prepare_case_inputs(
    tts: Any,
    prompt_wav_path: str,
    prompt_text: str,
    gt_text: str,
    condition: str,
    input_seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Standard preprocessing matching official infer.py pipeline.
    Executed outside the core synthesis timed region.
    """
    prompt_raw, sr = torchaudio.load(prompt_wav_path)
    prompt_np = prompt_raw.squeeze().numpy()
    if condition == "noisy15":
        prompt_np, _, _ = add_gaussian_noise_15db(
            prompt_np, seed=input_seed, target_snr_db=15.0
        )

    temp_wav = os.path.join(tempfile.gettempdir(), f"temp_input_{input_seed}_{time.time_ns()}.wav")
    sf.write(temp_wav, prompt_np, sr)
    ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(temp_wav, prompt_text)

    # EXACT HISTORICAL CODE: audio_tensor is prompt_np (preserves exact frame alignment & drift)
    audio = torch.from_numpy(prompt_np).float().unsqueeze(0)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.3:
        audio = audio * 0.3 / rms
    if sr != tts.target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio)
    audio = audio.to(device)

    torch.manual_seed(input_seed)
    predicted_dur = tts.predict_duration(
        temp_wav, gt_text, prompt_text, dp_softmax_range=0.7, temperature=0
    )
    target_dur = max(10, int(predicted_dur))
    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass

    text_list = [ref_text_clean + gt_text]
    final_text_list = convert_char_to_pinyin(text_list)
    ref_audio_len = audio.shape[-1] // tts.hop_length
    total_duration = ref_audio_len + target_dur

    cond_tensor, text_ids, step_cond, cond_mask, _, dur_tensor = tts._prepare_inputs(
        audio, final_text_list, total_duration
    )
    prompt_len = cond_mask.sum().item()
    return prompt_np, cond_tensor, text_ids, step_cond, prompt_len, dur_tensor


def run_early_stop_teacher(
    tts: Any,
    step_cond: torch.Tensor,
    text: torch.Tensor,
    y0: torch.Tensor,
    prompt_len: int,
    prompt_snr: float,
    policy: str,
    lambda_thresh: float = 0.50,
    snr_thresh: float = 20.0,
    cfg_strength: float = CFG_STRENGTH,
) -> Tuple[torch.Tensor, int, int, Dict[int, float], float]:
    """
    Teacher Euler solver with genuine online early stopping per policy.
    Zero drift overhead for baselines (Point 1 of Verdict):
    - Fixed Step 8: No drift work, no intermediate endpoint projections.
    - SNR-Only: No drift work, no intermediate endpoint projections.
    - Proposed (SNR < 20 dB): Follows fixed Step 8 fallback with zero drift work.
    - Proposed (SNR >= 20 dB): Compares consecutive endpoints strictly at 4, 5, 6
      (preserving endpoint at 3 for d4, and previous endpoint for d5 and d6).
    """
    device = step_cond.device
    state = y0.clone()
    drift_trace: Dict[int, float] = {}
    prev_endpoint = None
    teacher_calls = 0
    decision_drift = -1.0

    # Determine execution path
    if policy == "step8":
        target_step = 8
        is_dynamic = False
    elif policy == "snr_only":
        target_step = 4 if prompt_snr >= snr_thresh else 8
        is_dynamic = False
    elif policy == "proposed":
        if prompt_snr < snr_thresh:
            target_step = 8  # Fallback due to noise: zero drift work
            is_dynamic = False
        else:
            target_step = None
            is_dynamic = True
    else:
        raise ValueError(f"Unknown policy: {policy}")

    with torch.inference_mode():
        # Fast path for deterministic target_step (step8, snr_only, proposed noisy fallback)
        if not is_dynamic:
            for i in range(target_step + 1):
                t_curr_val = SWAY_TEACHER_GRID[i]
                t_curr = torch.tensor(t_curr_val, device=device, dtype=torch.float32)

                pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=False, drop_text=False)
                null_pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=True, drop_text=True)
                teacher_calls += 2
                v_curr = pred + (pred - null_pred) * cfg_strength

                if i == target_step:
                    endpoint = state + (1.0 - t_curr_val) * v_curr
                    return endpoint, target_step, teacher_calls, {}, -1.0

                dt = SWAY_TEACHER_GRID[i + 1] - t_curr_val
                state = state + dt * v_curr

        # Dynamic path: Proposed policy on clean speech (prompt_snr >= 20 dB)
        # Evaluates candidate decisions only at indices 4, 5, 6
        for i in range(9):
            t_curr_val = SWAY_TEACHER_GRID[i]
            t_curr = torch.tensor(t_curr_val, device=device, dtype=torch.float32)

            pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=False, drop_text=False)
            null_pred = tts.teacher(x=state, cond=step_cond, text=text, time=t_curr, mask=None, drop_audio_cond=True, drop_text=True)
            teacher_calls += 2
            v_curr = pred + (pred - null_pred) * cfg_strength

            # Index 3: Preserve endpoint to compute d_4 at the next step
            if i == 3:
                prev_endpoint = state + (1.0 - t_curr_val) * v_curr

            # Indices 4, 5, 6: Consecutive drift evaluations
            elif i in [4, 5, 6]:
                endpoint_curr = state + (1.0 - t_curr_val) * v_curr
                d_i = compute_stabilization_drift(endpoint_curr, prev_endpoint, prompt_len)
                drift_trace[i] = d_i

                if d_i <= lambda_thresh:
                    return endpoint_curr, i, teacher_calls, drift_trace, d_i

                # Retain immediately previous endpoint for next consecutive candidate
                prev_endpoint = endpoint_curr

            # Index 8: Final fallback
            elif i == 8:
                endpoint_curr = state + (1.0 - t_curr_val) * v_curr
                decision_drift = drift_trace.get(6, -1.0)
                return endpoint_curr, 8, teacher_calls, drift_trace, decision_drift

            # State evolution
            dt = SWAY_TEACHER_GRID[i + 1] - t_curr_val
            state = state + dt * v_curr

    return endpoint_curr, 8, teacher_calls, drift_trace, decision_drift


def run_student_completion_timed(
    tts: Any,
    cond: torch.Tensor,
    text: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    prompt_len: int,
    input_seed: int,
) -> Tuple[np.ndarray, int]:
    """
    Executes student consistency completion (j=1, 3 passes) + Vocos vocoder decoding.
    Counts actual forward evaluations.
    """
    device = cond.device
    x_curr = teacher_endpoint.clone()
    student_calls = 0

    with torch.inference_mode():
        for tau in STUDENT_TAUS:
            student_seed = (int(input_seed) + int(tau * 100)) % (2**31 - 1)
            gen_s = make_local_generator(student_seed, device=device)
            eps = torch.randn(x_curr.shape, dtype=torch.float32, device=device, generator=gen_s)
            time_tensor = torch.tensor([tau], device=device, dtype=torch.float32)
            t_scalar = time_tensor.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - t_scalar) * eps + t_scalar * x_curr

            pred_s = tts.generator(
                x=phi, cond=cond, text=text, time=time_tensor,
                drop_audio_cond=False, drop_text=False
            )
            student_calls += 1
            x_curr = phi + (1.0 - t_scalar) * pred_s

        scale = getattr(tts, "scale", 1.0)
        mel = x_curr.permute(0, 2, 1) * scale
        generated_wave = tts.vocos.decode(mel[..., prompt_len:])
        audio = generated_wave.cpu().numpy().squeeze()

    return audio, student_calls


def load_all_split_manifests() -> List[Dict[str, Any]]:
    """Loads all 500 speakers preserving split labels."""
    splits_dir = os.path.join(HANDOVER_DIR, "data", "splits")
    manifest_files = [
        ("dev", os.path.join(splits_dir, "development_manifest.json")),
        ("calib", os.path.join(splits_dir, "calibration_manifest.json")),
        ("test", os.path.join(splits_dir, "test_manifest.json")),
    ]
    items = []
    for split_name, m_path in manifest_files:
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Missing split manifest: {m_path}")
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                d["split"] = split_name
                items.append(d)
    return items


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Core-Synthesis GPU Timing Benchmark")
    parser.add_argument("--scope", type=str, default="all", choices=["all", "test", "calib", "dev"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join(COLAB_DIR, "results"))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--warmup_runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gpu_full_pipeline_timing_3000.csv")
    parquet_path = os.path.join(output_dir, "gpu_full_pipeline_timing_3000.parquet")
    session_id = f"timing_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 80)
    print("DMOSPEECH 2: FULL PIPELINE CORE-SYNTHESIS TIMING BENCHMARK")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Output File: {csv_path}")
    print(f"Session ID: {session_id}")
    print("=" * 80)

    # 1. Load Model
    ckpt_model = os.path.join(ROOT_DIR, "ckpts", "model_85000.pt")
    ckpt_duration = os.path.join(ROOT_DIR, "ckpts", "model_1500.pt")
    tts = DMOInference(
        student_checkpoint_path=ckpt_model,
        duration_predictor_path=ckpt_duration,
        device=str(device),
    )

    # 2. Load Manifest
    all_items = load_all_split_manifests()
    if args.scope != "all":
        all_items = [it for it in all_items if it["split"] == args.scope]
    total_cases = len(all_items)
    print(f"Loaded {total_cases} condition-specific cases to evaluate.")

    # 3. Check for existing progress
    completed_keys = set()
    rows_buffer = []
    if args.resume and os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            rows_buffer = df_existing.to_dict("records")
            for r in rows_buffer:
                completed_keys.add((r["utt_id"], r["condition"], r["policy"]))
            print(f"Resuming: Found {len(rows_buffer)} existing runs ({len(completed_keys)} keys).")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")

    # 4. Excluded Warm-up Generations
    print(f"\nExecuting {args.warmup_runs} warm-up runs to prime GPU kernels and memory pools...")
    sample_item = all_items[0]
    prompt_wav_path = sample_item["prompt_wav_path"]
    if not os.path.exists(prompt_wav_path):
        candidates = [
            os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "prompt-wavs", os.path.basename(prompt_wav_path)),
            os.path.join(ROOT_DIR, "data", "seed-tts-eval", "prompt-wavs", os.path.basename(prompt_wav_path)),
        ]
        for c in candidates:
            if os.path.exists(c):
                prompt_wav_path = c
                break

    for w_i in range(args.warmup_runs):
        p_pol = ["step8", "snr_only", "proposed"][w_i % 3]
        prompt_np, cond_tensor, text_ids, step_cond, p_len, dur_tensor = prepare_case_inputs(
            tts=tts,
            prompt_wav_path=prompt_wav_path,
            prompt_text=sample_item["prompt_text"],
            gt_text=sample_item["gt_text"],
            condition="clean",
            input_seed=12345 + w_i,
            device=device,
        )
        gen_w = make_local_generator(12345 + w_i, device="cpu")
        y0_w = pad_sequence([torch.randn(dur, 100, dtype=torch.float32, generator=gen_w) for dur in dur_tensor],
                            padding_value=0, batch_first=True).to(device)
        snr_w = estimate_prompt_snr(prompt_np, 24000)
        ep, ex_s, _, _, _ = run_early_stop_teacher(tts, step_cond, text_ids, y0_w, p_len, snr_w, p_pol)
        _ = run_student_completion_timed(tts, cond_tensor, text_ids, ep, p_len, 12345)
    print("Warm-up completed successfully.\n")

    # 5. Timing Benchmark Loop
    policies = ["step8", "snr_only", "proposed"]
    t_start = time.time()

    for idx, item in enumerate(all_items, 1):
        utt_id = item["utt_id"]
        spk_id = item["speaker_id"]
        cond = item["condition"]
        split = item["split"]
        base_seed = item.get("base_seed", 42)
        input_seed = item.get("input_seed", derive_input_seed(base_seed, utt_id))

        # Check if all 3 policies already completed for this case
        if all((utt_id, cond, pol) in completed_keys for pol in policies):
            continue

        # Prepare audio paths
        p_path = item["prompt_wav_path"]
        if not os.path.exists(p_path):
            candidates = [
                os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "prompt-wavs", os.path.basename(p_path)),
                os.path.join(ROOT_DIR, "data", "seed-tts-eval", "prompt-wavs", os.path.basename(p_path)),
            ]
            for c in candidates:
                if os.path.exists(c):
                    p_path = c
                    break

        # Standard preprocessing outside timed region
        prompt_np, cond_tensor, text_ids, step_cond, prompt_len, dur_tensor = prepare_case_inputs(
            tts=tts,
            prompt_wav_path=p_path,
            prompt_text=item["prompt_text"],
            gt_text=item["gt_text"],
            condition=cond,
            input_seed=input_seed,
            device=device,
        )

        # Shared initial noise drawn identically outside timer
        gen = make_local_generator(input_seed, device="cpu")
        y0_list = [torch.randn(dur, 100, dtype=torch.float32, generator=gen) for dur in dur_tensor]
        y0 = pad_sequence(y0_list, padding_value=0, batch_first=True).to(device)

        # Rotate policy execution order to balance sequential timing bias
        order_rotation = idx % 3
        if order_rotation == 0:
            case_policies = ["step8", "snr_only", "proposed"]
        elif order_rotation == 1:
            case_policies = ["proposed", "step8", "snr_only"]
        else:
            case_policies = ["snr_only", "proposed", "step8"]

        for order_idx, pol in enumerate(case_policies, 1):
            if (utt_id, cond, pol) in completed_keys:
                continue

            # Core Synthesis Latency Timed Region
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 1. Prompt SNR (measured for SNR-only and Proposed)
            if pol == "step8":
                measured_snr = 0.0  # Not computed for fixed step8
            else:
                measured_snr = estimate_prompt_snr(prompt_np, sample_rate=24000)

            # 2. Teacher execution with genuine early stopping
            teacher_ep, exit_step, teacher_nfe, drift_trace, decision_drift = run_early_stop_teacher(
                tts=tts,
                step_cond=step_cond,
                text=text_ids,
                y0=y0,
                prompt_len=prompt_len,
                prompt_snr=measured_snr,
                policy=pol,
                lambda_thresh=0.50,
                snr_thresh=20.0,
            )

            # 3. Student completion (3 consistency steps) + Vocos vocoder
            syn_audio, student_nfe = run_student_completion_timed(
                tts=tts,
                cond=cond_tensor,
                text=text_ids,
                teacher_endpoint=teacher_ep,
                prompt_len=prompt_len,
                input_seed=input_seed,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            core_time = time.perf_counter() - t0

            audio_dur = len(syn_audio) / 24000.0
            rtf = core_time / audio_dur if audio_dur > 0 else 0.0
            total_nfe = teacher_nfe + student_nfe

            row = {
                "utt_id": utt_id,
                "speaker_id": spk_id,
                "split": split,
                "condition": cond,
                "policy": pol,
                "measurement_order": order_idx,
                "exit_step": exit_step,
                "prompt_snr": float(measured_snr),
                "decision_drift": float(decision_drift),
                "teacher_nfe": int(teacher_nfe),
                "student_nfe": int(student_nfe),
                "total_nfe": int(total_nfe),
                "core_synthesis_time": float(core_time),
                "audio_duration": float(audio_dur),
                "rtf": float(rtf),
                "input_seed": int(input_seed),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "session_id": session_id,
            }
            rows_buffer.append(row)
            completed_keys.add((utt_id, cond, pol))

        # Checkpoint every 10 cases (30 rows)
        if idx % 10 == 0 or idx == total_cases:
            df_save = pd.DataFrame(rows_buffer)
            df_save.to_csv(csv_path, index=False)
            df_save.to_parquet(parquet_path, index=False)
            elapsed = time.time() - t_start
            print(f"[Progress] Processed {idx}/{total_cases} cases ({len(rows_buffer)} runs saved) | Elapsed: {elapsed:.1f}s")

    # Final Save & Summary
    df_final = pd.DataFrame(rows_buffer)
    df_final.to_csv(csv_path, index=False)
    df_final.to_parquet(parquet_path, index=False)
    elapsed_total = time.time() - t_start

    print("\n" + "=" * 80)
    print("TIMING BENCHMARK COMPLETE!")
    print(f"Total Rows Saved: {len(df_final)} | Total Time: {elapsed_total:.2f}s ({elapsed_total/60:.1f} min)")
    print(f"Saved CSV: {csv_path}")
    print("=" * 80)

    # Generate full analysis and verification reports
    generate_full_analysis_reports(df_final, args.output_dir)


def speaker_cluster_bootstrap(
    df_subset: pd.DataFrame,
    val_col: str = "core_synthesis_time",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% bootstrap confidence intervals clustered at the speaker level.
    Resamples unique speaker IDs with replacement, keeping each speaker's clean
    and noisy15 conditions grouped together.
    """
    rng = np.random.default_rng(seed)
    unique_spks = df_subset["speaker_id"].unique()
    n_spk = len(unique_spks)

    # Pre-group by speaker
    spk_groups = {spk: grp for spk, grp in df_subset.groupby("speaker_id")}

    policies = sorted(df_subset["policy"].unique())
    boot_means = {pol: [] for pol in policies}
    boot_diffs_step8 = {pol: [] for pol in policies if pol != "step8"}
    boot_diff_prop_snr = []

    for _ in range(n_boot):
        sample_spks = rng.choice(unique_spks, size=n_spk, replace=True)
        sampled_rows = pd.concat([spk_groups[s] for s in sample_spks], ignore_index=True)

        pol_means = sampled_rows.groupby("policy")[val_col].mean().to_dict()
        for pol in policies:
            if pol in pol_means:
                boot_means[pol].append(pol_means[pol])

        if "step8" in pol_means:
            for pol in boot_diffs_step8:
                if pol in pol_means:
                    boot_diffs_step8[pol].append(pol_means["step8"] - pol_means[pol])

        if "proposed" in pol_means and "snr_only" in pol_means:
            boot_diff_prop_snr.append(pol_means["snr_only"] - pol_means["proposed"])

    results = {}
    for pol in policies:
        arr = np.array(boot_means[pol])
        results[pol] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    for pol, arr_diff in boot_diffs_step8.items():
        arr = np.array(arr_diff)
        results[f"{pol}_savings_vs_step8"] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    if boot_diff_prop_snr:
        arr = np.array(boot_diff_prop_snr)
        results["proposed_vs_snr_only"] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    return results


def generate_full_analysis_reports(df: pd.DataFrame, output_dir: str):
    """
    Generates all 6 analytical outputs specified by VERDICT_GPU_FULL_PIPELINE_TIMING_RERUN.md:
    1. Summary by policy (mean, median, std, IQR, mean RTF, mean NFE).
    2. Paired latency differences & % savings vs Step 8 using matched cases.
    3. Direct Proposed vs SNR-Only timing comparison.
    4. Condition breakdown (Clean vs Noisy15).
    5. Split breakdown (Dev 100, Calib 200, Test 200) with speaker cluster bootstrap.
    6. Correctness verification of selected steps and forward evaluation counts.
    """
    from scipy import stats

    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE TIMING ANALYSIS & VERIFICATION REPORTS")
    print("=" * 80)

    # 1. Primary Evaluation: Held-Out Test Cohort (200 Speakers / 400 Cases)
    df_test = df[df["split"] == "test"].copy()
    iqr_fn = lambda x: np.percentile(x, 75) - np.percentile(x, 25)

    test_summary = df_test.groupby("policy").agg(
        mean_latency=("core_synthesis_time", "mean"),
        median_latency=("core_synthesis_time", "median"),
        std_latency=("core_synthesis_time", "std"),
        iqr_latency=("core_synthesis_time", iqr_fn),
        mean_rtf=("rtf", "mean"),
        mean_nfe=("total_nfe", "mean"),
    ).reset_index()

    step8_test_mean = test_summary.loc[test_summary["policy"] == "step8", "mean_latency"].values[0]
    test_summary["latency_savings_%"] = (step8_test_mean - test_summary["mean_latency"]) / step8_test_mean * 100.0

    print("\n--- [Table 1] Primary Evaluation: Held-Out Test Cohort (200 Speakers) ---")
    print(test_summary.to_string(index=False))

    # 2. Paired Differences on Held-Out Test Cohort (Speaker-Level Aggregation, N=200)
    # Pivot by speaker_id and condition
    piv_test = df_test.pivot_table(index=["speaker_id", "condition"], columns="policy", values="core_synthesis_time").reset_index()
    piv_test["diff_prop"] = piv_test["step8"] - piv_test["proposed"]
    piv_test["diff_snr"] = piv_test["step8"] - piv_test["snr_only"]
    piv_test["diff_prop_vs_snr"] = piv_test["snr_only"] - piv_test["proposed"]

    # Aggregate clean and noisy differences to one prespecified speaker-level value (N=200)
    spk_agg = piv_test.groupby("speaker_id").agg(
        mean_step8=("step8", "mean"),
        mean_prop=("proposed", "mean"),
        mean_snr=("snr_only", "mean"),
        spk_diff_prop=("diff_prop", "mean"),
        spk_diff_snr=("diff_snr", "mean"),
        spk_diff_prop_vs_snr=("diff_prop_vs_snr", "mean"),
    ).reset_index()

    # Pre-specified hypothesis test directions (Point 2 of Verdict):
    # Proposed vs Step 8: greater (Proposed is faster, latency difference > 0)
    w_prop = stats.wilcoxon(spk_agg["spk_diff_prop"], alternative="greater")
    # SNR-Only vs Step 8: greater (SNR-Only is faster, latency difference > 0)
    w_snr = stats.wilcoxon(spk_agg["spk_diff_snr"], alternative="greater")
    # Proposed vs SNR-Only: two-sided
    w_prop_vs_snr = stats.wilcoxon(spk_agg["spk_diff_prop_vs_snr"], alternative="two-sided")

    # Percentage savings metrics:
    # 1. Ratio of mean latencies: (Mean(Step8) - Mean(Pol)) / Mean(Step8) * 100%
    ratio_savings_prop = (spk_agg["mean_step8"].mean() - spk_agg["mean_prop"].mean()) / spk_agg["mean_step8"].mean() * 100.0
    ratio_savings_snr = (spk_agg["mean_step8"].mean() - spk_agg["mean_snr"].mean()) / spk_agg["mean_step8"].mean() * 100.0

    # 2. Mean of speaker-level percentage savings: Mean_s( (Step8(s) - Pol(s)) / Step8(s) * 100% )
    spk_pct_savings_prop = (spk_agg["spk_diff_prop"] / spk_agg["mean_step8"] * 100.0).mean()
    spk_pct_savings_snr = (spk_agg["spk_diff_snr"] / spk_agg["mean_step8"] * 100.0).mean()

    print("\n--- [Table 2] Paired Latency Differences vs. Step 8 (200 Held-Out Speakers, Clean & Noisy Aggregated) ---")
    print(f"Proposed vs. Step 8 : Mean Diff = {spk_agg['spk_diff_prop'].mean():.4f}s | Ratio Savings = {ratio_savings_prop:.2f}% | Mean Per-Speaker Savings = {spk_pct_savings_prop:.2f}% | Wilcoxon p = {w_prop.pvalue:.4e}")
    print(f"SNR-Only vs. Step 8 : Mean Diff = {spk_agg['spk_diff_snr'].mean():.4f}s | Ratio Savings = {ratio_savings_snr:.2f}% | Mean Per-Speaker Savings = {spk_pct_savings_snr:.2f}% | Wilcoxon p = {w_snr.pvalue:.4e}")
    print(f"Proposed vs SNR-Only: Mean Delta = {spk_agg['spk_diff_prop_vs_snr'].mean():.4f}s | Median Delta = {spk_agg['spk_diff_prop_vs_snr'].median():.4f}s | Wilcoxon two-sided p = {w_prop_vs_snr.pvalue:.4e}")

    # 3. Speaker-Level Cluster Bootstrap (200 Test Speakers)
    boot_test = speaker_cluster_bootstrap(df_test, val_col="core_synthesis_time", n_boot=1000)
    print("\n--- [Table 3] Speaker-Clustered Bootstrap 95% CIs (200 Test Speakers, Clean & Noisy Paired) ---")
    for k, v in boot_test.items():
        print(f"  {k:25s}: Mean = {v['mean']:.4f}s [95% CI: {v['ci_low']:.4f}s, {v['ci_high']:.4f}s]")

    # 4. Clean vs Noisy Breakdown
    cond_summary = df_test.groupby(["condition", "policy"]).agg(
        mean_latency=("core_synthesis_time", "mean"),
        median_latency=("core_synthesis_time", "median"),
        mean_rtf=("rtf", "mean"),
        mean_nfe=("total_nfe", "mean"),
    ).reset_index()
    print("\n--- [Table 4] Breakdown by Acoustic Condition (Test Cohort) ---")
    print(cond_summary.to_string(index=False))

    # 5. Split Breakdown Across All 500 Speakers
    split_summary = df.groupby(["split", "policy"]).agg(
        mean_latency=("core_synthesis_time", "mean"),
        mean_rtf=("rtf", "mean"),
        mean_nfe=("total_nfe", "mean"),
    ).reset_index()

    # 6. Correctness Verification of Forward Evaluations (NFE Accounting)
    print("\n--- [Verification] Neural Function Evaluation (NFE) Accounting ---")
    expected_nfe = {4: (10, 3, 13), 5: (12, 3, 15), 6: (14, 3, 17), 8: (18, 3, 21)}
    mismatches = 0
    for _, r in df.iterrows():
        st = int(r["exit_step"])
        exp_t, exp_s, exp_tot = expected_nfe[st]
        if (int(r["teacher_nfe"]), int(r["student_nfe"]), int(r["total_nfe"])) != (exp_t, exp_s, exp_tot):
            mismatches += 1
    print(f"Forward Evaluation Mismatches: {mismatches} / {len(df)} (100% Verified Consistent: {mismatches == 0})")

    # Save summary tables to disk
    test_summary.to_csv(os.path.join(output_dir, "gpu_timing_summary_test.csv"), index=False)
    cond_summary.to_csv(os.path.join(output_dir, "gpu_timing_summary_by_condition.csv"), index=False)
    split_summary.to_csv(os.path.join(output_dir, "gpu_timing_summary_by_split.csv"), index=False)

    df_boot = pd.DataFrame([{"metric": k, **v} for k, v in boot_test.items()])
    df_boot.to_csv(os.path.join(output_dir, "gpu_timing_cluster_bootstrap_ci.csv"), index=False)
    print(f"\nAll analytical reports saved to: {output_dir}")


if __name__ == "__main__":
    main()

