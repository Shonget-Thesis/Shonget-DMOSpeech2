"""Drift Justification Benchmark (SNR gate vs. drift gate under prompt corruptions).

Question: does trajectory drift catch prompt degradations that the energy-percentile
SNR estimator cannot see? Existing data (clean + 15 dB Gaussian) shows drift and SNR are
redundant, so this runner tests corruptions that are NOT additive stationary noise:

- reverb   : synthetic exponential-decay room impulse responses (RT60 0.3 / 0.6 / 0.9 s)
- babble   : speech babble mixed from 6 other Seed-TTS prompts (15 / 20 dB)
- demand   : real DEMAND cafeteria noise PCAFETER ch01 (15 / 20 dB, optional download)
- phone    : 8 kHz telephone band (300-3400 Hz) + 8-bit mu-law quantization
- gauss    : borderline white Gaussian noise (18 / 20 / 22 / 25 dB)

Speakers: by default the 166 Seed-TTS speakers NOT used in the dev/calib/test splits,
so nothing here is tuned on data the gates were chosen from.

All decision rules are frozen in a preregistration JSON written BEFORE any synthesis.
The analysis only reports computed numbers; nothing is hardcoded.

Usage:
    python run_drift_justification_benchmark.py --device cuda --conditions quick
    python run_drift_justification_benchmark.py --device cuda --conditions all
    python run_drift_justification_benchmark.py --analyze_only
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Paths setup (mirrors run_gpu_expanded_benchmark.py)
COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOVER_DIR = os.path.dirname(COLAB_DIR)
ROOT_DIR = os.path.dirname(HANDOVER_DIR)

for p in [ROOT_DIR, os.path.join(ROOT_DIR, "src"), HANDOVER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_OUT_DIR = os.path.join(COLAB_DIR, "results", "drift_justification")
META_LST = os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "meta.lst")
SPLITS_DIR = os.path.join(HANDOVER_DIR, "data", "splits")
DEMAND_WAV = os.path.join(ROOT_DIR, "data", "demand", "PCAFETER", "ch01.wav")

# ============================================================================
# FROZEN PREREGISTRATION (do not edit after the first run)
# ============================================================================
# Thresholds come from the existing dev/calib CPU grids (500 speakers):
#   - SNR gate  : S_hat >= 20 dB  (the certified theta from Chapter 3)
#   - Drift gate: d_4 > 0.30      (clean min d_4 ~0.18, noisy15 max ~0.38; 0.30 matched the
#                                  SNR gate on calib: 2.5% risk, 18.06 vs 18.02 NFE)
# Direction: LOW drift = degraded prompt (found on dev/calib, confirmed here on new conditions).
PREREG = {
    "purpose": "Does trajectory drift detect prompt degradations that the SNR estimator misses?",
    "early_handover_step": 5,
    "reference_step": 8,
    "policies": {
        "snr_only": "early handover iff S_hat >= 20.0 dB",
        "drift_only": "early handover iff d_4 > 0.30",
        "snr_and_drift": "early handover iff S_hat >= 20.0 dB AND d_4 > 0.30",
    },
    "theta_snr_db": 20.0,
    "tau_drift_d4": 0.30,
    "failure_definition": "Definition 3.1 via core.metrics.check_failure (same as Chapter 3 grids)",
    "noise_floor_p0": 0.025,
    "prerequisite_rule": (
        "A condition is ELIGIBLE only if the Step-5 failure rate is significantly above the "
        "seed-noise floor p0=0.025 (one-sided exact binomial, p < 0.05). Otherwise there is "
        "no degradation for any gate to catch and the condition cannot count for or against drift."
    ),
    "primary_hypothesis": (
        "On at least one ELIGIBLE condition, snr_and_drift has fewer Step-5 failures than "
        "snr_only (one-sided exact McNemar on paired utterances), Holm-corrected across "
        "eligible conditions at family alpha = 0.05."
    ),
    "secondary_reports": [
        "drift_only vs snr_only (two-sided exact McNemar) per condition",
        "ROC-AUC of S_hat and d_4 for predicting Step-5 failure, per condition",
        "gate trigger rates and mean NFE per policy per condition",
    ],
    "family_alpha": 0.05,
}

CONDITION_SPECS: Dict[str, Dict[str, Any]] = {
    "clean": {"kind": "clean"},
    "reverb03": {"kind": "reverb", "rt60": 0.3},
    "reverb06": {"kind": "reverb", "rt60": 0.6},
    "reverb09": {"kind": "reverb", "rt60": 0.9},
    "babble15": {"kind": "babble", "snr_db": 15.0},
    "babble20": {"kind": "babble", "snr_db": 20.0},
    "demand15": {"kind": "demand", "snr_db": 15.0},
    "demand20": {"kind": "demand", "snr_db": 20.0},
    "phone": {"kind": "phone"},
    "gauss18": {"kind": "gauss", "snr_db": 18.0},
    "gauss20": {"kind": "gauss", "snr_db": 20.0},
    "gauss22": {"kind": "gauss", "snr_db": 22.0},
    "gauss25": {"kind": "gauss", "snr_db": 25.0},
}
CONDITION_PRESETS = {
    "quick": ["clean", "reverb06", "phone"],
    "reverb": ["clean", "reverb03", "reverb06", "reverb09"],
    "all": list(CONDITION_SPECS.keys()),
}


# ============================================================================
# 1. PREREGISTRATION
# ============================================================================

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return "unknown"


def freeze_preregistration(out_dir: str, speaker_ids: List[str], conditions: List[str]) -> Dict[str, Any]:
    """Writes the preregistration once. Later runs must match the frozen rules exactly."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "drift_justification_preregistration.json")
    spk_hash = hashlib.sha256("\n".join(sorted(speaker_ids)).encode("utf-8")).hexdigest()

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            frozen = json.load(f)
        if frozen["frozen_rules"] != PREREG:
            raise RuntimeError(
                f"Frozen rules in {path} differ from the rules in this script. Decision rules must not "
                "change after preregistration. Delete the file only if you are starting a NEW study."
            )
        if frozen["speaker_list_sha256"] != spk_hash:
            raise RuntimeError(f"Speaker list differs from the preregistered one in {path}.")
        new_conds = [c for c in conditions if c not in frozen["conditions"]]
        if new_conds:
            frozen["conditions"].extend(new_conds)
            frozen.setdefault("conditions_added_later", []).append(
                {"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "conditions": new_conds}
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(frozen, f, indent=2)
        print(f"[Prereg] Verified existing preregistration: {path}")
        return frozen

    frozen = {
        "preregistration_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "code_commit": _git_commit(),
        "n_speakers": len(speaker_ids),
        "speaker_list_sha256": spk_hash,
        "conditions": list(conditions),
        "frozen_rules": PREREG,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    print(f"[Prereg] Wrote NEW preregistration before any synthesis: {path}")
    return frozen


# ============================================================================
# 2. SPEAKER POOL
# ============================================================================

def load_speaker_pool(pool: str, n_speakers: Optional[int]) -> List[Dict[str, Any]]:
    """One utterance per speaker. 'fresh' = Seed-TTS speakers unused by dev/calib/test."""
    from core.sampler import derive_input_seed
    from data.make_splits import load_dataset_metadata, BASE_SEED

    entries = load_dataset_metadata(META_LST)
    used = set()
    split_spk: Dict[str, set] = {}
    for name in ["development", "calibration", "test"]:
        p = os.path.join(SPLITS_DIR, f"{name}_manifest.json")
        with open(p, "r", encoding="utf-8") as f:
            spks = {r["speaker_id"] for r in json.load(f)}
        split_spk[name] = spks
        used |= spks

    if pool == "fresh":
        keep = lambda s: s not in used
    elif pool == "test":
        keep = lambda s: s in split_spk["test"]
    else:
        raise ValueError(f"Unknown pool: {pool}")

    by_spk: Dict[str, Dict[str, Any]] = {}
    for e in sorted(entries, key=lambda e: e["utt_id"]):
        if keep(e["speaker_id"]) and e["speaker_id"] not in by_spk:
            by_spk[e["speaker_id"]] = e
    rows = [by_spk[s] for s in sorted(by_spk)]
    if n_speakers is not None:
        rows = rows[:n_speakers]
    for r in rows:
        r["input_seed"] = derive_input_seed(BASE_SEED, r["utt_id"])
    print(f"[Dataset] Pool '{pool}': {len(rows)} speakers (1 utterance each).")
    return rows


# ============================================================================
# 3. PROMPT CORRUPTIONS
# ============================================================================

def _cond_seed(input_seed: int, condition: str) -> int:
    digest = hashlib.sha256(f"{input_seed}-{condition}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise[: len(clean)]
    gain = _rms(clean) / (_rms(noise) * 10.0 ** (snr_db / 20.0))
    return np.clip(clean + gain * noise, -1.0, 1.0).astype(np.float32)


def _tile_crop(x: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))) + 1)
    start = rng.randint(0, len(x) - n + 1)
    return x[start : start + n]


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    import torchaudio
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def apply_reverb(x: np.ndarray, sr: int, rt60: float, seed: int) -> np.ndarray:
    """Synthetic RIR: direct path + exponentially decaying Gaussian tail (-60 dB at rt60)."""
    from scipy.signal import fftconvolve
    rng = np.random.RandomState(seed)
    n = int(1.2 * rt60 * sr)
    t = np.arange(n) / sr
    rir = rng.randn(n) * np.exp(-6.908 * t / rt60)
    rir[: int(0.002 * sr)] = 0.0  # 2 ms gap between direct path and tail
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir**2))
    y = fftconvolve(x, rir)[: len(x)]
    y = y * (_rms(x) / _rms(y))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def apply_phone(x: np.ndarray, sr: int) -> np.ndarray:
    """8 kHz telephone channel: band-limit 300-3400 Hz + 8-bit mu-law, back to original rate."""
    import torch
    import torchaudio.functional as AF
    w = torch.from_numpy(x).float().unsqueeze(0)
    w = AF.resample(w, sr, 8000)
    w = AF.highpass_biquad(w, 8000, 300.0)
    w = AF.lowpass_biquad(w, 8000, 3400.0)
    w = w.clamp(-1.0, 1.0)
    w = AF.mu_law_decoding(AF.mu_law_encoding(w, 256), 256)
    w = AF.resample(w, 8000, sr)
    y = w.squeeze(0).numpy()[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    y = y * (_rms(x) / _rms(y))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def make_babble(sr: int, n: int, self_spk: str, pool_paths: List[Tuple[str, str]], seed: int) -> np.ndarray:
    """Sum of 6 other speakers' prompts, each RMS-normalized."""
    rng = np.random.RandomState(seed)
    others = [p for s, p in pool_paths if s != self_spk]
    picks = rng.choice(len(others), size=6, replace=False)
    out = np.zeros(n, dtype=np.float32)
    for i in picks:
        src = _load_mono(others[i], sr)
        src = _tile_crop(src, n, rng)
        out += src / _rms(src)
    return out


def corrupt_prompt(
    x: np.ndarray, sr: int, condition: str, input_seed: int, speaker_id: str,
    babble_pool: List[Tuple[str, str]],
) -> np.ndarray:
    from data.make_splits import add_gaussian_noise_15db
    spec = CONDITION_SPECS[condition]
    seed = _cond_seed(input_seed, condition)
    kind = spec["kind"]
    if kind == "clean":
        return x.astype(np.float32)
    if kind == "gauss":
        y, _, _ = add_gaussian_noise_15db(x, seed=seed, target_snr_db=spec["snr_db"])
        return y.astype(np.float32)
    if kind == "reverb":
        return apply_reverb(x, sr, spec["rt60"], seed)
    if kind == "phone":
        return apply_phone(x, sr)
    if kind == "babble":
        return _mix_at_snr(x, make_babble(sr, len(x), speaker_id, babble_pool, seed), spec["snr_db"])
    if kind == "demand":
        noise = _tile_crop(_load_mono(DEMAND_WAV, sr), len(x), np.random.RandomState(seed))
        return _mix_at_snr(x, noise, spec["snr_db"])
    raise ValueError(f"Unknown condition kind: {kind}")


# ============================================================================
# 4. SYNTHESIS + SCORING (same pipeline as run_gpu_expanded_benchmark.py)
# ============================================================================

def run_benchmark(
    conditions: List[str],
    pool: str = "fresh",
    n_speakers: Optional[int] = None,
    device_name: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    import soundfile as sf
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence

    out_dir = output_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if any(CONDITION_SPECS[c]["kind"] == "demand" for c in conditions) and not os.path.exists(DEMAND_WAV):
        skipped = [c for c in conditions if CONDITION_SPECS[c]["kind"] == "demand"]
        print(f"[DEMAND] {DEMAND_WAV} not found. SKIPPING {skipped} (no substitute numbers are generated).")
        conditions = [c for c in conditions if c not in skipped]

    # Preregister the FULL pool so a --n_speakers smoke test does not change the frozen speaker list
    full_pool = load_speaker_pool(pool, None)
    freeze_preregistration(out_dir, [r["speaker_id"] for r in full_pool], conditions)
    speakers = full_pool[:n_speakers] if n_speakers is not None else full_pool

    from infer import DMOInference
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
    from core.sampler import (
        REFERENCE_STEP, ALL_TARGET_STEPS, STEP_METADATA, CFG_STRENGTH,
        make_local_generator, estimate_prompt_snr, run_teacher_single_pass, run_student_completion,
    )
    from core.metrics import MetricSuite, check_failure, load_failure_config

    device_str = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Target Compute Device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    tts = DMOInference(
        student_checkpoint_path=os.path.join(ROOT_DIR, "ckpts", "model_85000.pt"),
        duration_predictor_path=os.path.join(ROOT_DIR, "ckpts", "model_1500.pt"),
        device=device,
    )
    metric_suite = MetricSuite(device=device_str)
    failure_cfg = load_failure_config()
    print("[OK] All models initialized.")

    babble_pool = [(r["speaker_id"], r["prompt_wav_path"]) for r in load_speaker_pool("test", None)]

    csv_path = os.path.join(out_dir, f"drift_justification_grid_{pool}.csv")
    rows_buffer: List[Dict[str, Any]] = []
    completed = set()
    if os.path.exists(csv_path):
        df_exist = pd.read_csv(csv_path)
        rows_buffer = df_exist.to_dict("records")
        completed = {(str(r["utt_id"]), str(r["condition"])) for r in rows_buffer}
        print(f"[Resume] Found {len(completed)} completed (utterance, condition) pairs in {csv_path}.")

    jobs = [(r, c) for c in conditions for r in speakers if (r["utt_id"], c) not in completed]
    print(f"[Plan] {len(jobs)} (utterance, condition) pairs to run x {len(ALL_TARGET_STEPS)} steps each.")
    t_start = time.time()

    for j, (item, condition) in enumerate(jobs, start=1):
        utt_id, spk_id = item["utt_id"], item["speaker_id"]
        prompt_text, gt_text, input_seed = item["prompt_text"], item["gt_text"], item["input_seed"]

        if os.path.getsize(item["prompt_wav_path"]) < 1000:
            raise RuntimeError(f"{item['prompt_wav_path']} is a Git LFS pointer. Run the Seed-TTS download cell first.")

        prompt_raw, sr = torchaudio.load(item["prompt_wav_path"])
        clean_np = prompt_raw.mean(dim=0).numpy().astype(np.float32)
        prompt_np = corrupt_prompt(clean_np, sr, condition, input_seed, spk_id, babble_pool)
        prompt_snr = estimate_prompt_snr(prompt_np, sample_rate=sr)

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
        predicted_dur = tts.predict_duration(temp_prompt_path, gt_text, prompt_text, dp_softmax_range=0.7, temperature=0)
        target_dur = max(10, int(predicted_dur))
        os.remove(temp_prompt_path)

        final_text_list = convert_char_to_pinyin([ref_text_clean + gt_text])
        total_duration = audio_tensor.shape[-1] // tts.hop_length + target_dur
        cond_tensor, text_ids, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
            audio_tensor, final_text_list, total_duration
        )
        prompt_len = cond_mask.sum().item()

        # Same initial noise across conditions -> paired comparisons
        gen = make_local_generator(input_seed)
        y0 = [torch.randn(dur, 100, dtype=torch.float32, generator=gen) for dur in dur_tensor]
        y0 = pad_sequence(y0, padding_value=0, batch_first=True).to(device)

        endpoints, drift_trace, _ = run_teacher_single_pass(
            tts=tts, step_cond=step_cond, text=text_ids, y0=y0, prompt_len=prompt_len,
            target_steps=ALL_TARGET_STEPS, cfg_strength=CFG_STRENGTH,
        )
        audios = {}
        for step in ALL_TARGET_STEPS:
            _, audios[step] = run_student_completion(
                tts=tts, cond=cond_tensor, text=text_ids, teacher_endpoint=endpoints[step],
                prompt_len=prompt_len, input_seed=input_seed,
            )

        # Same fallback as run_gpu_expanded_benchmark.py when the GT recording is missing
        gt_path = item["gt_wav_path"]
        if os.path.exists(gt_path) and os.path.getsize(gt_path) >= 1000:
            gt_raw, gt_sr = torchaudio.load(gt_path)
            gt_np = gt_raw.squeeze().numpy()
        else:
            print(f"[Warning] GT wav missing for {utt_id}; using clean prompt for SIM.")
            gt_np, gt_sr = clean_np, sr

        ref_audio = audios[REFERENCE_STEP]
        wer_ref = metric_suite.compute_wer(ref_audio, gt_text, sample_rate=24000)
        sim_ref = metric_suite.compute_similarity(ref_audio, gt_np, sample_rate=24000, gt_sr=gt_sr)
        utmos_ref = metric_suite.compute_utmos(ref_audio, sample_rate=24000)

        drift_cols = {f"d_{i}": float(drift_trace.get(i, -1.0)) for i in range(1, 9)}
        for step in ALL_TARGET_STEPS:
            if step == REFERENCE_STEP:
                wer_k, sim_k, utmos_k, is_fail = wer_ref, sim_ref, utmos_ref, False
            else:
                wer_k = metric_suite.compute_wer(audios[step], gt_text, sample_rate=24000)
                sim_k = metric_suite.compute_similarity(audios[step], gt_np, sample_rate=24000, gt_sr=gt_sr)
                utmos_k = metric_suite.compute_utmos(audios[step], sample_rate=24000)
                is_fail, _ = check_failure(wer_k, wer_ref, sim_k, sim_ref, config=failure_cfg)
            rows_buffer.append({
                "utt_id": utt_id,
                "speaker_id": spk_id,
                "condition": condition,
                "teacher_step": step,
                "nfe": STEP_METADATA[step]["nfe"],
                "prompt_snr": float(prompt_snr),
                **drift_cols,
                "wer": float(wer_k),
                "similarity": float(sim_k),
                "utmos": float(utmos_k),
                "wer_ref": float(wer_ref),
                "similarity_ref": float(sim_ref),
                "utmos_ref": float(utmos_ref),
                "is_failure": bool(is_fail),
            })

        elapsed = time.time() - t_start
        eta_min = elapsed / j * (len(jobs) - j) / 60.0
        print(f"[{j}/{len(jobs)}] {utt_id} ({condition}) S_hat={prompt_snr:.1f} dB d_4={drift_cols['d_4']:.3f} | ETA {eta_min:.1f} min")
        if j % 10 == 0 or j == len(jobs):
            pd.DataFrame(rows_buffer).to_csv(csv_path, index=False)

    pd.DataFrame(rows_buffer).to_csv(csv_path, index=False)
    print(f"[Done] Grid saved to {csv_path}")
    return csv_path


# ============================================================================
# 5. ANALYSIS (computed numbers only)
# ============================================================================

def _auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, score))


def _holm(pvals: Dict[str, float], alpha: float) -> Dict[str, bool]:
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    reject, still = {}, True
    for i, k in enumerate(order):
        still = still and pvals[k] <= alpha / (m - i)
        reject[k] = still
    return reject


def analyze(output_dir: Optional[str] = None, pool: str = "fresh") -> Dict[str, Any]:
    from scipy.stats import binomtest

    out_dir = output_dir or DEFAULT_OUT_DIR
    csv_path = os.path.join(out_dir, f"drift_justification_grid_{pool}.csv")
    df = pd.read_csv(csv_path)
    df["is_failure"] = df["is_failure"].astype(str).str.lower().isin(["true", "1"])

    k_early = PREREG["early_handover_step"]
    theta, tau, p0 = PREREG["theta_snr_db"], PREREG["tau_drift_d4"], PREREG["noise_floor_p0"]
    nfe_early = int(df[df.teacher_step == k_early].nfe.iloc[0])
    nfe_ref = int(df[df.teacher_step == PREREG["reference_step"]].nfe.iloc[0])

    per_cond = []
    for cond, g in df.groupby("condition", sort=False):
        w = g[g.teacher_step == k_early].set_index("utt_id")
        n = len(w)
        fail5 = w.is_failure.values.astype(int)
        snr_ok = (w.prompt_snr >= theta).values
        drift_ok = (w.d_4 > tau).values
        gates = {"snr_only": snr_ok, "drift_only": drift_ok, "snr_and_drift": snr_ok & drift_ok}
        # A policy fails only when it hands over early on a case where Step 5 fails (Step 8 never fails).
        pol_fail = {k: fail5 & v for k, v in gates.items()}

        def mcnemar(a: np.ndarray, b: np.ndarray, alt: str) -> Dict[str, Any]:
            b_disc = int(((a == 1) & (b == 0)).sum())
            c_disc = int(((a == 0) & (b == 1)).sum())
            p = 1.0 if b_disc + c_disc == 0 else float(binomtest(b_disc, b_disc + c_disc, 0.5, alternative=alt).pvalue)
            return {"b_only_first_fails": b_disc, "c_only_second_fails": c_disc, "p_value": p}

        n_fail = int(fail5.sum())
        row = {
            "condition": cond,
            "n": n,
            "step_fail_rate": {int(s): float(gs.is_failure.mean()) for s, gs in g.groupby("teacher_step")},
            "step5_failures": n_fail,
            "prereq_p_value": float(binomtest(n_fail, n, p0, alternative="greater").pvalue),
            "s_hat_mean_db": float(w.prompt_snr.mean()),
            "d4_mean": float(w.d_4.mean()),
            "auc_step5_fail_low_snr": _auc(fail5, -w.prompt_snr.values),
            "auc_step5_fail_low_d4": _auc(fail5, -w.d_4.values),
            "policies": {
                k: {
                    "early_rate": float(v.mean()),
                    "failures": int(pol_fail[k].sum()),
                    "risk": float(pol_fail[k].mean()),
                    "mean_nfe": float(np.where(v, nfe_early, nfe_ref).mean()),
                }
                for k, v in gates.items()
            },
            "mcnemar_snr_only_vs_snr_and_drift": mcnemar(pol_fail["snr_only"], pol_fail["snr_and_drift"], "greater"),
            "mcnemar_snr_only_vs_drift_only": mcnemar(pol_fail["snr_only"], pol_fail["drift_only"], "two-sided"),
        }
        row["eligible"] = row["prereq_p_value"] < 0.05
        per_cond.append(row)

    eligible = {r["condition"]: r["mcnemar_snr_only_vs_snr_and_drift"]["p_value"] for r in per_cond if r["eligible"]}
    holm = _holm(eligible, PREREG["family_alpha"]) if eligible else {}
    for r in per_cond:
        r["primary_holm_reject"] = holm.get(r["condition"])

    if not eligible:
        verdict = ("INCONCLUSIVE: no condition degraded Step 5 above the seed-noise floor, "
                   "so there was nothing for either gate to catch.")
    elif any(holm.values()):
        wins = [c for c, rej in holm.items() if rej]
        verdict = (f"SUPPORTED: snr_and_drift has significantly fewer failures than snr_only on {wins} "
                   "(Holm-corrected). Drift catches degradation the SNR gate misses.")
    else:
        verdict = ("NOT SUPPORTED: on every eligible condition, adding drift to the SNR gate did not "
                   "significantly reduce failures. Drift is redundant with SNR under these corruptions.")

    summary = {"preregistration": PREREG, "n_rows": len(df), "conditions": per_cond, "primary_verdict": verdict}
    with open(os.path.join(out_dir, "drift_justification_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    flat = []
    for r in per_cond:
        for k, v in r["policies"].items():
            flat.append({"condition": r["condition"], "eligible": r["eligible"], "policy": k, **v})
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, "drift_justification_policy_table.csv"), index=False)

    _write_report(summary, out_dir)
    return summary


def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def _write_report(summary: Dict[str, Any], out_dir: str):
    lines = [
        "# Drift Justification Benchmark Report",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Rows:** {summary['n_rows']}  ",
        f"**Gates (frozen):** SNR `S_hat >= {PREREG['theta_snr_db']} dB`, drift `d_4 > {PREREG['tau_drift_d4']}`, "
        f"early step {PREREG['early_handover_step']}, fallback step {PREREG['reference_step']}",
        "",
        f"## Primary verdict\n\n**{summary['primary_verdict']}**",
        "",
        "## Degradation check (prerequisite)",
        "",
        "| Condition | n | S_hat mean (dB) | d_4 mean | Step 4 fail | Step 5 fail | Step 6 fail | p vs 2.5% floor | Eligible |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["conditions"]:
        fr = r["step_fail_rate"]
        lines.append(
            f"| {r['condition']} | {r['n']} | {r['s_hat_mean_db']:.1f} | {r['d4_mean']:.3f} | "
            f"{fr.get(4, 0):.1%} | {fr.get(5, 0):.1%} | {fr.get(6, 0):.1%} | {r['prereq_p_value']:.4f} | "
            f"{'yes' if r['eligible'] else 'no'} |"
        )
    lines += [
        "",
        "## Policy comparison (early handover at Step 5)",
        "",
        "| Condition | Policy | Early rate | Failures | Risk | Mean NFE |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["conditions"]:
        for k, v in r["policies"].items():
            lines.append(f"| {r['condition']} | {k} | {v['early_rate']:.1%} | {v['failures']}/{r['n']} | {v['risk']:.1%} | {v['mean_nfe']:.2f} |")
    lines += [
        "",
        "## Paired tests and failure prediction",
        "",
        "| Condition | SNR vs SNR+drift (b, c, p one-sided) | Holm reject | SNR vs drift-only (b, c, p) | AUC low-S_hat | AUC low-d_4 |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["conditions"]:
        a, b = r["mcnemar_snr_only_vs_snr_and_drift"], r["mcnemar_snr_only_vs_drift_only"]
        hr = r["primary_holm_reject"]
        lines.append(
            f"| {r['condition']} | {a['b_only_first_fails']}, {a['c_only_second_fails']}, {a['p_value']:.4f} | "
            f"{'n/a (ineligible)' if hr is None else ('yes' if hr else 'no')} | "
            f"{b['b_only_first_fails']}, {b['c_only_second_fails']}, {b['p_value']:.4f} | "
            f"{_fmt(r['auc_step5_fail_low_snr'])} | {_fmt(r['auc_step5_fail_low_d4'])} |"
        )
    lines += [
        "",
        "AUC > 0.5 means lower S_hat (or lower d_4) predicts a Step-5 failure. "
        "Fallback to Step 8 never fails by definition, so a gate can only reduce risk by falling back more, at higher NFE.",
    ]
    path = os.path.join(out_dir, "DRIFT_JUSTIFICATION_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Report] {path}")


def parse_conditions(arg: str) -> List[str]:
    if arg in CONDITION_PRESETS:
        return CONDITION_PRESETS[arg]
    conds = [c.strip() for c in arg.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITION_SPECS]
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}. Choose from {list(CONDITION_SPECS)} or presets {list(CONDITION_PRESETS)}.")
    return conds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift Justification Benchmark")
    parser.add_argument("--conditions", type=str, default="quick",
                        help=f"Preset {list(CONDITION_PRESETS)} or comma list of {list(CONDITION_SPECS)}")
    parser.add_argument("--pool", type=str, default="fresh", choices=["fresh", "test"],
                        help="fresh = Seed-TTS speakers unused by dev/calib/test (default)")
    parser.add_argument("--n_speakers", type=int, default=None, help="Limit speakers (smoke test)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        run_benchmark(parse_conditions(args.conditions), args.pool, args.n_speakers, args.device, args.output_dir)
    s = analyze(args.output_dir, args.pool)
    print("\n" + "=" * 80)
    print(s["primary_verdict"])
    print("=" * 80)
