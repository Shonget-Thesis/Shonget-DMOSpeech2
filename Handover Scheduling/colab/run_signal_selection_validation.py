"""Signal-Selection Validation: why the SNR gate, and why drift?

The proposed policy (Definition 3.4) uses two zero-NFE signals:
  1. a prompt-quality gate (energy-percentile prompt SNR, theta = 20 dB), and
  2. a stopping signal (endpoint drift d_i at candidate Steps 4/5/6).
This runner tests each choice against alternatives under one protocol:

  Q1  Gate:      SNR vs. other pre-generation prompt signals
                 (spectral-percentile SNR, spectral flatness, high-frequency ratio,
                  noise floor, prompt UTMOS, prompt duration, speaking rate, text length;
                  oracle = true clean/noisy label).
  Q2  Stopping:  drift vs. other zero-NFE teacher-trajectory signals
                 (cosine change, per-frame max drift, CFG gap, velocity norm),
                 all with the SNR gate fixed at 20 dB.

Protocol (frozen in signal_selection_preregistration.json before any analysis):
  development (100 spk) -> AUC and signal orientation
  calibration (200 spk) -> threshold by Learn-Then-Test (alpha = 0.05, delta = 0.10)
  test        (200 spk) -> held-out risk and NFE, McNemar vs. the proposal's signal

Failure labels come from colab/results/gpu_grid_all1000.csv (no re-synthesis). The teacher
pass reproduces the stored d_4/d_5/d_6 as a correctness check.

Usage:
  python run_signal_selection_validation.py --device cuda             # extract + analyse
  python run_signal_selection_validation.py --device cuda --n_pairs 10 --output_dir <smoke>
  python run_signal_selection_validation.py --analyze_only            # re-run the analysis
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.stats import binom, binomtest, rankdata

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOVER_DIR = os.path.dirname(COLAB_DIR)
ROOT_DIR = os.path.dirname(HANDOVER_DIR)
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), HANDOVER_DIR, COLAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

GRID_CSV = os.path.join(COLAB_DIR, "results", "gpu_grid_all1000.csv")
DEFAULT_OUT = os.path.join(COLAB_DIR, "results", "signal_selection")
FEATURES_CSV = "signal_features_all1000.csv"
PREREG_JSON = "signal_selection_preregistration.json"
SUMMARY_JSON = "signal_selection_summary.json"
REPORT_MD = "SIGNAL_SELECTION_REPORT.md"

CANDIDATES = (4, 5, 6)
REFERENCE = 8
MAX_TEACHER_STEP = max(CANDIDATES)          # stopping signals only need steps <= 6

PREREG: Dict[str, Any] = {
    "version": 1,
    "question_1_gate": {
        "signals": {
            "snr_energy": "PROPOSAL: energy-percentile prompt SNR (Eq. 3.9), dB",
            "snr_spectral": "spectral-percentile SNR (per-bin 10th-percentile noise PSD), dB",
            "spectral_flatness": "mean spectral flatness (Wiener entropy) of the prompt",
            "hf_ratio": "energy above 4 kHz / total energy",
            "noise_floor_dbfs": "mean energy of the quietest 10% of 25 ms frames, dBFS",
            "prompt_utmos": "UTMOS (utmos22_strong) of the prompt; one extra non-TTS network pass",
            "prompt_duration_s": "prompt duration, seconds (input difficulty, not quality)",
            "prompt_char_rate": "prompt characters per second (speaking rate)",
            "target_words": "number of words in the target text",
        },
        "oracle": "true condition label (clean = safe)",
        "proposal_signal": "snr_energy",
        "proposal_fixed_threshold_db": 20.0,
        "templates": {
            "gate_to_step4": "safe -> Step 4 (proposal's SNR-only rule, Table 3.8), else Step 8",
            "gate_to_step5": "safe -> Step 5, else Step 8",
        },
        "auc_target": "Step-5 failure, all development utterances (clean + noisy15)",
    },
    "question_2_stopping": {
        "gate": "snr_energy >= 20 dB (fixed, as in the proposal)",
        "signals": {
            "drift": "PROPOSAL: d_i = ||x1(t_i) - x1(t_i-1)|| / ||x1(t_i)|| over generated frames (Eq. 3.8)",
            "cos_change": "1 - cosine(x1(t_i), x1(t_i-1)) over generated frames",
            "frame_max_drift": "max over generated frames of per-frame relative drift",
            "cfg_gap": "||v_c - v_u|| / ||v_c|| over generated frames",
            "velocity_norm": "RMS of the guided velocity over generated frames",
        },
        "proposal_signal": "drift",
        "proposal_fixed_lambda": 0.50,
        "template": "Definition 3.4: first i in (4, 5, 6) with score_i <= lambda, else Step 8",
        "auc_target": "Step-4 failure among gated development utterances, score at step 4",
        "baselines": ["SNR gate -> fixed Step 4 / 5 / 6", "matched-random step assignment"],
    },
    "ltt_justification": {
        "policy": "proposal policy: SNR gate 20 dB + drift stopping rule (Definition 3.4)",
        "lambda_grid": "0.20, 0.25, ..., 1.20 (phase2_calib_ltt_200spk/config.json), conservative -> aggressive",
        "resplits": 200,
        "pool": "calibration + test speakers, random speaker-disjoint 50/50 re-splits (calibrate on one half, check on the other)",
        "methods": {
            "naive": "most aggressive lambda whose calibration risk is <= alpha (no guarantee)",
            "hoeffding": "fixed-sequence test with Hoeffding p = exp(-2n(alpha - R)^2)",
            "ltt_binomial": "PROPOSAL: fixed-sequence test with exact binomial p = P(Bin(n, alpha) <= k)",
        },
        "monotonicity_check": "on the full calibration split: does robust risk ever decrease as lambda increases?",
        "label_sources": {
            "gpu": "colab/results/gpu_grid_all1000.csv (same labels as Q1 and Q2)",
            "cpu": "phase2/phase3 CPU grids (the labels behind the original lambda-hat = 0.50 calibration)",
        },
    },
    "orientation_rule": "chosen on development only: the direction in which the signal predicts failure (AUC >= 0.5)",
    "threshold_grid_rule": "development quantiles q = 0, 0.05, ..., 1, ordered from conservative to aggressive",
    "ltt": {"alpha": 0.05, "delta": 0.10, "test": "fixed-sequence binomial, P(Bin(n, alpha) <= k) <= delta"},
    "risk": "robust per speaker: a speaker fails if either the clean or the noisy15 utterance fails",
    "verdict_rule": {
        "JUSTIFIED": "proposal signal is certified on calibration and safe on test (risk <= alpha), and no "
                     "alternative of equal or lower cost class is safe on test with mean NFE lower by > 0.5",
        "MATCHED": "an alternative of equal or lower cost class is safe on test with |delta NFE| <= 0.5",
        "OUTPERFORMED": "an alternative of equal or lower cost class is safe on test with mean NFE lower by > 0.5",
        "FAILS_GUARANTEE": "proposal signal exceeds alpha on test",
    },
    "cost_classes": {"metadata": 0, "dsp": 1, "neural": 2, "zero_nfe_trajectory": 1, "fixed_rule": 0},
    "caveat": "post-hoc validation: the test split was inspected in earlier analyses",
}

GATE_COST = {
    "snr_energy": "dsp", "snr_spectral": "dsp", "spectral_flatness": "dsp", "hf_ratio": "dsp",
    "noise_floor_dbfs": "dsp", "prompt_utmos": "neural", "prompt_duration_s": "metadata",
    "prompt_char_rate": "metadata", "target_words": "metadata",
}
STOP_SIGNALS = list(PREREG["question_2_stopping"]["signals"].keys())
GATE_SIGNALS = list(PREREG["question_1_gate"]["signals"].keys())


def nfe(step: int) -> int:
    return 2 * (step + 1) + 3


# ============================================================================
# 1. PROMPT (GATE) SIGNALS: computed from the prompt waveform before generation
# ============================================================================

def prompt_dsp_features(wav: np.ndarray, sr: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (features, per-feature CPU time in ms). Uses the same 25 ms / 10 ms framing as Eq. 3.9."""
    from core.sampler import estimate_prompt_snr

    feats: Dict[str, float] = {}
    cost: Dict[str, float] = {}
    eps = 1e-12

    t0 = time.perf_counter()
    feats["snr_energy"] = float(estimate_prompt_snr(wav, sample_rate=sr))
    cost["snr_energy"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    nper = round(0.025 * sr)
    hop = round(0.010 * sr)
    freqs, _, z = stft(wav, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    power = np.abs(z) ** 2
    stft_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    noise_psd = np.percentile(power, 10, axis=1)
    total = float(power.sum())
    noise_total = float(noise_psd.sum() * power.shape[1])
    feats["snr_spectral"] = float(np.clip(10 * np.log10(max(total - noise_total, eps) / (noise_total + eps)), 0.0, 50.0))
    cost["snr_spectral"] = stft_ms + (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    geo = np.exp(np.mean(np.log(power + eps), axis=0))
    arith = np.mean(power, axis=0) + eps
    feats["spectral_flatness"] = float(np.mean(geo / arith))
    cost["spectral_flatness"] = stft_ms + (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    feats["hf_ratio"] = float(power[freqs > 4000].sum() / (total + eps)) if sr / 2 > 4000 else float("nan")
    cost["hf_ratio"] = stft_ms + (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    n_frames = 1 + (len(wav) - nper) // hop
    if n_frames >= 2:
        frames = np.lib.stride_tricks.as_strided(
            wav, shape=(n_frames, nper), strides=(wav.strides[0] * hop, wav.strides[0]))
        energy = np.sort(np.mean(frames ** 2, axis=-1))
        floor = float(np.mean(energy[:max(1, round(0.10 * len(energy)))]))
        feats["noise_floor_dbfs"] = float(10 * np.log10(floor + eps))
    else:
        feats["noise_floor_dbfs"] = float("nan")
    cost["noise_floor_dbfs"] = (time.perf_counter() - t0) * 1e3
    return feats, cost


# ============================================================================
# 2. TEACHER TRAJECTORY (STOPPING) SIGNALS: same math as core.sampler.run_teacher_single_pass
# ============================================================================

def teacher_trajectory_features(tts: Any, step_cond, text, y0, prompt_len: int, cfg_strength: float) -> Dict[str, float]:
    import torch
    from core.sampler import SWAY_TEACHER_GRID, compute_stabilization_drift

    feats: Dict[str, float] = {}
    state = y0.clone()
    prev_end = None
    eps = 1e-8
    with torch.inference_mode():
        for i in range(MAX_TEACHER_STEP + 1):
            t_val = SWAY_TEACHER_GRID[i]
            t = torch.tensor(t_val, device=step_cond.device, dtype=torch.float32)
            pred = tts.teacher(x=state, cond=step_cond, text=text, time=t, mask=None,
                               drop_audio_cond=False, drop_text=False)
            null_pred = tts.teacher(x=state, cond=step_cond, text=text, time=t, mask=None,
                                    drop_audio_cond=True, drop_text=True)
            v = pred + (pred - null_pred) * cfg_strength
            end = state + (1.0 - t_val) * v

            v_gen = v[:, prompt_len:, :]
            pc_gen = pred[:, prompt_len:, :]
            feats[f"cfg_gap_{i}"] = float(torch.norm(pc_gen - null_pred[:, prompt_len:, :]) / (torch.norm(pc_gen) + eps))
            feats[f"velocity_norm_{i}"] = float(torch.sqrt(torch.mean(v_gen.float() ** 2)))
            if prev_end is not None:
                e_gen = end[:, prompt_len:, :].float()
                p_gen = prev_end[:, prompt_len:, :].float()
                feats[f"drift_{i}"] = compute_stabilization_drift(end, prev_end, prompt_len)
                cos = torch.nn.functional.cosine_similarity(e_gen.flatten(), p_gen.flatten(), dim=0)
                feats[f"cos_change_{i}"] = float(1.0 - cos)
                per_frame = torch.norm(e_gen - p_gen, dim=-1) / (torch.norm(e_gen, dim=-1) + eps)
                feats[f"frame_max_drift_{i}"] = float(per_frame.max())
            if i < MAX_TEACHER_STEP:
                state = state + (SWAY_TEACHER_GRID[i + 1] - t_val) * v
                prev_end = end
    return feats


# ============================================================================
# 3. EXTRACTION
# ============================================================================

def extract(device_name: Optional[str], output_dir: str, n_pairs: Optional[int]) -> str:
    import soundfile as sf
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    from infer import DMOInference
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
    from core.sampler import CFG_STRENGTH, make_local_generator
    from core.metrics import UTMOSEvaluator
    from data.make_splits import add_gaussian_noise_15db
    from run_gpu_expanded_benchmark import load_expanded_manifest

    device_str = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[Device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    tts = DMOInference(
        student_checkpoint_path=os.path.join(ROOT_DIR, "ckpts", "model_85000.pt"),
        duration_predictor_path=os.path.join(ROOT_DIR, "ckpts", "model_1500.pt"),
        device=device,
    )
    utmos = UTMOSEvaluator(device=device_str)

    rows = load_expanded_manifest("all1000")
    if n_pairs is not None:
        rows = rows[:n_pairs]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, FEATURES_CSV)

    done, records = set(), []
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        records = prev.to_dict("records")
        done = {(r["utt_id"], r["condition"]) for r in records}
        print(f"[Resume] {len(done)} rows already extracted.")

    t_start = time.time()
    for idx, item in enumerate(rows, start=1):
        key = (item["utt_id"], item["condition"])
        if key in done:
            continue
        wav_path = item["prompt_wav_path"]
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(f"Prompt audio missing or an LFS pointer: {wav_path}. Run the dataset cell first.")

        raw, sr = torchaudio.load(wav_path)
        wav = raw.squeeze().numpy()
        if item["condition"] == "noisy15":
            wav, _, _ = add_gaussian_noise_15db(wav, seed=item["input_seed"], target_snr_db=15.0)
        wav = np.ascontiguousarray(wav, dtype=np.float32)

        rec: Dict[str, Any] = {
            "utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": item["condition"],
            "split": item.get("split", ""),
        }
        feats, cost = prompt_dsp_features(wav, sr)
        rec.update(feats)
        rec.update({f"cost_ms_{k}": v for k, v in cost.items()})

        t0 = time.perf_counter()
        rec["prompt_utmos"] = float(utmos.score(wav, sample_rate=sr))
        if device.type == "cuda":
            torch.cuda.synchronize()
        rec["cost_ms_prompt_utmos"] = (time.perf_counter() - t0) * 1e3

        dur_s = len(wav) / sr
        rec["prompt_duration_s"] = dur_s
        rec["prompt_char_rate"] = len(item["prompt_text"]) / dur_s if dur_s > 0 else float("nan")
        rec["target_words"] = len(item["gt_text"].split())
        for k in ("prompt_duration_s", "prompt_char_rate", "target_words"):
            rec[f"cost_ms_{k}"] = 0.0

        # Teacher inputs, identical to run_gpu_expanded_benchmark.py
        tmp = os.path.join(output_dir, f"temp_{item['utt_id']}_{item['condition']}.wav")
        sf.write(tmp, wav, sr)
        _, ref_text_clean = preprocess_ref_audio_text(tmp, item["prompt_text"])
        audio = torch.from_numpy(wav).float().unsqueeze(0)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < 0.3:
            audio = audio * 0.3 / rms
        if sr != tts.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio)
        audio = audio.to(device)
        torch.manual_seed(item["input_seed"])
        pred_dur = tts.predict_duration(tmp, item["gt_text"], item["prompt_text"], dp_softmax_range=0.7, temperature=0)
        if os.path.exists(tmp):
            os.remove(tmp)
        text_list = convert_char_to_pinyin([ref_text_clean + item["gt_text"]])
        total_dur = audio.shape[-1] // tts.hop_length + max(10, int(pred_dur))
        _, text_ids, step_cond, cond_mask, _, dur_tensor = tts._prepare_inputs(audio, text_list, total_dur)
        prompt_len = cond_mask.sum().item()
        gen = make_local_generator(item["input_seed"])
        y0 = pad_sequence([torch.randn(d, 100, dtype=torch.float32, generator=gen) for d in dur_tensor],
                          padding_value=0, batch_first=True).to(device)

        rec.update(teacher_trajectory_features(tts, step_cond, text_ids, y0, prompt_len, CFG_STRENGTH))
        records.append(rec)

        if idx % 20 == 0 or idx == len(rows):
            pd.DataFrame(records).to_csv(csv_path, index=False)
            el = time.time() - t_start
            print(f"[{idx}/{len(rows)}] {item['utt_id']} ({item['condition']}) | {el / 60:.1f} min elapsed")

    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"[OK] Features written: {csv_path} ({len(records)} rows)")
    return csv_path


# ============================================================================
# 4. ANALYSIS
# ============================================================================

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score of a failure > score of a pass)."""
    labels = labels.astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(df: pd.DataFrame, score_col: str, label_col: str, rng: np.random.Generator, b: int = 1000) -> Tuple[float, float]:
    """Speaker-clustered bootstrap 95% CI of the AUC."""
    groups = list(df.groupby("speaker_id").indices.values())
    scores, labels = df[score_col].values, df[label_col].values
    vals = []
    for _ in range(b):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        a = auc(scores[idx], labels[idx])
        if not math.isnan(a):
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def robust_risk(df: pd.DataFrame, steps: np.ndarray) -> Tuple[int, int, float]:
    """Robust failures (speaker fails if any of its utterances fails), speakers, mean NFE."""
    fail = np.zeros(len(df), dtype=bool)
    for k in CANDIDATES:
        m = steps == k
        fail[m] = df[f"fail_{k}"].values[m]
    codes = df["_spk"].values
    n_spk = int(codes.max()) + 1
    k = int((np.bincount(codes, weights=fail, minlength=n_spk) > 0).sum())
    return k, n_spk, float(np.mean(2 * (steps + 1) + 3))


def per_speaker_fail(df: pd.DataFrame, steps: np.ndarray) -> pd.Series:
    fail = np.zeros(len(df), dtype=bool)
    for k in CANDIDATES:
        m = steps == k
        fail[m] = df[f"fail_{k}"].values[m].astype(bool)
    return pd.Series(fail, index=df["speaker_id"].values).groupby(level=0).max()


def ltt(calib: pd.DataFrame, policy, grid: List[float]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Fixed-sequence LTT over a grid ordered conservative -> aggressive. Returns the last certified value."""
    a, d = PREREG["ltt"]["alpha"], PREREG["ltt"]["delta"]
    chosen, trace = None, []
    for g in grid:
        k, n, mean_nfe = robust_risk(calib, policy(calib, g))
        p = float(binom.cdf(k, n, a))
        trace.append({"value": g, "failures": k, "n": n, "p": p, "nfe": mean_nfe, "certified": p <= d})
        if p > d:
            break
        chosen = g
    return chosen, trace


def mcnemar(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    j = a.index.intersection(b.index)
    a, b = a.loc[j], b.loc[j]
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b > 0 else 1.0
    return {"only_first_fails": only_a, "only_second_fails": only_b, "p_two_sided": float(p)}


def load_labeled(output_dir: str) -> pd.DataFrame:
    feats = pd.read_csv(os.path.join(output_dir, FEATURES_CSV))
    grid = pd.read_csv(GRID_CSV)
    lab = grid.pivot_table(index=["utt_id", "condition"], columns="teacher_step", values="is_failure", aggfunc="first")
    lab = lab.rename(columns={k: f"fail_{k}" for k in lab.columns}).reset_index()
    ref = grid[grid.teacher_step == REFERENCE][["utt_id", "condition", "d_4", "d_5", "d_6"]]
    df = feats.merge(lab, on=["utt_id", "condition"], how="inner").merge(ref, on=["utt_id", "condition"], how="left")
    for k in CANDIDATES:
        df[f"fail_{k}"] = df[f"fail_{k}"].astype(str).str.lower().isin(["true", "1", "1.0"])
    if (df["split"] == "").all() or df["split"].isna().all():
        raise RuntimeError("Split labels missing from the features CSV.")
    df["split"] = df["split"].replace({"development": "dev", "calibration": "calib"})
    return df


def orient(dev: pd.DataFrame, col: str, label: str) -> int:
    a = auc(dev[col].values, dev[label].values)
    return 1 if (math.isnan(a) or a >= 0.5) else -1


def analyze(output_dir: str) -> Dict[str, Any]:
    prereg_path = os.path.join(output_dir, PREREG_JSON)
    with open(prereg_path, "r", encoding="utf-8") as f:
        stored = json.load(f)
    if stored["protocol"] != PREREG:
        raise RuntimeError("Preregistration mismatch: the frozen protocol differs from this script. Do not edit after freezing.")

    df = load_labeled(output_dir)
    df["oracle_clean"] = (df.condition == "clean").astype(float)
    df["cost_ms_oracle_clean"] = 0.0
    rng = np.random.default_rng(42)
    a = PREREG["ltt"]["alpha"]
    dev, calib, test = (df[df.split == s].reset_index(drop=True) for s in ("dev", "calib", "test"))
    for x in (dev, calib, test):
        x["_spk"] = pd.factorize(x["speaker_id"])[0]
    out: Dict[str, Any] = {"n_rows": len(df), "n_speakers": {s: int(x.speaker_id.nunique()) for s, x in
                                                             (("dev", dev), ("calib", calib), ("test", test))}}

    # Reproduction check against the stored grid drift
    rep = {}
    for i in CANDIDATES:
        m = df[f"d_{i}"].notna() & df[f"drift_{i}"].notna()
        diff = (df.loc[m, f"drift_{i}"] - df.loc[m, f"d_{i}"]).abs()
        rep[f"d_{i}"] = {"max_abs_diff": float(diff.max()), "corr": float(np.corrcoef(df.loc[m, f"drift_{i}"], df.loc[m, f"d_{i}"])[0, 1])}
    out["reproduction"] = rep

    # ---------------- Q1: gate ----------------
    gate_rows = []
    gate_signals = [s for s in GATE_SIGNALS if df[s].notna().all()] + ["oracle_clean"]
    for s in gate_signals:
        sign = orient(dev, s, "fail_5")
        risk_col = f"_risk_{s}"
        for x in (dev, calib, test, df):
            x[risk_col] = sign * x[s]
        lo, hi = auc_ci(dev, risk_col, "fail_5", rng)
        row = {
            "signal": s, "cost_class": GATE_COST.get(s, "oracle"),
            "cost_ms_median": float(df[f"cost_ms_{s}"].median()) if f"cost_ms_{s}" in df else 0.0,
            "safe_direction": "lower" if sign > 0 else "higher",
            "dev_auc_fail5": auc(dev[risk_col].values, dev["fail_5"].values), "dev_auc_ci": [lo, hi],
            "dev_auc_fail5_clean": auc(dev[dev.condition == "clean"][risk_col].values, dev[dev.condition == "clean"]["fail_5"].values),
            "dev_auc_fail5_noisy": auc(dev[dev.condition == "noisy15"][risk_col].values, dev[dev.condition == "noisy15"]["fail_5"].values),
            "dev_auc_noisy_vs_clean": auc(dev[risk_col].values, (dev.condition == "noisy15").values),
        }
        qs = np.quantile(dev[risk_col].values, np.linspace(0, 1, 21))
        grid_vals = sorted(set(float(q) for q in qs))            # risk threshold: early if risk <= value
        for tpl, early in (("gate_to_step4", 4), ("gate_to_step5", 5)):
            pol = (lambda d, g, rc=risk_col, e=early: np.where(d[rc].values <= g, e, REFERENCE))
            g_hat, trace = ltt(calib, pol, grid_vals)
            if g_hat is None:
                row[tpl] = {"certified": False}
                continue
            k, n, m_nfe = robust_risk(test, pol(test, g_hat))
            row[tpl] = {"certified": True, "threshold_risk_score": g_hat,
                        "calib": trace[[t["value"] for t in trace].index(g_hat)],
                        "test_failures": k, "test_n": n, "test_risk": k / n, "test_nfe": m_nfe,
                        "test_safe": k / n <= a, "_fails": per_speaker_fail(test, pol(test, g_hat))}
        gate_rows.append(row)

    # The proposal's own fixed rule (theta = 20 dB) for reference
    fixed = {}
    for tpl, early in (("gate_to_step4", 4), ("gate_to_step5", 5)):
        steps = np.where(test["snr_energy"].values >= 20.0, early, REFERENCE)
        k, n, m_nfe = robust_risk(test, steps)
        fixed[tpl] = {"test_failures": k, "test_n": n, "test_risk": k / n, "test_nfe": m_nfe}
    out["q1_proposal_fixed_theta20"] = fixed

    q1_verdict = {}
    for tpl in ("gate_to_step4", "gate_to_step5"):
        q1_verdict[tpl] = verdict(gate_rows, "snr_energy", tpl, exclude=("oracle_clean",))
    out["q1_gate"] = [{k: v for k, v in r.items()} for r in gate_rows]
    out["q1_verdict"] = q1_verdict

    # ---------------- Q2: stopping ----------------
    gated = lambda d: d[d.snr_energy >= 20.0]
    stop_rows = []
    for s in STOP_SIGNALS:
        sign = orient(gated(dev), f"{s}_4", "fail_4")
        for x in (dev, calib, test):
            for i in CANDIDATES:
                x[f"_r_{s}_{i}"] = sign * x[f"{s}_{i}"]
        dev_g = gated(dev)
        lo, hi = auc_ci(dev_g, f"_r_{s}_4", "fail_4", rng)
        pooled = np.concatenate([dev_g[f"_r_{s}_{i}"].values for i in CANDIDATES])
        grid_vals = [float(np.min(pooled)) - 1.0] + sorted(set(float(q) for q in np.quantile(pooled, np.linspace(0, 1, 21))))

        def pol(d, lam, s=s):
            steps = np.full(len(d), REFERENCE)
            open_ = (d.snr_energy.values >= 20.0)
            for i in CANDIDATES:
                hit = open_ & (d[f"_r_{s}_{i}"].values <= lam)
                steps[hit] = i
                open_ &= ~hit
            return steps

        row = {"signal": s, "cost_class": "zero_nfe_trajectory", "risk_direction": "higher" if sign > 0 else "lower",
               "dev_auc_fail4_gated": auc(dev_g[f"_r_{s}_4"].values, dev_g["fail_4"].values), "dev_auc_ci": [lo, hi],
               "dev_auc_fail5_gated": auc(dev_g[f"_r_{s}_5"].values, dev_g["fail_5"].values)}
        lam, trace = ltt(calib, pol, grid_vals)
        if lam is None:
            row["policy"] = {"certified": False}
        else:
            st = pol(test, lam)
            k, n, m_nfe = robust_risk(test, st)
            row["policy"] = {"certified": True, "lambda_risk_score": lam,
                             "calib": trace[[t["value"] for t in trace].index(lam)],
                             "test_failures": k, "test_n": n, "test_risk": k / n, "test_nfe": m_nfe,
                             "test_safe": k / n <= a, "_fails": per_speaker_fail(test, st),
                             "matched_random_p": matched_random(test, st, rng)}
        stop_rows.append(row)

    # Baselines: gate -> fixed step, and the proposal's frozen lambda = 0.50
    base = []
    for k_fixed in CANDIDATES:
        st = np.where(test.snr_energy.values >= 20.0, k_fixed, REFERENCE)
        k, n, m_nfe = robust_risk(test, st)
        base.append({"signal": f"fixed_step{k_fixed}", "cost_class": "fixed_rule",
                     "policy": {"certified": True, "test_failures": k, "test_n": n, "test_risk": k / n,
                                "test_nfe": m_nfe, "test_safe": k / n <= a, "_fails": per_speaker_fail(test, st)}})
    st = np.full(len(test), REFERENCE)
    open_ = test.snr_energy.values >= 20.0
    for i in CANDIDATES:
        hit = open_ & (test[f"drift_{i}"].values <= PREREG["question_2_stopping"]["proposal_fixed_lambda"])
        st[hit] = i
        open_ &= ~hit
    k, n, m_nfe = robust_risk(test, st)
    out["q2_proposal_frozen_lambda050"] = {"test_failures": k, "test_n": n, "test_risk": k / n, "test_nfe": m_nfe,
                                           "matched_random_p": matched_random(test, st, rng)}
    out["q2_stopping"] = stop_rows + base
    out["q2_verdict"] = verdict(stop_rows + base, "drift", "policy")

    # McNemar: proposal signal vs. best alternative
    out["q1_mcnemar"] = {tpl: best_vs(gate_rows, "snr_energy", tpl, exclude=("oracle_clean",)) for tpl in ("gate_to_step4", "gate_to_step5")}
    out["q2_mcnemar"] = best_vs(stop_rows + base, "drift", "policy")

    # ---------------- Why LTT? ----------------
    lj = {"gpu": ltt_justification(calib, test, rng)}
    if all(os.path.exists(p) for p in CPU_GRIDS.values()):
        lj["cpu"] = ltt_justification(load_cpu_split(CPU_GRIDS["calib"]), load_cpu_split(CPU_GRIDS["test"]), rng)
    a, dl = PREREG["ltt"]["alpha"], PREREG["ltt"]["delta"]
    lj["hoeffding_min_n_zero_failures"] = math.ceil(math.log(1 / dl) / (2 * a ** 2))
    lj["binomial_min_n_zero_failures"] = math.ceil(math.log(dl) / math.log(1 - a))
    out["ltt_justification"] = lj

    clean = strip_private(out)
    with open(os.path.join(output_dir, SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    write_report(clean, output_dir)
    return clean


LTT_LAMBDA_GRID = [round(0.20 + 0.05 * i, 2) for i in range(21)]
CPU_GRIDS = {
    "calib": os.path.join(HANDOVER_DIR, "phase2_calib_ltt_200spk", "results", "calib_grid.csv"),
    "test": os.path.join(HANDOVER_DIR, "phase3_test_benchmark_200spk", "results", "test_grid.csv"),
}


def load_cpu_split(path: str) -> pd.DataFrame:
    """One row per (utterance, condition) from a CPU grid, in the column layout used by ltt_justification."""
    g = pd.read_csv(path)
    lab = g.pivot_table(index=["utt_id", "condition"], columns="teacher_step", values="fail", aggfunc="first")
    lab = lab.rename(columns={k: f"fail_{k}" for k in lab.columns}).reset_index()
    base = g[g.teacher_step == REFERENCE][["utt_id", "speaker_id", "condition", "prompt_snr", "d_4", "d_5", "d_6"]]
    d = base.merge(lab, on=["utt_id", "condition"]).rename(
        columns={"prompt_snr": "snr_energy", "d_4": "drift_4", "d_5": "drift_5", "d_6": "drift_6"})
    for k in CANDIDATES:
        d[f"fail_{k}"] = d[f"fail_{k}"].astype(bool)
    d["_spk"] = pd.factorize(d["speaker_id"])[0]
    return d.reset_index(drop=True)


def drift_policy_steps(d: pd.DataFrame, lam: float) -> np.ndarray:
    steps = np.full(len(d), REFERENCE)
    open_ = d.snr_energy.values >= 20.0
    for i in CANDIDATES:
        hit = open_ & (d[f"drift_{i}"].values <= lam)
        steps[hit] = i
        open_ &= ~hit
    return steps


def ltt_justification(calib: pd.DataFrame, test: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Any]:
    """Naive vs. Hoeffding vs. exact-binomial LTT on repeated speaker re-splits of calibration + test."""
    a, dl = PREREG["ltt"]["alpha"], PREREG["ltt"]["delta"]
    grid = LTT_LAMBDA_GRID

    # Monotonicity on the full calibration split
    calib_trace = []
    for lam in grid:
        k, n, m_nfe = robust_risk(calib, drift_policy_steps(calib, lam))
        calib_trace.append({"lambda": lam, "failures": k, "n": n, "risk": k / n, "nfe": m_nfe})
    decreases = [(calib_trace[j]["lambda"], calib_trace[j + 1]["lambda"], calib_trace[j]["failures"], calib_trace[j + 1]["failures"])
                 for j in range(len(grid) - 1) if calib_trace[j + 1]["failures"] < calib_trace[j]["failures"]]

    # Per-speaker failure matrix F[lambda, speaker] and NFE totals on the pooled speakers
    pool = pd.concat([calib, test], ignore_index=True)
    codes, uniq = pd.factorize(pool["speaker_id"])
    n_spk = len(uniq)
    F = np.zeros((len(grid), n_spk), dtype=bool)
    nfe_sum = np.zeros((len(grid), n_spk))
    utt_count = np.bincount(codes, minlength=n_spk)
    for g, lam in enumerate(grid):
        st = drift_policy_steps(pool, lam)
        fail = np.zeros(len(pool), dtype=bool)
        for k in CANDIDATES:
            m = st == k
            fail[m] = pool[f"fail_{k}"].values[m]
        F[g] = np.bincount(codes, weights=fail, minlength=n_spk) > 0
        nfe_sum[g] = np.bincount(codes, weights=2 * (st + 1) + 3, minlength=n_spk)

    def choose(risks: np.ndarray, n: int, method: str) -> Optional[int]:
        if method == "naive":
            ok = np.where(risks <= a)[0]
            return int(ok.max()) if len(ok) else None
        chosen = None
        for g, r in enumerate(risks):
            if method == "hoeffding":
                p = math.exp(-2 * n * (a - r) ** 2) if r < a else 1.0
            else:
                p = float(binom.cdf(int(round(r * n)), n, a))
            if p > dl:
                break
            chosen = g
        return chosen

    reps = PREREG["ltt_justification"]["resplits"]
    res = {m: {"exceed": 0, "risk": [], "nfe": [], "none": 0, "lam": []} for m in ("naive", "hoeffding", "ltt_binomial")}
    for _ in range(reps):
        perm = rng.permutation(n_spk)
        cal_idx, hold_idx = perm[: n_spk // 2], perm[n_spk // 2:]
        cal_risk = F[:, cal_idx].mean(axis=1)
        for m in res:
            g = choose(cal_risk, len(cal_idx), m)
            if g is None:
                res[m]["none"] += 1
                res[m]["risk"].append(0.0)
                res[m]["nfe"].append(float(nfe(REFERENCE)))
                continue
            r_hold = float(F[g, hold_idx].mean())
            res[m]["risk"].append(r_hold)
            res[m]["nfe"].append(float(nfe_sum[g, hold_idx].sum() / utt_count[hold_idx].sum()))
            res[m]["lam"].append(grid[g])
            res[m]["exceed"] += r_hold > a
    summary = {m: {"pct_holdout_risk_above_alpha": 100 * v["exceed"] / reps,
                   "mean_holdout_risk": float(np.mean(v["risk"])),
                   "p95_holdout_risk": float(np.percentile(v["risk"], 95)),
                   "mean_holdout_nfe": float(np.mean(v["nfe"])),
                   "pct_no_threshold": 100 * v["none"] / reps,
                   "modal_lambda": (float(pd.Series(v["lam"]).mode().iloc[0]) if v["lam"] else None)}
               for m, v in res.items()}
    return {"calibration_trace": calib_trace, "risk_decreases_with_larger_lambda": decreases,
            "resplits": reps, "pooled_speakers": int(n_spk), "methods": summary}


def matched_random(test: pd.DataFrame, steps: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> float:
    """P(random reassignment of the same early steps among gated utterances has <= the policy's failures)."""
    k_pol, _, _ = robust_risk(test, steps)
    gated = np.where(test.snr_energy.values >= 20.0)[0]
    pool = steps[gated].copy()
    hits = 0
    for _ in range(reps):
        st = steps.copy()
        st[gated] = rng.permutation(pool)
        k, _, _ = robust_risk(test, st)
        hits += k <= k_pol
    return hits / reps


def _policy(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    return row.get(key, {}) or {}


def verdict(rows: List[Dict[str, Any]], proposal: str, key: str, exclude=()) -> Dict[str, Any]:
    classes = PREREG["cost_classes"]
    prop = next(r for r in rows if r["signal"] == proposal)
    pp = _policy(prop, key)
    if not pp.get("certified"):
        return {"verdict": "NOT_CERTIFIED", "detail": "proposal signal certifies no threshold on calibration"}
    if not pp["test_safe"]:
        return {"verdict": "FAILS_GUARANTEE", "detail": f"test risk {pp['test_risk']:.3f} > alpha"}
    best = None
    for r in rows:
        if r["signal"] in exclude or r["signal"] == proposal:
            continue
        rp = _policy(r, key)
        if not (rp.get("certified") and rp.get("test_safe")):
            continue
        if classes[r["cost_class"]] > classes[prop["cost_class"]]:
            continue
        if best is None or rp["test_nfe"] < _policy(best, key)["test_nfe"]:
            best = r
    if best is None:
        return {"verdict": "JUSTIFIED", "detail": "no cheaper-or-equal alternative is safe on test"}
    delta = pp["test_nfe"] - _policy(best, key)["test_nfe"]
    v = "OUTPERFORMED" if delta > 0.5 else ("MATCHED" if delta >= -0.5 else "JUSTIFIED")
    return {"verdict": v, "best_alternative": best["signal"], "proposal_nfe": pp["test_nfe"],
            "alternative_nfe": _policy(best, key)["test_nfe"], "delta_nfe": delta}


def best_vs(rows, proposal, key, exclude=()):
    prop = _policy(next(r for r in rows if r["signal"] == proposal), key)
    cands = [r for r in rows if r["signal"] not in exclude and r["signal"] != proposal
             and _policy(r, key).get("certified") and _policy(r, key).get("test_safe")]
    if not prop.get("certified") or not cands:
        return None
    best = min(cands, key=lambda r: _policy(r, key)["test_nfe"])
    res = mcnemar(prop["_fails"], _policy(best, key)["_fails"])
    res["alternative"] = best["signal"]
    return res


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================================
# 5. REPORT
# ============================================================================

def _fmt_pol(p: Dict[str, Any]) -> str:
    if not p or not p.get("certified"):
        return "not certified | — | — | —"
    return f"{p['test_failures']}/{p['test_n']} ({100 * p['test_risk']:.1f}%) | {p['test_nfe']:.2f} | {'✅' if p['test_safe'] else '❌'}"


def write_report(s: Dict[str, Any], output_dir: str) -> None:
    L = []
    L.append("# Signal-Selection Validation: Why the SNR Gate, and Why Drift?\n")
    L.append("Generated by `colab/run_signal_selection_validation.py`. Protocol frozen in "
             f"`{PREREG_JSON}` before analysis. Development → AUC and orientation; calibration → LTT threshold "
             "(α = 0.05, δ = 0.10); test → held-out robust risk and mean NFE.\n")
    L.append(f"> Caveat: {PREREG['caveat']}.\n")
    L.append(f"Speakers: dev {s['n_speakers']['dev']}, calibration {s['n_speakers']['calib']}, test {s['n_speakers']['test']}.\n")

    L.append("## Reproduction check\n")
    L.append("| Signal | Max abs. difference vs. stored grid | Correlation |\n|---|---|---|")
    for k, v in s["reproduction"].items():
        L.append(f"| {k} | {v['max_abs_diff']:.6f} | {v['corr']:.4f} |")

    L.append("\n## Q1. Why a prompt-SNR gate?\n")
    for tpl, label in (("gate_to_step4", "Gate → Step 4 (proposal's SNR-only template)"), ("gate_to_step5", "Gate → Step 5")):
        v = s["q1_verdict"][tpl]
        L.append(f"**{label}: verdict {v['verdict']}**" + (f" (best alternative: `{v.get('best_alternative')}`, ΔNFE = {v['delta_nfe']:+.2f})" if "delta_nfe" in v else "") + "\n")
    L.append("| Signal | Cost class | Median cost (ms) | Dev AUC, Step-5 failure [95% CI] | AUC within clean | AUC within noisy | AUC noisy vs. clean | Gate→4: test failures | NFE | ≤ α | Gate→5: test failures | NFE | ≤ α |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["q1_gate"]:
        name = f"**{r['signal']}** (proposal)" if r["signal"] == "snr_energy" else r["signal"]
        L.append(f"| {name} | {r['cost_class']} | {r['cost_ms_median']:.2f} | {r['dev_auc_fail5']:.3f} [{r['dev_auc_ci'][0]:.2f}, {r['dev_auc_ci'][1]:.2f}] | "
                 f"{r['dev_auc_fail5_clean']:.3f} | {r['dev_auc_fail5_noisy']:.3f} | {r['dev_auc_noisy_vs_clean']:.3f} | "
                 f"{_fmt_pol(r.get('gate_to_step4'))} | {_fmt_pol(r.get('gate_to_step5'))} |")
    f20 = s["q1_proposal_fixed_theta20"]
    L.append(f"\nProposal's fixed rule (SNR ≥ 20 dB): → Step 4: {f20['gate_to_step4']['test_failures']}/{f20['gate_to_step4']['test_n']}, "
             f"{f20['gate_to_step4']['test_nfe']:.2f} NFE; → Step 5: {f20['gate_to_step5']['test_failures']}/{f20['gate_to_step5']['test_n']}, "
             f"{f20['gate_to_step5']['test_nfe']:.2f} NFE.\n")
    for tpl, m in s["q1_mcnemar"].items():
        if m:
            L.append(f"- McNemar ({tpl}), snr_energy vs. `{m['alternative']}`: {m['only_first_fails']} vs. {m['only_second_fails']}, p = {m['p_two_sided']:.3f}")

    L.append("\n## Q2. Why drift as the stopping signal? (SNR gate fixed at 20 dB)\n")
    v = s["q2_verdict"]
    L.append(f"**Verdict {v['verdict']}**" + (f" (best alternative: `{v.get('best_alternative')}`, ΔNFE = {v['delta_nfe']:+.2f})" if "delta_nfe" in v else "") + "\n")
    L.append("| Signal | Cost class | Dev AUC, Step-4 failure (gated) [95% CI] | Dev AUC, Step-5 failure | Test failures | Mean NFE | ≤ α | Matched-random p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in s["q2_stopping"]:
        p = r.get("policy", {})
        name = f"**{r['signal']}** (proposal)" if r["signal"] == "drift" else r["signal"]
        auc4 = f"{r['dev_auc_fail4_gated']:.3f} [{r['dev_auc_ci'][0]:.2f}, {r['dev_auc_ci'][1]:.2f}]" if "dev_auc_fail4_gated" in r else "—"
        auc5 = f"{r['dev_auc_fail5_gated']:.3f}" if "dev_auc_fail5_gated" in r else "—"
        mr = f"{p['matched_random_p']:.3f}" if "matched_random_p" in p else "—"
        L.append(f"| {name} | {r['cost_class']} | {auc4} | {auc5} | {_fmt_pol(p)} | {mr} |")
    fz = s["q2_proposal_frozen_lambda050"]
    L.append(f"\nProposal's frozen policy (λ̂ = 0.50): {fz['test_failures']}/{fz['test_n']} test failures, {fz['test_nfe']:.2f} NFE, "
             f"matched-random p = {fz['matched_random_p']:.3f}.\n")
    if s["q2_mcnemar"]:
        m = s["q2_mcnemar"]
        L.append(f"- McNemar, drift vs. `{m['alternative']}`: {m['only_first_fails']} vs. {m['only_second_fails']}, p = {m['p_two_sided']:.3f}")

    lj = s["ltt_justification"]
    L.append("\n## Q3. Why Learn-Then-Test for calibration?\n")
    L.append("The calibration step must give a statement of the form *\"the selected threshold adds failures to at most "
             "5% of new inputs, with 90% confidence\"*, for a threshold chosen from many candidates.\n")
    L.append("| Framework | What it guarantees | Fit for this problem |")
    L.append("|---|---|---|")
    L.append("| Naive selection (most aggressive threshold with calibration risk ≤ α) | Nothing | ❌ Overfits the calibration split (see the re-split check below) |")
    L.append("| Cross-validation (Appendix D pilot) | No finite-sample guarantee | ❌ With 36 inputs it could not certify α = 5% at all |")
    L.append("| Split conformal prediction | Coverage of prediction sets | ❌ Answers a different question (intervals, not the risk of a chosen threshold) |")
    L.append("| Conformal Risk Control (Angelopoulos et al.) | Risk ≤ α in expectation | ❌ Weaker (average, not 1 − δ confidence) and needs risk monotone in λ |")
    L.append("| RCPS (Bates et al.) | Risk ≤ α with probability 1 − δ | ❌ Needs risk monotone in λ |")
    L.append("| **Learn-Then-Test** (Angelopoulos et al.), exact binomial, fixed sequence | Risk ≤ α with probability 1 − δ for the selected threshold | ✅ No monotonicity needed (Remark 3.2); controls error across many thresholds; exact test for failure counts |\n")
    L.append(f"**Sample size needed to certify α = 5% at δ = 10% even with zero failures:** exact binomial "
             f"n ≥ {lj['binomial_min_n_zero_failures']}; Hoeffding n ≥ {lj['hoeffding_min_n_zero_failures']}. "
             "With 200 calibration speakers (or 200 per re-split half), a Hoeffding-based test can never certify any threshold, "
             "while the exact binomial test allows up to 5 failures.\n")
    names = {"naive": "Naive selection", "hoeffding": "Hoeffding fixed-sequence", "ltt_binomial": "**LTT, exact binomial (proposal)**"}
    src_label = {"gpu": "GPU labels (`gpu_grid_all1000.csv`)", "cpu": "CPU labels (phase 2 / phase 3 grids; original λ̂ = 0.50 calibration)"}
    for src in ("cpu", "gpu"):
        if src not in lj:
            continue
        r = lj[src]
        L.append(f"### {src_label[src]}\n")
        dec = r["risk_decreases_with_larger_lambda"]
        L.append(f"**Monotonicity (full calibration split, λ grid 0.20–1.20):** robust risk **decreased** at {len(dec)} of "
                 f"{len(r['calibration_trace']) - 1} steps to a larger λ"
                 + (": " + "; ".join(f"λ {x[0]:.2f} → {x[1]:.2f}: {x[2]} → {x[3]} failures" for x in dec[:6]) if dec else "")
                 + ". " + ("Risk is not monotone in λ, so the CRC and RCPS assumptions do not hold, while LTT needs no such assumption.\n"
                           if dec else "Risk was monotone on this split; LTT remains valid either way.\n"))
        L.append(f"**Re-split check** ({r['resplits']} random 50/50 speaker re-splits of {r['pooled_speakers']} calibration + test "
                 "speakers; calibrate on one half, check on the other):\n")
        L.append("| Method | Held-out risk > 5% | Mean held-out risk | 95th pct. held-out risk | Mean held-out NFE | No threshold certified | Modal λ |")
        L.append("|---|---|---|---|---|---|---|")
        for m, v in r["methods"].items():
            lam = f"{v['modal_lambda']:.2f}" if v["modal_lambda"] is not None else "—"
            L.append(f"| {names[m]} | {v['pct_holdout_risk_above_alpha']:.1f}% | {100 * v['mean_holdout_risk']:.2f}% | "
                     f"{100 * v['p95_holdout_risk']:.2f}% | {v['mean_holdout_nfe']:.2f} | {v['pct_no_threshold']:.1f}% | {lam} |")
        nv, lt, hf = r["methods"]["naive"], r["methods"]["ltt_binomial"], r["methods"]["hoeffding"]
        notes = []
        if nv["pct_holdout_risk_above_alpha"] > lt["pct_holdout_risk_above_alpha"]:
            notes.append(f"naive selection exceeded 5% on held-out halves more often than LTT "
                         f"({nv['pct_holdout_risk_above_alpha']:.1f}% vs. {lt['pct_holdout_risk_above_alpha']:.1f}%), "
                         f"in exchange for {lt['mean_holdout_nfe'] - nv['mean_holdout_nfe']:.2f} fewer NFE")
        elif abs(nv["mean_holdout_nfe"] - lt["mean_holdout_nfe"]) < 0.05:
            notes.append("naive selection and LTT chose nearly the same thresholds, so failures were too rare on these labels "
                         "for overfitting to show; the case for LTT then rests on its guarantee, not on this check")
        else:
            notes.append(f"naive selection and LTT exceeded 5% equally often ({lt['pct_holdout_risk_above_alpha']:.1f}%), "
                         f"with mean NFE {nv['mean_holdout_nfe']:.2f} vs. {lt['mean_holdout_nfe']:.2f}")
        if hf["pct_no_threshold"] >= 99.5:
            notes.append("Hoeffding certified no threshold at all, as the sample-size bound predicts")
        if lt["pct_holdout_risk_above_alpha"] <= 100 * PREREG["ltt"]["delta"]:
            notes.append(f"LTT exceeded 5% on {lt['pct_holdout_risk_above_alpha']:.1f}% of held-out halves, within its δ = 10% allowance")
        else:
            notes.append(f"LTT exceeded 5% on {lt['pct_holdout_risk_above_alpha']:.1f}% of held-out halves, above δ = 10%; "
                         "held-out halves are finite samples, so this is noisy, but it should be discussed")
        L.append("\nReading: " + "; ".join(notes) + ".\n")
    L.append("**Is LTT a contribution?** No: LTT, its binomial p-values and fixed-sequence testing are existing methods "
             "(Angelopoulos et al.; Bauer, 1991) and must be cited as such. The contribution is the adaptation: casting the "
             "teacher-to-student handover point as a risk-controlled selection problem, defining a reference-relative "
             "added-failure risk (Definition 3.1), building a data design that meets LTT's independence assumption "
             "(speaker-disjoint splits, one utterance per speaker), and showing the guarantee holds on held-out speakers. "
             "CALM (Schuster et al., 2022) made the same kind of contribution for early exit in language models.\n")

    L.append("\n## Reading the verdicts\n")
    for k, d in PREREG["verdict_rule"].items():
        L.append(f"- **{k}**: {d}")
    with open(os.path.join(output_dir, REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[OK] Report written: {os.path.join(output_dir, REPORT_MD)}")


# ============================================================================
# 6. MAIN
# ============================================================================

def freeze_protocol(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PREREG_JSON)
    with open(os.path.abspath(__file__), "rb") as f:
        code_sha = hashlib.sha256(f.read()).hexdigest()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if stored["protocol"] != PREREG:
            raise RuntimeError(f"{path} holds a different protocol. Use a new --output_dir instead of editing a frozen one.")
        print(f"[Prereg] Verified frozen protocol ({path}).")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "script_sha256": code_sha, "protocol": PREREG}, f, indent=2)
    print(f"[Prereg] Protocol frozen: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--n_pairs", type=int, default=None, help="smoke test: first N rows only (analysis skipped)")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    freeze_protocol(args.output_dir)
    if not args.analyze_only:
        extract(args.device, args.output_dir, args.n_pairs)
    if args.n_pairs is not None:
        print("[Smoke] Extraction finished; analysis needs all three splits and is skipped.")
        return
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
