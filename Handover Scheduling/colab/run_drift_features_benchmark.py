"""Improved Drift Benchmark: richer zero-NFE stabilization signals for noisy prompts.

Why: global relative-L2 drift (d_i) does not beat the SNR gate (Phase 12). On clean prompts
there is nothing to predict (Step-5 failures sit at the 2.5% seed-noise floor), but noisy
prompts fail at Step 5 only about half the time. The SNR gate sends ALL of them to Step 8.
If an improved stabilization signal can tell which gated prompts are still safe at Step 5,
SNR + drift saves compute that SNR-only cannot, under the same LTT guarantee.

Role of drift tested here (extends Definition 3.4):
    S_hat >= theta                       -> hand over at Step 5   (SNR-only path)
    S_hat <  theta and risk score <= lam -> hand over at Step 5   (drift rescues the prompt)
    otherwise                            -> Step 8 (reference)

Every feature is computed from tensors the teacher already produces (endpoint estimates,
conditional / unconditional velocities), so the policy still adds ZERO forward passes.
Only features from steps <= 5 are allowed, because the decision is made at Step 5.

Protocol (frozen in code, mirrors Phase 12):
  1. Teacher-only pass for all 500 speakers x {clean, noisy15} (no student, no Whisper /
     WavLM / UTMOS). Failure labels are reused from gpu_grid_all1000.csv; the stored
     d_4 / d_5 / d_6 must be reproduced, which checks that trajectories match the labels.
  2. EXPLORE on the 100 development speakers only: pick the best single feature and fit a
     regularized logistic score. The choice is written to a frozen JSON.
  3. CONFIRM on the 400 calibration + test speakers: AUC with Holm correction, LTT
     certification on calibration, held-out risk / NFE on test, matched-random control.

Usage:
    python run_drift_features_benchmark.py --device cuda          # extract + analyze
    python run_drift_features_benchmark.py --analyze_only
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOVER_DIR = os.path.dirname(COLAB_DIR)
ROOT_DIR = os.path.dirname(HANDOVER_DIR)

for p in [ROOT_DIR, os.path.join(ROOT_DIR, "src"), HANDOVER_DIR, COLAB_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_OUT_DIR = os.path.join(COLAB_DIR, "results", "drift_features")
GPU_GRID = os.path.join(COLAB_DIR, "results", "gpu_grid_all1000.csv")
SPLITS_DIR = os.path.join(HANDOVER_DIR, "data", "splits")

DECISION_STEP = 5
MAX_STEP = 8
NFE = {4: 13, 5: 15, 6: 17, 8: 21}
BANDS = {"lo": (0, 33), "mid": (33, 66), "hi": (66, 100)}

# ============================================================================
# FROZEN RULES (do not edit after the first analysis)
# ============================================================================
PREREG = {
    "purpose": "Can an improved zero-NFE stabilization signal identify gated (low-SNR) prompts that are safe at Step 5?",
    "target": "Step-5 added failure (Definition 3.1 labels from gpu_grid_all1000.csv) among prompts with S_hat < theta",
    "theta_snr_db": 20.0,
    "decision_step": DECISION_STEP,
    "allowed_features": "only features computed at teacher steps <= 5 (available when the Step-5 decision is made)",
    "explore_split": "development (100 speakers)",
    "confirm_splits": "calibration + test (400 speakers)",
    "candidate_scores": {
        "S1_best_single": "single feature with the highest dev AUC (direction chosen on dev)",
        "S2_logistic": "L2 logistic regression (C=0.1) on all allowed features, standardized, fitted on dev",
    },
    "H1_signal": "On confirm gated prompts, a frozen score has AUC > 0.5 (one-sided Mann-Whitney, Holm over S1/S2, alpha 0.05)",
    "policy": "S_hat >= theta -> Step 5; S_hat < theta and score <= lambda -> Step 5; else Step 8",
    "lambda_grid": "dev gated-prompt score quantiles q in {0, 0.05, ..., 0.50}, tested conservative -> aggressive",
    "ltt": {"alpha": 0.05, "delta": 0.10, "risk": "robust per-speaker failure = max over clean and noisy15",
            "calibrate_on": "calibration split", "evaluate_on": "test split"},
    "H2_policy": ("The certified improved policy admits >= 1 gated test prompt, has test risk <= alpha, lower mean NFE "
                  "than SNR->Step 5, and fewer admitted failures than random admission of the same number "
                  "(one-sided permutation p < 0.05)"),
    "baselines": ["SNR->Step 4 (proposal SNR-only rule)", "SNR->Step 5", "Original proposed policy (lambda=0.50, steps 4->5->6)"],
}


# ============================================================================
# 1. FEATURE EXTRACTION (teacher-only, GPU)
# ============================================================================

def _step_features(torch, ep_curr, ep_prev, pred, null_pred, v, prompt_len: int, eps: float = 1e-8) -> Dict[str, float]:
    """Zero-NFE features at one teacher step, computed on generated frames only."""
    e = ep_curr[0, prompt_len:, :]
    d = (ep_curr - ep_prev)[0, prompt_len:, :]
    feats = {
        # Same formula as core.sampler.compute_stabilization_drift (used to verify reproduction)
        "d": float((torch.norm(d) / (torch.norm(e) + eps)).item()),
    }
    per_frame = torch.norm(d, dim=-1) / (torch.norm(e, dim=-1) + eps)
    feats["fr_mean"] = float(per_frame.mean().item())
    feats["fr_p90"] = float(torch.quantile(per_frame.float(), 0.90).item())
    feats["fr_max"] = float(per_frame.max().item())
    for name, (lo, hi) in BANDS.items():
        feats[f"band_{name}"] = float((torch.norm(d[:, lo:hi]) / (torch.norm(e[:, lo:hi]) + eps)).item())
    flat_c, flat_p = e.reshape(-1), ep_prev[0, prompt_len:, :].reshape(-1)
    feats["cos"] = float((1.0 - torch.dot(flat_c, flat_p) / (torch.norm(flat_c) * torch.norm(flat_p) + eps)).item())
    pc = pred[0, prompt_len:, :]
    feats["cfg"] = float((torch.norm(pc - null_pred[0, prompt_len:, :]) / (torch.norm(pc) + eps)).item())
    feats["vnorm"] = float((torch.norm(v[0, prompt_len:, :]) / (torch.norm(e) + eps)).item())
    feats["e_std"] = float(e.std().item())
    return feats


def extract_features(device_name: Optional[str], output_dir: str, n_pairs: Optional[int] = None) -> str:
    import soundfile as sf
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence

    from infer import DMOInference
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
    from core.sampler import SWAY_TEACHER_GRID, CFG_STRENGTH, make_local_generator, estimate_prompt_snr
    from data.make_splits import add_gaussian_noise_15db
    from run_gpu_expanded_benchmark import load_expanded_manifest

    os.makedirs(output_dir, exist_ok=True)
    device_str = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Target Compute Device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # same setting as the run that produced the labels

    tts = DMOInference(
        student_checkpoint_path=os.path.join(ROOT_DIR, "ckpts", "model_85000.pt"),
        duration_predictor_path=os.path.join(ROOT_DIR, "ckpts", "model_1500.pt"),
        device=device,
    )

    rows_manifest = load_expanded_manifest("all1000")
    if n_pairs is not None:
        rows_manifest = rows_manifest[:n_pairs]

    csv_path = os.path.join(output_dir, "drift_features_all1000.csv")
    rows: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(str(r["utt_id"]), str(r["condition"])) for r in rows}
        print(f"[Resume] {len(done)} pairs already extracted.")

    jobs = [r for r in rows_manifest if (r["utt_id"], r["condition"]) not in done]
    print(f"[Plan] {len(jobs)} teacher-only passes (8 guided steps each, no student or metrics).")
    t0 = time.time()

    for j, item in enumerate(jobs, start=1):
        utt_id, condition, input_seed = item["utt_id"], item["condition"], item["input_seed"]
        prompt_path = item["prompt_wav_path"]
        if os.path.getsize(prompt_path) < 1000:
            raise RuntimeError(f"{prompt_path} is a Git LFS pointer. Run the Seed-TTS download cell first.")

        # --- identical input preparation to run_gpu_expanded_benchmark.py ---
        prompt_raw, sr = torchaudio.load(prompt_path)
        prompt_np = prompt_raw.squeeze().numpy()
        if condition == "noisy15":
            prompt_np, _, _ = add_gaussian_noise_15db(prompt_np, seed=input_seed, target_snr_db=15.0)
        prompt_snr = estimate_prompt_snr(prompt_np, sample_rate=sr)

        temp_prompt_path = os.path.join(output_dir, f"temp_{utt_id}_{condition}.wav")
        sf.write(temp_prompt_path, prompt_np, sr)
        ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(temp_prompt_path, item["prompt_text"])

        audio_tensor = torch.from_numpy(prompt_np).float().unsqueeze(0)
        rms = torch.sqrt(torch.mean(torch.square(audio_tensor)))
        if rms < 0.3:
            audio_tensor = audio_tensor * 0.3 / rms
        if sr != tts.target_sample_rate:
            audio_tensor = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio_tensor)
        audio_tensor = audio_tensor.to(device)

        torch.manual_seed(input_seed)
        predicted_dur = tts.predict_duration(temp_prompt_path, item["gt_text"], item["prompt_text"],
                                             dp_softmax_range=0.7, temperature=0)
        target_dur = max(10, int(predicted_dur))
        os.remove(temp_prompt_path)

        final_text_list = convert_char_to_pinyin([ref_text_clean + item["gt_text"]])
        total_duration = audio_tensor.shape[-1] // tts.hop_length + target_dur
        cond_tensor, text_ids, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
            audio_tensor, final_text_list, total_duration
        )
        prompt_len = cond_mask.sum().item()

        gen = make_local_generator(input_seed)
        y0 = [torch.randn(dur, 100, dtype=torch.float32, generator=gen) for dur in dur_tensor]
        state = pad_sequence(y0, padding_value=0, batch_first=True).to(device)

        # --- same teacher loop as core.sampler.run_teacher_single_pass, keeping pred / null_pred ---
        row: Dict[str, Any] = {
            "utt_id": utt_id, "speaker_id": item["speaker_id"], "condition": condition,
            "split": item.get("split", ""), "prompt_snr": float(prompt_snr),
            "n_gen_frames": int(state.shape[1] - prompt_len),
        }
        prev_ep = None
        with torch.inference_mode():
            for i in range(MAX_STEP + 1):
                t_val = SWAY_TEACHER_GRID[i]
                t = torch.tensor(t_val, device=device, dtype=torch.float32)
                pred = tts.teacher(x=state, cond=step_cond, text=text_ids, time=t, mask=None,
                                   drop_audio_cond=False, drop_text=False)
                null_pred = tts.teacher(x=state, cond=step_cond, text=text_ids, time=t, mask=None,
                                        drop_audio_cond=True, drop_text=True)
                v = pred + (pred - null_pred) * CFG_STRENGTH
                ep = state + (1.0 - t_val) * v
                if prev_ep is not None:
                    for k, val in _step_features(torch, ep, prev_ep, pred, null_pred, v, prompt_len).items():
                        row[f"{k}_{i}"] = val
                if i < MAX_STEP:
                    state = state + (SWAY_TEACHER_GRID[i + 1] - t_val) * v
                    prev_ep = ep
        rows.append(row)

        if j % 25 == 0 or j == len(jobs):
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            eta = (time.time() - t0) / j * (len(jobs) - j) / 60.0
            print(f"[{j}/{len(jobs)}] {utt_id} ({condition}) d_5={row['d_5']:.3f} cfg_5={row['cfg_5']:.3f} | ETA {eta:.1f} min")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Done] Features saved to {csv_path}")
    return csv_path


# ============================================================================
# 2. ANALYSIS (computed numbers only)
# ============================================================================

def _split_map() -> Dict[tuple, str]:
    out = {}
    for name in ["development", "calibration", "test"]:
        with open(os.path.join(SPLITS_DIR, f"{name}_manifest.json"), "r", encoding="utf-8") as f:
            for r in json.load(f):
                out[(r["utt_id"], r["condition"])] = name
    return out


def _load_joined(output_dir: str) -> pd.DataFrame:
    feats = pd.read_csv(os.path.join(output_dir, "drift_features_all1000.csv"))
    grid = pd.read_csv(GPU_GRID)
    grid["is_failure"] = grid["is_failure"].astype(str).str.lower().isin(["true", "1"])
    labels = grid.pivot_table(index=["utt_id", "condition"], columns="teacher_step", values="is_failure", aggfunc="first")
    labels.columns = [f"fail_{c}" for c in labels.columns]
    stored = grid.groupby(["utt_id", "condition"])[["prompt_snr", "d_4", "d_5", "d_6"]].first()
    stored.columns = ["grid_snr", "grid_d_4", "grid_d_5", "grid_d_6"]
    df = feats.merge(labels.reset_index(), on=["utt_id", "condition"]).merge(stored.reset_index(), on=["utt_id", "condition"])
    smap = _split_map()
    df["split"] = [smap.get((u, c), "") for u, c in zip(df.utt_id, df.condition)]
    for k in [4, 5, 6, 8]:
        df[f"fail_{k}"] = df[f"fail_{k}"].astype(bool).astype(int)
    return df


def _allowed_features(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        base, _, step = c.rpartition("_")
        if step.isdigit() and base and not c.startswith(("fail_", "grid_")) and 1 <= int(step) <= DECISION_STEP:
            out.append(c)
    return sorted(out)


def _auc(y, s) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return None if y.min() == y.max() else float(roc_auc_score(y, s))


def _ltt(fail_counts: List[int], n: int, alpha: float, delta: float) -> int:
    """Fixed-sequence LTT: index of the last certified candidate, -1 if none."""
    from scipy.stats import binom
    last = -1
    for g, k in enumerate(fail_counts):
        if binom.cdf(k, n, alpha) <= delta:
            last = g
        else:
            break
    return last


def _policy_steps(d: pd.DataFrame, rule: str, score: Optional[np.ndarray] = None, lam: float = -np.inf) -> np.ndarray:
    theta = PREREG["theta_snr_db"]
    clean_gate = (d.grid_snr >= theta).values
    if rule == "snr4":
        return np.where(clean_gate, 4, 8)
    if rule == "snr5":
        return np.where(clean_gate, 5, 8)
    if rule == "orig":
        st = np.full(len(d), 8)
        for k in (6, 5, 4):  # earliest qualifying step wins
            st = np.where(clean_gate & (d[f"grid_d_{k}"].values <= 0.50), k, st)
        return st
    if rule == "improved":
        return np.where(clean_gate | (score <= lam), 5, 8)
    raise ValueError(rule)


def _evaluate(d: pd.DataFrame, steps: np.ndarray) -> Dict[str, Any]:
    fails = np.array([d[f"fail_{s}"].values[i] if s != 8 else 0 for i, s in enumerate(steps)])
    robust = pd.Series(fails).groupby(d.speaker_id.values).max()
    return {"failures": int(robust.sum()), "n_speakers": int(len(robust)), "risk": float(robust.mean()),
            "mean_nfe": float(np.mean([NFE[s] for s in steps])), "fails_vec": fails}


def analyze(output_dir: Optional[str] = None, explore_only: bool = False) -> Dict[str, Any]:
    from scipy.stats import binom, mannwhitneyu
    from sklearn.linear_model import LogisticRegression

    out_dir = output_dir or DEFAULT_OUT_DIR
    df = _load_joined(out_dir)
    theta, alpha, delta = PREREG["theta_snr_db"], PREREG["ltt"]["alpha"], PREREG["ltt"]["delta"]

    # --- Reproduction check: rerun trajectories must match the labelled GPU run ---
    repro = {}
    for k in (4, 5, 6):
        a, b = df[f"d_{k}"].values, df[f"grid_d_{k}"].values
        repro[f"d_{k}"] = {"pearson_r": float(np.corrcoef(a, b)[0, 1]), "max_abs_diff": float(np.max(np.abs(a - b)))}
    repro_ok = all(v["pearson_r"] > 0.99 for v in repro.values())

    feats = _allowed_features(df)
    gated = df.grid_snr < theta
    dev = df[(df.split == "development") & gated].copy()
    conf = df[(df.split.isin(["calibration", "test"])) & gated].copy()

    # --- EXPLORE (development only) ---
    y_dev = dev.fail_5.values
    dev_auc = {}
    for f in feats:
        a = _auc(y_dev, dev[f].values)
        if a is not None:
            dev_auc[f] = a
    # Direction: score oriented so that higher = more likely to fail
    best_f = max(dev_auc, key=lambda f: abs(dev_auc[f] - 0.5))
    best_sign = 1.0 if dev_auc[best_f] >= 0.5 else -1.0
    mu, sd = dev[feats].mean(), dev[feats].std().replace(0, 1.0)
    logit = LogisticRegression(C=0.1, max_iter=5000).fit(((dev[feats] - mu) / sd).values, y_dev)

    frozen = {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_dev_gated": int(len(dev)), "dev_failures": int(y_dev.sum()),
        "S1_best_single": {"feature": best_f, "sign": best_sign, "dev_auc_oriented": float(max(dev_auc[best_f], 1 - dev_auc[best_f]))},
        "S2_logistic": {"features": feats, "mean": mu.tolist(), "std": sd.tolist(),
                        "coef": logit.coef_[0].tolist(), "intercept": float(logit.intercept_[0])},
        "top10_dev_features": sorted(({"feature": f, "dev_auc": a} for f, a in dev_auc.items()),
                                     key=lambda r: -abs(r["dev_auc"] - 0.5))[:10],
    }
    frozen_path = os.path.join(out_dir, "drift_features_frozen_selection.json")
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    frozen_sha = hashlib.sha256(open(frozen_path, "rb").read()).hexdigest()

    if explore_only:
        # Development speakers only: calibration and test stay unseen while iterating
        print(f"[Explore] reproduction_ok={repro_ok} | gated dev prompts={len(dev)} ({int(y_dev.sum())} failures)")
        for r in frozen["top10_dev_features"]:
            print(f"  {r['feature']:<14} dev AUC {r['dev_auc']:.3f}")
        return {"reproduction_check": repro, "frozen_selection": frozen}

    def score(frame: pd.DataFrame, which: str) -> np.ndarray:
        if which == "S1":
            return best_sign * frame[best_f].values
        z = ((frame[feats] - mu) / sd).values
        return z @ logit.coef_[0] + logit.intercept_[0]

    # --- CONFIRM H1: signal on calibration + test gated prompts ---
    h1 = {}
    for which in ("S1", "S2"):
        s, y = score(conf, which), conf.fail_5.values
        p = float(mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue)
        rng = np.random.RandomState(0)
        boots = []
        for _ in range(2000):
            idx = rng.randint(0, len(y), len(y))
            a = _auc(y[idx], s[idx])
            if a is not None:
                boots.append(a)
        h1[which] = {"auc": _auc(y, s), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "p_one_sided": p}
    order = sorted(h1, key=lambda k: h1[k]["p_one_sided"])
    still = True
    for i, k in enumerate(order):
        still = still and h1[k]["p_one_sided"] <= 0.05 / (len(order) - i)
        h1[k]["holm_reject"] = still
    h1_pass = any(v["holm_reject"] for v in h1.values())

    # --- CONFIRM H2: LTT on calibration, evaluate on test ---
    calib = df[df.split == "calibration"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    qs = [round(0.05 * i, 2) for i in range(11)]
    policies = {}
    for which in ("S1", "S2"):
        dev_scores = np.sort(score(dev, which))
        lams = [-np.inf] + [float(np.quantile(dev_scores, q)) for q in qs[1:]]
        counts = [_evaluate(calib, _policy_steps(calib, "improved", score(calib, which), lam))["failures"] for lam in lams]
        g = _ltt(counts, calib.speaker_id.nunique(), alpha, delta)
        if g < 0:
            policies[which] = {"certified": False, "calib_failures_by_q": dict(zip(map(str, qs), counts))}
            continue
        lam_hat = lams[g]
        st = _policy_steps(test, "improved", score(test, which), lam_hat)
        ev = _evaluate(test, st)
        test_gated = (test.grid_snr < theta).values
        admitted = test_gated & (st == 5)
        m = int(admitted.sum())
        obs = int(test.fail_5.values[admitted].sum())
        pool = test.fail_5.values[test_gated]
        rng = np.random.RandomState(1)
        rand = [int(rng.choice(pool, m, replace=False).sum()) for _ in range(5000)] if m > 0 else [0]
        policies[which] = {
            "certified": True, "q_hat": qs[g], "lambda_hat": lam_hat,
            "calib_failures_by_q": dict(zip(map(str, qs), counts)),
            "test": {k: v for k, v in ev.items() if k != "fails_vec"},
            "test_gated_admitted": m, "test_admitted_failures": obs,
            "random_admit_mean_failures": float(np.mean(rand)),
            "random_perm_p": float(np.mean(np.array(rand) <= obs)) if m > 0 else 1.0,
        }

    baselines = {}
    for rule, name in [("snr4", "SNR->Step 4"), ("snr5", "SNR->Step 5"), ("orig", "Original proposed (lambda=0.50)")]:
        c_ev, t_ev = _evaluate(calib, _policy_steps(calib, rule)), _evaluate(test, _policy_steps(test, rule))
        baselines[name] = {
            "calib_failures": c_ev["failures"],
            "calib_ltt_p": float(binom.cdf(c_ev["failures"], c_ev["n_speakers"], alpha)),
            "test": {k: v for k, v in t_ev.items() if k != "fails_vec"},
        }
    snr5_nfe = baselines["SNR->Step 5"]["test"]["mean_nfe"]

    h2 = {}
    for which, pol in policies.items():
        h2[which] = bool(pol.get("certified") and pol["test_gated_admitted"] >= 1 and pol["test"]["risk"] <= alpha
                         and pol["test"]["mean_nfe"] < snr5_nfe and pol["random_perm_p"] < 0.05)
    h2_pass = any(h2.values())

    if h1_pass and h2_pass:
        verdict = ("SUPPORTED: the improved drift score predicts which low-SNR prompts are safe at Step 5, and the "
                   "LTT-certified SNR + drift policy saves compute over SNR-only on held-out speakers.")
    elif h1_pass:
        verdict = ("SIGNAL ONLY: the improved drift score predicts noisy-prompt failure better than chance on held-out "
                   "speakers, but not strongly enough to save compute under the 5% risk bound.")
    else:
        verdict = "NOT SUPPORTED: no frozen drift score predicted noisy-prompt failure on held-out speakers."

    summary = {
        "preregistration": PREREG, "frozen_selection_sha256": frozen_sha, "reproduction_check": repro,
        "reproduction_ok": repro_ok, "frozen_selection": {k: v for k, v in frozen.items() if k != "S2_logistic"},
        "confirm_n_gated": int(len(conf)), "confirm_failures": int(conf.fail_5.sum()),
        "H1": h1, "H1_pass": h1_pass, "policies": policies, "baselines": baselines, "H2": h2, "H2_pass": h2_pass,
        "verdict": verdict,
    }
    with open(os.path.join(out_dir, "drift_features_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    _write_report(summary, out_dir)
    return summary


def _write_report(s: Dict[str, Any], out_dir: str):
    L = [
        "# Improved Drift Benchmark Report",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Frozen selection SHA-256:** `{s['frozen_selection_sha256']}`",
        "",
        f"## Verdict\n\n**{s['verdict']}**",
        "",
        "## Reproduction check (rerun trajectories vs. labelled GPU run)",
        "",
        "| Signal | Pearson r | Max abs diff |", "|---|---|---|",
    ]
    for k, v in s["reproduction_check"].items():
        L.append(f"| {k} | {v['pearson_r']:.4f} | {v['max_abs_diff']:.4f} |")
    if not s["reproduction_ok"]:
        L.append("\n**WARNING: rerun drift does not match the labelled run (r <= 0.99). Results below are not trustworthy.**")
    sel = s["frozen_selection"]
    L += [
        "",
        "## Exploration on development speakers (frozen before confirmation)",
        "",
        f"Gated dev prompts: {sel['n_dev_gated']} ({sel['dev_failures']} Step-5 failures). "
        f"Best single feature: `{sel['S1_best_single']['feature']}` (sign {sel['S1_best_single']['sign']:+.0f}, "
        f"dev AUC {sel['S1_best_single']['dev_auc_oriented']:.3f}).",
        "",
        "| Top dev features | Dev AUC (raw) |", "|---|---|",
    ]
    L += [f"| `{r['feature']}` | {r['dev_auc']:.3f} |" for r in sel["top10_dev_features"]]
    L += [
        "",
        f"## H1: signal on held-out calibration + test speakers ({s['confirm_n_gated']} gated prompts, {s['confirm_failures']} failures)",
        "",
        "| Score | AUC | 95% CI | p (one-sided) | Holm reject |", "|---|---|---|---|---|",
    ]
    for k, v in s["H1"].items():
        L.append(f"| {k} | {v['auc']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['p_one_sided']:.4f} | {'yes' if v['holm_reject'] else 'no'} |")
    L += ["", "## H2: LTT-certified policies (calibrate on calibration, evaluate on test)", "",
          "| Policy | Certified | Test failures | Test risk | Test mean NFE | Gated admitted | Admitted fails (random mean) | Perm p |",
          "|---|---|---|---|---|---|---|---|"]
    for name, b in s["baselines"].items():
        t = b["test"]
        L.append(f"| {name} | calib p={b['calib_ltt_p']:.3f} | {t['failures']}/{t['n_speakers']} | {t['risk']:.1%} | {t['mean_nfe']:.2f} | — | — | — |")
    for k, p in s["policies"].items():
        if not p.get("certified"):
            L.append(f"| SNR + improved drift ({k}) | no | — | — | — | — | — | — |")
            continue
        t = p["test"]
        L.append(f"| SNR + improved drift ({k}), q={p['q_hat']} | yes | {t['failures']}/{t['n_speakers']} | {t['risk']:.1%} | "
                 f"{t['mean_nfe']:.2f} | {p['test_gated_admitted']} | {p['test_admitted_failures']} ({p['random_admit_mean_failures']:.1f}) | {p['random_perm_p']:.4f} |")
    L += ["", "Risk is the robust per-speaker failure rate (max over clean and noisy15). "
          "All features come from teacher steps <= 5 and add zero forward passes."]
    path = os.path.join(out_dir, "DRIFT_FEATURES_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[Report] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Drift Benchmark")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_pairs", type=int, default=None, help="Limit pairs (smoke test; analysis is skipped)")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--explore_only", action="store_true", help="Development speakers only; do not score calib/test")
    args = parser.parse_args()

    out = args.output_dir or DEFAULT_OUT_DIR
    if not args.analyze_only:
        extract_features(args.device, out, args.n_pairs)
    if args.explore_only:
        analyze(out, explore_only=True)
    elif args.n_pairs is None or args.analyze_only:
        s = analyze(out)
        print("\n" + "=" * 80)
        print(s["verdict"])
        print("=" * 80)
