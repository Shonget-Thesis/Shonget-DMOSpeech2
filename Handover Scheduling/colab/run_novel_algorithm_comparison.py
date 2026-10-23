"""Novel-Algorithm Comparison: three candidate handover algorithms vs. the current proposal.

Policies compared on the same 500 Seed-TTS speakers, with the same failure rule, the same
Learn-Then-Test calibration (alpha = 0.05, delta = 0.10) and the same held-out test split:

  Baselines   B0 Fixed Step 8 (DMOSpeech 2 reference)
              B1 SNR gate -> Step 4 (proposal's SNR-only rule)   B2 SNR gate -> Step 5
  Proposal    P0 SNR + drift, frozen lambda = 0.50 (Definition 3.4)
              P1 SNR + drift, lambda re-certified by LTT through this script's pipeline
  C1  Student-Probe Handover. At candidate step k the student takes its real first step
      from the teacher's endpoint (same seeded noise); the teacher-student disagreement
      a_k = ||y_hat - x1_hat|| / ||x1_hat|| decides the handover. An accepted probe is reused
      as the student's first step (0 extra NFE); each rejected probe costs 1 pass.
      C1a: probe only on prompts that pass the SNR gate.  C1b: probe on every prompt.
  C2  Joint handover-and-restart selection. Configurations (k, j) with handover step
      k in {4, 5, 6, 8} and student start j in {1, 2} (j = 2 restarts at tau = 0.2929 with
      2 student passes). Gate-passed and gate-failed prompts each get one configuration,
      chosen by Pareto testing (dev ordering, LTT on calibration).
  C3  Risk-budgeted SNR-band schedule. One handover step per SNR band
      [<15, 15-20, 20-30, >=30 dB], chosen by Pareto testing over all 4^4 schedules.

Labels: j = 1 handovers reuse colab/results/gpu_grid_all1000.csv (C1's output is identical to a
Step-k handover, so its labels are exact). C2's j = 2 configurations are synthesized and scored
here (Whisper large-v3, WavLM SV, UTMOS) against a Step-8 reference re-synthesized in the same run.

Usage:
  python run_novel_algorithm_comparison.py --device cuda                  # extract + analyse
  python run_novel_algorithm_comparison.py --device cuda --skip_c2        # C1 / C3 only (fast)
  python run_novel_algorithm_comparison.py --device cuda --n_pairs 4 --output_dir <smoke>
  python run_novel_algorithm_comparison.py --analyze_only
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOVER_DIR = os.path.dirname(COLAB_DIR)
ROOT_DIR = os.path.dirname(HANDOVER_DIR)
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), HANDOVER_DIR, COLAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_signal_selection_validation import auc, auc_ci, mcnemar  # noqa: E402

GRID_CSV = os.path.join(COLAB_DIR, "results", "gpu_grid_all1000.csv")
DEFAULT_OUT = os.path.join(COLAB_DIR, "results", "novel_algorithms")
FEATURES_CSV = "novel_features_all1000.csv"
PREREG_JSON = "novel_algorithms_preregistration.json"
SUMMARY_JSON = "novel_algorithms_summary.json"
REPORT_MD = "NOVEL_ALGORITHMS_REPORT.md"

CANDIDATES = (4, 5, 6)
REFERENCE = 8
STEPS = (4, 5, 6, 8)
SNR_GATE_DB = 20.0
SNR_BANDS = (15.0, 20.0, 30.0)                 # edges -> bands [<15, 15-20, 20-30, >=30]

PREREG: Dict[str, Any] = {
    "version": 1,
    "policies": {
        "B0_step8": "fixed Step 8, j = 1 (DMOSpeech 2 reference)",
        "B1_snr_step4": "SNR >= 20 dB -> Step 4, else Step 8 (proposal's SNR-only rule)",
        "B2_snr_step5": "SNR >= 20 dB -> Step 5, else Step 8",
        "P0_proposal_frozen": "SNR >= 20 dB and first k in (4,5,6) with d_k <= 0.50, else Step 8",
        "P1_proposal_ltt": "as P0, lambda certified by LTT over 0.20, 0.25, ..., 1.20",
        "C1a_probe_gated": "SNR >= 20 dB and first k in (4,5,6) with a_k <= lambda; lambda by LTT",
        "C1b_probe_all": "first k in (4,5,6) with a_k <= lambda on every prompt; lambda by LTT",
        "C2_joint_restart": "gate-passed / gate-failed prompts each get one (k, j); Pareto testing",
        "C3_snr_bands": "one step per SNR band [<15, 15-20, 20-30, >=30 dB]; Pareto testing over 256 schedules",
    },
    "probe": "a_k = ||y_hat - x1_hat(t_k)|| / ||x1_hat(t_k)|| over generated frames, y_hat = first student step "
             "(tau = 0.0761, the same seeded noise as the real student); higher a_k = riskier (fixed a priori)",
    "probe_cost": "accepted probe reused as the first student step (0 extra NFE); each rejected probe +1 NFE",
    "nfe": "teacher 2(k + 1) + student (4 - j)",
    "threshold_grid": "C1: development quantiles q = 0, 0.05, ..., 1 of a_k pooled over steps 4-6 "
                      "(gated or all prompts), preceded by a no-early-handover value; ordered conservative -> aggressive",
    "pareto_testing": "development: robust risk and NFE of every candidate; keep the Pareto front; order by development "
                      "risk ascending (ties: higher NFE first), always starting from the Step-8 reference; "
                      "calibration: fixed-sequence LTT; keep the last certified",
    "ltt": {"alpha": 0.05, "delta": 0.10, "p_value": "P(Bin(n, alpha) <= k)"},
    "risk": "robust per speaker: a speaker fails if either its clean or noisy15 utterance fails",
    "verdict_vs_P0": {
        "IMPROVES": "certified on calibration, test risk <= alpha, and test mean NFE lower than P0 by more than 0.5",
        "MATCHES": "certified, test risk <= alpha, |delta NFE| <= 0.5",
        "WORSE": "certified, test risk <= alpha, test mean NFE higher than P0 by more than 0.5",
        "FAILS_GUARANTEE": "test risk > alpha",
        "NOT_CERTIFIED": "LTT certifies nothing on calibration",
    },
    "caveat": "the test split was inspected in earlier analyses; a winning candidate needs confirmation on fresh speakers",
}


def nfe(k: int, j: int = 1) -> int:
    return 2 * (k + 1) + (4 - j)


# ============================================================================
# 1. EXTRACTION (GPU)
# ============================================================================

def student_complete(tts, cond, text, endpoint, prompt_len: int, input_seed: int, taus: List[float]):
    """core.sampler.run_student_completion with a configurable student grid (start index j)."""
    import torch
    from core.sampler import make_local_generator

    device = cond.device
    x_curr = endpoint.clone()
    first_pred = None
    with torch.inference_mode():
        for tau in taus:
            gen_s = make_local_generator((int(input_seed) + int(tau * 100)) % (2 ** 31 - 1), device=device)
            eps = torch.randn(x_curr.shape, dtype=torch.float32, device=device, generator=gen_s)
            time_tensor = torch.tensor([tau], device=device, dtype=torch.float32)
            t_scalar = time_tensor.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - t_scalar) * eps + t_scalar * x_curr
            pred_s = tts.generator(x=phi, cond=cond, text=text, time=time_tensor,
                                   drop_audio_cond=False, drop_text=False)
            x_curr = phi + (1.0 - t_scalar) * pred_s
            if first_pred is None:
                first_pred = x_curr.clone()
    return x_curr, first_pred


def decode(tts, x, prompt_len: int) -> np.ndarray:
    import torch
    with torch.inference_mode():
        mel = x.permute(0, 2, 1) * getattr(tts, "scale", 1.0)
        return tts.vocos.decode(mel[..., prompt_len:]).cpu().numpy().squeeze()


def probe_disagreement(tts, cond, text, endpoint, prompt_len: int, input_seed: int) -> float:
    """First student step from the teacher endpoint (identical to the real student's first step)."""
    import torch
    from core.sampler import STUDENT_TAUS

    _, y_hat = student_complete(tts, cond, text, endpoint, prompt_len, input_seed, [STUDENT_TAUS[0]])
    e = endpoint[:, prompt_len:, :].float()
    y = y_hat[:, prompt_len:, :].float()
    return float(torch.norm(y - e) / (torch.norm(e) + 1e-8))


def extract(device_name: Optional[str], output_dir: str, n_pairs: Optional[int], skip_c2: bool) -> None:
    import soundfile as sf
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    from infer import DMOInference
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
    from core.sampler import (CFG_STRENGTH, STUDENT_TAUS, estimate_prompt_snr, make_local_generator,
                              run_teacher_single_pass)
    from core.metrics import MetricSuite, check_failure, load_failure_config
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
    metrics = None if skip_c2 else MetricSuite(device=device_str)
    failure_cfg = load_failure_config()

    rows = load_expanded_manifest("all1000")
    if n_pairs is not None:
        rows = rows[:n_pairs]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, FEATURES_CSV)
    records, done = [], set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        if not skip_c2 and "fail_k8j2" not in prev.columns:
            raise RuntimeError(f"{csv_path} was made with --skip_c2. Use a new --output_dir for the full C2 run.")
        records = prev.to_dict("records")
        done = {(r["utt_id"], r["condition"]) for r in records}
        print(f"[Resume] {len(done)} rows already extracted.")

    j2_taus = list(STUDENT_TAUS[1:])
    t_start = time.time()
    for idx, item in enumerate(rows, start=1):
        if (item["utt_id"], item["condition"]) in done:
            continue
        wav_path = item["prompt_wav_path"]
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(f"Prompt audio missing or an LFS pointer: {wav_path}. Run the dataset cell first.")
        raw, sr = torchaudio.load(wav_path)
        wav = raw.squeeze().numpy()
        if item["condition"] == "noisy15":
            wav, _, _ = add_gaussian_noise_15db(wav, seed=item["input_seed"], target_snr_db=15.0)
        wav = np.ascontiguousarray(wav, dtype=np.float32)

        rec: Dict[str, Any] = {"utt_id": item["utt_id"], "speaker_id": item["speaker_id"],
                               "condition": item["condition"], "split": item.get("split", ""),
                               "snr_energy": float(estimate_prompt_snr(wav, sample_rate=sr))}

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
        cond_tensor, text_ids, step_cond, cond_mask, _, dur_tensor = tts._prepare_inputs(audio, text_list, total_dur)
        prompt_len = cond_mask.sum().item()
        gen = make_local_generator(item["input_seed"])
        y0 = pad_sequence([torch.randn(d, 100, dtype=torch.float32, generator=gen) for d in dur_tensor],
                          padding_value=0, batch_first=True).to(device)

        endpoints, drift, _ = run_teacher_single_pass(tts=tts, step_cond=step_cond, text=text_ids, y0=y0,
                                                      prompt_len=prompt_len, target_steps=STEPS,
                                                      cfg_strength=CFG_STRENGTH)
        for i in CANDIDATES:
            rec[f"drift_{i}"] = float(drift[i])
            rec[f"probe_{i}"] = probe_disagreement(tts, cond_tensor, text_ids, endpoints[i], prompt_len, item["input_seed"])

        if not skip_c2:
            gt_path = item["gt_wav_path"]
            if os.path.exists(gt_path) and os.path.getsize(gt_path) >= 1000:
                g, gsr = torchaudio.load(gt_path)
                gt_np, rec["gt_fallback_to_prompt"] = g.squeeze().numpy(), False
            else:
                gt_np, gsr, rec["gt_fallback_to_prompt"] = wav, sr, True

            def score(x):
                a = decode(tts, x, prompt_len)
                return (metrics.compute_wer(a, item["gt_text"], sample_rate=24000),
                        metrics.compute_similarity(a, gt_np, sample_rate=24000, gt_sr=gsr),
                        metrics.compute_utmos(a, sample_rate=24000))

            x_ref, _ = student_complete(tts, cond_tensor, text_ids, endpoints[REFERENCE], prompt_len,
                                        item["input_seed"], list(STUDENT_TAUS))
            rec["wer_ref"], rec["sim_ref"], rec["utmos_ref"] = score(x_ref)
            for k in STEPS:
                x_k, _ = student_complete(tts, cond_tensor, text_ids, endpoints[k], prompt_len,
                                          item["input_seed"], j2_taus)
                w, s, u = score(x_k)
                fail, _ = check_failure(w, rec["wer_ref"], s, rec["sim_ref"], config=failure_cfg)
                rec.update({f"wer_k{k}j2": w, f"sim_k{k}j2": s, f"utmos_k{k}j2": u, f"fail_k{k}j2": bool(fail)})

        records.append(rec)
        if idx % 10 == 0 or idx == len(rows):
            pd.DataFrame(records).to_csv(csv_path, index=False)
            print(f"[{idx}/{len(rows)}] {item['utt_id']} ({item['condition']}) | {(time.time() - t_start) / 60:.1f} min elapsed")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"[OK] Features written: {csv_path} ({len(records)} rows)")


# ============================================================================
# 2. ANALYSIS
# ============================================================================

class Table:
    """Per-utterance failure matrix over configurations, plus speaker codes for robust risk."""

    def __init__(self, df: pd.DataFrame, configs: List[str]):
        self.df = df.reset_index(drop=True)
        self.configs = configs
        self.F = np.stack([self.df[f"fail_{c}"].values.astype(bool) for c in configs], axis=1)
        self.cost = np.array([nfe(int(c[1]), int(c[3])) for c in configs], dtype=float)
        self.spk = pd.factorize(self.df["speaker_id"])[0]
        self.n_spk = int(self.spk.max()) + 1
        self.idx = {c: i for i, c in enumerate(configs)}

    def evaluate(self, cfg: np.ndarray, extra: Optional[np.ndarray] = None) -> Tuple[int, int, float, pd.Series]:
        fail = self.F[np.arange(len(cfg)), cfg]
        per = np.bincount(self.spk, weights=fail, minlength=self.n_spk) > 0
        cost = self.cost[cfg] + (extra if extra is not None else 0.0)
        fails = pd.Series(per, index=pd.factorize(self.df["speaker_id"])[1])
        return int(per.sum()), self.n_spk, float(cost.mean()), fails


def ltt_select(table: "Table", policies: List[Tuple[Any, Callable]]) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    a, d = PREREG["ltt"]["alpha"], PREREG["ltt"]["delta"]
    chosen, trace = None, []
    for label, fn in policies:
        k, n, cost, _ = table.evaluate(*fn(table))
        p = float(binom.cdf(k, n, a))
        trace.append({"value": label, "failures": k, "n": n, "nfe": cost, "p": p, "certified": p <= d})
        if p > d:
            break
        chosen = (label, fn)
    return chosen, trace


def pareto_order(dev: "Table", policies: List[Tuple[Any, Callable]], reference: Any) -> List[Tuple[Any, Callable]]:
    """Reference first (risk 0 by definition), then the development Pareto front of the other candidates."""
    ref = next(p for p in policies if p[0] == reference)
    scored = []
    for label, fn in policies:
        if label == reference:
            continue
        k, n, cost, _ = dev.evaluate(*fn(dev))
        scored.append((k / n, cost, label, fn))
    front = [s for s in scored if not any((o[0] <= s[0] and o[1] < s[1]) or (o[0] < s[0] and o[1] <= s[1]) for o in scored)]
    front.sort(key=lambda s: (s[0], -s[1]))
    return [ref] + [(s[2], s[3]) for s in front]


def seq_policy(scores: Dict[int, np.ndarray], open_mask: np.ndarray, lam: float, t: "Table",
               probe_cost: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Definition 3.4 template: first k in (4,5,6) with score_k <= lam among open rows, else Step 8."""
    cfg = np.full(len(open_mask), t.idx["k8j1"])
    extra = np.zeros(len(open_mask))
    still = open_mask.copy()
    for n_rejected, k in enumerate(CANDIDATES):
        hit = still & (scores[k] <= lam)
        cfg[hit] = t.idx[f"k{k}j1"]
        if probe_cost:
            extra[hit] = n_rejected
        still &= ~hit
    if probe_cost:
        extra[still] = len(CANDIDATES)
    return cfg, extra


def load_table(output_dir: str) -> Tuple[pd.DataFrame, bool]:
    feats = pd.read_csv(os.path.join(output_dir, FEATURES_CSV))
    grid = pd.read_csv(GRID_CSV)
    lab = grid.pivot_table(index=["utt_id", "condition"], columns="teacher_step", values="is_failure", aggfunc="first")
    lab = lab.rename(columns={k: f"fail_k{k}j1" for k in lab.columns}).reset_index()
    ref = grid[grid.teacher_step == REFERENCE][["utt_id", "condition", "d_4", "d_5", "d_6", "wer_ref", "similarity_ref"]]
    ref = ref.rename(columns={"wer_ref": "grid_wer_ref", "similarity_ref": "grid_sim_ref"})
    df = feats.merge(lab, on=["utt_id", "condition"]).merge(ref, on=["utt_id", "condition"], how="left")
    for k in STEPS:
        df[f"fail_k{k}j1"] = df[f"fail_k{k}j1"].astype(str).str.lower().isin(["true", "1", "1.0"])
    has_c2 = "fail_k8j2" in df.columns
    if has_c2:
        for k in STEPS:
            df[f"fail_k{k}j2"] = df[f"fail_k{k}j2"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["split"] = df["split"].replace({"development": "dev", "calibration": "calib"})
    return df, has_c2


def analyze(output_dir: str) -> Dict[str, Any]:
    with open(os.path.join(output_dir, PREREG_JSON), "r", encoding="utf-8") as f:
        if json.load(f)["protocol"] != PREREG:
            raise RuntimeError("Preregistration mismatch: the frozen protocol differs from this script.")
    df, has_c2 = load_table(output_dir)
    configs = [f"k{k}j1" for k in STEPS] + ([f"k{k}j2" for k in STEPS] if has_c2 else [])
    split = {s: df[df.split == s].reset_index(drop=True) for s in ("dev", "calib", "test")}
    T = {s: Table(x, configs) for s, x in split.items()}
    a = PREREG["ltt"]["alpha"]
    rng = np.random.default_rng(42)
    out: Dict[str, Any] = {"has_c2": has_c2, "n_speakers": {s: T[s].n_spk for s in T}, "configs": configs}

    # Reproduction checks
    rep = {}
    for i in CANDIDATES:
        diff = (df[f"drift_{i}"] - df[f"d_{i}"]).abs()
        rep[f"d_{i}"] = {"max_abs_diff": float(diff.max())}
    if has_c2:
        rep["wer_ref_vs_grid"] = {"mean_abs_diff": float((df.wer_ref - df.grid_wer_ref).abs().mean()),
                                  "pct_identical": float(100 * np.mean(np.isclose(df.wer_ref, df.grid_wer_ref)))}
        rep["sim_ref_vs_grid"] = {"mean_abs_diff": float((df.sim_ref - df.grid_sim_ref).abs().mean())}
        rep["gt_fallback_rows"] = int(df.get("gt_fallback_to_prompt", pd.Series(dtype=bool)).astype(bool).sum())
    out["reproduction"] = rep

    # Signal quality: probe vs. drift on development
    dev = split["dev"]
    gated = dev.snr_energy >= SNR_GATE_DB
    sig = {}
    for name in ("drift", "probe"):
        sig[name] = {
            "auc_fail4_gated": auc(dev.loc[gated, f"{name}_4"].values, dev.loc[gated, "fail_k4j1"].values),
            "auc_fail4_gated_ci": auc_ci(dev[gated], f"{name}_4", "fail_k4j1", rng),
            "auc_fail5_noisy": auc(dev.loc[dev.condition == "noisy15", f"{name}_5"].values,
                                   dev.loc[dev.condition == "noisy15", "fail_k5j1"].values),
            "auc_fail5_all": auc(dev[f"{name}_5"].values, dev["fail_k5j1"].values),
        }
    out["signal_auc_dev"] = sig

    policies: Dict[str, Dict[str, Any]] = {}

    def record(name: str, chosen, trace, family: str, describe: Callable[[Any], str]):
        if chosen is None:
            policies[name] = {"family": family, "certified": False, "calib_trace": trace}
            return
        label, fn = chosen
        k, n, cost, fails = T["test"].evaluate(*fn(T["test"]))
        cal = next(t for t in trace if t["value"] == label)
        policies[name] = {"family": family, "certified": True, "choice": describe(label), "calib": cal,
                          "test_failures": k, "test_n": n, "test_risk": k / n, "test_nfe": cost,
                          "test_safe": k / n <= a, "_fails": fails, "_fn": fn}

    def fixed(name: str, fn, family: str, describe: str):
        k, n, cost, fails = T["test"].evaluate(*fn(T["test"]))
        kc, nc, cc, _ = T["calib"].evaluate(*fn(T["calib"]))
        pc = float(binom.cdf(kc, nc, a))
        policies[name] = {"family": family, "certified": pc <= PREREG["ltt"]["delta"], "choice": describe,
                          "calib": {"failures": kc, "n": nc, "nfe": cc, "p": pc},
                          "test_failures": k, "test_n": n, "test_risk": k / n, "test_nfe": cost,
                          "test_safe": k / n <= a, "_fails": fails, "_fn": fn}

    gate = lambda t: t.df.snr_energy.values >= SNR_GATE_DB
    sc = lambda t, name: {k: t.df[f"{name}_{k}"].values for k in CANDIDATES}

    # Baselines and the proposal
    fixed("B0_step8", lambda t: (np.full(len(t.df), t.idx["k8j1"]), None), "baseline", "always Step 8")
    fixed("B1_snr_step4", lambda t: (np.where(gate(t), t.idx["k4j1"], t.idx["k8j1"]), None), "baseline", "SNR >= 20 -> Step 4")
    fixed("B2_snr_step5", lambda t: (np.where(gate(t), t.idx["k5j1"], t.idx["k8j1"]), None), "baseline", "SNR >= 20 -> Step 5")
    fixed("P0_proposal_frozen", lambda t: seq_policy(sc(t, "drift"), gate(t), 0.50, t, False), "proposal", "SNR + drift, lambda = 0.50")
    lam_grid = [round(0.20 + 0.05 * i, 2) for i in range(21)]
    pol = [(lam, (lambda t, lam=lam: seq_policy(sc(t, "drift"), gate(t), lam, t, False))) for lam in lam_grid]
    record("P1_proposal_ltt", *ltt_select(T["calib"], pol), "proposal", lambda l: f"SNR + drift, lambda = {l:.2f}")

    # C1: student probe
    for name, use_gate in (("C1a_probe_gated", True), ("C1b_probe_all", False)):
        mask_dev = gate(T["dev"]) if use_gate else np.ones(len(dev), dtype=bool)
        pooled = np.concatenate([dev.loc[mask_dev, f"probe_{k}"].values for k in CANDIDATES])
        grid_vals = [float(pooled.min()) - 1.0] + sorted(set(float(q) for q in np.quantile(pooled, np.linspace(0, 1, 21))))
        pol = [(lam, (lambda t, lam=lam, g=use_gate: seq_policy(
            sc(t, "probe"), gate(t) if g else np.ones(len(t.df), dtype=bool), lam, t, True))) for lam in grid_vals]
        record(name, *ltt_select(T["calib"], pol), "C1", lambda l: f"probe lambda = {l:.4f}")

    # C2: joint (k, j) per gate group
    if has_c2:
        cands = []
        for cg, cn in itertools.product(configs, configs):
            fn = (lambda t, cg=cg, cn=cn: (np.where(gate(t), t.idx[cg], t.idx[cn]), None))
            cands.append(((cg, cn), fn))
        order = pareto_order(T["dev"], cands, reference=("k8j1", "k8j1"))
        record("C2_joint_restart", *ltt_select(T["calib"], order), "C2",
               lambda l: f"SNR >= 20 -> {l[0]}, else {l[1]}")
        out["c2_pareto_front_size"] = len(order)
        out["c2_config_failure_rates"] = {
            c: {cond: float(df.loc[df.condition == cond, f"fail_{c}"].mean()) for cond in ("clean", "noisy15")}
            for c in configs}

    # C3: SNR-band schedule
    def band(t):
        return np.digitize(t.df.snr_energy.values, SNR_BANDS)
    cands = []
    for sched in itertools.product(STEPS, repeat=len(SNR_BANDS) + 1):
        fn = (lambda t, sched=sched: (np.array([t.idx[f"k{sched[b]}j1"] for b in band(t)]), None))
        cands.append((sched, fn))
    order = pareto_order(T["dev"], cands, reference=(8, 8, 8, 8))
    record("C3_snr_bands", *ltt_select(T["calib"], order), "C3",
           lambda l: "bands <15 / 15-20 / 20-30 / >=30 dB -> Steps " + " / ".join(str(s) for s in l))
    out["c3_pareto_front_size"] = len(order)

    # Verdicts and McNemar vs. the frozen proposal
    p0 = policies["P0_proposal_frozen"]
    for name, p in policies.items():
        if name == "P0_proposal_frozen":
            continue
        if not p.get("certified"):
            p["verdict_vs_P0"] = "NOT_CERTIFIED"
            continue
        if not p["test_safe"]:
            p["verdict_vs_P0"] = "FAILS_GUARANTEE"
        else:
            dlt = p["test_nfe"] - p0["test_nfe"]
            p["verdict_vs_P0"] = "IMPROVES" if dlt < -0.5 else ("WORSE" if dlt > 0.5 else "MATCHES")
        p["mcnemar_vs_P0"] = mcnemar(p["_fails"], p0["_fails"])
    out["policies"] = policies

    clean = _strip(out)
    with open(os.path.join(output_dir, SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    write_report(clean, output_dir)
    return clean


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ============================================================================
# 3. REPORT
# ============================================================================

NAMES = {
    "B0_step8": "B0 Fixed Step 8 (DMOSpeech 2)", "B1_snr_step4": "B1 SNR → Step 4 (SNR-only)",
    "B2_snr_step5": "B2 SNR → Step 5", "P0_proposal_frozen": "**P0 Proposal: SNR + drift (λ = 0.50)**",
    "P1_proposal_ltt": "P1 Proposal, λ re-certified", "C1a_probe_gated": "**C1a Student probe (gated)**",
    "C1b_probe_all": "**C1b Student probe (all prompts)**", "C2_joint_restart": "**C2 Joint handover + restart**",
    "C3_snr_bands": "**C3 SNR-band schedule**",
}


def write_report(s: Dict[str, Any], output_dir: str) -> None:
    L = ["# Novel-Algorithm Comparison: Three Candidates vs. the Current Proposal\n",
         "Generated by `colab/run_novel_algorithm_comparison.py`. Protocol frozen in "
         f"`{PREREG_JSON}` before analysis. Development → ordering and thresholds grid; calibration → Learn-Then-Test "
         "(α = 0.05, δ = 0.10); test → held-out robust risk and mean NFE (one look).\n",
         f"> Caveat: {PREREG['caveat']}.\n",
         f"Speakers: dev {s['n_speakers']['dev']}, calibration {s['n_speakers']['calib']}, test {s['n_speakers']['test']}. "
         f"C2 (j = 2 synthesis) included: {'yes' if s['has_c2'] else 'no (run without --skip_c2)'}.\n",
         "## Reproduction checks\n"]
    for k, v in s["reproduction"].items():
        L.append(f"- `{k}`: " + ", ".join(f"{a} = {b:.6f}" if isinstance(b, float) else f"{a} = {b}" for a, b in v.items())
                 if isinstance(v, dict) else f"- `{k}`: {v}")

    L.append("\n## Held-out test results (200 speakers)\n")
    L.append("| Policy | Choice (certified on calibration) | Test failures | Mean NFE | ≤ α | Verdict vs. P0 | McNemar vs. P0 (only this fails / only P0 fails, p) |")
    L.append("|---|---|---|---|---|---|---|")
    for name, p in s["policies"].items():
        if not p.get("certified") and "test_failures" not in p:
            L.append(f"| {NAMES.get(name, name)} | not certified | — | — | — | NOT_CERTIFIED | — |")
            continue
        m = p.get("mcnemar_vs_P0")
        mc = f"{m['only_first_fails']} / {m['only_second_fails']}, p = {m['p_two_sided']:.3f}" if m else "—"
        L.append(f"| {NAMES.get(name, name)} | {p.get('choice', '')} | {p['test_failures']}/{p['test_n']} "
                 f"({100 * p['test_risk']:.1f}%) | {p['test_nfe']:.2f} | {'✅' if p['test_safe'] else '❌'} | "
                 f"{p.get('verdict_vs_P0', '(reference)')} | {mc} |")

    sa = s["signal_auc_dev"]
    L.append("\n## C1 signal quality on development: student probe vs. drift\n")
    L.append("| Signal | AUC, Step-4 failure (gated) [95% CI] | AUC, Step-5 failure (noisy prompts) | AUC, Step-5 failure (all) |")
    L.append("|---|---|---|---|")
    for n in ("drift", "probe"):
        v = sa[n]
        L.append(f"| {n} | {v['auc_fail4_gated']:.3f} [{v['auc_fail4_gated_ci'][0]:.2f}, {v['auc_fail4_gated_ci'][1]:.2f}] | "
                 f"{v['auc_fail5_noisy']:.3f} | {v['auc_fail5_all']:.3f} |")
    L.append("\nAUC 0.5 = chance. The noisy-prompt column is where a probe could rescue prompts the SNR gate sends to Step 8.\n")

    if s["has_c2"]:
        L.append("## C2 failure rate per configuration (all 500 speakers)\n")
        L.append("| Configuration | NFE | Clean failure | Noisy15 failure |")
        L.append("|---|---|---|---|")
        for c, v in s["c2_config_failure_rates"].items():
            L.append(f"| {c} (Step {c[1]}, j = {c[3]}) | {nfe(int(c[1]), int(c[3]))} | {100 * v['clean']:.1f}% | {100 * v['noisy15']:.1f}% |")
        L.append(f"\nPareto front used for testing: {s['c2_pareto_front_size']} candidates.\n")
    L.append(f"C3 Pareto front used for testing: {s['c3_pareto_front_size']} of 256 schedules.\n")

    L.append("## Verdict rules\n")
    for k, d in PREREG["verdict_vs_P0"].items():
        L.append(f"- **{k}**: {d}")
    L.append("\n## Before claiming novelty\n")
    L.append("- Run a literature search for each candidate that IMPROVES or MATCHES.")
    L.append("- Confirm any winner once on fresh speakers (e.g., the 166 unused Seed-TTS speakers) before Chapter 3 changes.")
    L.append("- Get the adviser's approval before replacing Definition 3.4.")
    with open(os.path.join(output_dir, REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[OK] Report written: {os.path.join(output_dir, REPORT_MD)}")


# ============================================================================
# 4. MAIN
# ============================================================================

def freeze_protocol(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PREREG_JSON)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if json.load(f)["protocol"] != PREREG:
                raise RuntimeError(f"{path} holds a different protocol. Use a new --output_dir.")
        print(f"[Prereg] Verified frozen protocol ({path}).")
        return
    with open(os.path.abspath(__file__), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "script_sha256": sha, "protocol": PREREG}, f, indent=2)
    print(f"[Prereg] Protocol frozen: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--n_pairs", type=int, default=None, help="smoke test: first N rows only (analysis skipped)")
    ap.add_argument("--skip_c2", action="store_true", help="skip the j = 2 synthesis and scoring (C2)")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    freeze_protocol(args.output_dir)
    if not args.analyze_only:
        extract(args.device, args.output_dir, args.n_pairs, args.skip_c2)
    if args.n_pairs is not None:
        print("[Smoke] Extraction finished; analysis needs all three splits and is skipped.")
        return
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
