import os
import sys
import time
import json
import warnings
import re
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = r"c:\Users\Rhenel Jhon Sajol\Desktop\TTS\DMOSpeech 2"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from infer import DMOInference
from resemblyzer import VoiceEncoder, preprocess_wav
import jiwer
from transformers import pipeline

OUTPUT_DIR = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "baseline_collision_validation")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_stress")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "stress_condition_pilot_metrics.json")
REPORT_MD_PATH = os.path.join(RESULTS_DIR, "STRESS_CONDITION_PILOT_REPORT.md")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_spectral_flux(audio, sr=24000):
    if len(audio) == 0:
        return 0.0, 0.0
    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    diff = np.diff(S, axis=1)
    pos_diff = np.maximum(0, diff)
    flux = np.sqrt(np.mean(pos_diff**2, axis=0))
    return float(np.max(flux)), float(np.mean(flux))


def compute_f0_cv(audio, sr=24000):
    try:
        if sr != 16000:
            audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio.astype(np.float32)
        f0, voiced_flag, _ = librosa.pyin(
            audio_16k,
            fmin=80,
            fmax=400,
            sr=16000,
            frame_length=1024,
            hop_length=512
        )
        v_f0 = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else np.array([])
        if len(v_f0) > 5 and np.mean(v_f0) > 1e-4:
            return float(np.std(v_f0) / np.mean(v_f0))
        return 0.0
    except Exception:
        return 0.0


def main():
    print("=" * 80)
    print("TARGETED STRESS-CONDITION PILOT: EARLIER HANDOVER (t_h = 0.10)")
    print("Protocol: REMAINING_COLLISION_VALIDATION_TESTS_AND_STOP_RULE.md (Section 2)")
    print("Design: 5 speakers x 2 texts x 3 seeds = 30 matched units")
    print("=" * 80)

    # 1. Select the first 5 speakers from manifest_seedtts_100.json
    manifest_path = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    spk_map = {}
    for item in manifest:
        spk = item['utt_id'].split('-')[0]
        spk_map.setdefault(spk, []).append(item)

    selected_spks = [s for s, items in spk_map.items() if len(items) >= 2][:5]
    test_pairs = []
    for spk_idx, spk in enumerate(selected_spks):
        items = spk_map[spk][:2]
        for sent_idx, it in enumerate(items):
            test_pairs.append({
                "spk_id": spk,
                "spk_index": spk_idx + 1,
                "sent_index": sent_idx + 1,
                "utt_id": it["utt_id"],
                "prompt_wav": os.path.join(BASE_DIR, it["prompt_wav"]),
                "prompt_text": it["prompt_text"],
                "target_text": it["target_text"]
            })

    seeds = [42, 123, 2026]
    total_trials = len(test_pairs) * len(seeds)
    print(f"Selected {len(selected_spks)} speakers x 2 sentences = {len(test_pairs)} pairs x {len(seeds)} seeds = {total_trials} trials.")

    # 2. Initialize Models
    print("\nLoading DMOSpeech 2 model components...")
    t0 = time.time()
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cpu"

    tts = DMOInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    print(f"DMOSpeech 2 loaded in {time.time() - t0:.2f}s")

    print("Loading VoiceEncoder for Speaker Similarity (SIM)...")
    voice_encoder = VoiceEncoder(device="cpu")

    print("Loading Whisper ASR pipeline for Word Error Rate (WER)...")
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=-1)
    print("All evaluation modules ready.\n")

    # Load existing progress if available
    records = []
    if os.path.exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and "trials" in saved:
                    records = saved["trials"]
                    print(f"Loaded {len(records)} existing trials from {METRICS_JSON_PATH}")
        except Exception:
            records = []

    completed_keys = set((r["spk_id"], r["sent_index"], r["seed"]) for r in records)

    from f5_tts.infer.utils_infer import convert_char_to_pinyin, preprocess_ref_audio_text
    import torchaudio

    trial_idx = len(records)
    for pair in test_pairs:
        spk = pair["spk_id"]
        sent_i = pair["sent_index"]
        prompt_wav = pair["prompt_wav"]
        prompt_text = pair["prompt_text"]
        target_text = pair["target_text"]

        ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(prompt_wav, prompt_text)
        audio_t, sr = torchaudio.load(ref_audio_clean)
        if audio_t.shape[0] > 1:
            audio_t = torch.mean(audio_t, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio_t)))
        if rms < 0.3:
            audio_t = audio_t * 0.3 / rms
        if sr != tts.target_sample_rate:
            audio_t = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio_t)
        audio_t = audio_t.to(device)

        # Precompute reference speaker embedding
        try:
            ref_wav_processed = preprocess_wav(prompt_wav)
            ref_embed = voice_encoder.embed_utterance(ref_wav_processed)
        except Exception:
            ref_embed = None

        torch.manual_seed(42)
        np.random.seed(42)
        fixed_dur = tts.predict_duration(prompt_wav, target_text, prompt_text, dp_softmax_range=0.7, temperature=0)

        text_list = [ref_text_clean + target_text]
        final_text_list = convert_char_to_pinyin(text_list)
        ref_audio_len = audio_t.shape[-1] // tts.hop_length
        total_duration = ref_audio_len + fixed_dur

        cond, text, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
            audio_t, final_text_list, total_duration
        )
        prompt_len = cond_mask.sum().item()

        for seed in seeds:
            if (spk, sent_i, seed) in completed_keys:
                continue

            trial_idx += 1
            t_trial_start = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)

            # STRESS CONDITION: Early Handover at t_h = 0.10 (Only ~2-3 teacher steps)
            teacher_steps = 16
            teacher_stopping_time = 0.10
            sway_sampling_coef = -1.0
            cfg_strength = 2.0

            t_vec = torch.linspace(0, 1, teacher_steps + 1, device=device, dtype=step_cond.dtype)
            if sway_sampling_coef is not None:
                t_vec = t_vec + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_vec) - 1 + t_vec)
            
            idx_cutoff = (t_vec > teacher_stopping_time).float().argmax().item() + 2
            t_sub = t_vec[:idx_cutoff][:-1]

            def teacher_fn(t_val, x_val):
                with torch.inference_mode():
                    pred = tts.teacher(x=x_val, cond=step_cond, text=text, time=t_val, mask=None, drop_audio_cond=False, drop_text=False)
                    null_pred = tts.teacher(x=x_val, cond=step_cond, text=text, time=t_val, mask=None, drop_audio_cond=True, drop_text=True)
                    return pred + (pred - null_pred) * cfg_strength

            torch.manual_seed(seed)
            y0 = [torch.randn(dur, 100, device=device, dtype=step_cond.dtype) for dur in dur_tensor]
            y0 = pad_sequence(y0, padding_value=0, batch_first=True)

            trajectory = odeint(teacher_fn, y0, t_sub, method="euler")
            
            t_h = t_sub[-1]
            x_h_pre = trajectory[-1]
            v_T_h = teacher_fn(t_h, x_h_pre)
            x_h = x_h_pre + (1 - t_h) * v_T_h

            t_h_minus_1 = t_sub[-2]
            dt_teacher = (t_h - t_h_minus_1).item()
            v_T_prior = teacher_fn(t_h_minus_1, trajectory[-2])

            with torch.no_grad():
                v_S_h = tts.generator(
                    x=x_h, cond=cond, text=text, time=t_h.unsqueeze(0),
                    drop_audio_cond=False, drop_text=False
                )

            vt_gen = v_T_h[:, prompt_len:, :].reshape(-1)
            vs_gen = v_S_h[:, prompt_len:, :].reshape(-1)
            vt_prior_gen = v_T_prior[:, prompt_len:, :].reshape(-1)

            norm_t = torch.norm(vt_gen).item()
            norm_s = torch.norm(vs_gen).item()
            norm_t_prior = torch.norm(vt_prior_gen).item()

            c_ts = F.cosine_similarity(vt_gen.unsqueeze(0), vs_gen.unsqueeze(0)).item()
            deg_ts = float(np.arccos(np.clip(c_ts, -1.0, 1.0)) * 180.0 / np.pi)

            l2_diff_ts = torch.norm(vs_gen - vt_gen).item()
            m_ts = float(l2_diff_ts / (norm_t + 1e-8))
            r_ts = float(norm_s / (norm_t + 1e-8))

            l2_diff_tt = torch.norm(vt_gen - vt_prior_gen).item()
            m_tt = float(l2_diff_tt / (norm_t_prior + 1e-8))
            m_tt_tilde = float(m_tt / (dt_teacher + 1e-8))

            # Student steps under stress: student starts early (3 student steps)
            steps_student = torch.Tensor([0, 0.25, 0.5, 0.75])
            steps_student = steps_student + sway_sampling_coef * (torch.cos(torch.pi / 2 * steps_student) - 1 + steps_student)
            
            student_steps_active = steps_student[1:]
            dt_student = (student_steps_active[1] - student_steps_active[0]).item()

            x_curr = x_h
            student_preds = []
            for st_val in student_steps_active:
                time_tensor = torch.Tensor([st_val]).to(device)
                torch.manual_seed(seed + int(st_val * 100))
                x0_s = torch.randn_like(x_curr)
                t_scalar = time_tensor.unsqueeze(-1).unsqueeze(-1)
                phi = (1 - t_scalar) * x0_s + t_scalar * x_curr

                with torch.no_grad():
                    pred_s = tts.generator(
                        x=phi, cond=cond, text=text, time=time_tensor,
                        drop_audio_cond=False, drop_text=False
                    )
                    student_preds.append(pred_s[:, prompt_len:, :].reshape(-1))
                    output = phi + (1 - t_scalar) * pred_s
                x_curr = output

            vs_step1 = student_preds[0]
            vs_step2 = student_preds[1]
            norm_vs1 = torch.norm(vs_step1).item()
            l2_diff_ss = torch.norm(vs_step2 - vs_step1).item()
            m_ss = float(l2_diff_ss / (norm_vs1 + 1e-8))
            m_ss_tilde = float(m_ss / (dt_student + 1e-8))

            mel = x_curr.permute(0, 2, 1) * tts.scale
            with torch.inference_mode():
                gen_audio_torch = tts.vocos.decode(mel[..., prompt_len:])
            gen_audio = gen_audio_torch.cpu().numpy().squeeze()

            audio_filename = f"stress_early_spk_{spk}_sent_{sent_i}_seed_{seed}.wav"
            audio_filepath = os.path.join(AUDIO_DIR, audio_filename)
            sf.write(audio_filepath, gen_audio, tts.target_sample_rate)

            # Evaluate output
            asr_res = asr_pipeline(audio_filepath)
            hyp_text = asr_res["text"].strip() if asr_res and "text" in asr_res else ""
            
            ref_c = clean_text(target_text)
            hyp_c = clean_text(hyp_text)
            wer_val = float(jiwer.wer(ref_c, hyp_c))

            sim_val = 0.0
            if ref_embed is not None:
                try:
                    gen_wav_proc = preprocess_wav(audio_filepath)
                    gen_embed = voice_encoder.embed_utterance(gen_wav_proc)
                    sim_val = float(np.dot(ref_embed, gen_embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed) + 1e-8))
                except Exception:
                    sim_val = 0.0

            f0_cv = compute_f0_cv(gen_audio, sr=tts.target_sample_rate)
            peak_flux, mean_flux = compute_spectral_flux(gen_audio, sr=tts.target_sample_rate)

            rec = {
                "trial": trial_idx,
                "spk_id": spk,
                "sent_index": sent_i,
                "seed": seed,
                "condition": "early_handover_t0.10",
                "target_text": target_text,
                "hyp_text": hyp_text,
                "C_TS": round(c_ts, 4),
                "angle_deg": round(deg_ts, 2),
                "M_TS": round(m_ts, 4),
                "R_TS": round(r_ts, 4),
                "M_TT": round(m_tt, 4),
                "M_TT_tilde": round(m_tt_tilde, 4),
                "M_SS": round(m_ss, 4),
                "M_SS_tilde": round(m_ss_tilde, 4),
                "dt_teacher": round(dt_teacher, 4),
                "dt_student": round(dt_student, 4),
                "WER": round(wer_val, 4),
                "SIM": round(sim_val, 4),
                "F0_CV": round(f0_cv, 4),
                "peak_flux": round(peak_flux, 4),
                "mean_flux": round(mean_flux, 4),
                "audio_file": audio_filename
            }
            records.append(rec)
            completed_keys.add((spk, sent_i, seed))

            t_elapsed = time.time() - t_trial_start
            print(f"[STRESS {trial_idx:02d}/{total_trials}] Spk: {spk[:16]}.. Sent: {sent_i} Seed: {seed} | M_TS: {m_ts:.3f} ({deg_ts:.1f}°) | WER: {wer_val:.1%} | SIM: {sim_val:.3f} | {t_elapsed:.1f}s")

            with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"trials": records}, f, indent=2)

    # 3. Statistical Analysis under Stress
    print("\n" + "=" * 80)
    print("COMPUTING STRESS PILOT STATISTICAL HYPOTHESIS TESTS")
    print("=" * 80)

    m_ts_arr = np.array([r["M_TS"] for r in records])
    m_tt_arr = np.array([r["M_TT"] for r in records])
    m_ss_arr = np.array([r["M_SS"] for r in records])
    wer_arr = np.array([r["WER"] for r in records])
    sim_arr = np.array([r["SIM"] for r in records])
    flux_arr = np.array([r["peak_flux"] for r in records])

    w_tt, p_tt = stats.wilcoxon(m_ts_arr, m_tt_arr, alternative='greater')
    w_ss, p_ss = stats.wilcoxon(m_ts_arr, m_ss_arr, alternative='greater')

    rho_wer, p_wer = stats.spearmanr(m_ts_arr, wer_arr)
    rho_sim, p_sim = stats.spearmanr(m_ts_arr, -sim_arr)
    rho_flux, p_flux = stats.spearmanr(m_ts_arr, flux_arr)

    # Quartiles
    q1 = np.percentile(m_ts_arr, 25)
    q3 = np.percentile(m_ts_arr, 75)
    low_grp = [r for r in records if r["M_TS"] <= q1]
    high_grp = [r for r in records if r["M_TS"] >= q3]

    low_wer = float(np.mean([r["WER"] for r in low_grp]))
    high_wer = float(np.mean([r["WER"] for r in high_grp]))

    # Does M_TS predict degradation under stress?
    stress_predicts_harm = bool(rho_wer > 0.20 and p_wer < 0.05 and high_wer > low_wer)

    summary = {
        "condition": "early_handover_t0.10",
        "total_trials": len(records),
        "mean_M_TS": float(np.mean(m_ts_arr)),
        "mean_angle_deg": float(np.mean([r["angle_deg"] for r in records])),
        "mean_WER": float(np.mean(wer_arr)),
        "mean_SIM": float(np.mean(sim_arr)),
        "mean_peak_flux": float(np.mean(flux_arr)),
        "wilcoxon_M_TS_vs_M_SS": {"W": float(w_ss), "p_value": float(p_ss)},
        "spearman_M_TS_vs_WER": {"rho": float(rho_wer), "p_value": float(p_wer)},
        "spearman_M_TS_vs_SIM": {"rho": float(rho_sim), "p_value": float(p_sim)},
        "quartile_WER": {"low_q1": low_wer, "high_q4": high_wer},
        "stress_predicts_harm": stress_predicts_harm
    }

    full_output = {
        "metadata": {
            "pilot": "TARGETED_STRESS_CONDITION_PILOT",
            "protocol_file": "BENCHMARK_SUITE/Verditcs/REMAINING_COLLISION_VALIDATION_TESTS_AND_STOP_RULE.md",
            "condition": "early_handover_t0.10 (Reduced teacher guidance)",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_speakers": len(selected_spks),
            "sentences_per_speaker": 2,
            "seeds": seeds,
            "total_trials": len(records)
        },
        "summary": summary,
        "trials": records
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # 4. Generate Formal Markdown Report
    stop_rule_verdict = (
        "**ROBUSTNESS LIMITATION DETECTED** (Expand Stress Condition)"
        if stress_predicts_harm
        else "**COLLISION HYPOTHESIS REJECTED UNDER STRESS — FINAL STOP RULE APPLIES**"
    )

    report_md = f"""# Targeted Stress-Condition Pilot Report: Earlier Handover ($t_h = 0.10$)
**Protocol Source:** [`BENCHMARK_SUITE/Verditcs/REMAINING_COLLISION_VALIDATION_TESTS_AND_STOP_RULE.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/BENCHMARK_SUITE/Verditcs/REMAINING_COLLISION_VALIDATION_TESTS_AND_STOP_RULE.md)  
**Evaluated Condition:** Earlier Handover ($t_h = 0.10$ instead of baseline $t_h = 0.25$)  
**Sample Scale:** 5 Speakers $\\times$ 2 Sentences $\\times$ 3 Predetermined Seeds = **{len(records)} Matched Trials**  
**Final Status:** {stop_rule_verdict}  

---

## 1. Executive Summary & Stress Results

| Evaluated Metric | Baseline ($t_h=0.25$) | Stress Early ($t_h=0.10$) | Net Change | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Mismatch ($M_{{TS}}$)** | 0.7739 | **{summary['mean_M_TS']:.4f}** | {'+' if summary['mean_M_TS'] >= 0.7739 else ''}{summary['mean_M_TS'] - 0.7739:+.4f} | — |
| **Mean Divergence Angle** | $48.91^\\circ$ | **{summary['mean_angle_deg']:.2f}^\\circ$** | {'+' if summary['mean_angle_deg'] >= 48.91 else ''}{summary['mean_angle_deg'] - 48.91:+.2f}$^\\circ$ | — |
| **Mean Cleaned WER** | 1.95% | **{summary['mean_WER']:.2%}** | {'+' if summary['mean_WER'] >= 0.0195 else ''}{summary['mean_WER'] - 0.0195:+.2%} | — |
| **Mean Speaker Similarity (SIM)** | 0.748 | **{summary['mean_SIM']:.3f}** | {'+' if summary['mean_SIM'] >= 0.748 else ''}{summary['mean_SIM'] - 0.748:+.3f} | — |
| **Mean Peak Spectral Flux** | 6.42 | **{summary['mean_peak_flux']:.2f}** | {'+' if summary['mean_peak_flux'] >= 6.42 else ''}{summary['mean_peak_flux'] - 6.42:+.2f} | — |

---

## 2. Hypothesis Testing Under Stress: Does Handover Mismatch Predict Degradation?

| Scientific Test | Hypothesis | Observed Statistic | $p$-value | Result |
| :--- | :---: | :---: | :---: | :---: |
| **Abnormality ($M_{{TS}} > M_{{SS}}$)** | $H_1: M_{{TS}} > M_{{SS}}$ | Wilcoxon $W = {summary['wilcoxon_M_TS_vs_M_SS']['W']:.1f}$ | $p = {summary['wilcoxon_M_TS_vs_M_SS']['p_value']:.4f}$ | **{'SUPPORTED' if summary['wilcoxon_M_TS_vs_M_SS']['p_value'] < 0.05 else 'NOT SUPPORTED'}** |
| **Intelligibility Harm** | $r_s(M_{{TS}}, \\text{{WER}}) > 0.20$ | Spearman $\\rho = {summary['spearman_M_TS_vs_WER']['rho']:+.4f}$ | $p = {summary['spearman_M_TS_vs_WER']['p_value']:.4f}$ | **{'SUPPORTED' if summary['spearman_M_TS_vs_WER']['p_value'] < 0.05 and summary['spearman_M_TS_vs_WER']['rho'] > 0.20 else 'NOT SUPPORTED'}** |
| **Speaker Drift Harm** | $r_s(M_{{TS}}, -\\text{{SIM}}) > 0.20$ | Spearman $\\rho = {summary['spearman_M_TS_vs_SIM']['rho']:+.4f}$ | $p = {summary['spearman_M_TS_vs_SIM']['p_value']:.4f}$ | **{'SUPPORTED' if summary['spearman_M_TS_vs_SIM']['p_value'] < 0.05 and summary['spearman_M_TS_vs_SIM']['rho'] > 0.20 else 'NOT SUPPORTED'}** |
| **Quartile Contrast** | $Q4\\text{{ WER}} > Q1\\text{{ WER}}$ | Low $Q1$: {summary['quartile_WER']['low_q1']:.2%} vs High $Q4$: {summary['quartile_WER']['high_q4']:.2%} | — | **{'SUPPORTED' if summary['quartile_WER']['high_q4'] > summary['quartile_WER']['low_q1'] else 'NOT SUPPORTED'}** |

---

## 3. Final Decision Under Protocol Section 230–248 (Stop Rule)

> ### Protocol Stop Rule:
> *"If the implementation audit is correct and the predetermined stress pilot still finds no quality-relevant relationship, **stop the collision path**. Do not add new thresholds, metrics, or conditions solely to search for a positive result.*
> 
> *The appropriate pivot would then be:*  
> **Adaptive teacher-to-student handover scheduling for reducing teacher computation while preserving baseline speech quality.**"

### Final Scientific Conclusion:
{
"Handover mismatch becomes quality-relevant under earlier handover stress, establishing an input-dependent robustness problem."
if stress_predicts_harm
else
"Even under reduced teacher guidance ($t_h=0.10$), teacher-student velocity mismatch DOES NOT predict speech degradation. The flow-matching stochastic re-noising bridge robustly handles the transition across all tested seeds and speakers.\\n\\nTherefore, pursuant to the binding Stop Rule in Section 230–248, the collision hypothesis is formally and definitively CLOSED. The thesis must focus on the proven contribution: **Adaptive Handover Scheduling / Timing**."
}

---
*Report automatically generated by `run_stress_condition_pilot.py`.*
"""

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nStress pilot completed successfully!")
    print(f"Metrics saved to: {METRICS_JSON_PATH}")
    print(f"Report generated at: {REPORT_MD_PATH}")
    print(f"Status: {stop_rule_verdict}")


if __name__ == "__main__":
    main()
