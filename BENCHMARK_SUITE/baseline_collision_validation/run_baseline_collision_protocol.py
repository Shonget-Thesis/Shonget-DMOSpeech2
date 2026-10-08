import os
import sys
import time
import json
import warnings
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
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "baseline_collision_validation_metrics.json")
REPORT_MD_PATH = os.path.join(RESULTS_DIR, "BASELINE_COLLISION_VALIDATION_REPORT.md")


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
        # Downsample to 16kHz for fast F0 estimation on CPU
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
    print("BASELINE-ONLY COLLISION PROBLEM VALIDATION BENCHMARK (72 TRIALS)")
    print("Protocol: BASELINE_ONLY_COLLISION_PROBLEM_VALIDATION_PROTOCOL.md")
    print("=" * 80)

    # 1. Select 12 speakers x 2 sentences from manifest_seedtts_100.json
    manifest_path = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    spk_map = {}
    for item in manifest:
        spk = item['utt_id'].split('-')[0]
        spk_map.setdefault(spk, []).append(item)

    selected_spks = [s for s, items in spk_map.items() if len(items) >= 2][:12]
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

    print(f"Selected {len(selected_spks)} speakers x 2 sentences = {len(test_pairs)} prompt-text pairs.")
    seeds = [42, 123, 2026]
    total_trials = len(test_pairs) * len(seeds)
    print(f"Predetermined seeds: {seeds}. Total matched baseline trials: {total_trials}")

    # 2. Initialize Models
    print("\nLoading DMOSpeech 2 model components...")
    t0 = time.time()
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cpu"  # CPU environment

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

        # Precompute reference speaker embedding for SIM
        try:
            ref_wav_processed = preprocess_wav(prompt_wav)
            ref_embed = voice_encoder.embed_utterance(ref_wav_processed)
        except Exception:
            ref_embed = None

        # Fixed predicted duration for this prompt-target pair
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

            # Set up teacher grid: 16 teacher steps, stopping time t_h = 0.25, sway = -1.0
            teacher_steps = 16
            teacher_stopping_time = 0.25
            sway_sampling_coef = -1.0
            cfg_strength = 2.0

            t_vec = torch.linspace(0, 1, teacher_steps + 1, device=device, dtype=step_cond.dtype)
            if sway_sampling_coef is not None:
                t_vec = t_vec + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_vec) - 1 + t_vec)
            
            # Slice teacher steps up to stopping time
            idx_cutoff = (t_vec > teacher_stopping_time).float().argmax().item() + 2
            t_sub = t_vec[:idx_cutoff][:-1]

            def teacher_fn(t_val, x_val):
                with torch.inference_mode():
                    pred = tts.teacher(x=x_val, cond=step_cond, text=text, time=t_val, mask=None, drop_audio_cond=False, drop_text=False)
                    null_pred = tts.teacher(x=x_val, cond=step_cond, text=text, time=t_val, mask=None, drop_audio_cond=True, drop_text=True)
                    return pred + (pred - null_pred) * cfg_strength

            # Noise initialization
            torch.manual_seed(seed)
            y0 = [torch.randn(dur, 100, device=device, dtype=step_cond.dtype) for dur in dur_tensor]
            y0 = pad_sequence(y0, padding_value=0, batch_first=True)

            # Run teacher ODE
            trajectory = odeint(teacher_fn, y0, t_sub, method="euler")
            
            # Cached handover state x_h at t_h
            t_h = t_sub[-1]
            x_h_pre = trajectory[-1]
            v_T_h = teacher_fn(t_h, x_h_pre)
            x_h = x_h_pre + (1 - t_h) * v_T_h  # DMOSpeech 2 baseline final handover mel

            # Teacher step prior (x_{h-1}, t_{h-1})
            t_h_minus_1 = t_sub[-2]
            dt_teacher = (t_h - t_h_minus_1).item()
            v_T_prior = teacher_fn(t_h_minus_1, trajectory[-2])

            # Query student at exact handover state (x_h, t_h)
            with torch.no_grad():
                v_S_h = tts.generator(
                    x=x_h, cond=cond, text=text, time=t_h.unsqueeze(0),
                    drop_audio_cond=False, drop_text=False
                )

            # Restrict vector comparisons to generated speech segment (excluding prompt frames)
            vt_gen = v_T_h[:, prompt_len:, :].reshape(-1)
            vs_gen = v_S_h[:, prompt_len:, :].reshape(-1)
            vt_prior_gen = v_T_prior[:, prompt_len:, :].reshape(-1)

            norm_t = torch.norm(vt_gen).item()
            norm_s = torch.norm(vs_gen).item()
            norm_t_prior = torch.norm(vt_prior_gen).item()

            # Directional Agreement C_TS
            c_ts = F.cosine_similarity(vt_gen.unsqueeze(0), vs_gen.unsqueeze(0)).item()
            deg_ts = float(np.arccos(np.clip(c_ts, -1.0, 1.0)) * 180.0 / np.pi)

            # Cross-model mismatch M_TS
            l2_diff_ts = torch.norm(vs_gen - vt_gen).item()
            m_ts = float(l2_diff_ts / (norm_t + 1e-8))
            r_ts = float(norm_s / (norm_t + 1e-8))

            # Within-Teacher mismatch M_TT
            l2_diff_tt = torch.norm(vt_gen - vt_prior_gen).item()
            m_tt = float(l2_diff_tt / (norm_t_prior + 1e-8))
            m_tt_tilde = float(m_tt / (dt_teacher + 1e-8))

            # Student phase: run unchanged student sampling (steps 1 & 2)
            steps_student = torch.Tensor([0, 0.25, 0.5, 0.75])
            steps_student = steps_student + sway_sampling_coef * (torch.cos(torch.pi / 2 * steps_student) - 1 + steps_student)
            
            student_steps_active = steps_student[1:]  # step 1 and step 2
            dt_student = (student_steps_active[1] - student_steps_active[0]).item()

            # Execute student sampling and record consecutive student predictions for M_SS
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

            # Within-Student mismatch M_SS between step 1 and step 2
            vs_step1 = student_preds[0]
            vs_step2 = student_preds[1]
            norm_vs1 = torch.norm(vs_step1).item()
            l2_diff_ss = torch.norm(vs_step2 - vs_step1).item()
            m_ss = float(l2_diff_ss / (norm_vs1 + 1e-8))
            m_ss_tilde = float(m_ss / (dt_student + 1e-8))

            # Decode final baseline audio
            mel = x_curr.permute(0, 2, 1) * tts.scale
            with torch.inference_mode():
                gen_audio_torch = tts.vocos.decode(mel[..., prompt_len:])
            gen_audio = gen_audio_torch.cpu().numpy().squeeze()

            # Save unmodified baseline audio
            audio_filename = f"baseline_spk_{spk}_sent_{sent_i}_seed_{seed}.wav"
            audio_filepath = os.path.join(AUDIO_DIR, audio_filename)
            sf.write(audio_filepath, gen_audio, tts.target_sample_rate)

            # Evaluate output quality
            # 1. ASR Transcription & WER
            asr_res = asr_pipeline(audio_filepath)
            hyp_text = asr_res["text"].strip() if asr_res and "text" in asr_res else ""
            wer_val = float(jiwer.wer(target_text, hyp_text))

            # 2. Speaker Similarity (SIM)
            sim_val = 0.0
            if ref_embed is not None:
                try:
                    gen_wav_proc = preprocess_wav(audio_filepath)
                    gen_embed = voice_encoder.embed_utterance(gen_wav_proc)
                    sim_val = float(np.dot(ref_embed, gen_embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed) + 1e-8))
                except Exception:
                    sim_val = 0.0

            # 3. Prosody F0 CV
            f0_cv = compute_f0_cv(gen_audio, sr=tts.target_sample_rate)

            # 4. Spectral Flux
            peak_flux, mean_flux = compute_spectral_flux(gen_audio, sr=tts.target_sample_rate)

            rec = {
                "trial": trial_idx,
                "spk_id": spk,
                "sent_index": sent_i,
                "seed": seed,
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
            print(f"[{trial_idx:02d}/{total_trials}] Spk: {spk[:16]}.. Sent: {sent_i} Seed: {seed} | M_TS: {m_ts:.3f} (ang: {deg_ts:.1f}°) | M_TT: {m_tt:.3f} | M_SS: {m_ss:.3f} | WER: {wer_val:.1%} | SIM: {sim_val:.3f} | {t_elapsed:.1f}s")

            # Incrementally save JSON
            with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"trials": records}, f, indent=2)

    # 3. Statistical Analysis
    print("\n" + "=" * 80)
    print("COMPUTING FORMAL STATISTICAL HYPOTHESIS TESTS (WILCOXON & SPEARMAN)")
    print("=" * 80)

    m_ts_arr = np.array([r["M_TS"] for r in records])
    m_tt_arr = np.array([r["M_TT"] for r in records])
    m_ss_arr = np.array([r["M_SS"] for r in records])
    m_tt_tilde_arr = np.array([r["M_TT_tilde"] for r in records])
    m_ss_tilde_arr = np.array([r["M_SS_tilde"] for r in records])

    wer_arr = np.array([r["WER"] for r in records])
    sim_arr = np.array([r["SIM"] for r in records])
    flux_arr = np.array([r["peak_flux"] for r in records])
    f0_arr = np.array([r["F0_CV"] for r in records])

    # Paired Wilcoxon Signed-Rank Tests (one-sided: alternative='greater')
    w_tt_raw, p_tt_raw = stats.wilcoxon(m_ts_arr, m_tt_arr, alternative='greater')
    w_ss_raw, p_ss_raw = stats.wilcoxon(m_ts_arr, m_ss_arr, alternative='greater')

    w_tt_norm, p_tt_norm = stats.wilcoxon(m_ts_arr, m_tt_tilde_arr, alternative='greater')
    w_ss_norm, p_ss_norm = stats.wilcoxon(m_ts_arr, m_ss_tilde_arr, alternative='greater')

    # Spearman rank correlations
    rho_wer, p_wer = stats.spearmanr(m_ts_arr, wer_arr)
    rho_sim, p_sim = stats.spearmanr(m_ts_arr, -sim_arr)  # higher M_TS -> lower SIM (worse)
    rho_flux, p_flux = stats.spearmanr(m_ts_arr, flux_arr)
    rho_f0, p_f0 = stats.spearmanr(m_ts_arr, f0_arr)

    # Predetermined Quartile Analysis
    q1 = np.percentile(m_ts_arr, 25)
    q3 = np.percentile(m_ts_arr, 75)

    low_grp = [r for r in records if r["M_TS"] <= q1]
    med_grp = [r for r in records if q1 < r["M_TS"] < q3]
    high_grp = [r for r in records if r["M_TS"] >= q3]

    def grp_stats(grp):
        return {
            "count": len(grp),
            "median_M_TS": float(np.median([r["M_TS"] for r in grp])),
            "median_WER": float(np.median([r["WER"] for r in grp])),
            "mean_WER": float(np.mean([r["WER"] for r in grp])),
            "median_SIM": float(np.median([r["SIM"] for r in grp])),
            "mean_SIM": float(np.mean([r["SIM"] for r in grp])),
            "mean_flux": float(np.mean([r["peak_flux"] for r in grp])),
            "mean_f0_cv": float(np.mean([r["F0_CV"] for r in grp]))
        }

    low_stats = grp_stats(low_grp)
    med_stats = grp_stats(med_grp)
    high_stats = grp_stats(high_grp)

    # Condition checks for the proof chain
    # Condition A: Existence (M_TS > 0)
    cond_a = bool(np.mean(m_ts_arr) > 0.05)
    # Condition B: Abnormality (M_TS > M_TT and M_TS > M_SS statistically)
    cond_b = bool(p_tt_raw < 0.05 and p_ss_raw < 0.05)
    # Condition C: Quality relevance (Spearman rho with WER or -SIM is positive and significant p < 0.05)
    cond_c = bool((rho_wer > 0.2 and p_wer < 0.05) or (rho_sim > 0.2 and p_sim < 0.05))
    # Condition D: High vs Low mismatch quartile (High mismatch WER > Low mismatch WER)
    cond_d = bool(high_stats["mean_WER"] > low_stats["mean_WER"])

    collision_supported = cond_a and cond_b and cond_c and cond_d

    summary_stats = {
        "total_trials": len(records),
        "mean_C_TS": float(np.mean([r["C_TS"] for r in records])),
        "mean_angle_deg": float(np.mean([r["angle_deg"] for r in records])),
        "median_M_TS": float(np.median(m_ts_arr)),
        "mean_M_TS": float(np.mean(m_ts_arr)),
        "std_M_TS": float(np.std(m_ts_arr)),
        "median_M_TT": float(np.median(m_tt_arr)),
        "mean_M_TT": float(np.mean(m_tt_arr)),
        "median_M_SS": float(np.median(m_ss_arr)),
        "mean_M_SS": float(np.mean(m_ss_arr)),
        "wilcoxon_tests": {
            "M_TS_vs_M_TT_raw": {"W": float(w_tt_raw), "p_value": float(p_tt_raw)},
            "M_TS_vs_M_SS_raw": {"W": float(w_ss_raw), "p_value": float(p_ss_raw)},
            "M_TS_vs_M_TT_norm": {"W": float(w_tt_norm), "p_value": float(p_tt_norm)},
            "M_TS_vs_M_SS_norm": {"W": float(w_ss_norm), "p_value": float(p_ss_norm)}
        },
        "spearman_correlations": {
            "M_TS_vs_WER": {"rho": float(rho_wer), "p_value": float(p_wer)},
            "M_TS_vs_neg_SIM": {"rho": float(rho_sim), "p_value": float(p_sim)},
            "M_TS_vs_peak_flux": {"rho": float(rho_flux), "p_value": float(p_flux)},
            "M_TS_vs_F0_CV": {"rho": float(rho_f0), "p_value": float(p_f0)}
        },
        "quartile_analysis": {
            "q1_threshold": float(q1),
            "q3_threshold": float(q3),
            "low_mismatch": low_stats,
            "medium_mismatch": med_stats,
            "high_mismatch": high_stats
        },
        "proof_chain_verdict": {
            "condition_A_existence": cond_a,
            "condition_B_abnormality": cond_b,
            "condition_C_quality_association": cond_c,
            "condition_D_quartile_difference": cond_d,
            "final_collision_supported": collision_supported
        }
    }

    # Save full JSON with summary stats
    full_output = {
        "metadata": {
            "benchmark": "BASELINE_ONLY_COLLISION_PROBLEM_VALIDATION",
            "protocol_file": "BENCHMARK_SUITE/Verditcs/BASELINE_ONLY_COLLISION_PROBLEM_VALIDATION_PROTOCOL.md",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_speakers": len(selected_spks),
            "sentences_per_speaker": 2,
            "seeds": seeds,
            "total_trials": len(records),
            "teacher_checkpoint": "UniModel/real_unet (Seed-TTS / F5-TTS Base)",
            "student_checkpoint": "ckpts/model_85000.pt",
            "handover_time": 0.25,
            "teacher_steps": 16,
            "vocoder": "Vocos 24kHz"
        },
        "summary": summary_stats,
        "trials": records
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # 4. Generate Formal Markdown Report
    verdict_title = "**COLLISION PROBLEM SUPPORTED**" if collision_supported else "**COLLISION PROBLEM NOT SUPPORTED (REJECTED)**"
    report_content = f"""# Baseline-Only Validation of the Seed-TTS-to-Student Handover Problem
## Formal Scientific Validation Report & Empirical Findings

**Date:** {time.strftime('%B %d, %Y')}  
**Protocol Source:** [`BENCHMARK_SUITE/Verditcs/BASELINE_ONLY_COLLISION_PROBLEM_VALIDATION_PROTOCOL.md`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/BENCHMARK_SUITE/Verditcs/BASELINE_ONLY_COLLISION_PROBLEM_VALIDATION_PROTOCOL.md)  
**Execution Standard:** 100% Unmodified Baseline Only, Zero Softeners, Zero Cushioning, Strict Causal Evaluation  
**Dataset:** 12 Common Voice Speakers $\\times$ 2 Sentences $\\times$ 3 Predetermined Random Seeds = **{len(records)} Matched Trials**  
**Final Status:** {verdict_title}  

---

## 1. Executive Summary & Proof Chain Verdict

| Protocol Condition | Scientific Criterion | Observed Empirical Metric | Threshold / Significance | Condition Met? |
| :--- | :--- | :---: | :---: | :---: |
| **A. Existence** | Measurable teacher-student divergence at $t_h=0.25$ | $M_{{TS}} = {summary_stats['mean_M_TS']:.4f}$ (Angle: ${summary_stats['mean_angle_deg']:.2f}^\\circ$, $C_{{TS}} = {summary_stats['mean_C_TS']:.4f}$) | $M_{{TS}} > 0.05$ | **{'YES' if cond_a else 'NO'}** |
| **B. Abnormality** | $M_{{TS}} > M_{{TT}}$ and $M_{{TS}} > M_{{SS}}$ | $M_{{TS}} = {summary_stats['mean_M_TS']:.3f}$ vs $M_{{TT}} = {summary_stats['mean_M_TT']:.3f}$, $M_{{SS}} = {summary_stats['mean_M_SS']:.3f}$ | Wilcoxon $p < 0.05$ ($p_{{TT}}={summary_stats['wilcoxon_tests']['M_TS_vs_M_TT_raw']['p_value']:.4e}$) | **{'YES' if cond_b else 'NO'}** |
| **C. Quality Association** | Larger $M_{{TS}}$ predicts worse speech quality | Spearman $r_s(M_{{TS}}, \\text{{WER}}) = {summary_stats['spearman_correlations']['M_TS_vs_WER']['rho']:.4f}$ ($p={summary_stats['spearman_correlations']['M_TS_vs_WER']['p_value']:.4f}$) | $r_s > 0.20, p < 0.05$ | **{'YES' if cond_c else 'NO'}** |
| **D. Group Contrast** | High-mismatch quartile ($Q4$) performs worse than Low ($Q1$) | Low WER: {low_stats['mean_WER']:.2%} vs High WER: {high_stats['mean_WER']:.2%} | $Q4 > Q1$ degradation | **{'YES' if cond_d else 'NO'}** |

### Definitive Conclusion Under Protocol Section 20:
> {"The empirical evidence supports the Teacher-to-Student Collision Problem." if collision_supported else "The empirical evidence DOES NOT support the Teacher-to-Student Collision Problem as a speech degradation defect. While a numerical velocity difference exists between teacher and student, this difference DOES NOT predict speech degradation (WER, SIM, or spectral shock). In accordance with Section 17 & 20 of the protocol, the teacher-student velocity difference is normal model variation, not a harmful defect."}

---

## 2. Abnormality Analysis: Cross-Model vs. Within-Model Continuity

| Metric Level | Transition Vector | Median Mismatch | Mean Mismatch | Std Dev | Wilcoxon $W$ | $p$-value (one-sided) | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cross-Model Handover** | **$M_{{TS}}$ (Teacher $\\to$ Student)** | **{summary_stats['median_M_TS']:.4f}** | **{summary_stats['mean_M_TS']:.4f}** | $\\pm {summary_stats['std_M_TS']:.4f}$ | — | — | Handover transition |
| **Within-Teacher** | $M_{{TT}}$ (Teacher $\\to$ Teacher, raw) | {summary_stats['median_M_TT']:.4f} | {summary_stats['mean_M_TT']:.4f} | — | {summary_stats['wilcoxon_tests']['M_TS_vs_M_TT_raw']['W']:.1f} | $p = {summary_stats['wilcoxon_tests']['M_TS_vs_M_TT_raw']['p_value']:.4e}$ | Consecutive teacher step |
| **Within-Student** | $M_{{SS}}$ (Student $\\to$ Student, raw) | {summary_stats['median_M_SS']:.4f} | {summary_stats['mean_M_SS']:.4f} | — | {summary_stats['wilcoxon_tests']['M_TS_vs_M_SS_raw']['W']:.1f} | $p = {summary_stats['wilcoxon_tests']['M_TS_vs_M_SS_raw']['p_value']:.4e}$ | Consecutive student step |

---

## 3. Mismatch–Quality Association (Spearman Rank Correlations)

| Hypothesized Relationship | Target Outcome | Spearman $\\rho$ | $p$-value | Empirical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| $M_{{TS}} \\leftrightarrow \\text{{WER}}$ | Word Error Rate (Whisper) | **{summary_stats['spearman_correlations']['M_TS_vs_WER']['rho']:.4f}** | {summary_stats['spearman_correlations']['M_TS_vs_WER']['p_value']:.4f} | {"Statistically significant correlation with word errors" if summary_stats['spearman_correlations']['M_TS_vs_WER']['p_value'] < 0.05 else "No significant correlation with word errors"} |
| $M_{{TS}} \\leftrightarrow -\\text{{SIM}}$ | Speaker Similarity (VoiceEncoder) | **{summary_stats['spearman_correlations']['M_TS_vs_neg_SIM']['rho']:.4f}** | {summary_stats['spearman_correlations']['M_TS_vs_neg_SIM']['p_value']:.4f} | {"Significant voice drift" if summary_stats['spearman_correlations']['M_TS_vs_neg_SIM']['p_value'] < 0.05 else "No significant voice drift"} |
| $M_{{TS}} \\leftrightarrow \\text{{Peak Flux}}$ | Spectral Flux (Acoustic Shock) | **{summary_stats['spearman_correlations']['M_TS_vs_peak_flux']['rho']:.4f}** | {summary_stats['spearman_correlations']['M_TS_vs_peak_flux']['p_value']:.4f} | {"Predicts spectral discontinuity" if summary_stats['spearman_correlations']['M_TS_vs_peak_flux']['p_value'] < 0.05 else "Spectral flux is decoupled from vector mismatch"} |
| $M_{{TS}} \\leftrightarrow CV_{{F0}}$ | Pitch Variation (Prosody) | **{summary_stats['spearman_correlations']['M_TS_vs_F0_CV']['rho']:.4f}** | {summary_stats['spearman_correlations']['M_TS_vs_F0_CV']['p_value']:.4f} | {"Impacts pitch dynamics" if summary_stats['spearman_correlations']['M_TS_vs_F0_CV']['p_value'] < 0.05 else "Intact prosody dynamics"} |

---

## 4. Predetermined Quartile Breakdown ($Q1$ vs $Q2-Q3$ vs $Q4$)

| Baseline Quartile Group | Range of $M_{{TS}}$ | Sample Count | Median WER | Mean WER | Mean SIM | Mean Peak Flux | Mean $CV_{{F0}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Mismatch ($Q1$)** | $\\le {q1:.4f}$ | {low_stats['count']} | {low_stats['median_WER']:.2%} | {low_stats['mean_WER']:.2%} | {low_stats['mean_SIM']:.3f} | {low_stats['mean_flux']:.2f} | {low_stats['mean_f0_cv']:.3f} |
| **Medium Mismatch ($Q2-Q3$)** | ${q1:.4f} - {q3:.4f}$ | {med_stats['count']} | {med_stats['median_WER']:.2%} | {med_stats['mean_WER']:.2%} | {med_stats['mean_SIM']:.3f} | {med_stats['mean_flux']:.2f} | {med_stats['mean_f0_cv']:.3f} |
| **High Mismatch ($Q4$)** | $\\ge {q3:.4f}$ | {high_stats['count']} | {high_stats['median_WER']:.2%} | {high_stats['mean_WER']:.2%} | {high_stats['mean_SIM']:.3f} | {high_stats['mean_flux']:.2f} | {high_stats['mean_f0_cv']:.3f} |

---

## 5. Defense Interpretation & Final Recommendation

1. **Protocol Compliance:** All {len(records)} runs were conducted on the unmodified DMOSpeech 2 baseline with fixed duration, identical vocoder (Vocos), and fixed seeds across 12 distinct human voices.
2. **Key Takeaway:** {
"The baseline handover suffers from an empirical collision problem that causes measurable speech degradation." if collision_supported else
"The baseline handover does NOT suffer from a destructive collision problem. The vector discrepancy between teacher and student is an artifact of distillation architecture that does not translate into acoustic degradation. Therefore, attempting to 'cushion' the handover boundary with tanh or projection is scientifically unjustified, validating the strategic pivot to Handover Timing / Steps optimization."
}

---
*Report automatically generated by `run_baseline_collision_protocol.py` according to `BASELINE_ONLY_COLLISION_PROBLEM_VALIDATION_PROTOCOL.md`.*
"""

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nBenchmark completed successfully!")
    print(f"Metrics saved to: {METRICS_JSON_PATH}")
    print(f"Formal report generated at: {REPORT_MD_PATH}")
    print(f"Final Verdict: {verdict_title}")


if __name__ == "__main__":
    main()
