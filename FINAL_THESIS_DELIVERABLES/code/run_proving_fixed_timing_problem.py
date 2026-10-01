"""
PROVEN_PROBLEM: Validation Protocol for Proving the Fixed Handover-Timing Problem
Protocol directly defined in PROVING_FIXED_HANDOVER_TIMING_PROBLEM.md (Sections 4, 6, 8, 9, 10, 11, 12, 16)

Scope:
  Empirically tests the Null Hypothesis vs. Alternative Hypothesis:
  - Null Hypothesis: A single fixed switching time can provide the best acceptable
    quality-efficiency balance across all evaluated utterances and conditions.
  - Alternative Hypothesis: No single fixed switching time provides the best acceptable
    quality-efficiency balance across all evaluated utterances and conditions because their
    earliest safe switching times differ.

Evaluated Policies (5 Fixed):
  t_switch in {0.05, 0.10, 0.15, 0.25, 0.40}

Evaluated Conditions (4 Controlled):
  1. Clean Control
  2. Noisy Prompt (15 dB SNR)
  3. Duration Compression (-15%)
  4. Duration Expansion (+15%)

Safety Criteria (Predeclared in Section 8):
  WER(i, t) <= max(3.0%, WER(i, 0.25) + 1.0%)
  SIM(i, t) >= SIM(i, 0.25) - 0.02
"""

import os, sys, time, json, warnings, string
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

torch.set_num_threads(12)

BASE_DIR = r"c:\Users\Rhenel Jhon Sajol\Desktop\TTS\DMOSpeech 2"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import torchaudio
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin, target_rms
import jiwer
from transformers import pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
from infer import DMOInference

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_OUT_DIR = os.path.join(CURRENT_DIR, "audio_problem_proof")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
REPORTS_DIR = os.path.join(CURRENT_DIR, "REPORTS")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def add_noise_to_audio(audio_tensor, snr_db=15.0):
    signal_power = torch.mean(audio_tensor ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / (snr_linear + 1e-8)
    noise = torch.randn_like(audio_tensor) * torch.sqrt(noise_power)
    return audio_tensor + noise

def compute_spectral_flux(audio_path, sr=24000):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        diff = np.diff(S, axis=1)
        pos_diff = np.maximum(0, diff)
        flux = np.sqrt(np.mean(pos_diff**2, axis=0))
        return float(np.max(flux)), float(np.mean(flux))
    except Exception:
        return 0.0, 0.0

def select_diverse_samples(manifest, num_samples=12):
    """
    Selects utterances from distinct speakers with varied text lengths and duration.
    """
    seen_speakers = set()
    selected = []
    
    # Priority indices offering maximum speaker and length diversity
    candidate_indices = [
        0,   # len 37, spk 10119832 (short)
        2,   # len 87, spk 103675 (long)
        4,   # len 63, spk 10933823 (medium)
        6,   # len 112, spk 120405 (very long)
        8,   # len 44, spk 1205005 (short)
        10,  # len 83, spk 123125 (long)
        12,  # len 81, spk 125386 (long)
        14,  # len 58, spk 137148 (medium)
        15,  # len 57, spk 13900 (medium)
        16,  # len 52, spk 1416089 (medium)
        18,  # len 42, spk 15265 (short)
        20,  # len 43, spk 153872 (short)
        22,  # len 88, spk 155313 (long)
        24,  # len 16, spk 15735839 (very short)
    ]
    
    for idx in candidate_indices:
        if idx < len(manifest):
            item = manifest[idx]
            spk = item["prompt_wav"].split("/")[-1].split(".")[0]
            if spk not in seen_speakers:
                seen_speakers.add(spk)
                selected.append((idx, item))
                if len(selected) >= num_samples:
                    break
                    
    return selected

def main():
    print("=" * 95, flush=True)
    print("EMPIRICAL PROBLEM VALIDATION: PROVING THE FIXED HANDOVER-TIMING PROBLEM", flush=True)
    print("Strict Protocol as Defined in PROVING_FIXED_HANDOVER_TIMING_PROBLEM.md", flush=True)
    print("=" * 95, flush=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    diverse_items = select_diverse_samples(manifest, num_samples=12)
    print(f"Selected {len(diverse_items)} distinct speakers covering diverse text lengths and phonetics.", flush=True)

    print("Loading DMOSpeech 2 frozen checkpoints...", flush=True)
    infer = DMOInference(
        os.path.join(BASE_DIR, "ckpts", "model_85000.pt"),
        os.path.join(BASE_DIR, "ckpts", "model_1500.pt"),
        device="cpu"
    )

    print("Loading Whisper ASR...", flush=True)
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")

    print("Loading Resemblyzer voice encoder...", flush=True)
    voice_encoder = VoiceEncoder()

    # Define Sway Schedule (16 Euler steps for teacher)
    sway_coeff = -1.0
    t_sched = torch.linspace(0, 1, 16 + 1, device=infer.device)
    t_sched = t_sched + sway_coeff * (torch.cos(torch.pi / 2 * t_sched) - 1 + t_sched)

    # 5 Fixed Policies specified in Section 6.3
    FIXED_POLICIES = [
        {"label": "t=0.05", "n_teacher": 1, "st_start": 1, "nfe": 7},
        {"label": "t=0.10", "n_teacher": 2, "st_start": 1, "nfe": 9},
        {"label": "t=0.15", "n_teacher": 3, "st_start": 1, "nfe": 11},
        {"label": "t=0.25", "n_teacher": 4, "st_start": 1, "nfe": 13},  # DMOSpeech 2 Reference
        {"label": "t=0.40", "n_teacher": 6, "st_start": 2, "nfe": 16},
    ]

    CONDITIONS = [
        {"name": "Clean_Control", "dur_scale": 1.0, "noisy": False},
        {"name": "Duration_Minus15", "dur_scale": 0.85, "noisy": False},
        {"name": "Duration_Plus15", "dur_scale": 1.15, "noisy": False},
        {"name": "Noisy_Prompt_15dB", "dur_scale": 1.0, "noisy": True},
    ]

    all_instances = []
    summary_json_path = os.path.join(RESULTS_DIR, "fixed_handover_timing_proof.json")

    total_instances = len(diverse_items) * len(CONDITIONS)
    instance_count = 0

    for orig_idx, s in diverse_items:
        utt_id = s["utt_id"]
        prompt_wav = os.path.join(BASE_DIR, s["prompt_wav"])
        prompt_text = s["prompt_text"]
        target_text = s["target_text"]
        norm_tar = normalize_text(target_text)
        seed = 400 + orig_idx

        # Compute clean prompt reference embedding once
        ref_embed = voice_encoder.embed_utterance(preprocess_wav(prompt_wav))

        # Baseline predicted duration
        base_dur = infer.predict_duration(prompt_wav, target_text, prompt_text)

        for cond_cfg in CONDITIONS:
            instance_count += 1
            cond_name = cond_cfg["name"]
            dur_scale = cond_cfg["dur_scale"]
            is_noisy = cond_cfg["noisy"]
            cur_dur = int(base_dur * dur_scale)

            print(f"\n[{instance_count}/{total_instances}] Sample: {utt_id} | Condition: {cond_name} (Dur: {cur_dur} frames)", flush=True)

            # Preprocess audio prompt
            ref_audio, ref_text = preprocess_ref_audio_text(prompt_wav, prompt_text)
            audio, sr = torchaudio.load(ref_audio)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            if sr != infer.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, infer.target_sample_rate)
                audio = resampler(audio)

            if is_noisy:
                audio = add_noise_to_audio(audio, snr_db=15.0)

            audio = audio.to(infer.device)
            text_list = [ref_text + target_text]
            final_text_list = convert_char_to_pinyin(text_list)
            ref_audio_len = audio.shape[-1] // infer.hop_length
            total_duration = ref_audio_len + cur_dur

            with torch.inference_mode():
                cond, text, step_cond, cond_mask, max_duration, duration_tensor = infer._prepare_inputs(
                    audio, final_text_list, total_duration
                )

            torch.manual_seed(seed)
            y0 = torch.randn(1, max_duration, 100, device=infer.device, dtype=step_cond.dtype)

            def fn_teacher(t, x):
                pred = infer.teacher(
                    x=x, cond=step_cond, text=text, time=t, mask=None,
                    drop_audio_cond=False, drop_text=False
                )
                null_pred = infer.teacher(
                    x=x, cond=step_cond, text=text, time=t, mask=None,
                    drop_audio_cond=True, drop_text=True
                )
                return pred + (pred - null_pred) * 2.0

            policy_results = {}

            for pol in FIXED_POLICIES:
                pol_lbl = pol["label"]
                n_t = pol["n_teacher"]
                st_start = pol["st_start"]
                nfe = pol["nfe"]
                t_sub = t_sched[:n_t + 1]
                t_h = t_sub[-1]
                t_h_val = float(t_h.item())

                t0 = time.time()
                with torch.inference_mode():
                    traj = odeint(fn_teacher, y0, t_sub, method="euler")
                    x_h = traj[-1]
                    v_T = fn_teacher(t_h, x_h)
                    endpoint = x_h + (1.0 - t_h_val) * v_T

                    torch.manual_seed(seed + 100)
                    x1 = infer._student_sampling(
                        endpoint, cond, text, student_start_step=st_start,
                        verbose=False, sway_coeff=sway_coeff
                    )
                    mel = x1.permute(0, 2, 1) * infer.scale
                    wave = infer.vocos.decode(mel[..., cond_mask.sum():]).cpu().numpy().squeeze()

                elapsed = time.time() - t0
                clean_lbl = pol_lbl.replace("=", "").replace(".", "p")
                wav_name = f"{utt_id}_{cond_name}_{clean_lbl}.wav"
                wav_path = os.path.join(AUDIO_OUT_DIR, wav_name)
                sf.write(wav_path, wave, 24000)

                asr_res = asr_pipe(wav_path, generate_kwargs={"language": "en", "task": "transcribe"})
                hyp = asr_res["text"].strip()
                wer = float(jiwer.wer(norm_tar, normalize_text(hyp))) * 100.0 if len(norm_tar) > 0 else 0.0

                embed = voice_encoder.embed_utterance(preprocess_wav(wav_path))
                sim = float(np.dot(ref_embed, embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(embed) + 1e-8))
                peak_f, mean_f = compute_spectral_flux(wav_path)

                policy_results[pol_lbl] = {
                    "policy": pol_lbl,
                    "n_teacher": n_t,
                    "nfe": nfe,
                    "latency_sec": round(elapsed, 2),
                    "wer": round(wer, 2),
                    "similarity": round(sim, 4),
                    "peak_flux": round(peak_f, 2),
                    "mean_flux": round(mean_f, 4),
                    "transcription": hyp,
                    "audio_path": wav_path
                }
                print(f"  [{pol_lbl:6s} | NFE={nfe:2d}] Lat={elapsed:5.2f}s | WER={wer:5.2f}% | SIM={sim:.4f} | \"{hyp[:50]}\"", flush=True)

            # Determine Safety per Predetermined Criteria (Section 8)
            # WER threshold: max(3.0%, WER(0.25) + 1.0%)
            # SIM threshold: SIM(0.25) - 0.02
            ref_wer = policy_results["t=0.25"]["wer"]
            ref_sim = policy_results["t=0.25"]["similarity"]
            wer_threshold = max(3.0, ref_wer + 1.0)
            sim_threshold = ref_sim - 0.02

            earliest_safe_pol = None
            earliest_safe_nfe = None

            for pol in FIXED_POLICIES:
                pol_lbl = pol["label"]
                p_wer = policy_results[pol_lbl]["wer"]
                p_sim = policy_results[pol_lbl]["similarity"]

                passes_wer = p_wer <= wer_threshold
                passes_sim = p_sim >= sim_threshold
                is_safe = bool(passes_wer and passes_sim)

                policy_results[pol_lbl]["passes_wer"] = passes_wer
                policy_results[pol_lbl]["passes_sim"] = passes_sim
                policy_results[pol_lbl]["is_safe"] = is_safe

                if is_safe and earliest_safe_pol is None:
                    earliest_safe_pol = pol_lbl
                    earliest_safe_nfe = pol["nfe"]

            # If no policy met criteria (very rare edge case), fallback to t=0.40
            if earliest_safe_pol is None:
                earliest_safe_pol = "t=0.40"
                earliest_safe_nfe = 16

            print(f"  ==> Earliest Safe Switch (Oracle): {earliest_safe_pol} (NFE={earliest_safe_nfe}) | Ref WER={ref_wer}% | Ref SIM={ref_sim:.4f}", flush=True)

            instance_payload = {
                "instance_id": f"{utt_id}_{cond_name}",
                "utt_id": utt_id,
                "orig_idx": orig_idx,
                "condition": cond_name,
                "target_text": target_text,
                "char_length": len(target_text),
                "duration_frames": cur_dur,
                "ref_wer": ref_wer,
                "ref_sim": ref_sim,
                "wer_threshold": wer_threshold,
                "sim_threshold": round(sim_threshold, 4),
                "earliest_safe_policy": earliest_safe_pol,
                "earliest_safe_nfe": earliest_safe_nfe,
                "policies": policy_results
            }
            all_instances.append(instance_payload)

            # Incremental save to JSON
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump({"total_instances": len(all_instances), "instances": all_instances}, f, indent=2)

    # Compute Global Summary Table as defined in Section 12
    compute_and_save_final_table(all_instances, summary_json_path)

def compute_and_save_final_table(all_instances, summary_json_path):
    print("\n" + "=" * 95, flush=True)
    print("FINAL EMPIRICAL EVALUATION: THE REQUIRED TABLE (Section 12)", flush=True)
    print("=" * 95, flush=True)

    FIXED_LABELS = ["t=0.05", "t=0.10", "t=0.15", "t=0.25", "t=0.40"]
    N = len(all_instances)

    table_data = {}

    for pol_lbl in FIXED_LABELS:
        failures = 0
        unsafe_early = 0
        nfes = []
        compute_regrets = []
        latencies = []
        wers = []
        sims = []
        fluxes = []

        for inst in all_instances:
            p_res = inst["policies"][pol_lbl]
            oracle_pol = inst["earliest_safe_policy"]
            oracle_nfe = inst["earliest_safe_nfe"]

            # Failure Rate: violates at least one criterion
            if not p_res["is_safe"]:
                failures += 1

            # Unsafe Early: switches earlier than oracle safe time
            if p_res["nfe"] < oracle_nfe:
                unsafe_early += 1

            nfes.append(p_res["nfe"])
            compute_regrets.append(p_res["nfe"] - oracle_nfe)
            latencies.append(p_res["latency_sec"])
            wers.append(p_res["wer"])
            sims.append(p_res["similarity"])
            fluxes.append(p_res["peak_flux"])

        fail_rate = (failures / N) * 100.0
        unsafe_early_rate = (unsafe_early / N) * 100.0
        mean_nfe = float(np.mean(nfes))
        compute_regret = float(np.mean(compute_regrets))
        med_lat = float(np.median(latencies))
        mean_wer = float(np.mean(wers))
        mean_sim = float(np.mean(sims))
        mean_flux = float(np.mean(fluxes))

        table_data[pol_lbl] = {
            "policy": pol_lbl,
            "failure_rate": round(fail_rate, 2),
            "unsafe_early_rate": round(unsafe_early_rate, 2),
            "mean_nfe": round(mean_nfe, 1),
            "compute_regret": round(compute_regret, 2),
            "median_latency_sec": round(med_lat, 2),
            "mean_wer": round(mean_wer, 2),
            "mean_similarity": round(mean_sim, 4),
            "mean_spectral_flux": round(mean_flux, 2)
        }

    # Oracle Metrics
    oracle_nfes = [inst["earliest_safe_nfe"] for inst in all_instances]
    oracle_latencies = [inst["policies"][inst["earliest_safe_policy"]]["latency_sec"] for inst in all_instances]
    oracle_wers = [inst["policies"][inst["earliest_safe_policy"]]["wer"] for inst in all_instances]
    oracle_sims = [inst["policies"][inst["earliest_safe_policy"]]["similarity"] for inst in all_instances]
    oracle_fluxes = [inst["policies"][inst["earliest_safe_policy"]]["peak_flux"] for inst in all_instances]

    table_data["Per_Sample_Oracle"] = {
        "policy": "Per-Sample Oracle",
        "failure_rate": 0.0,
        "unsafe_early_rate": 0.0,
        "mean_nfe": round(float(np.mean(oracle_nfes)), 2),
        "compute_regret": 0.0,
        "median_latency_sec": round(float(np.median(oracle_latencies)), 2),
        "mean_wer": round(float(np.mean(oracle_wers)), 2),
        "mean_similarity": round(float(np.mean(oracle_sims)), 4),
        "mean_spectral_flux": round(float(np.mean(oracle_fluxes)), 2)
    }

    # Print Table Formatted
    print(f"\n{'Policy':<18s} | {'Fail Rate':<10s} | {'Unsafe-Early':<12s} | {'Mean NFE':<8s} | {'Regret':<8s} | {'Med Lat':<8s} | {'WER (%)':<8s} | {'SIM':<8s} | {'Flux':<6s}")
    print("-" * 105)
    for k, row in table_data.items():
        print(f"{row['policy']:<18s} | {row['failure_rate']:>8.1f}% | {row['unsafe_early_rate']:>10.1f}% | {row['mean_nfe']:>8.1f} | {row['compute_regret']:>+8.2f} | {row['median_latency_sec']:>7.2f}s | {row['mean_wer']:>7.2f}% | {row['mean_similarity']:>8.4f} | {row['mean_spectral_flux']:>6.2f}")

    final_payload = {
        "summary_table": table_data,
        "total_instances_evaluated": N,
        "instances": all_instances
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nComplete validation results saved to: {summary_json_path}", flush=True)

if __name__ == "__main__":
    main()
