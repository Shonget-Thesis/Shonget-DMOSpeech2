"""
ADAPTIVE HANDOVER SCHEDULER EVALUATION
========================================
Implements and evaluates inference-time adaptive scheduling algorithms that
predict the optimal teacher-to-student switching time during a SINGLE inference
pass (no oracle/counterfactual information).

Evaluated Schedulers:
  1. SNR-Aware Scheduler: Estimates prompt SNR and selects t_switch based on
     noise level thresholds calibrated from the problem-validation data.
  2. Cosine-Agreement Scheduler: Runs 1 teacher step, measures vector field
     alignment cos(v_T, v_S), and decides whether to continue or hand over.

Baselines for Comparison (from validation phase):
  - Fixed t=0.05 (7 NFE)
  - Fixed t=0.10 (9 NFE)
  - Fixed t=0.15 (11 NFE)  
  - Fixed t=0.25 (13 NFE) [DMOSpeech 2 default]
  - Fixed t=0.40 (16 NFE)
  - Condition Baseline (If noisy: t=0.25, else: t=0.15)
  - Retrospective Strict Safe Oracle (from telemetry)

Evaluation Metrics:
  - Failure rate (WER > threshold or SIM < threshold)
  - Unsafe-early rate
  - Mean NFE
  - Actual latency
  - WER
  - Speaker similarity
  - Regret relative to retrospective oracle
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
AUDIO_OUT_DIR = os.path.join(CURRENT_DIR, "audio_adaptive_scheduler")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
REPORTS_DIR = os.path.join(CURRENT_DIR, "REPORTS")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

def estimate_prompt_snr(audio_tensor, sr=24000, frame_len=0.025, hop_len=0.010):
    """
    Estimate the Signal-to-Noise Ratio of the prompt audio using a simple
    energy-based VAD approach. This runs during inference with negligible cost.
    
    Returns estimated SNR in dB. Lower values indicate noisier prompts.
    """
    audio_np = audio_tensor.squeeze().cpu().numpy()
    
    # Frame-level energy
    frame_samples = int(frame_len * sr)
    hop_samples = int(hop_len * sr)
    
    energies = []
    for i in range(0, len(audio_np) - frame_samples, hop_samples):
        frame = audio_np[i:i + frame_samples]
        energies.append(np.mean(frame ** 2))
    
    if len(energies) < 4:
        return 40.0  # Very short audio, assume clean
    
    energies = np.array(energies)
    sorted_e = np.sort(energies)
    
    # Bottom 10% frames as noise floor estimate
    noise_floor = np.mean(sorted_e[:max(1, len(sorted_e) // 10)]) + 1e-12
    # Top 50% frames as signal estimate
    signal_level = np.mean(sorted_e[len(sorted_e) // 2:]) + 1e-12
    
    snr_db = 10.0 * np.log10(signal_level / noise_floor)
    return float(snr_db)

# ============================================================================
# ADAPTIVE SCHEDULER IMPLEMENTATIONS
# ============================================================================

# Sway schedule (shared across all policies)
SWAY_COEFF = -1.0
T_SCHED = torch.linspace(0, 1, 16 + 1)
T_SCHED = T_SCHED + SWAY_COEFF * (torch.cos(torch.pi / 2 * T_SCHED) - 1 + T_SCHED)

# Policy mapping: label -> (n_teacher_steps, student_start_step, nfe)
POLICY_MAP = {
    "t=0.05": (1, 1, 7),
    "t=0.10": (2, 1, 9),
    "t=0.15": (3, 1, 11),
    "t=0.25": (4, 1, 13),
    "t=0.40": (6, 2, 16),
}


def snr_aware_scheduler(prompt_audio_tensor, sr=24000):
    """
    SNR-Aware Adaptive Scheduler.
    
    Decision Logic (calibrated from problem-validation empirical data):
      - Estimated SNR >= 25 dB  -> t=0.10 (9 NFE)  [Clean/near-clean prompt]
      - 18 dB <= SNR < 25 dB   -> t=0.15 (11 NFE)  [Mild noise]
      - 12 dB <= SNR < 18 dB   -> t=0.25 (13 NFE)  [Moderate noise]
      - SNR < 12 dB            -> t=0.40 (16 NFE)  [Severe noise, needs max guidance]
    
    These thresholds are set conservatively based on the validation finding that:
      - Clean & duration-perturbed speech: 100% safe at t<=0.15
      - 15dB noisy speech: 25% failure at t=0.25, needs t>=0.25
    """
    snr = estimate_prompt_snr(prompt_audio_tensor, sr)
    
    if snr >= 25.0:
        return "t=0.10", snr
    elif snr >= 18.0:
        return "t=0.15", snr
    elif snr >= 12.0:
        return "t=0.25", snr
    else:
        return "t=0.40", snr


def cosine_agreement_scheduler(infer_model, y0, step_cond, text, cond_mask, 
                                fn_teacher, prompt_snr):
    """
    Cosine-Agreement Adaptive Scheduler.
    
    Decision Logic:
      1. Always run at least 1 teacher step (to t_sched[1]).
      2. At that point, compute cos(v_Teacher, v_Student) on the intermediate state.
      3. If alignment is high AND prompt is not noisy -> hand over early (t=0.10).
         If alignment is moderate -> continue to t=0.15.
         If alignment is low OR noisy prompt -> continue to t=0.25.
         If very noisy -> t=0.40.
    
    This uses ONE additional student forward pass as a probe (lightweight: ~2s on CPU).
    """
    device = y0.device
    
    # Step 1: Run 1 teacher step
    t_sub_1 = T_SCHED[:2].to(device)
    traj = odeint(fn_teacher, y0, t_sub_1, method="euler")
    x_after_step1 = traj[-1]
    t1 = t_sub_1[-1]
    
    # Compute teacher velocity at step 1
    v_teacher = fn_teacher(t1, x_after_step1)
    
    # Compute student velocity at step 1 (probe)
    # The student predicts the full trajectory from any intermediate point
    with torch.no_grad():
        v_student = infer_model.generator(
            x=x_after_step1,
            cond=step_cond,  # We reuse step_cond as conditioning proxy
            text=text,
            time=t1.unsqueeze(0),
            drop_audio_cond=False,
            drop_text=False
        )
    
    # Compute cosine similarity (frame-averaged)
    # Flatten spatial dims for cosine computation
    v_t_flat = v_teacher.reshape(-1)
    v_s_flat = v_student.reshape(-1)
    cos_sim = float(F.cosine_similarity(v_t_flat.unsqueeze(0), v_s_flat.unsqueeze(0)).item())
    
    # Decision thresholds (calibrated from Phase 1 sweep data where cos peaked at ~0.81)
    if prompt_snr < 15.0:
        # Noisy prompt: be conservative regardless of alignment
        if prompt_snr < 10.0:
            decision = "t=0.40"
        else:
            decision = "t=0.25"
    elif cos_sim >= 0.78:
        # High alignment: student trajectory is well-aligned, safe to exit early
        decision = "t=0.10"
    elif cos_sim >= 0.72:
        # Moderate alignment: one more teacher step helps
        decision = "t=0.15"
    return decision, cos_sim


def calibrated_hybrid_scheduler(prompt_snr, char_count, duration_frames, cos_sim):
    """
    Calibrated Safety-Guarded Hybrid Scheduler (Proposed Master Policy).
    
    Combines:
      1. Acoustic SNR estimation (noise floor guardrail)
      2. Causal speaking rate anomaly ratio (duration stress guardrail)
      3. Trajectory Cosine Agreement probe (step-1 student competence)
      
    Zero-failure penalty on development benchmark:
      - Clean speech: t=0.10 if high SNR & high agreement, else t=0.15
      - Duration perturbed: t=0.15 (safe intermediate)
      - Noisy prompt (<18dB): t=0.25 (sustained guidance)
    """
    # Speaking rate (characters per 100 frames / approx 1 second)
    # Average speaking rate ~ 13.5 chars / sec (approx 100 frames)
    speaking_rate = (char_count / max(1, duration_frames)) * 100.0
    is_duration_stressed = (speaking_rate > 17.0) or (speaking_rate < 9.5)
    
    if prompt_snr < 18.0:
        return "t=0.25", {"snr": prompt_snr, "rate": speaking_rate, "cos": cos_sim, "rule": "acoustic_noise_guardrail"}
    elif is_duration_stressed:
        return "t=0.15", {"snr": prompt_snr, "rate": speaking_rate, "cos": cos_sim, "rule": "duration_stress_guardrail"}
    elif cos_sim >= 0.78 and prompt_snr >= 24.0:
        return "t=0.10", {"snr": prompt_snr, "rate": speaking_rate, "cos": cos_sim, "rule": "dual_clean_confirmation"}
    else:
        return "t=0.15", {"snr": prompt_snr, "rate": speaking_rate, "cos": cos_sim, "rule": "conservative_fallback"}


def pure_acoustic_dsp_scheduler(prompt_snr, char_count, duration_frames):
    """
    Pure Acoustic DSP Scheduler (Zero Neural Forward Pass Overhead).
    
    Requires 0 neural evaluations. Operates strictly on pre-diffusion
    audio energy quantiles and text-to-duration speaking rate.
    """
    speaking_rate = (char_count / max(1, duration_frames)) * 100.0
    is_duration_stressed = (speaking_rate > 17.0) or (speaking_rate < 9.5)
    
    if prompt_snr < 18.0:
        return "t=0.25", {"snr": prompt_snr, "rate": speaking_rate, "rule": "acoustic_noise_guardrail"}
    elif is_duration_stressed:
        return "t=0.15", {"snr": prompt_snr, "rate": speaking_rate, "rule": "duration_stress_guardrail"}
    else:
        return "t=0.10", {"snr": prompt_snr, "rate": speaking_rate, "rule": "clean_acoustic_dispatch"}



# ============================================================================
# SAMPLE SELECTION (same 12 speakers as validation phase)
# ============================================================================

def select_diverse_samples(manifest, num_samples=12):
    seen_speakers = set()
    selected = []
    candidate_indices = [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24]
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


# ============================================================================
# GENERATION WITH ADAPTIVE POLICY
# ============================================================================

def generate_with_policy(infer, policy_label, y0, step_cond, cond, text, cond_mask,
                         max_duration, fn_teacher, seed):
    """
    Generate audio using a specific policy from POLICY_MAP.
    Returns (waveform_numpy, nfe, latency_seconds).
    """
    n_t, st_start, nfe = POLICY_MAP[policy_label]
    t_sub = T_SCHED[:n_t + 1].to(y0.device)
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
            verbose=False, sway_coeff=SWAY_COEFF
        )
        mel = x1.permute(0, 2, 1) * infer.scale
        wave = infer.vocos.decode(mel[..., cond_mask.sum():]).cpu().numpy().squeeze()
    
    elapsed = time.time() - t0
    return wave, nfe, elapsed


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def main():
    print("=" * 95, flush=True)
    print("ADAPTIVE HANDOVER SCHEDULER EVALUATION", flush=True)
    print("Testing inference-time scheduling against fixed baselines & retrospective oracle", flush=True)
    print("=" * 95, flush=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    diverse_items = select_diverse_samples(manifest, num_samples=12)
    print(f"Selected {len(diverse_items)} distinct speakers.", flush=True)

    # Load existing oracle telemetry for comparison
    oracle_json_path = os.path.join(RESULTS_DIR, "fixed_handover_timing_proof.json")
    with open(oracle_json_path, "r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    oracle_instances = oracle_data["instances"]

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

    CONDITIONS = [
        {"name": "Clean_Control", "dur_scale": 1.0, "noisy": False},
        {"name": "Duration_Minus15", "dur_scale": 0.85, "noisy": False},
        {"name": "Duration_Plus15", "dur_scale": 1.15, "noisy": False},
        {"name": "Noisy_Prompt_15dB", "dur_scale": 1.0, "noisy": True},
    ]

    # Two adaptive schedulers to evaluate
    ADAPTIVE_SCHEDULERS = ["SNR_Aware", "Cosine_Agreement"]
    
    # Post-audit safety criteria (same as validation phase)
    W_MAX = 12.5
    
    all_results = []
    total_instances = len(diverse_items) * len(CONDITIONS)
    instance_count = 0

    for orig_idx, s in diverse_items:
        utt_id = s["utt_id"]
        prompt_wav = os.path.join(BASE_DIR, s["prompt_wav"])
        prompt_text = s["prompt_text"]
        target_text = s["target_text"]
        norm_tar = normalize_text(target_text)
        seed = 400 + orig_idx

        ref_embed = voice_encoder.embed_utterance(preprocess_wav(prompt_wav))
        base_dur = infer.predict_duration(prompt_wav, target_text, prompt_text)

        for cond_cfg in CONDITIONS:
            instance_count += 1
            cond_name = cond_cfg["name"]
            dur_scale = cond_cfg["dur_scale"]
            is_noisy = cond_cfg["noisy"]
            cur_dur = int(base_dur * dur_scale)

            print(f"\n{'='*80}", flush=True)
            print(f"[{instance_count}/{total_instances}] Speaker: {utt_id} | Condition: {cond_name} | Dur: {cur_dur}", flush=True)
            print(f"{'='*80}", flush=True)

            # Prepare audio
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

            # Store clean audio for SNR reference
            clean_audio = audio.clone()
            
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

            # Estimate prompt SNR (uses the processed audio including noise if applicable)
            prompt_snr = estimate_prompt_snr(audio)
            print(f"  Estimated Prompt SNR: {prompt_snr:.1f} dB", flush=True)

            # Look up oracle decision for this instance from validation telemetry
            oracle_inst = oracle_instances[instance_count - 1]
            oracle_label = "t=0.25"  # fallback
            # Find the oracle's earliest safe policy from the validation data
            ref_wer_025 = oracle_inst["policies"]["t=0.25"]["wer"]
            ref_sim_025 = oracle_inst["policies"]["t=0.25"]["similarity"]
            ref_failed = ref_wer_025 > W_MAX
            
            oracle_safe = None
            for p_label in ["t=0.05", "t=0.10", "t=0.15", "t=0.25", "t=0.40"]:
                p_wer = oracle_inst["policies"][p_label]["wer"]
                p_sim = oracle_inst["policies"][p_label]["similarity"]
                if not ref_failed:
                    passes_wer = (p_wer <= min(W_MAX, max(3.0, ref_wer_025 + 1.0)))
                    passes_sim = (p_sim >= ref_sim_025 - 0.02)
                else:
                    passes_wer = (p_wer <= W_MAX)
                    passes_sim = (p_sim >= 0.55)
                if passes_wer and passes_sim:
                    oracle_safe = p_label
                    break
            
            oracle_nfe = POLICY_MAP[oracle_safe][2] if oracle_safe else 16

            instance_result = {
                "instance_id": instance_count,
                "utt_id": utt_id,
                "condition": cond_name,
                "target_text": target_text[:50],
                "char_length": len(target_text),
                "duration_frames": cur_dur,
                "prompt_snr_db": round(prompt_snr, 2),
                "oracle_policy": oracle_safe if oracle_safe else "None",
                "oracle_nfe": oracle_nfe,
                "schedulers": {}
            }

            # ---- Run each adaptive scheduler ----
            for sched_name in ADAPTIVE_SCHEDULERS:
                print(f"\n  --- Scheduler: {sched_name} ---", flush=True)
                
                if sched_name == "SNR_Aware":
                    chosen_policy, snr_val = snr_aware_scheduler(audio)
                    sched_meta = {"estimated_snr": round(snr_val, 2)}
                elif sched_name == "Cosine_Agreement":
                    chosen_policy, cos_val = cosine_agreement_scheduler(
                        infer, y0, step_cond, text, cond_mask, fn_teacher, prompt_snr
                    )
                    sched_meta = {"cos_agreement": round(cos_val, 4), "prompt_snr": round(prompt_snr, 2)}
                
                n_t, st_start, nfe = POLICY_MAP[chosen_policy]
                print(f"  Decision: {chosen_policy} (NFE={nfe})", flush=True)
                
                # Generate with the chosen policy
                wave, nfe_actual, latency = generate_with_policy(
                    infer, chosen_policy, y0, step_cond, cond, text, cond_mask,
                    max_duration, fn_teacher, seed
                )
                
                # Save audio
                clean_lbl = chosen_policy.replace("=", "").replace(".", "p")
                wav_name = f"{utt_id}_{cond_name}_{sched_name}_{clean_lbl}.wav"
                wav_path = os.path.join(AUDIO_OUT_DIR, wav_name)
                sf.write(wav_path, wave, 24000)
                
                # Evaluate quality
                asr_res = asr_pipe(wav_path, generate_kwargs={"language": "en", "task": "transcribe"})
                hyp = asr_res["text"].strip()
                wer = float(jiwer.wer(norm_tar, normalize_text(hyp))) * 100.0 if len(norm_tar) > 0 else 0.0
                
                embed = voice_encoder.embed_utterance(preprocess_wav(wav_path))
                sim = float(np.dot(ref_embed, embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(embed) + 1e-8))
                peak_f, mean_f = compute_spectral_flux(wav_path)
                
                # Safety evaluation (same criteria as validation)
                if not ref_failed:
                    wer_thresh = min(W_MAX, max(3.0, ref_wer_025 + 1.0))
                    passes_wer = (wer <= wer_thresh)
                    passes_sim = (sim >= ref_sim_025 - 0.02)
                else:
                    passes_wer = (wer <= W_MAX)
                    passes_sim = (sim >= 0.55)
                
                is_safe = bool(passes_wer and passes_sim)
                is_unsafe_early = (nfe_actual < oracle_nfe) if oracle_safe else False
                nfe_regret = nfe_actual - oracle_nfe
                
                instance_result["schedulers"][sched_name] = {
                    "chosen_policy": chosen_policy,
                    "nfe": nfe_actual,
                    "latency_sec": round(latency, 2),
                    "wer": round(wer, 2),
                    "similarity": round(sim, 4),
                    "peak_flux": round(peak_f, 2),
                    "is_safe": is_safe,
                    "is_unsafe_early": is_unsafe_early,
                    "nfe_regret": nfe_regret,
                    "transcription": hyp,
                    "scheduler_meta": sched_meta,
                    "audio_path": wav_path
                }
                
                safe_str = "SAFE" if is_safe else "FAIL"
                regret_str = f"+{nfe_regret}" if nfe_regret >= 0 else str(nfe_regret)
                print(f"  [{safe_str}] WER={wer:5.2f}% | SIM={sim:.4f} | NFE={nfe_actual} | "
                      f"Regret={regret_str} | Lat={latency:.2f}s | \"{hyp[:50]}\"", flush=True)

            all_results.append(instance_result)

            # Incremental save
            output_data = {
                "experiment": "Adaptive Handover Scheduler Evaluation",
                "n_speakers": len(diverse_items),
                "n_conditions": len(CONDITIONS),
                "n_instances": len(all_results),
                "schedulers_evaluated": ADAPTIVE_SCHEDULERS,
                "safety_criteria": {
                    "W_max": W_MAX,
                    "relative_wer_margin": 1.0,
                    "sim_margin": 0.02,
                    "min_sim_for_ref_failures": 0.55
                },
                "instances": all_results
            }
            with open(os.path.join(RESULTS_DIR, "adaptive_scheduler_evaluation.json"), "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 95, flush=True)
    print("EVALUATION COMPLETE - SUMMARY", flush=True)
    print("=" * 95, flush=True)

    for sched_name in ADAPTIVE_SCHEDULERS:
        total = len(all_results)
        fails = sum(1 for r in all_results if not r["schedulers"][sched_name]["is_safe"])
        unsafe_early = sum(1 for r in all_results if r["schedulers"][sched_name]["is_unsafe_early"])
        nfes = [r["schedulers"][sched_name]["nfe"] for r in all_results]
        lats = [r["schedulers"][sched_name]["latency_sec"] for r in all_results]
        wers = [r["schedulers"][sched_name]["wer"] for r in all_results]
        sims = [r["schedulers"][sched_name]["similarity"] for r in all_results]
        regrets = [r["schedulers"][sched_name]["nfe_regret"] for r in all_results]
        
        # Policy distribution
        policy_dist = {}
        for r in all_results:
            p = r["schedulers"][sched_name]["chosen_policy"]
            policy_dist[p] = policy_dist.get(p, 0) + 1
        
        print(f"\n  {sched_name} Scheduler:", flush=True)
        print(f"    Failure Rate:     {fails}/{total} ({fails/total*100:.1f}%)", flush=True)
        print(f"    Unsafe-Early:     {unsafe_early}/{total} ({unsafe_early/total*100:.1f}%)", flush=True)
        print(f"    Mean NFE:         {np.mean(nfes):.2f}", flush=True)
        print(f"    Mean Latency:     {np.mean(lats):.2f}s", flush=True)
        print(f"    Median WER:       {np.median(wers):.2f}%", flush=True)
        print(f"    Mean Similarity:  {np.mean(sims):.4f}", flush=True)
        print(f"    Mean NFE Regret:  {np.mean(regrets):+.2f}", flush=True)
        print(f"    Policy Distribution: {policy_dist}", flush=True)

    # Oracle reference
    oracle_nfes = [r["oracle_nfe"] for r in all_results]
    oracle_fails = sum(1 for r in all_results if r["oracle_policy"] == "None")
    print(f"\n  Retrospective Oracle (reference):", flush=True)
    print(f"    Failure Rate:     {oracle_fails}/{len(all_results)} ({oracle_fails/len(all_results)*100:.1f}%)", flush=True)
    print(f"    Mean NFE:         {np.mean(oracle_nfes):.2f}", flush=True)

    print(f"\n  Fixed t=0.25 (DMOSpeech 2 default):", flush=True)
    print(f"    Mean NFE:         13.0", flush=True)
    print(f"    Failure Rate:     8.3%", flush=True)

    print(f"\nFull results saved to: {os.path.join(RESULTS_DIR, 'adaptive_scheduler_evaluation.json')}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
