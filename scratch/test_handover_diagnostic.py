import os, sys, time, json, warnings, string
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
import pystoi

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

OUTPUT_DIR = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "handover_diagnostic_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_spectral_flux(audio_path, sr=24000):
    y, _ = librosa.load(audio_path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    diff = np.diff(S, axis=1)
    pos_diff = np.maximum(0, diff)
    flux = np.sqrt(np.mean(pos_diff**2, axis=0))
    return float(np.max(flux)), float(np.mean(flux))

def compute_mcd(synth_path, ref_path, sr=24000):
    try:
        y_synth, _ = librosa.load(synth_path, sr=sr)
        y_ref, _ = librosa.load(ref_path, sr=sr)
        mfcc_synth = librosa.feature.mfcc(y=y_synth, sr=sr, n_mfcc=13)[1:]
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)[1:]
        min_len = min(mfcc_synth.shape[1], mfcc_ref.shape[1])
        diff = mfcc_synth[:, :min_len] - mfcc_ref[:, :min_len]
        return float(np.mean(np.sqrt(2 * np.sum(diff**2, axis=0))))
    except Exception as e:
        return 999.0

def compute_stoi(synth_path, ref_path, sr=24000):
    try:
        y_synth, _ = librosa.load(synth_path, sr=16000)
        y_ref, _ = librosa.load(ref_path, sr=16000)
        min_len = min(len(y_synth), len(y_ref))
        if min_len < 1600:
            return 0.0
        return float(pystoi.stoi(y_ref[:min_len], y_synth[:min_len], 16000, extended=False))
    except Exception:
        return 0.0

def main():
    print("=" * 70)
    print("EMPIRICAL HANDOVER PROBLEM DIAGNOSTIC TEST")
    print("=" * 70)

    # 1. Load test sample from Seed-TTS manifest
    manifest_path = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample = manifest[0]
    prompt_wav = os.path.join(BASE_DIR, sample["prompt_wav"])
    prompt_text = sample["prompt_text"]
    target_text = sample["target_text"]
    
    # Target audio path
    target_id = sample["utt_id"].split("-")[-1]
    target_wav = os.path.join(BASE_DIR, "data", "seed-tts-eval", "en", "wavs", f"{target_id}.wav")

    print(f"Sample ID:        {sample['utt_id']}")
    print(f"Prompt Text:      {prompt_text}")
    print(f"Target Text:      {target_text}")
    print(f"Prompt Wav:       {prompt_wav}")
    print(f"Target Wav exists: {os.path.exists(target_wav)} ({target_wav})")

    # Ground truth audio stats
    gt_peak, gt_mean = compute_spectral_flux(prompt_wav)
    tgt_peak, tgt_mean = compute_spectral_flux(target_wav) if os.path.exists(target_wav) else (0, 0)
    print(f"\n[Human Ground Truth Prompt] Peak Flux: {gt_peak:.4f}, Mean Flux: {gt_mean:.4f}")
    if os.path.exists(target_wav):
        print(f"[Human Ground Truth Target] Peak Flux: {tgt_peak:.4f}, Mean Flux: {tgt_mean:.4f}")

    # 2. Initialize TTS Model
    print("\nLoading DMOSpeech 2 model...")
    t0 = time.time()
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt      = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tts = DMOInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Fix duration across all regimes to eliminate duration confounder!
    fixed_duration = tts.predict_duration(prompt_wav, target_text, prompt_text)
    print(f"Fixed Predicted Duration: {fixed_duration} frames")

    # 3. Test Vector Field Collision at Handover Points
    print("\n" + "-" * 70)
    print("PART 1: VECTOR FIELD COLLISION ANALYSIS AT TRANSITION POINT")
    print("-" * 70)

    import torchaudio
    from f5_tts.infer.utils_infer import convert_char_to_pinyin, preprocess_ref_audio_text
    ref_audio_clean, ref_text_clean = preprocess_ref_audio_text(prompt_wav, prompt_text)
    audio_t, sr = torchaudio.load(ref_audio_clean)
    if audio_t.shape[0] > 1:
        audio_t = torch.mean(audio_t, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio_t)))
    if rms < 0.3:
        audio_t = audio_t * 0.3 / rms
    if sr != tts.target_sample_rate:
        audio_t = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio_t)
    audio_t = audio_t.to(tts.device)

    text_list = [ref_text_clean + target_text]
    final_text_list = convert_char_to_pinyin(text_list)
    ref_audio_len = audio_t.shape[-1] // tts.hop_length
    total_duration = ref_audio_len + fixed_duration

    cond, text, step_cond, cond_mask, max_dur, dur_tensor = tts._prepare_inputs(
        audio_t, final_text_list, total_duration
    )

    vector_collision_results = {}
    for test_t_val in [0.07, 0.25, 0.50]:
        print(f"\n--- Testing Handover Transition at t = {test_t_val} ---")
        t_steps = 24
        t_vec = torch.linspace(0, 1, t_steps + 1, device=device, dtype=step_cond.dtype)
        t_vec = t_vec - 1.0 * (torch.cos(torch.pi / 2 * t_vec) - 1 + t_vec) # sway=-1
        t_vec = t_vec[:(t_vec > test_t_val).float().argmax() + 2][:-1]
        
        y0 = [torch.randn(dur, 100, device=device, dtype=step_cond.dtype) for dur in dur_tensor]
        from torch.nn.utils.rnn import pad_sequence
        from torchdiffeq import odeint
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        def fn(t_cur, x_cur):
            with torch.inference_mode():
                pred = tts.teacher(x=x_cur, cond=step_cond, text=text, time=t_cur, mask=None, drop_audio_cond=False, drop_text=False)
                null_pred = tts.teacher(x=x_cur, cond=step_cond, text=text, time=t_cur, mask=None, drop_audio_cond=True, drop_text=True)
                return pred + (pred - null_pred) * 2.0

        traj = odeint(fn, y0, t_vec, method="euler")
        x_at_t = traj[-1]
        
        t_tensor = t_vec[-1]
        v_teacher = fn(t_tensor, x_at_t)

        with torch.no_grad():
            v_student = tts.generator(
                x=x_at_t, cond=cond, text=text, time=t_tensor.unsqueeze(0),
                drop_audio_cond=False, drop_text=False
            )
        
        prompt_len = cond_mask.sum().item()
        vt_gen = v_teacher[:, prompt_len:, :].reshape(-1)
        vs_gen = v_student[:, prompt_len:, :].reshape(-1)

        cos_sim = F.cosine_similarity(vt_gen.unsqueeze(0), vs_gen.unsqueeze(0)).item()
        l2_diff = torch.norm(vs_gen - vt_gen).item()
        norm_t = torch.norm(vt_gen).item()
        norm_s = torch.norm(vs_gen).item()
        rel_diff = (l2_diff / (norm_t + 1e-8)) * 100.0
        angle_deg = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180.0 / np.pi

        res_v = {
            "teacher_norm": round(norm_t, 4),
            "student_norm": round(norm_s, 4),
            "cosine_similarity": round(cos_sim, 4),
            "angle_deg": round(angle_deg, 2),
            "relative_collision_pct": round(rel_diff, 2)
        }
        vector_collision_results[f"t_{test_t_val}"] = res_v

        print(f"  Teacher Vector Norm:       {norm_t:.4f}")
        print(f"  Student Vector Norm:       {norm_s:.4f}")
        print(f"  Vector Cosine Similarity:  {cos_sim:.4f}")
        print(f"  Directional Angle:         {angle_deg:.2f}°")
        print(f"  Relative Vector Collision: {rel_diff:.2f}%")

    # 4. Generate Audio Across the 4 Regimes
    print("\n" + "-" * 70)
    print("PART 2: ACOUSTIC SYNTHESIS ACROSS 4 REGIMES")
    print("-" * 70)

    regimes = [
        {
            "name": "Regime_1_Student_Only_4S",
            "desc": "Pure Student (0T + 4S, no handover)",
            "teacher_steps": 0,
            "teacher_stopping_time": 0.0,
            "student_start_step": 0,
            "mode": "standard"
        },
        {
            "name": "Regime_2_Handover_Early_5T3S",
            "desc": "Default Handover (5T + 3S, t_stop=0.07)",
            "teacher_steps": 16,
            "teacher_stopping_time": 0.07,
            "student_start_step": 1,
            "mode": "standard"
        },
        {
            "name": "Regime_3_Handover_Paper_14T2S",
            "desc": "Paper Handover (14T + 2S, t_stop=0.25)",
            "teacher_steps": 24,
            "teacher_stopping_time": 0.25,
            "student_start_step": 2,
            "mode": "standard"
        },
        {
            "name": "Regime_4_Teacher_Only_32T",
            "desc": "Pure Teacher Gold Standard (32T)",
            "teacher_steps": 32,
            "mode": "teacher_only"
        }
    ]

    results = []

    for reg in regimes:
        print(f"\nRunning {reg['desc']}...")
        t_start = time.time()
        
        if reg["mode"] == "teacher_only":
            wav = tts.generate_teacher_only(
                gen_text=target_text,
                audio_path=prompt_wav,
                prompt_text=prompt_text,
                teacher_steps=reg["teacher_steps"],
                duration=fixed_duration
            )
        else:
            wav = tts.generate(
                gen_text=target_text,
                audio_path=prompt_wav,
                prompt_text=prompt_text,
                teacher_steps=reg["teacher_steps"],
                teacher_stopping_time=reg["teacher_stopping_time"],
                student_start_step=reg["student_start_step"],
                duration=fixed_duration
            )
            
        elapsed = time.time() - t_start
        dur_sec = len(wav) / 24000.0
        rtf = elapsed / dur_sec

        # Save audio
        wav_path = os.path.join(OUTPUT_DIR, f"{reg['name']}.wav")
        sf.write(wav_path, wav, 24000)

        # Compute metrics
        peak_f, mean_f = compute_spectral_flux(wav_path)
        mcd_prompt = compute_mcd(wav_path, prompt_wav)
        mcd_target = compute_mcd(wav_path, target_wav) if os.path.exists(target_wav) else None
        stoi_target = compute_stoi(wav_path, target_wav) if os.path.exists(target_wav) else None

        res = {
            "regime": reg["name"],
            "desc": reg["desc"],
            "elapsed_sec": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "peak_flux": round(peak_f, 4),
            "mean_flux": round(mean_f, 4),
            "mcd_vs_target": round(mcd_target, 2) if mcd_target else None,
            "stoi_vs_target": round(stoi_target, 4) if stoi_target else None,
            "wav_path": wav_path
        }
        results.append(res)
        print(f"  -> RTF: {rtf:.2f}x | Peak Flux: {peak_f:.4f} | Mean Flux: {mean_f:.4f} | MCD vs Target: {mcd_target} | STOI: {stoi_target}")

    # 5. Whisper Transcription & Speaker Similarity
    print("\n" + "-" * 70)
    print("PART 3: ASR ACCURACY & SPEAKER SIMILARITY")
    print("-" * 70)
    
    print("Loading Whisper...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float32,
        device="cpu"
    )

    print("Loading Resemblyzer VoiceEncoder...")
    voice_enc = VoiceEncoder()
    prompt_wav_pre = preprocess_wav(prompt_wav)
    prompt_emb = voice_enc.embed_utterance(prompt_wav_pre)
    
    tgt_emb = None
    if os.path.exists(target_wav):
        tgt_wav_pre = preprocess_wav(target_wav)
        tgt_emb = voice_enc.embed_utterance(tgt_wav_pre)

    for res in results:
        # ASR
        trans = asr(res["wav_path"], generate_kwargs={"task": "transcribe", "language": "en"})["text"].strip()
        ref_clean = target_text.lower().translate(str.maketrans("", "", string.punctuation))
        hyp_clean = trans.lower().translate(str.maketrans("", "", string.punctuation))
        wer = jiwer.wer(ref_clean, hyp_clean) * 100.0
        
        # SIM
        synth_pre = preprocess_wav(res["wav_path"])
        synth_emb = voice_enc.embed_utterance(synth_pre)
        from scipy.spatial.distance import cosine as cosine_dist
        sim_prompt = 1.0 - cosine_dist(prompt_emb, synth_emb)
        sim_target = 1.0 - cosine_dist(tgt_emb, synth_emb) if tgt_emb is not None else None

        res["asr_text"] = trans
        res["wer"] = round(wer, 2)
        res["sim_prompt"] = round(float(sim_prompt), 4)
        res["sim_target"] = round(float(sim_target), 4) if sim_target is not None else None

        print(f"\n[{res['desc']}]")
        print(f"  Transcription: \"{trans}\" (Target: \"{target_text}\")")
        print(f"  WER: {wer:.2f}% | SIM-prompt: {sim_prompt:.4f} | SIM-target: {sim_target} | Peak Flux: {res['peak_flux']}")

    # Save summary report
    summary_path = os.path.join(OUTPUT_DIR, "handover_test_summary.json")
    final_payload = {
        "sample_info": sample,
        "ground_truth_human": {
            "prompt_peak_flux": round(gt_peak, 4),
            "prompt_mean_flux": round(gt_mean, 4),
            "target_peak_flux": round(tgt_peak, 4) if os.path.exists(target_wav) else None,
            "target_mean_flux": round(tgt_mean, 4) if os.path.exists(target_wav) else None
        },
        "vector_field_collisions": vector_collision_results,
        "regimes_evaluation": results
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"TEST COMPLETED. Results written to: {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
