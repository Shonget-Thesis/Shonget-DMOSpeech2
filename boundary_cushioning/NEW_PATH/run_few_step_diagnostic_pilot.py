"""
THESIS_EXPERIMENT/NEW_PATH: Diagnostic Pilot for Few-Step Inference Collapse in DMOSpeech 2
Protocol as specified in FEW_STEP_COLLAPSE_CRITICAL_VERDICT.md (Section 5)

Configurations Evaluated:
1. Official 4-step Sway Euler (Reference Baseline, NFE=4)
   Schedule: [0.0, 0.0761, 0.2929, 0.6173]
2. 3-step Sampler (NFE=3)
   Schedule: [0.0, 0.15, 0.50]
3. Standard 2-step Sway Euler (NFE=2)
   Schedule: [0.0, 0.2929]
4. Uniform 2-step Euler (NFE=2)
   Schedule: [0.0, 0.50]
5. Non-uniform 2-step Euler (NFE=2)
   Schedule: [0.0, 0.35]
6. 1-step Euler Jump (NFE=1)
   Schedule: [0.0]
7. 2-step Predictor-Corrector Heun (Diagnostic Higher-Order, NFE=4)
   Schedule: [0.0, 0.50] with 2 evaluations per step

Evaluates on 10 Predetermined Seed-TTS Utterances:
- Exact Whisper ASR Transcriptions & Word Error Rate (WER %)
- Resemblyzer Speaker Similarity (SIM)
- Wall-Clock Generation Latency (seconds)
- Neural Function Evaluations (NFE)
- Real-Time Factor (RTF)
"""

import os, sys, time, json, warnings, string
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F

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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_OUT_DIR = os.path.join(CURRENT_DIR, "diagnostic_pilot_audio")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(BASE_DIR, "BENCHMARK_SUITE", "data", "manifest_seedtts_100.json")

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

class FewStepDiagnosticPilot(DMOInference):
    def sample_custom_euler(self, cond, text, cond_mask, max_duration, total_duration, steps_list, seed=42):
        device = self.device
        torch.manual_seed(seed)
        
        # Prepare inputs
        x1 = torch.zeros(1, max_duration, 100, device=device)
        
        nfe_count = 0
        t0 = time.time()
        
        for idx, step_val in enumerate(steps_list):
            time_tensor = torch.Tensor([step_val]).to(device)
            t = time_tensor.unsqueeze(-1).unsqueeze(-1)
            
            torch.manual_seed(seed + idx * 100)
            x0 = torch.randn(1, max_duration, 100, device=device)
            
            # Stochastic interpolation
            phi = (1.0 - t) * x0 + t * x1
            
            with torch.no_grad():
                pred = self.generator(
                    x=phi,
                    cond=cond,
                    text=text,
                    time=time_tensor,
                    drop_audio_cond=False,
                    drop_text=False
                )
                nfe_count += 1
                output = phi + (1.0 - t) * pred
                x1 = output
                
        elapsed = time.time() - t0
        
        # Decode
        mel = x1.permute(0, 2, 1) * self.scale
        wav = self.vocos.decode(mel[..., cond_mask.sum():]).cpu().numpy().squeeze()
        return wav, elapsed, nfe_count

    def sample_heun_2step(self, cond, text, cond_mask, max_duration, total_duration, seed=42):
        device = self.device
        t0 = time.time()
        nfe_count = 0
        
        torch.manual_seed(seed)
        x0 = torch.randn(1, max_duration, 100, device=device)
        
        # Step 1: from t=0.0 to t=0.5
        # Evaluation 1 at t=0.0
        time0 = torch.Tensor([0.0]).to(device)
        with torch.no_grad():
            v0 = self.generator(x=x0, cond=cond, text=text, time=time0, drop_audio_cond=False, drop_text=False)
            nfe_count += 1
            
        # Predictor at t=0.5
        h1 = 0.5
        x_pred1 = x0 + h1 * v0
        
        # Evaluation 2 at t=0.5
        time1 = torch.Tensor([0.5]).to(device)
        with torch.no_grad():
            v1_pred = self.generator(x=x_pred1, cond=cond, text=text, time=time1, drop_audio_cond=False, drop_text=False)
            nfe_count += 1
            
        # Heun trapezoidal update for Step 1
        x_mid = x0 + (h1 / 2.0) * (v0 + v1_pred)
        
        # Step 2: from t=0.5 to t=1.0
        # Evaluation 3 at t=0.5 on x_mid
        with torch.no_grad():
            v1_corr = self.generator(x=x_mid, cond=cond, text=text, time=time1, drop_audio_cond=False, drop_text=False)
            nfe_count += 1
            
        # Predictor at t=1.0
        h2 = 0.5
        x_pred2 = x_mid + h2 * v1_corr
        
        # Evaluation 4 at t=1.0
        time2 = torch.Tensor([1.0]).to(device)
        with torch.no_grad():
            v2_pred = self.generator(x=x_pred2, cond=cond, text=text, time=time2, drop_audio_cond=False, drop_text=False)
            nfe_count += 1
            
        # Final Heun update
        x_final = x_mid + (h2 / 2.0) * (v1_corr + v2_pred)
        
        elapsed = time.time() - t0
        mel = x_final.permute(0, 2, 1) * self.scale
        wav = self.vocos.decode(mel[..., cond_mask.sum():]).cpu().numpy().squeeze()
        return wav, elapsed, nfe_count

def run_pilot(num_samples=10):
    print("=" * 85, flush=True)
    print("THESIS DIAGNOSTIC PILOT: FEW-STEP INFERENCE COLLAPSE IN DMOSPEECH 2", flush=True)
    print("Counting Exact NFE & Measured Latency across 7 Solver/Schedule Regimes", flush=True)
    print("=" * 85, flush=True)
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    samples = manifest[:num_samples]
    print(f"Loaded {len(samples)} predetermined evaluation utterances from Seed-TTS.\n", flush=True)
    
    tester = FewStepDiagnosticPilot(
        student_checkpoint_path="ckpts/model_85000.pt",
        duration_predictor_path="ckpts/model_1500.pt",
        device="cpu"
    )
    
    voice_encoder = VoiceEncoder()
    print("Loading Whisper ASR model (openai/whisper-base)...", flush=True)
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")
    
    regimes = [
        ("cfg1_4step_sway",    "4-Step Sway Euler (Official Baseline)", [0.0, 0.0761, 0.2929, 0.6173], "euler"),
        ("cfg2_3step_sampler", "3-Step Sampler",                       [0.0, 0.15, 0.50],             "euler"),
        ("cfg3_2step_sway",    "2-Step Sway Euler (Tested Collapse)",  [0.0, 0.2929],                 "euler"),
        ("cfg4_2step_uniform", "2-Step Uniform Euler",                 [0.0, 0.50],                   "euler"),
        ("cfg5_2step_nonunif", "2-Step Non-Uniform Euler (Early)",     [0.0, 0.35],                   "euler"),
        ("cfg6_1step_euler",   "1-Step Euler Jump",                    [0.0],                         "euler"),
        ("cfg7_2step_heun",    "2-Step Heun Solver (Higher-Order)",    [0.0, 0.50],                   "heun")
    ]
    
    all_utterance_results = []
    
    for i, s in enumerate(samples):
        utt_id = s["utt_id"]
        prompt_wav = os.path.join(BASE_DIR, s["prompt_wav"])
        prompt_text = s["prompt_text"]
        target_text = s["target_text"]
        current_seed = 42 + i
        
        print(f"\n[{i+1}/{len(samples)}] Utterance: {utt_id} (Seed: {current_seed})", flush=True)
        print(f"  Target Text: \"{target_text}\"", flush=True)
        
        # Prepare inputs identically
        import torchaudio
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin, target_rms
        
        duration_pred = tester.predict_duration(prompt_wav, target_text, prompt_text)
        ref_audio, ref_text = preprocess_ref_audio_text(prompt_wav, prompt_text)
        audio, sr = torchaudio.load(ref_audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != tester.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, tester.target_sample_rate)(audio)
        audio = audio.to(tester.device)
        
        final_text = convert_char_to_pinyin([ref_text + target_text])
        ref_len = audio.shape[-1] // tester.hop_length
        total_duration = ref_len + duration_pred
        
        cond, text_ids, step_cond, cond_mask, max_duration, duration_tensor = tester._prepare_inputs(
            audio, final_text, total_duration
        )
        
        ref_embed = voice_encoder.embed_utterance(preprocess_wav(prompt_wav))
        norm_target = normalize_text(target_text)
        
        utt_regime_results = {}
        
        for tag, label, sched, mode in regimes:
            if mode == "heun":
                wav, elapsed, nfe = tester.sample_heun_2step(cond, text_ids, cond_mask, max_duration, total_duration, seed=current_seed)
            else:
                wav, elapsed, nfe = tester.sample_custom_euler(cond, text_ids, cond_mask, max_duration, total_duration, sched, seed=current_seed)
                
            dur_sec = len(wav) / 24000.0
            rtf = elapsed / dur_sec
            
            # Save audio
            wav_path = os.path.join(AUDIO_OUT_DIR, f"{utt_id}_{tag}.wav")
            sf.write(wav_path, wav, 24000)
            
            # Whisper ASR
            asr_res = asr_pipe(wav_path, generate_kwargs={"language": "en", "task": "transcribe"})
            hyp_text = asr_res["text"].strip()
            norm_hyp = normalize_text(hyp_text)
            wer_val = float(jiwer.wer(norm_target, norm_hyp)) * 100.0 if len(norm_target) > 0 else 0.0
            
            # SIM
            synth_embed = voice_encoder.embed_utterance(preprocess_wav(wav_path))
            sim_score = float(np.dot(ref_embed, synth_embed) / (np.linalg.norm(ref_embed) * np.linalg.norm(synth_embed) + 1e-8))
            
            utt_regime_results[tag] = {
                "label": label,
                "nfe": nfe,
                "elapsed_sec": round(elapsed, 2),
                "rtf": round(rtf, 2),
                "wer": round(wer_val, 2),
                "sim": round(sim_score, 4),
                "transcription": hyp_text,
                "audio_path": wav_path
            }
            print(f"    [{label}] NFE={nfe} | RTF={rtf:.2f}x | WER={wer_val:.1f}% | SIM={sim_score:.4f} | \"{hyp_text}\"", flush=True)
            
        all_utterance_results.append({
            "utt_id": utt_id,
            "target_text": target_text,
            "seed": current_seed,
            "regimes": utt_regime_results
        })
        
    summary_path = os.path.join(RESULTS_DIR, "few_step_diagnostic_pilot_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_utterance_results, f, indent=2)
        
    print("\n" + "=" * 85, flush=True)
    print(f"DIAGNOSTIC PILOT COMPLETE! Full data saved to: {summary_path}", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    run_pilot(num_samples=10)
