"""
Run Proposed Smooth Manifold DMOSpeech 2 Inference

Executes 6-step hybrid synthesis (3 Teacher + 3 Student) with the
C^1-Continuous Smooth Manifold Handover.
"""

import os
import sys
import time
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from proposed_system.smooth_dmospeech import SmoothDMOSpeechInference

def main():
    print("==================================================================")
    print("      PROPOSED SMOOTH MANIFOLD DMOSPEECH 2 INFERENCE ENGINE       ")
    print("==================================================================")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    
    tts = SmoothDMOSpeechInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    
    prompt_audio = os.path.join(SRC_DIR, "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "Hello! DMOSpeech 2 is successfully running on this system."
    
    start_t = time.time()
    audio = tts.generate(
        gen_text=gen_text,
        audio_path=prompt_audio,
        prompt_text=ref_text,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        student_start_step=1,
        verbose=True
    )
    elapsed = time.time() - start_t
    print(f"\n[Generation Complete] Finished in {elapsed:.2f}s")
    
    out_path = os.path.join(BASE_DIR, "output_smooth.wav")
    sf.write(out_path, audio, 24000)
    print(f"[SUCCESS] Audio successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
