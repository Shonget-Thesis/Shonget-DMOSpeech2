import os
import sys
import time
import torch
import torchaudio
import soundfile as sf

# Set working directory to src or add src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from infer import DMOInference

def main():
    print("=== Testing DMOSpeech 2 Inference ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    
    print(f"Loading models:\n - Student: {student_ckpt}\n - Duration Predictor: {dp_ckpt}")
    start_load = time.time()
    tts = DMOInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    print(f"Models loaded in {time.time() - start_load:.2f}s")
    
    prompt_audio = os.path.join(SRC_DIR, "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "Hello! DMOSpeech 2 is successfully running on this system."
    
    print("\nStarting generation...")
    print(f"Prompt text: {ref_text}")
    print(f"Target text: {gen_text}")
    
    start_gen = time.time()
    # Using 4 steps (student only) or teacher-guided (8 steps)
    generated_audio = tts.generate(
        gen_text=gen_text,
        audio_path=prompt_audio,
        prompt_text=ref_text,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        student_start_step=1,
        verbose=True
    )
    gen_time = time.time() - start_gen
    print(f"Generation finished in {gen_time:.2f}s")
    
    out_path = os.path.join(BASE_DIR, "test_output.wav")
    sf.write(out_path, generated_audio, 24000)
    print(f"\n[SUCCESS] Audio successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    main()
