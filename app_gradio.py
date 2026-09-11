import os
import sys
import time
import tempfile
import torch
import soundfile as sf
import gradio as gr

# Ensure src is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from infer import DMOInference

# Initialize TTS model globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STUDENT_CKPT = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
DP_CKPT = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")

tts_engine = None

def get_tts():
    global tts_engine
    if tts_engine is None:
        print(f"Loading DMOSpeech 2 model on {DEVICE}...", flush=True)
        tts_engine = DMOInference(
            student_checkpoint_path=STUDENT_CKPT,
            duration_predictor_path=DP_CKPT,
            device=DEVICE,
            model_type="F5TTS_Base"
        )
        print("Model loaded successfully!", flush=True)
    return tts_engine

def synthesize(
    ref_audio,
    ref_text,
    gen_text,
    sampling_mode,
    temperature
):
    if not ref_audio:
        raise gr.Error("Please upload or record reference audio.")
    if not ref_text or not ref_text.strip():
        raise gr.Error("Please enter the transcript for your reference audio (what is said in the audio).")
    if not gen_text or not gen_text.strip():
        raise gr.Error("Please enter text to synthesize.")
    
    tts = get_tts()
    
    # Configure sampling mode parameters
    if "Fast Student" in sampling_mode:
        teacher_steps = 0
        teacher_stopping_time = 0.0
        student_start_step = 0
    elif "High Diversity" in sampling_mode:
        teacher_steps = 24
        teacher_stopping_time = 0.3
        student_start_step = 2
    else:  # Default Teacher-Guided (8 steps)
        teacher_steps = 16
        teacher_stopping_time = 0.07
        student_start_step = 1
        
    start_time = time.time()
    
    # Audio input from Gradio
    if isinstance(ref_audio, tuple):
        sr, audio_data = ref_audio
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_wav.name, audio_data, sr)
        audio_path = temp_wav.name
    else:
        audio_path = ref_audio
        
    generated_audio = tts.generate(
        gen_text=gen_text.strip(),
        audio_path=audio_path,
        prompt_text=ref_text.strip() if ref_text else None,
        teacher_steps=teacher_steps,
        teacher_stopping_time=teacher_stopping_time,
        student_start_step=student_start_step,
        temperature=float(temperature),
        verbose=True
    )
    
    elapsed = time.time() - start_time
    status_msg = f"Generated in {elapsed:.2f} seconds ({sampling_mode}) on {DEVICE.upper()}."
    
    return (24000, generated_audio), status_msg


def build_app():
    sample_en = os.path.join(SRC_DIR, "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
    sample_zh = os.path.join(SRC_DIR, "f5_tts", "infer", "examples", "basic", "basic_ref_zh.wav")

    with gr.Blocks(title="DMOSpeech 2 Web Studio") as demo:
        gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 12px; margin-bottom: 24px; color: white;">
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">🎙️ DMOSpeech 2 Web Studio</h1>
                <p style="color: #94a3b8; margin: 8px 0 0 0;">Reinforcement Learning Duration Prediction & Teacher-Guided Flow-Matching TTS</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Reference Voice (Prompt)")
                ref_audio_input = gr.Audio(
                    label="Reference Audio (Upload or Record)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                ref_text_input = gr.Textbox(
                    label="Reference Audio Transcript (Optional if transcribed)",
                    placeholder="Type the exact text spoken in the reference audio...",
                    lines=2
                )
                
                gr.Markdown("### 2. Synthesis Settings")
                sampling_mode_input = gr.Radio(
                    choices=[
                        "Teacher-Guided (8 steps) [Recommended]",
                        "Fast Student (4 steps)",
                        "High Diversity / Quality (16 steps)"
                    ],
                    value="Teacher-Guided (8 steps) [Recommended]",
                    label="Sampling Mode"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Duration Predictor Temperature (0 = Argmax/Deterministic)"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 3. Text to Speak")
                gen_text_input = gr.Textbox(
                    label="Target Text to Generate",
                    placeholder="Enter the sentence you want the model to generate...",
                    lines=4,
                    value="Hello! This speech is synthesized using the DMOSpeech 2 Reinforcement Learning duration pipeline."
                )
                
                generate_btn = gr.Button("🚀 Generate Speech", variant="primary", size="lg")
                
                gr.Markdown("### 4. Output Audio")
                output_audio = gr.Audio(label="Synthesized Speech", type="numpy")
                status_output = gr.Markdown("")
                
        # Example presets for fast clicking
        examples = []
        if os.path.exists(sample_en):
            examples.append([
                sample_en,
                "Some call me nature, others call me mother nature.",
                "I don't really care what you call me. I am mighty and enduring.",
                "Teacher-Guided (8 steps) [Recommended]",
                0.0
            ])
        if os.path.exists(sample_zh):
            examples.append([
                sample_zh,
                "对，这就是我，万人敬仰的太乙真人。",
                "突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂。",
                "Teacher-Guided (8 steps) [Recommended]",
                0.0
            ])
            
        if examples:
            gr.Markdown("### 💡 Click an Example to Try Immediately")
            gr.Examples(
                examples=examples,
                inputs=[ref_audio_input, ref_text_input, gen_text_input, sampling_mode_input, temperature_slider],
            )
            
        generate_btn.click(
            fn=synthesize,
            inputs=[ref_audio_input, ref_text_input, gen_text_input, sampling_mode_input, temperature_slider],
            outputs=[output_audio, status_output]
        )
        
    return demo

if __name__ == "__main__":
    app = build_app()
    print("Starting DMOSpeech 2 Web Studio on http://127.0.0.1:7860 ...", flush=True)
    app.launch(server_name="127.0.0.1", server_port=7860)
