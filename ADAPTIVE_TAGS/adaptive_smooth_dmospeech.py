"""
Adaptive TAGS-Flow: Pure DMOSpeech 2 with Input-Adaptive Trajectory Cushioning
Option 1: Bounded Hyperbolic Tangent (tanh) Shock Ratio

Strictly contained within ADAPTIVE_TAGS directory.
Does NOT modify baseline or constant cushion implementations.
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from infer import DMOInference


class AdaptiveDMOSpeechInference(DMOInference):
    """
    DMOSpeech 2 architecture with Input-Adaptive Trajectory Cushioning.
    Dynamically senses tensor collision ratio r = ||Delta x|| / ||x_teacher||
    and computes w_adaptive = 1.0 - 0.26 * tanh(r).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_collision_ratio = None
        self.last_dynamic_w = None
        self.last_absorbed_shock = None

    def _student_sampling(self, x1, cond, text, student_start_step, verbose, sway_coeff=-1):
        """
        Student sampling with real-time Adaptive Trajectory Cushioning.
        """
        steps = torch.Tensor([0, 0.25, 0.5, 0.75])
        steps = steps + sway_coeff * (torch.cos(torch.pi / 2 * steps) - 1 + steps)
        steps = steps[student_start_step:]

        # Preserve the Teacher's anchor state
        x1_teacher = x1.clone()

        for idx, step in enumerate(steps):
            time = torch.Tensor([step]).to(x1.device)

            x0 = torch.randn_like(x1)
            t = time.unsqueeze(-1).unsqueeze(-1)
            phi = (1 - t) * x0 + t * x1

            if verbose:
                print(f"current step: {step}")
            with torch.no_grad():
                pred = self.generator(
                    x=phi,
                    cond=cond,
                    text=text,
                    time=time,
                    drop_audio_cond=False,
                    drop_text=False
                )

                output = phi + (1 - t) * pred

            # 🌟 OUR CONTRIBUTION (OPTION 1): Input-Adaptive Cushioning Manifold
            if idx == 0:
                delta = output - x1_teacher
                
                # 1. Measure live tensor Euclidean L2 norms
                norm_delta = torch.norm(delta)
                norm_teacher = torch.norm(x1_teacher) + 1e-8
                
                # 2. Compute Relative Collision Shock Ratio r
                ratio = (norm_delta / norm_teacher).item()
                
                # 3. Dynamic Bounded Hyperbolic Tangent Formulation
                # Strictly bounded in [0.74, 1.00]
                w_adaptive = float(np.clip(1.0 - 0.26 * np.tanh(ratio), 0.74, 1.00))
                absorbed_units = (1.0 - w_adaptive) * norm_delta.item()
                
                # 4. Apply Adaptive Handover Cushion
                x1 = x1_teacher + w_adaptive * delta
                
                # Cache diagnostics
                self.last_collision_ratio = ratio
                self.last_dynamic_w = w_adaptive
                self.last_absorbed_shock = absorbed_units
                
                if verbose:
                    print(f"   [Adaptive Trajectory Cushioning Applied (Option 1)]")
                    print(f"      Measured ||x_teacher|| : {norm_teacher.item():.4f}")
                    print(f"      Measured ||Delta x||   : {norm_delta.item():.4f}")
                    print(f"      Collision Ratio r      : {ratio:.4f}")
                    print(f"      Dynamic Cushion w      : {w_adaptive:.4f} (Transmits {w_adaptive*100:.1f}%)")
                    print(f"      Shock Absorbed         : {absorbed_units:.2f} units (Absorbs {(1-w_adaptive)*100:.1f}%)")
            else:
                x1 = output

        return x1


def main():
    print("==================================================================")
    print("      TESTING ADAPTIVE TAGS-FLOW (OPTION 1: BOUNDED TANH)        ")
    print("==================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    student_ckpt = os.path.join(BASE_DIR, "ckpts", "model_85000.pt")
    dp_ckpt = os.path.join(BASE_DIR, "ckpts", "model_1500.pt")
    
    tts = AdaptiveDMOSpeechInference(
        student_checkpoint_path=student_ckpt,
        duration_predictor_path=dp_ckpt,
        device=device,
        model_type="F5TTS_Base"
    )
    
    prompt_audio = os.path.join(SRC_DIR, "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "Hello! DMOSpeech 2 is successfully running on this system."
    
    generated_audio = tts.generate(
        gen_text=gen_text,
        audio_path=prompt_audio,
        prompt_text=ref_text,
        teacher_steps=8,
        teacher_stopping_time=0.07,
        student_start_step=1,
        verbose=True
    )
    
    adaptive_dir = os.path.join(BASE_DIR, "ADAPTIVE_TAGS")
    out_path = os.path.join(adaptive_dir, "output_adaptive.wav")
    sf.write(out_path, generated_audio, 24000)
    print(f"\n[SUCCESS] Adaptive Audio successfully saved to: {out_path}")
    print(f"Final Execution Metrics: Ratio r={tts.last_collision_ratio:.4f}, Dynamic w={tts.last_dynamic_w:.4f}")


if __name__ == "__main__":
    main()
