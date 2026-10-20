"""Generate publication-grade figures for Chapter 2 thesis deliverables:
- Task T54: Sway-Grid Plot (Ch2 §2.1.2, line 85)
- Task T55: Teacher-Guided Handover Timeline Figure (Ch2 §2.3, line 166)
"""

import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# 1. Task T54: Sway-Grid Plot (Ch2 §2.1.2)
# -----------------------------------------------------------------------------
def generate_sway_grid_plot():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # Grid definitions
    N = 16
    steps = np.arange(N + 1)
    uniform_grid = steps / N
    sway_grid = np.array([1.0 - math.cos(math.pi / 2.0 * i / N) for i in steps])
    
    # Plot curves
    ax.plot(steps, uniform_grid, label=r"Uniform Euler Grid ($t_i = i / 16$)", color="#7f8c8d", linestyle="--", linewidth=1.8, marker="o", markersize=4)
    ax.plot(steps, sway_grid, label=r"Swayed Cosine Grid ($t_i = 1 - \cos(\frac{\pi}{2} \cdot \frac{i}{16})$)", color="#2980b9", linewidth=2.4, marker="s", markersize=5)
    
    # Highlight candidate handover points: s in {4, 5, 6, 8}
    candidate_steps = [4, 5, 6, 8]
    candidate_colors = {"4": "#27ae60", "5": "#f39c12", "6": "#e67e22", "8": "#c0392b"}
    
    for s in candidate_steps:
        t_val = sway_grid[s]
        color = candidate_colors[str(s)]
        ax.scatter([s], [t_val], color=color, s=120, zorder=5, edgecolors="black", linewidth=1.5)
        
        # Add annotation box
        label_text = f"Step {s}\n$t={t_val:.4f}$\n({2*(s+1)+3} NFE)"
        offset_y = 0.05 if s != 5 else -0.09
        ax.annotate(
            label_text,
            xy=(s, t_val),
            xytext=(s, t_val + offset_y),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="#ffffff", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
        )
    
    # Region shading
    ax.axvspan(0, 8, color="#3498db", alpha=0.08, label=r"Teacher Handover Window ($s \leq 8$)")
    ax.axvspan(8, 16, color="#95a5a6", alpha=0.05)
    
    ax.set_title("DMOSpeech 2 Swayed Euler Integration Grid & Candidate Handover Points", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"Solver Step Index ($i \in \{0, \dots, 16\}$)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Normalized Diffusion Time ($t \in [0, 1]$)", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(steps)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "sway_grid_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

# -----------------------------------------------------------------------------
# 2. Task T55: Teacher-Guided Handover Timeline Figure (Ch2 §2.3)
# -----------------------------------------------------------------------------
def generate_teacher_timeline_figure():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    
    # Hide axes
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    
    # Title
    ax.text(5.5, 4.6, "Dual-Mechanism Input-Adaptive Teacher-to-Student Handover Timeline", 
            ha="center", va="center", fontsize=12, fontweight="bold", color="#2c3e50")
    
    # 1. Input Audio & Prompt SNR Gate
    rect_input = patches.FancyBboxPatch((0.4, 2.7), 2.2, 1.3, boxstyle="round,pad=0.1", 
                                        edgecolor="#2980b9", facecolor="#ebf5fb", lw=1.8)
    ax.add_patch(rect_input)
    ax.text(1.5, 3.5, "Input Reference Prompt", ha="center", va="center", fontsize=9, fontweight="bold", color="#1b4f72")
    ax.text(1.5, 3.1, r"Energy-VAD SNR Calc" "\n" r"Threshold: $\theta = 20.0$ dB", ha="center", va="center", fontsize=8, color="#2c3e50")
    
    # Decision Diamond: SNR Gate
    ax.annotate("", xy=(3.2, 3.35), xytext=(2.6, 3.35),
                arrowprops=dict(arrowstyle="->", color="#2980b9", lw=2))
    
    # 2. Teacher Phase
    rect_teacher = patches.FancyBboxPatch((3.2, 2.4), 3.2, 1.8, boxstyle="round,pad=0.1", 
                                         edgecolor="#8e44ad", facecolor="#f4ecf7", lw=1.8)
    ax.add_patch(rect_teacher)
    ax.text(4.8, 3.8, "Phase 1: Teacher Flow Matching", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#512e5f")
    teacher_desc = (
        r"Euler Steps: $s \in \{4, 5, 6, 8\}$" "\n"
        r"CFG Scale: 2.0 (2 NFE / step)" "\n"
        r"Endpoint: $\hat{x}_1(t_i) = x_{t_i} + (1-t_i)v_\theta$"
    )
    ax.text(4.8, 3.35, teacher_desc, ha="center", va="center", fontsize=7.5, color="#2c3e50")
    ax.text(4.8, 2.7, r"Tangent Drift: $d_i \leq \hat{\lambda} = 0.50$?", ha="center", va="center", fontsize=8, fontweight="bold", color="#c0392b")
    
    # Fallback arrow for SNR < 20 dB (routes directly to Step 8)
    ax.annotate("If SNR < 20 dB\n(Direct to Step 8)", xy=(4.8, 2.4), xytext=(1.5, 1.5),
                ha="center", fontsize=7.5, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5, connectionstyle="arc3,rad=-0.3"))
    
    # 3. Handover Arrow
    ax.annotate(r"Handover Triggered" "\n" r"($d_i \leq 0.50$ or Step 8)", xy=(7.1, 3.35), xytext=(6.4, 3.35),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2.5))
    
    # 4. Student Phase
    rect_student = patches.FancyBboxPatch((7.1, 2.5), 2.3, 1.7, boxstyle="round,pad=0.1", 
                                         edgecolor="#27ae60", facecolor="#eafaf1", lw=1.8)
    ax.add_patch(rect_student)
    ax.text(8.25, 3.8, "Phase 2: Student Consistency", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#196f3d")
    student_desc = (
        r"Multi-step Restart ($j = 1$)" "\n"
        r"Steps $\tau \in [0.076, 0.293, 0.617]$" "\n"
        r"Passes: exactly 3 NFE"
    )
    ax.text(8.25, 3.25, student_desc, ha="center", va="center", fontsize=7.5, color="#2c3e50")
    ax.text(8.25, 2.7, "Generates Mel Spec", ha="center", va="center", fontsize=8, fontweight="bold", color="#196f3d")
    
    # 5. Vocoder & Waveform
    ax.annotate("", xy=(9.9, 3.35), xytext=(9.4, 3.35),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
    
    rect_vocos = patches.FancyBboxPatch((9.9, 2.7), 0.9, 1.3, boxstyle="round,pad=0.1", 
                                       edgecolor="#d35400", facecolor="#fdf2e9", lw=1.8)
    ax.add_patch(rect_vocos)
    ax.text(10.35, 3.5, "Vocos", ha="center", va="center", fontsize=9, fontweight="bold", color="#a04000")
    ax.text(10.35, 3.1, "Waveform\n24 kHz", ha="center", va="center", fontsize=7.5, color="#2c3e50")
    
    # Compute Accounting Summary Box at Bottom
    rect_summary = patches.FancyBboxPatch((0.5, 0.3), 10.0, 0.85, boxstyle="round,pad=0.1", 
                                         edgecolor="#bdc3c7", facecolor="#f8f9fa", lw=1.2)
    ax.add_patch(rect_summary)
    summary_text = (
        "Total Forward NFE Formulation:  NFE(s) = 2 x (s + 1) + 3\n"
        "• Step 4: 13 NFE (38.1% savings)   • Step 5: 15 NFE (28.6% savings)   "
        "• Step 6: 17 NFE (19.0% savings)   • Step 8 Reference: 21 NFE (0.0% baseline)"
    )
    ax.text(5.5, 0.72, summary_text, ha="center", va="center", fontsize=8, fontweight="bold", color="#34495e")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "teacher_guided_timeline.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

if __name__ == "__main__":
    generate_sway_grid_plot()
    generate_teacher_timeline_figure()
