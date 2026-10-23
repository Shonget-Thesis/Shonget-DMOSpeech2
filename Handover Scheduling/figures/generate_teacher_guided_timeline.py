"""
Figure 2.4 (task T55): teacher-guided sampling in DMOSpeech 2 as a timeline on the flow axis.

Top: teacher guided evaluations on the 16-interval sway grid up to the handover
(t_switch = 0.25 -> t_h = t_8 = 0.2929), then the student's 3 steps (restart j = 1)
on its own grid tau = {0.0761, 0.2929, 0.6173}, each starting from a re-noised input.
Bottom: running total of forward passes (18 teacher + 3 student = 21).

Colours match figures/generate_sway_grid_check.py. No data needed.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"     # teacher (guided, 2 passes)
ORANGE = "#eb6834"   # student (1 pass)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

T_SWITCH = 0.25
K = 8                                    # first grid point above t_switch
TEACHER = [1.0 - math.cos(math.pi / 2.0 * i / 16) for i in range(K + 1)]
STUDENT = [1.0 - math.cos(math.pi / 2.0 * u) for u in (0.25, 0.5, 0.75)]   # j = 1
Y_T, Y_S = 1.0, 0.0


def brace(ax, x0, x1, y, text, color=INK_2):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=color, lw=1))
    ax.text((x0 + x1) / 2, y - 0.12, text, ha="center", va="top", fontsize=9, color=color)


def timeline(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_yticks([])

    # Tracks
    ax.hlines([Y_T, Y_S], 0, 1, color=GRID, lw=1.2, zorder=1)
    ax.text(-0.015, Y_T, "Teacher\n(guided, 2 passes each)", ha="right", va="center", fontsize=9.5, color=INK)
    ax.text(-0.015, Y_S, "Student\n(1 pass each)", ha="right", va="center", fontsize=9.5, color=INK)

    # Teacher: Euler steps t_0..t_7 and the endpoint evaluation at t_8
    ax.plot(TEACHER, [Y_T] * len(TEACHER), color=BLUE, lw=2, zorder=2)
    ax.scatter(TEACHER, [Y_T] * len(TEACHER), s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for i in (0, 4, 5, 6, 7, 8):
        ax.text(TEACHER[i], Y_T + 0.13, f"t{i}", ha="center", fontsize=8, color=INK_2)
    ax.annotate("9 guided evaluations × 2 passes = 18\n(8 Euler steps + endpoint at t₈)",
                xy=(0.12, Y_T + 0.05), xytext=(0.0, Y_T + 0.55), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))

    # Switching time and handover
    ax.axvline(T_SWITCH, ymin=0.19, ymax=0.95, color=INK, lw=1, ls="--")
    ax.text(T_SWITCH + 0.008, Y_T + 0.85,
            "switching time t_switch = 0.25\n(fixed in DMOSpeech 2; chosen per utterance in this study)",
            ha="left", va="top", fontsize=8.5, color=INK)
    ax.text(TEACHER[K] + 0.012, Y_T + 0.28,
            "handover at t_h = t₈ = 0.293\n(first grid point above t_switch)\nendpoint x̂₁ = x + (1 − t₈)·v",
            ha="left", va="center", fontsize=8.5, color=INK)

    # Student: re-noise to tau, one pass, predict x_1, repeat
    ax.add_patch(FancyArrowPatch((TEACHER[K], Y_T - 0.05), (STUDENT[0], Y_S + 0.07),
                                 arrowstyle="-|>", mutation_scale=10, color=INK_2, lw=1,
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(0.305, 0.62, "re-noise with fresh noise\nto τ = 0.076 (restart j = 1)", ha="left", fontsize=8.5, color=INK_2)
    for a, b in zip(STUDENT, STUDENT[1:]):
        ax.add_patch(FancyArrowPatch((a, Y_S), (b, Y_S), arrowstyle="-|>", mutation_scale=10,
                                     color=ORANGE, lw=1.5, connectionstyle="arc3,rad=-0.35"))
        ax.text((a + b) / 2, Y_S + 0.22, "predict x̂₁, re-noise", ha="center", fontsize=8, color=INK_2)
    ax.add_patch(FancyArrowPatch((STUDENT[-1], Y_S), (1.0, Y_S), arrowstyle="-|>", mutation_scale=10,
                                 color=ORANGE, lw=1.5, connectionstyle="arc3,rad=-0.35"))
    ax.text((STUDENT[-1] + 1.0) / 2, Y_S + 0.22, "final x̂₁", ha="center", fontsize=8, color=INK_2)
    ax.scatter(STUDENT, [Y_S] * 3, s=70, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=3)
    for n, tau in enumerate(STUDENT, start=1):
        ax.text(tau, Y_S - 0.17, f"τ = {tau:.3f}\n1 pass", ha="center", va="top", fontsize=8, color=INK)
    ax.scatter([1.0], [Y_S], s=70, marker="s", color=ORANGE, zorder=3)
    ax.text(1.0, Y_S - 0.17, "mel →\nvocoder", ha="center", va="top", fontsize=8, color=INK)

    # What each part of the flow decides (Li et al., 2026)
    brace(ax, 0.0, 0.29, -0.62, "early, high-noise steps: prosody, alignment")
    brace(ax, 0.31, 1.0, -0.62, "later steps: acoustic detail")

    ax.set_xlim(-0.005, 1.03)
    ax.set_ylim(-1.0, 2.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("flow time t (0 = noise, 1 = data)", fontsize=10, color=INK)
    ax.set_title("Teacher-guided sampling at the reference switching time (released code, 16-interval sway grid)",
                 fontsize=11, color=INK, loc="left")


def pass_counter(ax):
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    cells = [("T", BLUE)] * 18 + [("S", ORANGE)] * 3
    for n, (_, c) in enumerate(cells):
        ax.add_patch(Rectangle((n, 0), 0.92, 1, color=c, lw=0))
    for n in (2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21):
        ax.text(n - 0.54, -0.35, str(n), ha="center", va="top", fontsize=8, color=INK_2)
    ax.text(9 - 0.04, 1.35, "teacher: 18 passes", ha="center", fontsize=9, color=INK)
    ax.text(19.5 - 0.04, 1.35, "student: 3", ha="center", fontsize=9, color=INK)
    ax.text(-0.6, 0.5, "running total\nof forward passes", ha="right", va="center", fontsize=9.5, color=INK)
    ax.text(21.6, 0.5, "= 21 NFE\n(Proposition 3.1)", ha="left", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(-0.2, 24.5)
    ax.set_ylim(-1.1, 1.9)


def main():
    fig, (a, b) = plt.subplots(2, 1, figsize=(13, 6.6), dpi=200,
                               gridspec_kw={"height_ratios": [4.2, 1]})
    fig.patch.set_facecolor(SURFACE)
    timeline(a)
    pass_counter(b)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.03, hspace=0.25)
    out = os.path.join(OUTPUT_DIR, "teacher_guided_timeline_t55.png")
    fig.savefig(out, facecolor=SURFACE)
    print("Saved", out)


if __name__ == "__main__":
    main()
