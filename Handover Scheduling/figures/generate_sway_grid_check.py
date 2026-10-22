"""
Sway-grid figure for Chapter 2 (Figure 2.2) plus the grid check behind
"14 teacher steps at t_switch = 0.25" (Remark 3.1).

Panel (a): sway map t = 1 - cos(pi/2 * u) (s = -1) against the identity (s = 0),
           the 17 points of the released 16-interval teacher grid, t_switch = 0.25,
           and the candidate handover steps 4/5/6/8 with their NFE.
Panel (b): 16- vs 32-interval grids near t_switch = 0.25 and the teacher stop
           index under the two stopping rules (code: first t_k > s; paper: last t_k <= s).

No data needed; everything follows from the formula.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"     # categorical slot 1: sway grid
ORANGE = "#eb6834"   # categorical slot 2: stop points
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
T_SWITCH = 0.25


def sway(u):
    return 1.0 - np.cos(np.pi / 2.0 * np.asarray(u))


def grid(n):
    return [1.0 - math.cos(math.pi / 2.0 * i / n) for i in range(n + 1)]


def nfe(k):
    # Proposition 3.1: 2 * (k + 1) guided passes + 3 student passes (j = 1)
    return 2 * (k + 1) + 3


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def panel_sway_map(ax):
    style(ax)
    u = np.linspace(0, 1, 400)
    g16 = grid(16)
    ax.axhspan(0, T_SWITCH, color=BLUE, alpha=0.06, lw=0)
    ax.axhspan(T_SWITCH, 1.0, color=INK_2, alpha=0.04, lw=0)
    ax.plot(u, u, ls="--", lw=1.5, color=INK_2, label="s = 0 (uniform)")
    ax.plot(u, sway(u), lw=2, color=BLUE, label="s = -1 (sway, used)")
    ax.scatter([i / 16 for i in range(17)], g16, s=26, color=BLUE,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(T_SWITCH, color=INK, lw=1, ls=":")
    ax.text(0.99, T_SWITCH + 0.015, "t_switch = 0.25", ha="right", va="bottom", fontsize=9, color=INK)
    ax.text(0.02, 0.12, "teacher\n(t ≤ 0.25)", fontsize=9, color=INK_2, va="center")
    ax.text(0.02, 0.62, "student", fontsize=9, color=INK_2, va="center")

    label_y = {4: 0.03, 5: 0.09, 6: 0.15, 8: 0.33}
    for k, ty in label_y.items():
        x, y = k / 16, g16[k]
        ax.scatter([x], [y], s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"Step {k}: t = {y:.4f}, {nfe(k)} NFE", xy=(x, y), xytext=(0.62, ty),
                    fontsize=8.5, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    below = sum(t <= T_SWITCH for t in g16)
    ax.text(0.02, 0.78, f"{below} of 17 grid points lie at or below 0.25 (t0–t7);\n"
                        f"half the grid (u ≤ 0.5) covers only t ≤ {g16[8]:.3f}",
            fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("grid position u = i / 16", fontsize=10, color=INK)
    ax.set_ylabel("flow time t (0 = noise, 1 = data)", fontsize=10, color=INK)
    ax.set_title("(a) Sway sampling puts teacher steps at small t", fontsize=11, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=False)


def panel_grid_check(ax):
    style(ax)
    rows = [(1.0, 16, "16 intervals\n(released code)"), (0.0, 32, "32 intervals\n(paper)")]
    for y, n, name in rows:
        g = [t for t in grid(n) if t <= 0.36]
        ax.hlines(y, 0, 0.36, color=GRID, lw=1)
        ax.scatter(g, [y] * len(g), s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for i, t in enumerate(g):
            if i == 0 or (n == 16 and i >= 2) or (n == 32 and (i in (13, 15) or (i >= 4 and i % 2 == 0))):
                ax.text(t, y - 0.16, str(i), ha="center", fontsize=7.5, color=INK_2)
        ax.text(-0.012, y, name, ha="right", va="center", fontsize=9, color=INK)
    ax.axvline(T_SWITCH, color=INK, lw=1, ls=":")
    ax.text(T_SWITCH - 0.004, 1.55, "s = 0.25", ha="right", fontsize=9, color=INK)

    g16, g32 = grid(16), grid(32)
    marks = [
        (g16[8], 1.0, "code rule: first t_k > s\nk = 8, t = 0.2929 → 21 NFE", (0.300, 1.30)),
        (g32[15], 0.0, "code rule: k = 15, t = 0.2590", (0.268, -0.55)),
        (g32[14], 0.0, "paper rule: last t_k ≤ s\nk = 14, t = 0.2270 → 14 teacher steps", (0.075, 0.42)),
    ]
    for t, y, text, (tx, ty) in marks:
        ax.scatter([t], [y], s=80, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(text, xy=(t, y), xytext=(tx, ty), fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_xlim(-0.005, 0.36)
    ax.set_ylim(-0.8, 1.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("flow time t (zoomed to t ≤ 0.36)", fontsize=10, color=INK)
    ax.set_title("(b) Where '14 teacher steps at t_switch = 0.25' comes from", fontsize=11, color=INK, loc="left")


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200,
                               gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    panel_sway_map(a)
    panel_grid_check(b)
    fig.tight_layout(w_pad=6)
    out = os.path.join(OUTPUT_DIR, "sway_grid_tswitch.png")
    fig.savefig(out, facecolor=SURFACE)
    print("Saved", out)


if __name__ == "__main__":
    main()
