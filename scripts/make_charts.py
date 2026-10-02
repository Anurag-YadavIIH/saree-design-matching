"""
README result charts, drawn from the reported Kaggle numbers (95 test designs). Numbers only:
no images of the data are read or drawn.

Run:  python scripts/make_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1, 2, 3, fixed order

METHODS = ["RGB colour\nhistogram", "grayscale\npHash", "DINOv2\nzero-shot RGB",
           "DINOv2\nzero-shot gray", "trained\nRGB", "trained\ngray"]

# Test Rank-1 with 95% CI over designs (README, Kaggle build).
MAIN = [(0.101, 0.069, 0.135), (0.455, 0.404, 0.507), (0.897, 0.857, 0.935),
        (0.916, 0.874, 0.952), (0.981, 0.954, 0.998), (0.989, 0.979, 0.998)]
TONAL = [(0.013, 0.002, 0.025), (0.213, 0.181, 0.248), (0.558, 0.490, 0.621),
         (0.642, 0.573, 0.705), (0.943, 0.901, 0.977), (0.952, 0.918, 0.979)]

# Stress B ROC-AUC, tiers 1, 2, 3 (results_baselines.md / results_checkpoints.md, Kaggle build).
STRESS = [(0.538, 0.4917, 0.2193), (0.7882, 0.785, 0.5297), (0.9881, 0.9704, 0.9382),
          (0.9886, 0.9737, 0.9525), (0.9998, 0.9994, 0.9991), (0.9998, 0.9994, 0.999)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def headline(out: Path):
    x = np.arange(len(METHODS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160, facecolor=SURFACE)
    style(ax)
    for vals, off, col, lab in ((MAIN, -w / 2, S1, "main queries (held-out recolour)"),
                                (TONAL, w / 2, S2, "tonal queries (lightness changes)")):
        v = np.array([t[0] for t in vals])
        err = np.array([[t[0] - t[1] for t in vals], [t[2] - t[0] for t in vals]])
        ax.bar(x + off, v, w - 0.04, color=col, label=lab, edgecolor=SURFACE, linewidth=1.5)
        ax.errorbar(x + off, v, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for i in (4, 5):   # direct labels on the two trained models only
        ax.text(x[i] - w / 2, MAIN[i][2] + 0.015, f"{MAIN[i][0]:.3f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
        ax.text(x[i] + w / 2, TONAL[i][2] + 0.015, f"{TONAL[i][0]:.3f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, METHODS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test Rank-1 (95% CI)", color=INK, fontsize=11)
    ax.set_title("Identification Rank-1 on 95 unseen test designs", color=INK, fontsize=13,
                 loc="left", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def stress(out: Path):
    x = np.arange(len(METHODS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160, facecolor=SURFACE)
    style(ax)
    labels = ["tier 1: same palette", "tier 2: same palette and craft family",
              "tier 3: same tonal palette"]
    for k, (col, lab) in enumerate(zip((S1, S2, S3), labels)):
        ax.bar(x + (k - 1) * w, [t[k] for t in STRESS], w - 0.03, color=col, label=lab,
               edgecolor=SURFACE, linewidth=1.5)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(METHODS) - 0.45, 0.515, "chance", color=MUTED, fontsize=9, ha="right")
    ax.set_xticks(x, METHODS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC, genuine vs same-palette impostors", color=INK, fontsize=11)
    ax.set_title("Stress test B: different designs in the same palette must not match",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK, ncol=1)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    d = Path("docs/img")
    d.mkdir(parents=True, exist_ok=True)
    headline(d / "headline.png")
    stress(d / "stress.png")
    print("wrote", d / "headline.png", "and", d / "stress.png")
