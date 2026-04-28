"""Elsevier-format graphical abstract for the SUSQA manuscript.

Elsevier specification (2026):
- Single image summarizing the work, readable without caption
- Minimum dimensions: 531 (h) x 1328 (w) pixels  -> aspect ratio ~1:2.5 banner
- Acceptable formats: TIFF, EPS, PDF, JPG, PNG
- Sans-serif fonts; min ~11pt at final display
- White or light background; self-explanatory

This script renders at 300 DPI in both PNG (raster, for portal upload) and
PDF (vector, for use as figure in companion materials). Output:

    results/figures/graphical_abstract.png    (high-res PNG)
    results/figures/graphical_abstract.pdf    (vector)

Layout (three-panel banner, left -> right):
    [PROBLEM] -> [PROTOCOL] -> [RESULT]
    QFL papers     The four        C3 hybrid Pareto-dominates
    report         SUSQA axes      classical baselines on
    incomparable                   PneumoniaMNIST FL IID
    units

Run: python notebooks/05_graphical_abstract.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual identity matched to the Pareto plots in 03_susqa_analysis.py
COLORS = {
    "C1": "#2c7fb8",
    "C2": "#7fcdbb",
    "C3": "#d95f0e",
    "axis_p": "#1f77b4",
    "axis_c": "#2ca02c",
    "axis_q": "#9467bd",
    "axis_eta": "#d62728",
    "panel_bg": "#f7f7f7",
    "rule": "#bdbdbd",
    "ink": "#222222",
    "muted": "#666666",
    "highlight_bg": "#fff5e6",
}


def panel_problem(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_facecolor(COLORS["panel_bg"])
    ax.text(50, 94, "Problem", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLORS["ink"])

    # Three "QFL paper" cards, each reporting a different (incomparable) subset of axes
    paper_specs = [
        ("QFL paper A", "params: 8k\naccuracy: 89%",         70),
        ("QFL paper B", "comm: 12 MB\nF$_1$: 0.81",          45),
        ("QFL paper C", "qubits: 6\nlayers: 3\nshots: 1024", 20),
    ]
    for label, body, y in paper_specs:
        card = FancyBboxPatch(
            (15, y - 8), 70, 16,
            boxstyle="round,pad=0.4,rounding_size=2",
            linewidth=1.0, edgecolor=COLORS["rule"], facecolor="white",
        )
        ax.add_patch(card)
        ax.text(20, y + 4, label, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=COLORS["ink"])
        ax.text(82, y, body, ha="right", va="center",
                fontsize=9, color=COLORS["muted"], family="monospace")

    ax.text(50, 5, "Sustainability claims reported in incomparable units.",
            ha="center", va="center", fontsize=10, fontstyle="italic",
            color=COLORS["muted"])


def panel_protocol(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_facecolor(COLORS["panel_bg"])
    ax.text(50, 94, "SUSQA Protocol", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLORS["ink"])
    ax.text(50, 86, "Four axes, each tied to a real sustainability cost",
            ha="center", va="center", fontsize=9.5, color=COLORS["muted"],
            fontstyle="italic")

    # 2x2 grid of axis cards
    axes_specs = [
        # (col_x, row_y, color, symbol, name, subtitle)
        (28, 60, COLORS["axis_p"],  r"$P$",                      "Parameters",     "trainable count"),
        (72, 60, COLORS["axis_c"],  r"$C_{\mathrm{total}}$",     "Communication",  "MB per FL run"),
        (28, 25, COLORS["axis_q"],  r"$Q$",                      "Quantum cost",   "gate $\\cdot$ shots $\\cdot$ samples"),
        (72, 25, COLORS["axis_eta"], r"$\eta(p)$",               "Noise robustness", "AUC retention"),
    ]
    for cx, cy, color, sym, name, sub in axes_specs:
        card = FancyBboxPatch(
            (cx - 18, cy - 13), 36, 26,
            boxstyle="round,pad=0.5,rounding_size=2",
            linewidth=1.5, edgecolor=color, facecolor="white",
        )
        ax.add_patch(card)
        ax.text(cx, cy + 6, sym, ha="center", va="center",
                fontsize=18, fontweight="bold", color=color)
        ax.text(cx, cy - 3, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=COLORS["ink"])
        ax.text(cx, cy - 9, sub, ha="center", va="center",
                fontsize=8, color=COLORS["muted"])


def panel_result(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_facecolor(COLORS["highlight_bg"])
    ax.text(50, 94, "Result", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLORS["ink"])
    ax.text(50, 86, "PneumoniaMNIST under FL IID",
            ha="center", va="center", fontsize=9.5, color=COLORS["muted"],
            fontstyle="italic")

    # Mini Pareto plot
    inset = ax.inset_axes([0.13, 0.16, 0.74, 0.55])
    inset.set_xscale("log")
    inset.set_xlabel("Trainable parameters (log)", fontsize=9)
    inset.set_ylabel("AUC", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, which="both", alpha=0.3)

    # Three case-study points (real numbers from comparison_table.csv, PneumoniaMNIST IID)
    points = [
        ("C1", 26434, 0.916, COLORS["C1"], "o"),
        ("C2", 12714, 0.916, COLORS["C2"], "s"),
        ("C3", 3530,  0.924, COLORS["C3"], "^"),
    ]
    for label, x, y, color, marker in points:
        inset.scatter(x, y, color=color, marker=marker, s=140,
                      edgecolor="black", linewidth=0.8, zorder=3)
        # Offset annotations so labels don't overlap markers
        offset = (8, -4) if label != "C3" else (8, 6)
        inset.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                       fontsize=10, fontweight="bold", color=color)

    inset.set_xlim(2_000, 50_000)
    inset.set_ylim(0.910, 0.930)
    # Light arrow pointing toward C3 corner
    inset.annotate("", xy=(3700, 0.924), xytext=(15000, 0.916),
                   arrowprops=dict(arrowstyle="->", color=COLORS["C3"],
                                   linewidth=1.4, alpha=0.6))

    # Headline numbers
    ax.text(50, 9,
            r"$\mathbf{7.5\times}$ fewer parameters  $\cdot$  "
            r"$\mathbf{15\times}$ less communication  $\cdot$  "
            r"higher AUC",
            ha="center", va="center", fontsize=10, color=COLORS["ink"])


def main():
    # Figure size in inches matching Elsevier's 531 (h) x 1328 (w) pixel guidance
    # at 100 DPI base. We render at DPI=300 for print; the resulting PNG is
    # ~1593 (h) x 3984 (w) pixels.
    fig = plt.figure(figsize=(13.28, 5.31), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1, 14],
        width_ratios=[1, 1, 1],
        hspace=0.05, wspace=0.04,
        left=0.015, right=0.985, top=0.97, bottom=0.04,
    )

    # Title bar spanning all columns
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_xlim(0, 100); title_ax.set_ylim(0, 100); title_ax.axis("off")
    title_ax.text(
        50, 50,
        "Sustainable Quantum Federated Learning for Medical Image Analysis: "
        "A Four-Axis Accounting Protocol",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=COLORS["ink"],
    )

    # Three panels
    panel_problem(fig.add_subplot(gs[1, 0]))
    panel_protocol(fig.add_subplot(gs[1, 1]))
    panel_result(fig.add_subplot(gs[1, 2]))

    # Subtle vertical rules between panels (drawn in figure coordinates)
    fig.lines.append(plt.Line2D(
        [0.345, 0.345], [0.05, 0.92], transform=fig.transFigure,
        color=COLORS["rule"], linewidth=0.6,
    ))
    fig.lines.append(plt.Line2D(
        [0.668, 0.668], [0.05, 0.92], transform=fig.transFigure,
        color=COLORS["rule"], linewidth=0.6,
    ))

    # Arrows between panels
    arrow_kwargs = dict(
        arrowstyle="->,head_length=6,head_width=4",
        color=COLORS["muted"], linewidth=1.2,
    )
    fig.patches.append(FancyArrowPatch(
        (0.337, 0.55), (0.353, 0.55),
        transform=fig.transFigure, **arrow_kwargs,
    ))
    fig.patches.append(FancyArrowPatch(
        (0.660, 0.55), (0.676, 0.55),
        transform=fig.transFigure, **arrow_kwargs,
    ))

    out_png = FIG_DIR / "graphical_abstract.png"
    out_pdf = FIG_DIR / "graphical_abstract.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches=None, facecolor="white")
    fig.savefig(out_pdf, bbox_inches=None, facecolor="white")
    plt.close(fig)

    print(f"  wrote {out_png}")
    print(f"  wrote {out_pdf}")
    print("  resolution: ~3984 x 1593 px (well above Elsevier's 531 x 1328 minimum)")


if __name__ == "__main__":
    main()
