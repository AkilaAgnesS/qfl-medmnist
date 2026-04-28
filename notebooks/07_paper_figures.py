"""Paper-quality figures for the SUSQA manuscript.

Generates four figures suitable for inclusion in the manuscript or graphical
abstract. All output goes to results/figures/ at 300 DPI.

    fig_system_overview.png       System overview: 3 case studies + FL topology
    fig_encoding_schematic.png    Image -> CNN -> quantum circuit -> classifier
    fig_pareto_multidataset.png   3-panel Pareto, one per dataset
    fig_effect_size_heatmap.png   Cohen's d heatmap across comparisons

Run: python notebooks/07_paper_figures.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual identity
COLORS = {
    "C1": "#2c7fb8",
    "C2": "#7fcdbb",
    "C3": "#d95f0e",
    "axis_p": "#1f77b4",
    "axis_c": "#2ca02c",
    "axis_q": "#9467bd",
    "axis_eta": "#d62728",
    "ink": "#222222",
    "muted": "#666666",
    "panel_bg": "#f7f7f7",
    "rule": "#bdbdbd",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, label, color, fontsize=10, fontweight="normal", alpha=0.85):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=2",
        linewidth=1.4, edgecolor=color, facecolor=color, alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="white")


def _outline_box(ax, x, y, w, h, label, color, fontsize=10, fontweight="normal"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=2",
        linewidth=1.4, edgecolor=color, facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=color)


def _arrow(ax, x1, y1, x2, y2, color="#666666", style="->", linewidth=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"{style},head_length=5,head_width=4",
        color=color, linewidth=linewidth, mutation_scale=10,
    ))


# ---------------------------------------------------------------------------
# Figure 1: System overview
# ---------------------------------------------------------------------------

def figure_system_overview():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # Title
    ax.text(50, 95, "SUSQA System Overview: Three Case Studies under Federated Averaging",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLORS["ink"])

    # ---- Top row: three case-study architectures ----
    ax.text(50, 86, "Three contrasting case studies",
            ha="center", va="center",
            fontsize=10.5, color=COLORS["muted"], fontstyle="italic")

    # C1: classical CNN
    _outline_box(ax, 5, 60, 26, 18,
                 "C1: Classical CNN\nP = 26,434 (binary)\nP = 26,743 (Derma)\nQ = 0",
                 COLORS["C1"], fontsize=9.5, fontweight="bold")

    # C2: classical compressed
    _outline_box(ax, 37, 60, 26, 18,
                 "C2: Classical compressed (MLP)\nP = 12,714 (binary)\nP = 37,847 (Derma)\nQ = 0",
                 COLORS["C2"], fontsize=9.5, fontweight="bold")

    # C3: hybrid quantum
    _outline_box(ax, 69, 60, 26, 18,
                 "C3: Hybrid CNN-VQC\nP = 3,530 (binary)\nP = 3,647 (Derma)\nQ ~ 10^9 gate-shots",
                 COLORS["C3"], fontsize=9.5, fontweight="bold")

    # ---- Middle: FedAvg server ----
    _box(ax, 35, 36, 30, 12, "Server (FedAvg aggregation)",
         "#444444", fontsize=11, fontweight="bold", alpha=0.85)

    # Arrows from each case-study to server
    _arrow(ax, 18, 60, 40, 48, color=COLORS["C1"])
    _arrow(ax, 50, 60, 50, 48, color=COLORS["C2"])
    _arrow(ax, 82, 60, 60, 48, color=COLORS["C3"])

    # ---- Bottom row: 5 hospital silos ----
    ax.text(50, 28, "K = 5 hospital silos (IID, Dirichlet 0.5, or Dirichlet 0.1 partitioning)",
            ha="center", va="center",
            fontsize=10, color=COLORS["muted"], fontstyle="italic")

    silo_w, silo_h = 13, 10
    silo_y = 14
    silo_xs = [4, 22, 40, 58, 76]
    for i, x in enumerate(silo_xs):
        _outline_box(ax, x, silo_y, silo_w, silo_h,
                     f"Silo {i + 1}\nlocal data D_{i + 1}",
                     "#444444", fontsize=8.5)
        # Arrows up to server
        _arrow(ax, x + silo_w / 2, silo_y + silo_h, 50, 36,
               color="#888888", linewidth=0.8)

    # ---- SUSQA axes annotation on right ----
    ax.text(95.5, 70, "Reported on every run",
            ha="right", va="center",
            fontsize=9, color=COLORS["muted"], fontstyle="italic")
    axis_y = 65
    for label, color in [
        (f"P  trainable parameters", COLORS["axis_p"]),
        (f"C_total  comm. (bytes)", COLORS["axis_c"]),
        (f"Q  gate-shot cost", COLORS["axis_q"]),
        (f"eta(p)  noise robustness", COLORS["axis_eta"]),
    ]:
        ax.text(95.5, axis_y, label, ha="right", va="center",
                fontsize=8.5, color=color, fontweight="bold")
        axis_y -= 4

    fig.tight_layout()
    out = FIG_DIR / "fig_system_overview.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: Encoding schematic for C3 hybrid
# ---------------------------------------------------------------------------

def figure_encoding_schematic():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(50, 92, "C3 Hybrid Architecture: Image -> CNN -> Bottleneck -> VQC -> Classifier",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=COLORS["ink"])

    # Stage 1: input image
    _outline_box(ax, 2, 38, 10, 22,
                 "Input\nimage\n28x28xC", COLORS["ink"], fontsize=8.5)

    # Stage 2: CNN
    _outline_box(ax, 17, 38, 14, 22,
                 "CNN\nfeature\nextractor", COLORS["C3"], fontsize=9, fontweight="bold")

    # Stage 3: bottleneck
    _outline_box(ax, 36, 38, 12, 22,
                 "Linear\nbottleneck\n(8-dim)", COLORS["axis_p"], fontsize=9)

    # Stage 4: angle encoding
    _outline_box(ax, 53, 38, 13, 22,
                 "Angle\nencoding\nRY(theta)", COLORS["axis_q"], fontsize=9)

    # Stage 5: variational layers
    _outline_box(ax, 71, 38, 14, 22,
                 "VQC\nL=2 layers\nRY,RZ + CNOT", COLORS["axis_q"], fontsize=9, fontweight="bold")

    # Stage 6: readout
    _outline_box(ax, 90, 38, 8, 22,
                 "<Z_i>\nclass\nlogits", COLORS["axis_eta"], fontsize=9)

    # Arrows
    for x1, x2 in [(12, 17), (31, 36), (48, 53), (66, 71), (85, 90)]:
        _arrow(ax, x1, 49, x2, 49, color=COLORS["muted"])

    # Subtitle
    ax.text(50, 23, "Trainable parameters: bottleneck + variational angles + readout",
            ha="center", va="center",
            fontsize=9.5, color=COLORS["muted"])
    ax.text(50, 17, "P = 3530 (binary, n_qubits = 8, L = 2)   |   G = n + L*3n = 56 gates per forward pass",
            ha="center", va="center",
            fontsize=9.5, color=COLORS["ink"], family="monospace")

    fig.tight_layout()
    out = FIG_DIR / "fig_encoding_schematic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Multi-dataset Pareto (data-driven)
# ---------------------------------------------------------------------------

def figure_multidataset_pareto():
    csv = FIG_DIR / "comparison_table.csv"
    if not csv.exists():
        print(f"  [skip] no {csv} -- run notebooks/03_susqa_analysis.py first")
        return
    df = pd.read_csv(csv)
    # Parse mean from "0.827 ± 0.009" cells
    def parse_mean(s: str) -> float:
        return float(str(s).split("±")[0].split("+/-")[0].strip())
    df["AUC_mean"] = df["AUC"].apply(parse_mean)

    datasets = ["breastmnist", "pneumoniamnist", "dermamnist"]
    titles = ["BreastMNIST", "PneumoniaMNIST", "DermaMNIST"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=300, sharey=False)

    for i, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        for cs, color, marker in [
            ("C1_classical", COLORS["C1"], "o"),
            ("C2_compressed", COLORS["C2"], "s"),
            ("C3_hybrid_quantum", COLORS["C3"], "^"),
        ]:
            cs_rows = sub[sub["case_study"] == cs]
            if len(cs_rows) == 0:
                continue
            xs = cs_rows["P"].values
            ys = cs_rows["AUC_mean"].values
            ax.scatter(xs, ys, color=color, marker=marker, s=110,
                       edgecolor="black", linewidth=0.7, alpha=0.85, label=cs)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters P (log)")
        ax.set_ylabel("AUC (mean)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("SUSQA Pareto: AUC vs. Trainable Parameters across three datasets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig_pareto_multidataset.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Effect-size heatmap
# ---------------------------------------------------------------------------

def figure_effect_size_heatmap():
    csv = FIG_DIR / "effect_sizes.csv"
    if not csv.exists():
        print(f"  [skip] no {csv} -- run notebooks/06_effect_sizes.py first")
        return
    df = pd.read_csv(csv)

    # Build (dataset, setting) x comparison matrix
    df["row"] = df["dataset"] + " | " + df["setting"]
    pivot = df.pivot_table(index="row", columns="comparison",
                           values="cohens_d", aggfunc="mean")
    # Order rows logically
    row_order = [
        "breastmnist | centralized",
        "breastmnist | fl_iid",
        "breastmnist | fl_dirichlet_alpha=0.5",
        "breastmnist | fl_dirichlet_alpha=0.1",
        "pneumoniamnist | fl_iid",
        "pneumoniamnist | fl_dirichlet_alpha=0.5",
        "pneumoniamnist | fl_dirichlet_alpha=0.1",
        "dermamnist | centralized",
        "dermamnist | fl_iid",
        "dermamnist | fl_dirichlet_alpha=0.5",
        "dermamnist | fl_dirichlet_alpha=0.1",
    ]
    row_order = [r for r in row_order if r in pivot.index]
    pivot = pivot.reindex(row_order)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            txt = f"{v:+.1f}"
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=color)

    ax.set_title("Cohen's d for AUC: red favours classical (C1/C2), blue favours hybrid (C3)\n"
                 "|d| > 0.8 large; |d| > 1.2 very large",
                 fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cohen's d", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "fig_effect_size_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    figure_system_overview()
    figure_encoding_schematic()
    figure_multidataset_pareto()
    figure_effect_size_heatmap()
    print("\nDone. Four figures in", FIG_DIR)


if __name__ == "__main__":
    main()
