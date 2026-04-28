"""Reference diagram of the complete federated-learning process with the
quantum (C3 hybrid) architecture made explicit.

Top half: FL flow (server, communication band, 5 hospitals).
Bottom half: zoomed-in C3 hybrid local model showing image -> CNN -> bottleneck
-> angle encoding -> 8-qubit VQC with explicit gate sequence -> measurement
-> readout.

Output: results/figures/fig_fl_process.png
Run:    python notebooks/08_fl_process_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL_SERVER = "#2E5984"
COL_HOSPITAL = "#7FCDBB"
COL_DATA_BORDER = "#C0392B"
COL_DOWN = "#2CA02C"
COL_UP = "#D95F0E"
COL_INK = "#222222"
COL_MUTED = "#666666"
COL_AXIS_P = "#1F77B4"
COL_AXIS_C = "#2CA02C"
COL_AXIS_Q = "#9467BD"
COL_AXIS_ETA = "#D62728"
COL_QUANTUM = "#7B3F99"
COL_CLASSICAL = "#1F77B4"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def fbox(ax, x, y, w, h, label, *, fill, edge=None, text_col="white",
         fontsize=10, fontweight="normal", linewidth=1.4, alpha=0.92,
         dashed=False):
    if edge is None:
        edge = fill
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
        linewidth=linewidth, edgecolor=edge, facecolor=fill, alpha=alpha,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_col)


def fbox_outline(ax, x, y, w, h, label, *, edge, text_col=None,
                 fontsize=10, fontweight="normal", linewidth=1.4, dashed=False):
    text_col = text_col or edge
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
        linewidth=linewidth, edgecolor=edge, facecolor="white",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_col)


def arrow(ax, x1, y1, x2, y2, color, linewidth=2.0, head_size=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"->,head_length={head_size},head_width={head_size * 0.7}",
        color=color, linewidth=linewidth,
    ))


def step_circle(ax, x, y, n, *, color="#444444", radius=1.2):
    ax.add_patch(Circle((x, y), radius, color=color, zorder=10))
    ax.text(x, y, str(n), ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=11)


# ---------------------------------------------------------------------------
# Quantum circuit drawing
# ---------------------------------------------------------------------------

def draw_vqc(ax, x0, y0, n_qubits=8, n_layers=2, qubit_spacing=1.4,
             column_spacing=2.2):
    """Draw an n_qubits x n_layers variational ansatz starting at (x0, y0).

    Layout per layer column:
        [angle encoding RY] -> [layer 1: per-qubit RY+RZ] -> [layer 1: CNOT ring]
        -> [layer 2: per-qubit RY+RZ] -> [layer 2: CNOT ring] -> [Z-measurement]
    """
    # Qubit horizontal lines
    qubit_ys = [y0 + i * qubit_spacing for i in range(n_qubits)]
    total_columns = 1 + 2 * n_layers + 1   # encoding + (rotations+entangler) per layer + measurement
    line_x_start = x0
    line_x_end = x0 + (total_columns + 1) * column_spacing
    for y in qubit_ys:
        ax.plot([line_x_start, line_x_end], [y, y],
                color=COL_INK, linewidth=0.9, zorder=1)

    # Qubit labels on the left
    for i, y in enumerate(qubit_ys):
        ax.text(x0 - 0.5, y, f"$q_{{{i}}}$",
                ha="right", va="center", fontsize=7.5, color=COL_INK)

    def gate_box(cx, cy, label, color):
        size = 0.95
        rect = Rectangle((cx - size / 2, cy - size / 2), size, size,
                         facecolor="white", edgecolor=color, linewidth=1.2,
                         zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=6, color=color, fontweight="bold", zorder=4)

    # Column 0: angle encoding (RY)
    col_x = x0 + 1.0
    for y in qubit_ys:
        gate_box(col_x, y, "RY", COL_QUANTUM)
    ax.text(col_x, y0 - 1.2, "angle\nenc.",
            ha="center", va="center", fontsize=6.5, color=COL_QUANTUM, fontweight="bold")

    # For each variational layer: rotation column + entangler column
    for layer in range(n_layers):
        # Rotation column (RY then RZ in same box for compactness)
        col_x = x0 + 1.0 + (1 + 2 * layer) * column_spacing
        for y in qubit_ys:
            gate_box(col_x, y, "RY\nRZ", COL_QUANTUM)
        ax.text(col_x, y0 - 1.2, f"layer {layer + 1}\nrotations",
                ha="center", va="center", fontsize=6.5, color=COL_QUANTUM, fontweight="bold")

        # Entangler column (CNOT ring) - draw connecting lines + dots
        ent_x = x0 + 1.0 + (2 + 2 * layer) * column_spacing
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            yi, yj = qubit_ys[i], qubit_ys[j]
            # Skip the wrap-around for a cleaner picture (linear chain only)
            if j < i:
                continue
            # Control dot
            ax.add_patch(Circle((ent_x, yi), 0.16, color=COL_QUANTUM, zorder=5))
            # Target circle
            ax.add_patch(Circle((ent_x, yj), 0.34,
                                facecolor="white", edgecolor=COL_QUANTUM,
                                linewidth=1.0, zorder=5))
            ax.plot([ent_x, ent_x], [yj - 0.34, yj + 0.34],
                    color=COL_QUANTUM, linewidth=1.0, zorder=6)
            ax.plot([ent_x - 0.34, ent_x + 0.34], [yj, yj],
                    color=COL_QUANTUM, linewidth=1.0, zorder=6)
            # Vertical link
            ax.plot([ent_x, ent_x], [yi, yj], color=COL_QUANTUM,
                    linewidth=0.9, zorder=4)
        ax.text(ent_x, y0 - 1.2, f"layer {layer + 1}\nCNOTs",
                ha="center", va="center", fontsize=6.5, color=COL_QUANTUM, fontweight="bold")

    # Final measurement column (Z)
    meas_x = x0 + 1.0 + (1 + 2 * n_layers) * column_spacing
    for y in qubit_ys:
        gate_box(meas_x, y, "Z", COL_AXIS_ETA)
    ax.text(meas_x, y0 - 1.2, "$\\langle Z_i \\rangle$",
            ha="center", va="center", fontsize=7.5, color=COL_AXIS_ETA, fontweight="bold")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(15, 11.5), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- TITLE ---
    ax.text(50, 97,
            "Federated Learning of the C3 Hybrid Quantum Model",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=COL_INK)
    ax.text(50, 94,
            "K = 5 hospitals  |  R rounds  |  E = 1 local epoch  |  FedAvg aggregation  |  CNN feature extractor + 8-qubit VQC",
            ha="center", va="center",
            fontsize=9.5, color=COL_MUTED, style="italic")

    # ====================================================================
    # ZONE 1 — SERVER (top, ~80-92)
    # ====================================================================
    server_x, server_y, server_w, server_h = 30, 81, 40, 9
    fbox(ax, server_x, server_y, server_w, server_h,
         "GLOBAL SERVER", fill=COL_SERVER, fontsize=11, fontweight="bold")
    fbox_outline(ax, server_x + 2, server_y + 0.6, 17, 4,
                 "Global $\\theta^{(r)}$ (P parameters)",
                 edge=COL_SERVER, fontsize=8)
    fbox_outline(ax, server_x + 21, server_y + 0.6, 17, 4,
                 "FedAvg: $\\theta^{(r)} = \\sum_k (n_k / N) \\theta_k^{(r)}$",
                 edge=COL_SERVER, fontsize=8)
    fbox_outline(ax, 75, 83, 22, 7,
                 "Round $r = 1, \\ldots, R$\nEvaluate global on test set\nLog SUSQA: $P, C, Q, \\eta$",
                 edge=COL_MUTED, fontsize=8)

    # ====================================================================
    # ZONE 2 — COMMUNICATION (band, ~63-77)
    # ====================================================================
    ax.add_patch(Rectangle((4, 63), 92, 14, color="#FAFAFA",
                           ec=COL_MUTED, lw=0.6, alpha=0.6))
    ax.text(50, 76, "COMMUNICATION BAND",
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=COL_MUTED)
    ax.text(20, 73, "Downlink: $\\theta^{(r-1)}$  ($P\\cdot 4$ B / client)",
            ha="center", fontsize=9, fontweight="bold", color=COL_DOWN)
    ax.text(80, 73, "Uplink: $\\theta_k^{(r)}$  ($P\\cdot 4$ B / client)",
            ha="center", fontsize=9, fontweight="bold", color=COL_UP)
    fbox_outline(ax, 38, 65, 24, 4.5,
                 "$C_{total} = 2 K R \\cdot P \\cdot 4$  bytes",
                 edge=COL_AXIS_C, text_col=COL_AXIS_C,
                 fontsize=9.5, fontweight="bold")

    # ====================================================================
    # ZONE 3 — HOSPITALS (bottom-FL, ~38-60)
    # ====================================================================
    hospital_y, hospital_w, hospital_h = 38, 16, 22
    hospital_x = [4, 22.5, 41, 59.5, 78]
    for i, hx in enumerate(hospital_x):
        fbox(ax, hx, hospital_y, hospital_w, hospital_h,
             "", fill=COL_HOSPITAL, alpha=0.25, edge=COL_HOSPITAL)
        ax.text(hx + hospital_w / 2, hospital_y + hospital_h - 1.8,
                f"Hospital {i + 1}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=COL_INK)
        # Private data
        fbox_outline(ax, hx + 2, hospital_y + 14, hospital_w - 4, 4,
                     f"Private $D_{{{i+1}}}$ (stays local)",
                     edge=COL_DATA_BORDER, text_col=COL_DATA_BORDER,
                     fontsize=7.5, fontweight="bold", linewidth=1.5, dashed=True)
        # Local C3 hybrid model copy
        fbox_outline(ax, hx + 2, hospital_y + 8, hospital_w - 4, 4,
                     f"Local C3 model\n(CNN + 8-qubit VQC)",
                     edge=COL_QUANTUM, text_col=COL_QUANTUM,
                     fontsize=7.5, fontweight="bold")
        # Local training
        fbox_outline(ax, hx + 2, hospital_y + 2, hospital_w - 4, 4,
                     f"Local training\n$E$ epochs on $D_{{{i+1}}}$",
                     edge="#0A6B5B", text_col="#0A6B5B",
                     fontsize=7.5, fontweight="bold")

    # Arrows: server -> hospitals (downlink, green)
    server_bottom = server_y
    for hx in hospital_x:
        x_target = hx + hospital_w / 2 - 0.7
        arrow(ax, 50, server_bottom, x_target, hospital_y + hospital_h,
              color=COL_DOWN, linewidth=1.3)
    # Arrows: hospitals -> server (uplink, orange)
    for hx in hospital_x:
        x_source = hx + hospital_w / 2 + 0.7
        arrow(ax, x_source, hospital_y + hospital_h, 50, server_bottom,
              color=COL_UP, linewidth=1.3)

    # Numbered step circles
    step_circle(ax, 8, 72, 1)
    ax.text(11, 72, "Broadcast", fontsize=8, va="center", color=COL_INK)
    step_circle(ax, 8, 53, 2)
    ax.text(11, 53, "Receive  $\\theta^{(r-1)}$", fontsize=8, va="center", color=COL_INK)
    step_circle(ax, 8, 41, 3)
    ax.text(11, 41, "Local training", fontsize=8, va="center", color=COL_INK)
    step_circle(ax, 92, 70, 4)
    ax.text(89, 70, "Upload  $\\theta_k^{(r)}$",
            fontsize=8, ha="right", va="center", color=COL_INK)
    step_circle(ax, 92, 86, 5)
    ax.text(89, 86, "FedAvg aggregate",
            fontsize=8, ha="right", va="center", color=COL_INK)
    step_circle(ax, 75, 87, 6)
    ax.text(73, 87, "Evaluate + log",
            fontsize=8, ha="right", va="center", color=COL_INK)

    # ZONE 4 - C3 HYBRID ARCHITECTURE (bottom panel)
    panel_y0 = 5
    panel_y1 = 33
    ax.add_patch(Rectangle((2, panel_y0), 96, panel_y1 - panel_y0,
                           facecolor="#FAFAFA", ec=COL_QUANTUM,
                           linewidth=1.4, alpha=0.5))
    ax.text(50, panel_y1 - 1.5,
            "Inside each hospital's local C3 hybrid model: image -> CNN -> bottleneck -> 8-qubit VQC -> readout",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=COL_QUANTUM)

    fbox_outline(ax, 4, 18, 8, 8, "Input\n28x28xC", edge=COL_INK, fontsize=8)
    arrow(ax, 12.2, 22, 14, 22, color=COL_MUTED, linewidth=1.5)
    fbox_outline(ax, 14, 18, 10, 8, "CNN\nfeature\nextractor",
                 edge=COL_CLASSICAL, text_col=COL_CLASSICAL,
                 fontsize=8, fontweight="bold")
    arrow(ax, 24.2, 22, 26, 22, color=COL_MUTED, linewidth=1.5)
    fbox_outline(ax, 26, 18, 9, 8, "Linear\nbottleneck\n8-dim",
                 edge=COL_AXIS_P, text_col=COL_AXIS_P, fontsize=8)
    arrow(ax, 35.2, 22, 37, 22, color=COL_MUTED, linewidth=1.5)
    fbox_outline(ax, 37, 18, 8, 8, "tanh*pi\n-> angles",
                 edge=COL_QUANTUM, text_col=COL_QUANTUM, fontsize=7.5)
    arrow(ax, 45.2, 22, 47, 22, color=COL_MUTED, linewidth=1.5)

    draw_vqc(ax, x0=49, y0=8.5, n_qubits=8, n_layers=2,
             qubit_spacing=1.4, column_spacing=2.2)

    arrow(ax, 78, 13, 80.5, 13, color=COL_MUTED, linewidth=1.5)
    fbox_outline(ax, 80.5, 9, 8, 8, "Linear\nreadout\n-> logits",
                 edge=COL_AXIS_ETA, text_col=COL_AXIS_ETA, fontsize=8)
    arrow(ax, 88.7, 13, 90.5, 13, color=COL_MUTED, linewidth=1.5)
    fbox_outline(ax, 90.5, 9, 7.5, 8, "class\nprediction",
                 edge=COL_INK, fontsize=8, fontweight="bold")

    ax.text(50, 5.5,
            "P counted on the full pipeline   |   G = n + L*3n = 56 gates per forward pass   |   Q = G * s * N * E_tot",
            ha="center", va="center",
            fontsize=9, color=COL_AXIS_Q, fontweight="bold")

    legend_x, legend_y = 1.5, 80
    fbox_outline(ax, legend_x, legend_y, 26, 8,
                 "SUSQA axes per run", edge=COL_INK,
                 text_col=COL_INK, fontsize=9, fontweight="bold")
    items = [
        ("P - params (server)", COL_AXIS_P),
        ("C_total - comm. bytes (band)", COL_AXIS_C),
        ("Q - gate-shots (clients)", COL_AXIS_Q),
        ("eta(p) - noise robustness", COL_AXIS_ETA),
    ]
    iy = legend_y + 6.5
    for text, col in items:
        iy -= 1.4
        ax.text(legend_x + 1.5, iy, "*", color=col, fontsize=11,
                va="center", fontweight="bold")
        ax.text(legend_x + 3, iy, text, color=col, fontsize=8, va="center")

    ax.text(50, 1.5,
            "Privacy: patient images never leave hospital; only model parameters travel.",
            ha="center", va="center",
            fontsize=9, color=COL_DATA_BORDER, fontweight="bold", style="italic")

    fig.tight_layout()
    out = FIG_DIR / "fig_fl_process.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
