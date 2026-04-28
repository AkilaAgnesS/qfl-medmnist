"""Three separate architecture diagrams for case studies C1, C2, C3.

Outputs (results/figures/):
    fig_c1_architecture.png   Classical CNN  (~26.4k params)
    fig_c2_architecture.png   Compressed MLP (~12.7k params)
    fig_c3_architecture.png   Hybrid CNN-VQC (~3.5k params)

Each diagram shows tensor shapes flowing through layers, trainable parameter
counts per layer, activation functions, and the total parameter budget at
the bottom. The C3 diagram additionally renders the explicit 8-qubit VQC
gate sequence with rotations and entanglers.

Run: python notebooks/09_model_architectures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_INK = "#222222"
C_MUTED = "#666666"
C_C1 = "#2c7fb8"
C_C2 = "#7fcdbb"
C_C3 = "#d95f0e"
C_DATA = "#444444"
C_QUANTUM = "#7B3F99"
C_CLASSICAL = "#1F77B4"
C_PARAM = "#1F77B4"


def fbox_outline(ax, x, y, w, h, label, *, edge, text_col=None,
                 fontsize=10, fontweight="normal", linewidth=1.4):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
        linewidth=linewidth, edgecolor=edge, facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=(text_col or edge))


def arrow(ax, x1, y1, x2, y2, color="#666666", linewidth=1.5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=10,head_width=7",
        color=color, linewidth=linewidth,
    ))


def shape_label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=8, color=C_MUTED, family="monospace", style="italic")


def param_label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=8, color=C_PARAM, fontweight="bold")


# ---------------------------------------------------------------------------
# C1: Classical CNN
# ---------------------------------------------------------------------------

def draw_c1():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 47, "Case Study C1 — Classical CNN (parameter-rich baseline)",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_INK)
    ax.text(50, 43, "Two convolutional blocks + dense bottleneck + softmax head",
            ha="center", va="center",
            fontsize=10, color=C_MUTED, style="italic")

    # Layers laid out left-to-right
    layers = [
        # (x, w, label, edge_color, shape_below, params_above)
        (2,  10, "Input", C_DATA, "1x28x28", ""),
        (15, 11, "Conv2D\n8 filters\n3x3 + ReLU", C_C1, "8x28x28", "P=80"),
        (29, 8,  "MaxPool\n2x2", C_MUTED, "8x14x14", "P=0"),
        (40, 11, "Conv2D\n16 filters\n3x3 + ReLU", C_C1, "16x14x14", "P=1,168"),
        (54, 8,  "MaxPool\n2x2", C_MUTED, "16x7x7", "P=0"),
        (65, 8,  "Flatten", C_MUTED, "784", "P=0"),
        (76, 9,  "Dense 32\n+ ReLU", C_C1, "32", "P=25,120"),
        (88, 10, "Dense 2\n(logits)", C_C1, "2", "P=66"),
    ]

    y_box = 18
    h_box = 12
    for x, w, label, color, shape, params in layers:
        fbox_outline(ax, x, y_box, w, h_box, label,
                     edge=color, fontsize=9, fontweight="bold", linewidth=1.6)
        shape_label(ax, x + w / 2, y_box - 2.5, shape)
        if params:
            param_label(ax, x + w / 2, y_box + h_box + 2, params)

    # Arrows between layers
    for i in range(len(layers) - 1):
        x_end_a = layers[i][0] + layers[i][1]
        x_start_b = layers[i + 1][0]
        arrow(ax, x_end_a, y_box + h_box / 2,
              x_start_b, y_box + h_box / 2)

    # Total
    ax.text(50, 8,
            "Total trainable parameters: P = 26,434  (C_total under FedAvg-25 = 26.43 MB)",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_C1)
    ax.text(50, 4,
            "Quantum cost Q = 0  |  Q-gates per forward pass G = 0",
            ha="center", va="center",
            fontsize=9, color=C_MUTED, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "fig_c1_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# C2: Classical compressed MLP
# ---------------------------------------------------------------------------

def draw_c2():
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 47, "Case Study C2 — Classical Compressed MLP (parameter-poor baseline)",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_INK)
    ax.text(50, 43, "Two-hidden-layer multilayer perceptron over flattened image",
            ha="center", va="center",
            fontsize=10, color=C_MUTED, style="italic")

    layers = [
        (5,  12, "Input", C_DATA, "1x28x28", ""),
        (22, 10, "Flatten", C_MUTED, "784", "P=0"),
        (37, 13, "Dense 16\n+ ReLU", C_C2, "16", "P=12,560"),
        (55, 13, "Dense 8\n+ ReLU", C_C2, "8", "P=136"),
        (73, 12, "Dense 2\n(logits)", C_C2, "2", "P=18"),
    ]

    y_box = 18
    h_box = 12
    for x, w, label, color, shape, params in layers:
        fbox_outline(ax, x, y_box, w, h_box, label,
                     edge=color, fontsize=9, fontweight="bold", linewidth=1.6)
        shape_label(ax, x + w / 2, y_box - 2.5, shape)
        if params:
            param_label(ax, x + w / 2, y_box + h_box + 2, params)

    for i in range(len(layers) - 1):
        x_end_a = layers[i][0] + layers[i][1]
        x_start_b = layers[i + 1][0]
        arrow(ax, x_end_a, y_box + h_box / 2,
              x_start_b, y_box + h_box / 2)

    ax.text(50, 8,
            "Total trainable parameters: P = 12,714  (C_total under FedAvg-25 = 12.71 MB)",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#0A6B5B")
    ax.text(50, 4,
            "Quantum cost Q = 0  |  Q-gates per forward pass G = 0",
            ha="center", va="center",
            fontsize=9, color=C_MUTED, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "fig_c2_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# C3: Hybrid CNN-VQC
# ---------------------------------------------------------------------------

def draw_vqc(ax, x0, y0, n_qubits=8, n_layers=2, qubit_spacing=1.4,
             column_spacing=2.2):
    qubit_ys = [y0 + i * qubit_spacing for i in range(n_qubits)]
    total_columns = 1 + 2 * n_layers + 1
    line_x_end = x0 + (total_columns + 1) * column_spacing
    for y in qubit_ys:
        ax.plot([x0, line_x_end], [y, y], color=C_INK, linewidth=0.9, zorder=1)
    for i, y in enumerate(qubit_ys):
        ax.text(x0 - 0.5, y, f"$q_{{{i}}}$", ha="right", va="center",
                fontsize=7, color=C_INK)

    def gate_box(cx, cy, label, color):
        size = 0.95
        rect = Rectangle((cx - size / 2, cy - size / 2), size, size,
                         facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=5.5, color=color, fontweight="bold", zorder=4)

    col_x = x0 + 1.0
    for y in qubit_ys:
        gate_box(col_x, y, "RY", C_QUANTUM)
    ax.text(col_x, y0 - 1.4, "angle\nenc.", ha="center", va="center",
            fontsize=6.5, color=C_QUANTUM, fontweight="bold")

    for layer in range(n_layers):
        col_x = x0 + 1.0 + (1 + 2 * layer) * column_spacing
        for y in qubit_ys:
            gate_box(col_x, y, "RY\nRZ", C_QUANTUM)
        ax.text(col_x, y0 - 1.4, f"L{layer + 1}\nrot.",
                ha="center", va="center",
                fontsize=6.5, color=C_QUANTUM, fontweight="bold")

        ent_x = x0 + 1.0 + (2 + 2 * layer) * column_spacing
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            yi, yj = qubit_ys[i], qubit_ys[j]
            if j < i:
                continue
            ax.add_patch(Circle((ent_x, yi), 0.16, color=C_QUANTUM, zorder=5))
            ax.add_patch(Circle((ent_x, yj), 0.34, facecolor="white",
                                edgecolor=C_QUANTUM, linewidth=1.0, zorder=5))
            ax.plot([ent_x, ent_x], [yj - 0.34, yj + 0.34],
                    color=C_QUANTUM, linewidth=1.0, zorder=6)
            ax.plot([ent_x - 0.34, ent_x + 0.34], [yj, yj],
                    color=C_QUANTUM, linewidth=1.0, zorder=6)
            ax.plot([ent_x, ent_x], [yi, yj], color=C_QUANTUM,
                    linewidth=0.9, zorder=4)
        ax.text(ent_x, y0 - 1.4, f"L{layer + 1}\nCNOTs",
                ha="center", va="center",
                fontsize=6.5, color=C_QUANTUM, fontweight="bold")

    meas_x = x0 + 1.0 + (1 + 2 * n_layers) * column_spacing
    for y in qubit_ys:
        gate_box(meas_x, y, "Z", "#D62728")
    ax.text(meas_x, y0 - 1.4, "$\\langle Z_i \\rangle$",
            ha="center", va="center",
            fontsize=7, color="#D62728", fontweight="bold")


def draw_c3():
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 67, "Case Study C3 — Hybrid CNN-VQC Quantum Model",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_INK)
    ax.text(50, 63,
            "Classical CNN feature extractor + 8-qubit variational quantum classifier",
            ha="center", va="center",
            fontsize=10, color=C_MUTED, style="italic")

    # Top row: classical pipeline (left to right)
    y_box = 47
    h_box = 10
    classical_layers = [
        (1,   8, "Input", C_DATA, "1x28x28", ""),
        (11,  10, "Conv2D\n4 filters", C_CLASSICAL, "4x28x28", "P=40"),
        (23,  8, "MaxPool", C_MUTED, "4x14x14", "P=0"),
        (33,  10, "Conv2D\n8 filters", C_CLASSICAL, "8x14x14", "P=296"),
        (45,  8, "MaxPool", C_MUTED, "8x7x7", "P=0"),
        (55,  8, "Flatten", C_MUTED, "392", "P=0"),
        (65,  10, "Linear\nbottleneck", C_PARAM, "8", "P=3,144"),
        (77,  10, "tanh*pi\n-> angles", C_QUANTUM, "8 angles", "P=0"),
    ]
    for x, w, label, color, shape, params in classical_layers:
        fbox_outline(ax, x, y_box, w, h_box, label,
                     edge=color, fontsize=8, fontweight="bold", linewidth=1.4)
        shape_label(ax, x + w / 2, y_box - 1.5, shape)
        if params:
            param_label(ax, x + w / 2, y_box + h_box + 1.5, params)

    for i in range(len(classical_layers) - 1):
        x_end_a = classical_layers[i][0] + classical_layers[i][1]
        x_start_b = classical_layers[i + 1][0]
        arrow(ax, x_end_a, y_box + h_box / 2,
              x_start_b, y_box + h_box / 2)

    # Connector arrow from angles to VQC
    arrow(ax, 87, y_box + h_box / 2, 92, 35, color=C_QUANTUM, linewidth=2)
    ax.text(91, y_box - 1, "to VQC",
            ha="center", va="center",
            fontsize=7.5, color=C_QUANTUM, fontweight="bold")

    # VQC drawing
    ax.text(50, 36, "8-qubit Variational Quantum Circuit (VQC)",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_QUANTUM)
    draw_vqc(ax, x0=15, y0=15, n_qubits=8, n_layers=2,
             qubit_spacing=1.6, column_spacing=2.4)

    # Final readout
    fbox_outline(ax, 78, 18, 9, 8, "Linear\nreadout\n8->2",
                 edge=C_PARAM, fontsize=8, fontweight="bold", linewidth=1.4)
    param_label(ax, 82.5, 27.5, "P=18")
    arrow(ax, 75, 22, 78, 22, color=C_MUTED)
    arrow(ax, 87, 22, 90, 22, color=C_MUTED)
    fbox_outline(ax, 90, 18, 8, 8, "logits\n2 classes",
                 edge=C_INK, fontsize=8)

    # Total
    ax.text(50, 8,
            "Total trainable parameters: P = 3,530  (C_total under FedAvg-15 = 2.12 MB)",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_C3)
    ax.text(50, 4,
            "Quantum cost: G = 8 + 2*24 = 56 gates per forward pass  |  "
            "Q = G * shots * samples * epochs",
            ha="center", va="center",
            fontsize=9, color=C_QUANTUM, style="italic", fontweight="bold")
    ax.text(50, 1,
            "Trainable: 32 quantum angles (8 qubits * 2 layers * RY+RZ) + 3,498 classical bottleneck/readout",
            ha="center", va="center",
            fontsize=8.5, color=C_MUTED, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "fig_c3_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    draw_c1()
    draw_c2()
    draw_c3()
    print("\nDone. Three architecture diagrams in", FIG_DIR)


if __name__ == "__main__":
    main()
