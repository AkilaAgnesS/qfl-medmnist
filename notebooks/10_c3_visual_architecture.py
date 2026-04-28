"""Polished, paper-ready C3 hybrid CNN-VQC architecture diagram.

Visual style adapted from the reference: stacked feature-map plates for the
CNN, a dense connection layer rendered with circles+edges, a colour-coded
quantum circuit with rotation/entangling gates and a measurement column,
and class-probability output bars.

Architecture rendered (true to the implementation in models/hybrid_qnn.py):
    Input image (28x28xC)
    -> Conv2d(C->4, 3x3) -> ReLU -> MaxPool2x2
    -> Conv2d(4->8, 3x3) -> ReLU -> MaxPool2x2
    -> Flatten (8*7*7 = 392)
    -> Dense bottleneck (392 -> 8) -> tanh*pi
    -> Angle encoding RY on 8 qubits
    -> [layer 1: RY,RZ per qubit, CNOT ring] x L=2
    -> <Z_i> measurement per qubit
    -> Linear readout (8 -> num_classes)
    -> Class-probability bars

Output: results/figures/fig_c3_visual.png  (300 DPI, 16x9 inches)
Run:    python notebooks/10_c3_visual_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Palette matched to the reference image
C_INK = "#1A1A1A"
C_MUTED = "#5A5A5A"
C_INPUT = "#3A3A3A"
C_CONV1 = "#5BC2D8"
C_POOL = "#F4A6BC"
C_CONV2 = "#A6D88B"
C_FC_LINE = "#1F4E8C"
C_FC_NODE = "#1F4E8C"
C_QUANTUM_PURPLE = "#7B3F99"
C_QUANTUM_BLUE = "#3B7BB7"
C_GATE_HAD = "#F4D060"  # yellow Hadamard
C_GATE_MEAS = "#5B86E5"
C_BOX_BORDER = "#3F3F8F"
C_OUTPUT_A = "#E8C547"
C_OUTPUT_B = "#7FCDBB"


# ---------------------------------------------------------------------------
# Helpers: stacked feature-map plates
# ---------------------------------------------------------------------------

def stacked_plates(ax, x, y, w, h, n_plates=4, color="#5BC2D8",
                   offset=(0.6, 0.5), border="#222222"):
    """Draw n_plates overlapping rectangles to suggest a feature-map stack."""
    for i in range(n_plates - 1, -1, -1):
        rect = Rectangle((x + i * offset[0], y + i * offset[1]), w, h,
                         facecolor=color, edgecolor=border,
                         linewidth=1.2, alpha=0.85, zorder=2 + i)
        ax.add_patch(rect)


def labeled_block_below(ax, cx, cy, label, fontsize=9, color="#222222"):
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)


def fc_layer(ax, cx, y_top, y_bot, n_nodes=6, radius=0.6, color=C_FC_NODE):
    """Draw a vertical column of circle 'neurons'."""
    ys = np.linspace(y_bot, y_top, n_nodes)
    for y in ys:
        ax.add_patch(Circle((cx, y), radius, facecolor="white",
                            edgecolor=color, linewidth=1.4, zorder=4))
    return list(ys)


def fc_connect(ax, xs_left, ys_left, x_right, ys_right, color=C_FC_LINE,
               alpha=0.45, linewidth=0.5):
    for yl in ys_left:
        for yr in ys_right:
            ax.plot([xs_left, x_right], [yl, yr],
                    color=color, alpha=alpha, linewidth=linewidth, zorder=3)


def small_arrow(ax, x1, y1, x2, y2, color=C_MUTED, linewidth=1.6, head=10):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"->,head_length={head},head_width={head*0.7}",
        color=color, linewidth=linewidth,
    ))


def gate_box(ax, cx, cy, w, h, label, *, fill, edge=None, text_col="white",
             fontsize=9, fontweight="bold"):
    if edge is None:
        edge = fill
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.3",
        facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_col, zorder=6)


def cnot_pair(ax, x, y_ctrl, y_targ, color=C_QUANTUM_PURPLE):
    ax.add_patch(Circle((x, y_ctrl), 0.18, color=color, zorder=6))
    ax.add_patch(Circle((x, y_targ), 0.42, facecolor="white",
                        edgecolor=color, linewidth=1.4, zorder=6))
    ax.plot([x, x], [y_targ - 0.42, y_targ + 0.42],
            color=color, linewidth=1.2, zorder=7)
    ax.plot([x - 0.42, x + 0.42], [y_targ, y_targ],
            color=color, linewidth=1.2, zorder=7)
    ax.plot([x, x], [y_ctrl, y_targ], color=color, linewidth=1.0, zorder=5)


def meas_symbol(ax, cx, cy, w=0.9, h=0.9, color=C_GATE_MEAS):
    rect = Rectangle((cx - w/2, cy - h/2), w, h,
                     facecolor=color, edgecolor=color, linewidth=1.2, zorder=5)
    ax.add_patch(rect)
    # Inner arc (gauge)
    theta = np.linspace(np.pi, 2 * np.pi, 30)
    arc_r = 0.25
    ax.plot(cx + arc_r * np.cos(theta),
            cy - 0.05 + arc_r * np.sin(theta) * (-1),
            color="white", linewidth=1.5, zorder=6)
    # Pointer
    ax.plot([cx, cx + 0.18], [cy - 0.05, cy + 0.18],
            color="white", linewidth=1.5, zorder=6)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(50, 54.5, "C3 Hybrid CNN-VQC Architecture for Federated Medical Image Classification",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_INK)
    ax.text(50, 51.5,
            "Classical CNN feature extractor + 8-qubit variational quantum classifier",
            ha="center", va="center",
            fontsize=10, color=C_MUTED, style="italic")

    # ==================================================================
    # ROW 1: classical CNN pipeline (top)
    # ==================================================================
    row_y = 32

    # Input image: stylized as a blocky representation of a medical scan
    img_x, img_y, img_w, img_h = 2, row_y - 4, 8, 8
    rng = np.random.default_rng(0)
    pixels = rng.random((10, 10))
    ax.imshow(pixels, extent=(img_x, img_x + img_w, img_y, img_y + img_h),
              cmap="gray", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((img_x, img_y), img_w, img_h,
                           facecolor="none", edgecolor=C_INK,
                           linewidth=1.4, zorder=3))
    labeled_block_below(ax, img_x + img_w / 2, img_y - 1.5,
                        "Input image\n28x28x1", fontsize=8.5, color=C_INK)

    small_arrow(ax, img_x + img_w + 0.5, row_y, img_x + img_w + 3, row_y)

    # Conv1: 4 stacked plates
    conv1_x = img_x + img_w + 3.5
    stacked_plates(ax, conv1_x, row_y - 4, 7.5, 8, n_plates=4,
                   color=C_CONV1, offset=(0.55, 0.55))
    labeled_block_below(ax, conv1_x + 4.5, row_y - 6,
                        "Conv2D 4 filters\n3x3 + ReLU\nP=40",
                        fontsize=8, color=C_CONV1)

    small_arrow(ax, conv1_x + 11, row_y, conv1_x + 13, row_y)

    # Pool1
    pool1_x = conv1_x + 13
    stacked_plates(ax, pool1_x, row_y - 3, 6, 6, n_plates=4,
                   color=C_POOL, offset=(0.45, 0.45))
    labeled_block_below(ax, pool1_x + 4, row_y - 6,
                        "MaxPool 2x2\nP=0",
                        fontsize=8, color="#A04060")

    small_arrow(ax, pool1_x + 9, row_y, pool1_x + 11, row_y)

    # Conv2: 8 stacked plates
    conv2_x = pool1_x + 11
    stacked_plates(ax, conv2_x, row_y - 3, 6, 6, n_plates=6,
                   color=C_CONV2, offset=(0.45, 0.45))
    labeled_block_below(ax, conv2_x + 4.5, row_y - 6,
                        "Conv2D 8 filters\n3x3 + ReLU\nP=296",
                        fontsize=8, color="#5A8A40")

    small_arrow(ax, conv2_x + 9.5, row_y, conv2_x + 11.5, row_y)

    # Pool2
    pool2_x = conv2_x + 11.5
    stacked_plates(ax, pool2_x, row_y - 2, 4.5, 4.5, n_plates=6,
                   color=C_POOL, offset=(0.35, 0.35))
    labeled_block_below(ax, pool2_x + 3.5, row_y - 6,
                        "MaxPool 2x2\nP=0",
                        fontsize=8, color="#A04060")

    small_arrow(ax, pool2_x + 8, row_y, pool2_x + 10, row_y)

    # Flatten + Dense bottleneck (drawn as fully-connected with edges)
    flat_x = pool2_x + 10
    flat_ys = fc_layer(ax, flat_x, y_top=row_y + 5, y_bot=row_y - 5,
                       n_nodes=8, radius=0.55, color=C_FC_LINE)
    labeled_block_below(ax, flat_x, row_y - 7,
                        "Flatten 392", fontsize=8, color=C_FC_NODE)

    bn_x = flat_x + 7
    bn_ys = fc_layer(ax, bn_x, y_top=row_y + 4, y_bot=row_y - 4,
                     n_nodes=8, radius=0.7, color=C_FC_LINE)
    fc_connect(ax, flat_x, flat_ys, bn_x, bn_ys, alpha=0.35, linewidth=0.45)
    labeled_block_below(ax, bn_x, row_y - 6,
                        "Linear bottleneck\n8-dim, P=3,144",
                        fontsize=8, color=C_FC_NODE)

    small_arrow(ax, bn_x + 1.5, row_y, bn_x + 4, row_y, color=C_QUANTUM_PURPLE,
                linewidth=2.2)
    labeled_block_below(ax, bn_x + 2.5, row_y + 2.5,
                        "tanh*pi\n-> angles",
                        fontsize=7.5, color=C_QUANTUM_PURPLE)

    # ==================================================================
    # ROW 2: 8-qubit variational quantum circuit (bottom)
    # ==================================================================
    qc_y = 12
    qc_x_start = 12
    qc_x_end = 88
    n_qubits = 8

    # Box around quantum filter section
    qc_box = FancyBboxPatch(
        (qc_x_start - 4, qc_y - 4), qc_x_end - qc_x_start + 8, 13,
        boxstyle="round,pad=0.4,rounding_size=0.6",
        facecolor="white", edgecolor=C_BOX_BORDER, linewidth=1.4,
        linestyle="--", zorder=1,
    )
    ax.add_patch(qc_box)
    ax.text((qc_x_start + qc_x_end) / 2, qc_y + 9.5,
            "8-qubit Variational Quantum Circuit (VQC)",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_BOX_BORDER)

    # Qubit horizontal lines and |0> kets
    qubit_ys = np.linspace(qc_y - 3, qc_y + 6, n_qubits)
    for i, qy in enumerate(qubit_ys):
        ax.plot([qc_x_start, qc_x_end], [qy, qy],
                color=C_INK, linewidth=0.9, zorder=2)
        ax.text(qc_x_start - 1.6, qy, r"$|0\rangle$",
                ha="right", va="center", fontsize=8, color=C_INK)
        ax.text(qc_x_start - 5.5, qy, f"$q_{{{i}}}$",
                ha="right", va="center", fontsize=8, color=C_MUTED)

    # ----- Column: angle encoding RY(theta) -----
    enc_x = qc_x_start + 3
    for qy in qubit_ys:
        gate_box(ax, enc_x, qy, 2.8, 0.9, r"$R_Y(\theta)$",
                 fill=C_QUANTUM_PURPLE, edge=C_QUANTUM_PURPLE,
                 fontsize=7, fontweight="bold")
    ax.text(enc_x, qc_y - 5,
            "Angle\nencoding",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_QUANTUM_PURPLE)

    # ----- Two variational layers: RY,RZ rotations + CNOT ring -----
    layer_centers = []
    for layer in range(2):
        # rotation column
        rot_x = qc_x_start + 10 + layer * 24
        for qy in qubit_ys:
            gate_box(ax, rot_x, qy, 3.2, 0.9, r"$R_Y R_Z$",
                     fill=C_QUANTUM_BLUE, edge=C_QUANTUM_BLUE,
                     fontsize=7, fontweight="bold")
        ax.text(rot_x, qc_y - 5,
                f"Layer {layer + 1}\nrotations",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C_QUANTUM_BLUE)

        # CNOT chain (between adjacent qubits)
        cnot_x = rot_x + 7
        for i in range(n_qubits - 1):
            cnot_pair(ax, cnot_x + i * 1.4,
                      qubit_ys[i], qubit_ys[i + 1],
                      color=C_QUANTUM_PURPLE)
        ax.text(cnot_x + (n_qubits - 1) * 0.7, qc_y - 5,
                f"Layer {layer + 1}\nCNOT chain",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C_QUANTUM_PURPLE)
        layer_centers.append((rot_x, cnot_x))

    # ----- Measurement column -----
    meas_x = qc_x_end - 3
    for qy in qubit_ys:
        meas_symbol(ax, meas_x, qy, w=1.4, h=1.0, color=C_GATE_MEAS)
    ax.text(meas_x, qc_y - 5,
            r"$\langle Z_i \rangle$" + "\nmeasurement",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_GATE_MEAS)

    # ==================================================================
    # Connection from CNN row to VQC row, and from VQC to readout/output
    # ==================================================================

    # Curved arrow from bottleneck angles down to encoding column
    ax.add_patch(FancyArrowPatch(
        (bn_x + 4, row_y - 1), (enc_x - 1.5, qubit_ys[-1] + 1.5),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle="->,head_length=12,head_width=8",
        color=C_QUANTUM_PURPLE, linewidth=1.8,
    ))

    # ==================================================================
    # Right side: Linear readout + class-probability output bars
    # ==================================================================

    # Linear readout
    ro_x = 90
    ro_ys = fc_layer(ax, ro_x, y_top=qc_y + 6, y_bot=qc_y - 3,
                     n_nodes=8, radius=0.5, color="#666666")
    # connect from measurement column to readout
    fc_connect(ax, meas_x + 0.8, list(qubit_ys), ro_x, ro_ys,
               color="#777777", alpha=0.3, linewidth=0.45)
    labeled_block_below(ax, ro_x, qc_y - 5,
                        "Linear\nreadout",
                        fontsize=8, color="#444444")

    # Class probability output bars (binary or 7-class)
    bar_x = 95
    # show three bars to indicate "any number of classes"
    bar_labels = ["benign", "malignant"]
    bar_colors = [C_OUTPUT_A, C_OUTPUT_B]
    bar_heights = [4.5, 6.5]
    for i, (lbl, c, h) in enumerate(zip(bar_labels, bar_colors, bar_heights)):
        bx = bar_x + i * 2.2
        ax.add_patch(Rectangle((bx, qc_y - 1), 1.6, h,
                               facecolor=c, edgecolor=C_INK,
                               linewidth=1.0, zorder=5))
        ax.text(bx + 0.8, qc_y - 2.5, lbl,
                ha="center", va="center",
                fontsize=7.5, color=C_INK, rotation=0)
    ax.text(bar_x + 1.2, qc_y + 7.5, "Class\nprobabilities",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=C_INK)

    # Connect readout to output bars
    small_arrow(ax, ro_x + 1, qc_y + 1.5, bar_x - 0.3, qc_y + 1.5,
                color=C_MUTED, linewidth=1.4)

    # ==================================================================
    # Bottom totals
    # ==================================================================
    ax.text(50, 1.5,
            "Total trainable parameters P = 3,530   |   "
            "Quantum gates per forward pass G = n + L*3n = 56   |   "
            "Q = G * shots * samples * epochs",
            ha="center", va="center",
            fontsize=9, color=C_QUANTUM_PURPLE, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "fig_c3_visual.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
