"""Hybrid CNN feature extractor -> Variational Quantum Classifier.

Pipeline:
    image (1x28x28)
    -> small CNN -> N-dim feature vector
    -> linear bottleneck to `n_qubits` features
    -> angle encoding
    -> N variational layers (RY/RZ + entangling CNOT ring)
    -> Pauli-Z expectation values per qubit
    -> linear readout to logits

Implemented with PennyLane + PyTorch. Use the `default.qubit` simulator for
noise-free runs; switch to `default.mixed` with depolarizing noise for the
robustness study.
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_qnode(n_qubits: int, n_layers: int, noise_p: float = 0.0):
    """Build a PennyLane QNode + return TorchLayer-ready function."""
    if noise_p > 0:
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle encoding via AngleEmbedding — handles both unbatched (n_qubits,)
        # and batched (batch, n_qubits) inputs from TorchLayer correctly.
        # (Manual `qml.RY(inputs[i], ...)` indexing breaks for batched inputs because
        # inputs[i] then returns the i-th batch row, not the i-th feature.)
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # variational ansatz
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            if noise_p > 0:
                for i in range(n_qubits):
                    qml.DepolarizingChannel(noise_p, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return circuit, weight_shapes


class HybridQNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        n_qubits: int = 8,
        n_layers: int = 2,
        noise_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # CNN feature extractor (smaller than classical baseline; we want few params before the quantum head)
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Linear(8 * 7 * 7, n_qubits)

        circuit, weight_shapes = make_qnode(n_qubits, n_layers, noise_p)
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.readout = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.tanh(self.bottleneck(x)) * math.pi  # bound features to [-pi, pi] for angle encoding
        x = self.qlayer(x)
        return self.readout(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = HybridQNN(n_qubits=8, n_layers=2)
    dummy = torch.randn(2, 1, 28, 28)
    out = m(dummy)
    print(f"HybridQNN params: {m.num_params:,}, output shape: {out.shape}")
