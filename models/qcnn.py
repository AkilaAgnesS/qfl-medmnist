"""(Stretch goal) Pure Quantum Convolutional Neural Network.

Implements the Cong, Choi & Lukin (2019) QCNN structure: alternating
convolutional and pooling unitaries that reduce qubit count exponentially.
Left as a stub — fill in after the hybrid model is working end-to-end.
"""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


class QCNN(nn.Module):
    def __init__(self, n_qubits: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        raise NotImplementedError(
            "Stretch goal — implement after the hybrid model is producing publishable results."
        )
