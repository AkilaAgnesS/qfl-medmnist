"""Classical CNN baseline for MedMNIST binary classification.

Small CNN intentionally — we want a fair comparison vs hybrid QNN with
limited parameter budget, not a state-of-the-art classical result.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # 28 -> 14 -> 7
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ClassicalCNN()
    print(f"ClassicalCNN params: {m.num_params:,}")
