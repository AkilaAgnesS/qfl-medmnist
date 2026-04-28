"""Case study C2 — classical-compressed: a tiny MLP.

The point of this model is to occupy a different point on the SUSQA Pareto
frontier than the larger CNN baseline (C1) and the hybrid quantum model (C3),
so the protocol has three contrasting case studies to differentiate.

Architecture: flatten 28x28 -> hidden(16) -> hidden(8) -> num_classes.
~13k params with hidden=16,8 — about half of the C1 baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalCompressed(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        hidden: tuple[int, ...] = (16, 8),
        image_size: int = 28,
    ) -> None:
        super().__init__()
        flat = in_channels * image_size * image_size
        layers: list[nn.Module] = []
        prev = flat
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ClassicalCompressed()
    print(f"ClassicalCompressed params: {m.num_params:,}")
    dummy = torch.randn(2, 1, 28, 28)
    print(f"Output shape: {m(dummy).shape}")
