"""Demo models for ``torchlens_in_10_minutes.ipynb``.

These live in a real source file so TorchLens can use AST inspection to label
conditional branch arms in the rendered graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TinyBranchCNN(nn.Module):
    """Tiny conv net with one tensor-driven if/else branch.

    Each arm consumes ``x`` directly so the conditional fork at ``input_1`` has
    both the test edge (IF) and the executed body edge (THEN / ELSE) emanating
    from a single node in the rendered graph.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.up_head = nn.Linear(4, 3)
        self.down_head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            y = self.up_head(F.relu(self.conv(x)).mean(dim=(2, 3)))
        else:
            y = self.down_head(F.relu(self.conv(x)).mean(dim=(2, 3)).neg())
        return y


class TinyMLP(nn.Module):
    """MLP with stable layer labels for intervention and bundle demos."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.relu(self.in_proj(x)))
