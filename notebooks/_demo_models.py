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


class LoopModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.fc2(x)
            x = nn.functional.relu(x)
        return x


class InnerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loop_module = LoopModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        x = nn.functional.relu(x)
        x = self.loop_module(x)
        return x


class DemoModel(nn.Module):
    """Kitchen-sink demo model: cos op, buffer add, conditional branch, loop module.

    Defined in a real source file so TorchLens's AST inspector can label the
    conditional branch arms (IF / THEN / ELSE) in the rendered graph.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.register_buffer("example_buffer", torch.tensor([1, 2, 3, 4]))
        self.inner_module = InnerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cos(x)
        x = self.fc1(x) + self.example_buffer
        if x.mean() > 0:
            x = self.inner_module(x)
        x = x + torch.rand(x.shape)
        return x
