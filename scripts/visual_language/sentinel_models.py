"""Fixture models the sentinel strip needs beyond the teaching slides' own.

Copied from the visual audit pack's model files (``notebooks/audit/visual/_visual_models.py``)
or written for the strip where neither file had the combination a row asks for. The deck
never imports from ``notebooks/``; this module is the deck's own copy. Factories follow
``models.FIXTURES``: each returns ``(model, inputs)`` built under seed 0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from scripts.visual_language.models import _seeded


class ResidualNorm(nn.Module):
    """A nested residual block with BatchNorm in eval, a buffer, a constant and a functional op."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.register_buffer("scale", torch.full((3,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.block(x)
        bias = torch.ones(3)  # a constant: not from the input, drawn dashed
        return F.relu(y * self.scale + bias)


class _Sprig(nn.Module):
    """One inception-style block: parallel branches concatenated (audit pack ``InceptionSprig``)."""

    def __init__(self, cin: int, each: int = 2) -> None:
        super().__init__()
        self.b1 = nn.Conv2d(cin, each, 1)
        self.b3 = nn.Sequential(nn.Conv2d(cin, each, 1), nn.Conv2d(each, each, 3, padding=1))
        self.bp = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(cin, each, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.bp(x)], dim=1)


class BranchConv(nn.Module):
    """Conv stem, parallel branches with a concat, then a tensor-driven if/else."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 4, 3, padding=1)
        self.sprig = _Sprig(4)
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.sprig(F.relu(self.stem(x))).mean(dim=(2, 3)))
        if y.sum() > 0:
            return y * 2
        return y - 1


class RNNCellSeq(nn.Module):
    """``nn.RNNCell`` applied over a sequence in a Python loop (audit pack ``RNNCellSeq``)."""

    def __init__(self, steps: int = 3) -> None:
        super().__init__()
        self.steps = steps
        self.cell = nn.RNNCell(6, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.shape[1], 8)
        for t in range(self.steps):
            h = self.cell(x[t], h)
        return self.head(h)


class MiniTransformer(nn.Module):
    """Two TransformerEncoder layers (audit pack ``MiniTransformer``)."""

    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).mean(dim=1))


class MultiOut(nn.Module):
    """Multi-output ops (``max``, ``topk``, ``split``) returned in a dict."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        values, indices = torch.max(x, dim=1)
        top = torch.topk(x, 2, dim=1).values
        left, right = torch.split(x, 2, dim=1)
        return {"max": values + indices, "top": top.sum(1), "parts": left * right}


class NaNMidway(nn.Module):
    """NaN produced in the middle of the graph (audit pack ``NaNMidway``)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.fc1(x))
        y = torch.sqrt(y - 10.0)  # negative operand: NaN from here on
        return self.fc2(y)


class SingleOp(nn.Module):
    """Forward is exactly one op (audit pack ``SingleOp``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class ScalarOut(nn.Module):
    """Output is a 0-dim scalar tensor (audit pack ``ScalarOut``)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).sum()


SENTINEL_FIXTURES: dict[str, Callable[[], tuple[nn.Module, Any]]] = {
    "ResidualNorm": _seeded(lambda: (ResidualNorm().eval(), torch.randn(4, 3))),
    "BranchConv": _seeded(lambda: (BranchConv(), torch.randn(1, 1, 6, 6))),
    "RNNCellSeq": _seeded(lambda: (RNNCellSeq(), torch.randn(3, 1, 6))),
    "MiniTransformer": _seeded(lambda: (MiniTransformer().eval(), torch.randn(1, 4, 16))),
    "MultiOut": _seeded(lambda: (MultiOut(), torch.randn(1, 4))),
    "NaNMidway": _seeded(lambda: (NaNMidway(), torch.randn(1, 4))),
    "SingleOp": _seeded(lambda: (SingleOp(), torch.randn(1, 3))),
    "ScalarOut": _seeded(lambda: (ScalarOut(), torch.randn(1, 4))),
}
