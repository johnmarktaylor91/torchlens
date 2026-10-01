"""Shared fixtures for the F34 leverage-dividend suites (no tests here).

The toy nets reproduce the leverage panel's measured identity scenarios in
miniature: a reused-module cohort (one ``nn.ReLU`` instance called at several
sites), a VALUE-PRESERVING same-cohort insertion (the extra ``relu`` call on
an already-rectified value), and identity 1x1 convolutions so that EVERY op
output is bit-identical to the input — making the naive-join controls
payload-blind by construction, exactly like the panel's T6 measurement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "ReusedReluNet",
    "OptionalBranchNet",
    "identity_conv",
    "make_insertion_pair",
]


def identity_conv(conv: nn.Conv2d) -> None:
    """Make a 1x1 conv the identity map (payload-blind control substrate)."""

    with torch.no_grad():
        conv.weight.zero_()
        for index in range(conv.weight.shape[0]):
            conv.weight[index, index, 0, 0] = 1.0


class ReusedReluNet(nn.Module):
    """conv -> relu -> conv -> relu with ONE reused relu instance.

    ``extra=True`` inserts a value-preserving third call of the SAME relu
    instance (relu after relu) — the same-cohort insertion the guarded join
    must refuse.
    """

    def __init__(self, extra: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(2, 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.extra = extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        if self.extra:
            x = self.relu(x)
        x = self.relu(self.conv2(x))
        return x


class OptionalBranchNet(nn.Module):
    """A net with a module that only executes when ``use_tail=True``.

    Gives one capture a structural site the other never executes — the
    declared added/removed rows of the differential projection.
    """

    def __init__(self, use_tail: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=False)
        self.tail = nn.Conv2d(2, 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.use_tail = use_tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        if self.use_tail:
            x = self.tail(x)
        return x


def make_insertion_pair() -> tuple[nn.Module, nn.Module, torch.Tensor]:
    """Return (baseline, insertion variant, non-negative input), weights shared.

    Identity convs + non-negative input make every activation bit-identical
    to the input on BOTH nets, so payload equality is blind to the insertion.
    """

    torch.manual_seed(0)
    baseline = ReusedReluNet(extra=False)
    variant = ReusedReluNet(extra=True)
    variant.load_state_dict(baseline.state_dict())
    for net in (baseline, variant):
        identity_conv(net.conv1)
        identity_conv(net.conv2)
    x = torch.rand(1, 2, 3, 3)
    return baseline, variant, x
