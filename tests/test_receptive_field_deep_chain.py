"""Receptive-field path walks run without recursion, so deep CNNs do not overflow.

The receptive (reverse) and projective (forward) walks recursed once per hop
(two Python frames per op), so the always-on geometric metadata invariants
raised ``RecursionError`` on a ResNet image encoder (m21309). Both walks now run
on an explicit stack; a 450-block Conv2d + ReLU chain is far past the old limit.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._validation import check_geometric_metadata_invariants

_DEPTH = 450


class _DeepConvChain(nn.Module):
    """``depth`` pointwise convolutions, each followed by a ReLU."""

    def __init__(self, depth: int = _DEPTH) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[layer for _ in range(depth) for layer in (nn.Conv2d(2, 2, 1), nn.ReLU())]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


@pytest.mark.heavy
def test_geometric_invariants_and_queries_survive_a_450_block_chain() -> None:
    """The synthetic trigger: invariants and a receptive-field query on the deep chain."""

    torch.manual_seed(0)
    trace = tl.trace(_DeepConvChain().eval(), torch.randn(1, 2, 3, 3))
    try:
        assert check_geometric_metadata_invariants(trace) is True
        box = trace.output_ops[0].receptive_field.at("center")
        assert not box.empty
    finally:
        trace.cleanup()
