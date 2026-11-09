"""Smart collapse on a module object registered under several addresses.

One ``nn.ReLU`` object given to three blocks is registered at ``act`` (canonical) and at
``blocks.0.act``, ``blocks.1.act`` and ``blocks.2.act`` (aliases). ``draw(collapse="auto")``
crashed with ``KeyError: 'blocks.0.act'``: the collapse planner listed a parent's children by
their alias addresses while every per-module table is keyed by the canonical address.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl


class _Block(nn.Module):
    def __init__(self, act: nn.Module) -> None:
        super().__init__()
        self.lin1, self.lin2, self.act = nn.Linear(8, 8), nn.Linear(8, 8), act

    def forward(self, x: Tensor) -> Tensor:
        y = self.act(self.lin1(x))
        return self.act(self.lin2(y)) + x


class _SharedActNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList(_Block(self.act) for _ in range(3))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


@pytest.mark.parametrize("collapse", ["auto", "max"])
def test_smart_collapse_draws_a_module_registered_under_several_addresses(
    collapse: str, tmp_path: Path
) -> None:
    """Smart collapse plans the shared module by its canonical address and draws."""

    torch.manual_seed(0)
    trace = tl.trace(_SharedActNet().eval(), torch.randn(2, 8))
    assert trace.modules["blocks.0.act"].address == "act"
    dot = trace.draw(path=str(tmp_path / collapse), file_format="dot", collapse=collapse).dot
    assert "blocks" in dot
