"""A module built inside ``forward`` has attributed parameters.

Its Parameters are created by in-place initializer writes (``kaiming_uniform_`` and
``uniform_`` on freshly allocated storage) inside the captured pass. Those writes are the
Parameters' only origin, so the live Parameter takes the initializer op's label and the
module's later read binds to that op: the initializer ops stay in the graph instead of
being pruned as orphans, the read has provenance, and replay re-derives the weights
(the initializers restore the recorded RNG stream).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _ModuleInForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.late: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.late = nn.Linear(4, 3)
        return self.late(torch.relu(self.pre(x)))


@pytest.mark.smoke
def test_forward_created_module_validates() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_ModuleInForward(), torch.randn(2, 4), scope="forward") is True
