"""U15.1: ``tl.pluck`` honours ``stop_after=`` by passing it to the trace.

Before, ``pluck`` validated nothing and discarded the argument: the forward
always ran to the end and a bad spelling was silently ignored. The site now
rides ``CaptureOptions(stop_after=...)``, so the forward halts after it and an
unsupported spelling refuses typed through the trace's own check.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ArgumentTypeError


class _Counted(nn.Module):
    """Two stages; the second counts how often it runs."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)
        self.second_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.first(x))
        self.second_calls += 1
        return self.second(h)


def test_pluck_stop_after_halts_the_forward() -> None:
    """A valid site halts after it: the later stage never runs."""

    model, x = _Counted(), torch.randn(2, 4)
    out = tl.pluck(model, x, "linear_1_1", stop_after="first")
    assert torch.allclose(out, model.first(x))
    assert model.second_calls == 0

    tl.pluck(model, x, "linear_1_1")
    assert model.second_calls == 1, "without stop_after the full forward runs"


def test_pluck_stop_after_refuses_an_unsupported_spelling() -> None:
    """A bad spelling refuses typed instead of being silently ignored."""

    with pytest.raises(ArgumentTypeError) as info:
        tl.pluck(_Counted(), torch.randn(2, 4), "linear_1_1", stop_after=12345)
    assert info.value.fields["code"] == "stop_after_type_invalid"
