"""A trace of ``model.generate`` names the model and takes its bare module addresses.

The bound-method root registers the model as its ``owner`` submodule, so the
recorded addresses read ``owner.body.0``. The trace is still named after the
model class, and ``body.0`` resolves to the same Module a plain trace would
call ``body.0``. A plain model that has a real ``owner`` submodule gets no such
mapping.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).


class Gen(nn.Module):
    """A small model with a ``generate`` loop over its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    def generate(self, x: torch.Tensor, steps: int = 3) -> torch.Tensor:
        for _ in range(steps):
            x = self(x)
        return x


class HasOwner(nn.Module):
    """A plain model whose real submodule is called ``owner``."""

    def __init__(self) -> None:
        super().__init__()
        self.owner = nn.Sequential(nn.Linear(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.owner(x)


def test_bound_method_trace_is_named_after_the_model_class() -> None:
    trace = tl.trace(Gen().generate, torch.randn(1, 4))
    assert re.fullmatch(r"gen_[0-9]+", trace.name), trace.name


@pytest.mark.parametrize("address", ["body", "body.0", "body.1", "head"])
def test_bare_module_address_resolves_on_a_bound_method_trace(address: str) -> None:
    trace = tl.trace(Gen().generate, torch.randn(1, 4))
    assert address in trace.modules
    assert trace.modules[address] is trace.modules[f"owner.{address}"]
    assert trace.modules[address].num_calls == 3


def test_plain_trace_gains_no_owner_mapping() -> None:
    trace = tl.trace(HasOwner(), torch.randn(1, 4))
    assert "owner.0" in trace.modules
    assert "0" not in trace.modules
    with pytest.raises(KeyError):
        trace.modules["0"]
