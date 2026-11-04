"""A trace of ``model.generate`` names the model and uses its own module addresses.

The bound-method root presents the model as the trace's root ``self``, so the
recorded addresses are the model's own (``body.0``, never ``owner.body.0``):
the same spelling a plain trace of the model gives, in module lookups and in
selectors such as ``tl.in_module``. A plain model with a real submodule called
``owner`` keeps that name. The trace is named after the model class.
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
def test_module_addresses_are_the_models_own_on_a_bound_method_trace(address: str) -> None:
    model = Gen()
    plain = tl.trace(model, torch.randn(1, 4))
    trace = tl.trace(model.generate, torch.randn(1, 4))
    assert address in trace.modules
    assert f"owner.{address}" not in trace.modules
    assert trace.modules[address].num_calls == 3
    assert [str(m.address) for m in trace.modules] == [str(m.address) for m in plain.modules]


def test_in_module_selector_takes_the_bare_address_on_a_bound_method_trace() -> None:
    model = Gen()
    plain = tl.trace(model, torch.randn(1, 4))
    trace = tl.trace(model.generate, torch.randn(1, 4))
    plain_labels = [str(label) for label in plain[tl.in_module("body.0")].labels]
    gen_labels = [str(label) for label in trace[tl.in_module("body.0")].labels]
    assert len(plain_labels) == 1
    assert len(gen_labels) == 3  # one body.0 op per generate step
    assert all(label in trace.modules["body.0"].layer_labels for label in gen_labels)
    try:
        stale = list(trace[tl.in_module("owner.body.0")].labels)
    except LookupError:
        stale = []
    assert stale == []


def test_plain_trace_keeps_a_real_owner_submodule() -> None:
    trace = tl.trace(HasOwner(), torch.randn(1, 4))
    assert "owner.0" in trace.modules
    assert "0" not in trace.modules
    with pytest.raises(KeyError):
        trace.modules["0"]
