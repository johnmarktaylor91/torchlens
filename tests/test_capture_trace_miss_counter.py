"""Deterministic counter: Trace attribute misses must not scale with op count.

Optional session fields (``capture_events``, ``_predicate_save_options``,
``_deferred_gradient_selector``, ``_validation_active``) are absent on a plain
torch capture. Reading them through ``getattr(trace, name, default)`` on every
op paid the Trace miss path (a formatted ``AttributeError``) about 20 times per
layer; the hot sites now read the instance dict. This test pins that the number
of ``Trace.__getattr__`` misses during a capture is independent of the model's
size.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._data_classes import trace as trace_mod


def _chain(depth: int) -> torch.nn.Module:
    torch.manual_seed(0)
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        layers.extend((torch.nn.Linear(4, 4), torch.nn.ReLU()))
    return torch.nn.Sequential(*layers).eval()


def _misses_during_trace(monkeypatch: pytest.MonkeyPatch, depth: int) -> int:
    counts = {"misses": 0}
    original = trace_mod.Trace.__getattr__

    def counting(self: object, name: str) -> object:
        counts["misses"] += 1
        return original(self, name)

    monkeypatch.setattr(trace_mod.Trace, "__getattr__", counting)
    tl.trace(_chain(depth), torch.randn(2, 4))
    return counts["misses"]


def test_trace_attribute_misses_do_not_scale_with_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Twice the layers, the same number of Trace attribute misses."""

    small = _misses_during_trace(monkeypatch, 6)
    large = _misses_during_trace(monkeypatch, 12)
    assert large == small, f"misses grew with op count: {small} -> {large}"
