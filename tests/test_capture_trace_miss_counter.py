"""Deterministic counter: Trace attribute misses must not scale with op count.

Optional session fields (``capture_events``, ``_predicate_save_options``,
``_deferred_gradient_selector``, ``_validation_active``) are absent on a plain
torch capture. Reading them through ``getattr(trace, name, default)`` on every
op paid the Trace miss path (a formatted ``AttributeError``) about 20 times per
layer; the hot sites now read the instance dict. This test pins that the number
of ``Trace.__getattr__`` misses during a capture is independent of the model's
size.

The count is hermetic: a warm-up trace absorbs once-per-process work, pending
garbage is collected before the window and the cyclic collector is off inside
it, and only misses on this thread count, so the result does not depend on
which tests ran earlier in the same worker process.
"""

from __future__ import annotations

import threading

import pytest
import torch
from _hermetic_counters import quiet_gc

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
    owner = threading.get_ident()
    model, x = _chain(depth), torch.randn(2, 4)

    def counting(self: object, name: str) -> object:
        if threading.get_ident() == owner:
            counts["misses"] += 1
        return original(self, name)

    with monkeypatch.context() as patch, quiet_gc():
        patch.setattr(trace_mod.Trace, "__getattr__", counting)
        tl.trace(model, x)
    return counts["misses"]


def test_trace_attribute_misses_do_not_scale_with_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Twice the layers, the same number of Trace attribute misses."""

    tl.trace(_chain(2), torch.randn(2, 4))
    small = _misses_during_trace(monkeypatch, 6)
    large = _misses_during_trace(monkeypatch, 12)
    assert large == small, f"misses grew with op count: {small} -> {large}"
