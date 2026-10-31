"""Deterministic counters for per-op work on the validation and capture hot paths.

* The per-layer field invariant probed the trace-level ``_loaded_from_bundle``
  flag once per op through the Trace attribute-miss path; it now probes at
  most once per walk.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._data_classes import trace as trace_mod
from torchlens.validation._invariants_payloads import _check_op_log_fields


def _chain(depth: int) -> torch.nn.Module:
    torch.manual_seed(0)
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        layers.extend((torch.nn.Linear(4, 4), torch.nn.ReLU()))
    return torch.nn.Sequential(*layers).eval()


def _loaded_flag_probes(monkeypatch: pytest.MonkeyPatch, depth: int) -> int:
    trace = tl.trace(_chain(depth), torch.randn(2, 4))
    counts = {"probes": 0}
    original = trace_mod.Trace.__getattr__

    def counting(self: object, name: str) -> object:
        if name == "_loaded_from_bundle":
            counts["probes"] += 1
        return original(self, name)

    monkeypatch.setattr(trace_mod.Trace, "__getattr__", counting)
    _check_op_log_fields(trace)
    monkeypatch.setattr(trace_mod.Trace, "__getattr__", original)
    return counts["probes"]


def test_loaded_flag_probed_at_most_once_per_field_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-layer field invariant probes the loaded flag once, whatever the depth."""

    small = _loaded_flag_probes(monkeypatch, 3)
    large = _loaded_flag_probes(monkeypatch, 9)
    assert small == large == 1, (small, large)
