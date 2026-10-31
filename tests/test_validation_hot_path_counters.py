"""Deterministic counters for per-op work on the validation and capture hot paths.

* The per-layer field invariant probed the trace-level ``_loaded_from_bundle``
  flag once per op through the Trace attribute-miss path; it now probes at
  most once per walk.
* Five per-op helpers executed a function-level import statement on every
  call (about 22,600 per validate of a 1,200-op chain); each now binds its
  module once.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._data_classes import trace as trace_mod
from torchlens._ir import op_record_scatter
from torchlens._utils import tensor_utils
from torchlens.intervention import _module_boundary
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


_HOT_HELPERS = {
    tensor_utils._unwrapped_tensor_size_method.__code__: "_unwrapped_tensor_size_method",
    tensor_utils.materialize_deferred_for_call.__code__: "materialize_deferred_for_call",
    op_record_scatter.scatter_record_to_cells.__code__: "scatter_record_to_cells",
    op_record_scatter._resolve_dtype.__code__: "_resolve_dtype",
    _module_boundary._peek_tensor_live_fire_results.__code__: "_peek_tensor_live_fire_results",
    _module_boundary._record_tensor_live_fire_results.__code__: "_record_tensor_live_fire_results",
}


def test_hot_helpers_execute_no_import_statement(monkeypatch: pytest.MonkeyPatch) -> None:
    """After warm-up, the per-op helpers run no import statement during trace and validate."""

    model, x = _chain(3), torch.randn(2, 4)
    tl.trace(model, x)
    tl.validate(model, x, scope="forward", seed=0, validate_metadata=True)
    counts: dict[str, int] = {}
    real = builtins.__import__

    def counting(*args: Any, **kwargs: Any) -> Any:
        name = _HOT_HELPERS.get(sys._getframe(1).f_code)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
        return real(*args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting)
    tl.trace(model, x)
    tl.validate(model, x, scope="forward", seed=0, validate_metadata=True)
    monkeypatch.setattr(builtins, "__import__", real)
    assert counts == {}
