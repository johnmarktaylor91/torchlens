"""Step 15.5 (``_build_layer_logs``) over ``Op`` column views.

Two contracts. The counter contract: on a fixed trace, the column body of the
pass makes far fewer Python calls, and far fewer ``Op`` facade reads, than the
facade body it replaces. The semantic contract: both bodies leave identical
aggregate ``Layer`` records, per-op stored cells and autograd total, on models
that exercise every merge rule (multi-pass layers, a conditional, tied
parameters, a varying-shape recurrence the multi-pass reconcile marks).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import Op
from torchlens._postprocess import _column_passes_finalize, finalization
from torchlens._trace_core.op_store import _MISSING

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


class _LoopBranch(nn.Module):
    """A shared Linear applied three times, then a data-dependent branch."""

    def __init__(self) -> None:
        """Build the shared and head layers."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the loop, the branch and the head."""

        for _ in range(3):
            x = torch.relu(self.fc(x))
        if x.sum() > 0:
            x = x * 2
        else:
            x = x - 1
        return self.head(x)


class _TiedShrinking(nn.Module):
    """Tied input/output weights and a recurrence over a shrinking sequence."""

    def __init__(self) -> None:
        """Build the tied pair and the recurrent cell."""

        super().__init__()
        self.enc = nn.Linear(4, 4, bias=False)
        self.dec = nn.Linear(4, 4, bias=False)
        self.dec.weight = self.enc.weight
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, step the cell over 4, 3 and 2 rows, decode."""

        h = self.enc(x)
        for n in (4, 3, 2):
            h = torch.tanh(self.cell(h[:n]))
        return self.dec(h)


_MODELS: dict[str, Callable[[], tuple[nn.Module, torch.Tensor]]] = {
    "loop_branch": lambda: (_LoopBranch(), torch.ones(2, 4)),
    "tied_shrinking": lambda: (_TiedShrinking(), torch.ones(4, 4)),
}


def _canon(value: Any) -> str:
    """Return an address-free ``repr`` of ``value``."""

    return _ADDRESS.sub("0x", repr(value))


def _snapshot(trace: Any) -> dict[str, Any]:
    """Return the state step 15.5 builds: layers, the per-op cells, the autograd total."""

    layers = {}
    for label, layer in trace._layer_logs.items():
        state = {}
        for key, value in layer.__dict__.items():
            if key == "_source_trace_ref":
                continue
            if key == "ops":
                state[key] = [(number, op.label) for number, op in value.items()]
            else:
                state[key] = _canon(value)
        layers[label] = state
    store = object.__getattribute__(trace._layer_list[0], "_core")
    cells = {}
    for name in ("is_constant", "annotations"):
        fid = store.layout.fid_by_name[name]
        cells[name] = [
            _canon(store.cell_get(object.__getattribute__(op, "_row"), fid))
            for op in trace._layer_list
        ]
    return {
        "layers": layers,
        "order": list(trace._layer_logs),
        "cells": cells,
        "autograd": _canon(trace._cost_total_autograd_memory),
    }


def _python_calls(fn: Callable[[], Any]) -> tuple[int, int]:
    """Return ``(Python call events, Op.__getattribute__ entries)`` that ``fn`` makes."""

    getattribute_code = Op.__getattribute__.__code__
    counts = [0, 0]

    def profiler(frame: Any, event: str, arg: Any) -> None:
        """Count Python function entries and facade attribute reads."""

        if event == "call":
            counts[0] += 1
            if frame.f_code is getattribute_code:
                counts[1] += 1

    sys.setprofile(profiler)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return counts[0] - 1, counts[1]


def _trace_through_pass(
    model_name: str, monkeypatch: pytest.MonkeyPatch, *, facade: bool, count: bool
) -> dict[str, Any]:
    """Trace one model, hooking step 15.5; return its snapshot (and call counts)."""

    if facade:
        monkeypatch.setattr(_column_passes_finalize, "column_rows_by_label", lambda _ops: None)
    real_pass = finalization._build_layer_logs
    seen: dict[str, Any] = {}

    def hooked(trace_arg: Any) -> None:
        """Run the real pass (counted when asked) and snapshot its result."""

        if count:
            seen["calls"], seen["facade_reads"] = _python_calls(lambda: real_pass(trace_arg))
        else:
            real_pass(trace_arg)
        seen["snapshot"] = _snapshot(trace_arg)

    monkeypatch.setattr(finalization, "_build_layer_logs", hooked)
    torch.manual_seed(0)
    model, x = _MODELS[model_name]()
    tl.trace(model, x)
    monkeypatch.undo()
    return seen


@pytest.fixture
def isolated_patch() -> Iterator[pytest.MonkeyPatch]:
    """Yield a private monkeypatch context that is always undone."""

    patch = pytest.MonkeyPatch()
    try:
        yield patch
    finally:
        patch.undo()


def test_column_pass_cuts_python_calls_and_facade_reads(isolated_patch: pytest.MonkeyPatch) -> None:
    """The column body makes far fewer calls and facade reads than the facade body."""

    column = _trace_through_pass("loop_branch", isolated_patch, facade=False, count=True)
    facade = _trace_through_pass("loop_branch", isolated_patch, facade=True, count=True)
    assert column["snapshot"] == facade["snapshot"]
    assert column["calls"] * 10 < facade["calls"] * 7, (column["calls"], facade["calls"])
    assert column["facade_reads"] * 2 < facade["facade_reads"], (
        column["facade_reads"],
        facade["facade_reads"],
    )


@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_column_pass_matches_facade_pass(
    model_name: str, isolated_patch: pytest.MonkeyPatch
) -> None:
    """Column and facade bodies build identical layers, cells and autograd total."""

    column = _trace_through_pass(model_name, isolated_patch, facade=False, count=False)
    facade = _trace_through_pass(model_name, isolated_patch, facade=True, count=False)
    assert column["snapshot"] == facade["snapshot"]


def test_merge_rules_are_exercised() -> None:
    """The fixed models hit multi-pass merges, a conditional, tying and a varying shape."""

    torch.manual_seed(0)
    model, x = _MODELS["loop_branch"]()
    trace = tl.trace(model, x)
    assert any(len(layer.ops) == 3 for layer in trace._layer_logs.values())
    assert any(layer.conditional_role_stacks != [[]] for layer in trace._layer_logs.values())
    model, x = _MODELS["tied_shrinking"]()
    trace = tl.trace(model, x)
    assert any("tied_parameter_notation" in op.annotations for op in trace._layer_list)
    assert any("varying_across_passes" in layer.annotations for layer in trace._layer_logs.values())


def test_column_reader_matches_getattr_semantics() -> None:
    """The column reader returns what ``getattr(op, name[, default])`` returns."""

    torch.manual_seed(0)
    model, x = _MODELS["loop_branch"]()
    trace = tl.trace(model, x)
    ops = trace._layer_list
    lookup = _column_passes_finalize.column_rows_by_label(dict(enumerate(ops)))
    assert lookup is not None
    columns, rows_by_index = lookup
    row_of = {id(op): rows_by_index[index] for index, op in enumerate(ops)}
    get = {"label": columns.getter("label")}
    read = _column_passes_finalize._column_reader(columns, get, row_of)
    store = columns.store
    fid = store.layout.fid_by_name["transformed_out_shape"]
    unset = [op for op in ops if store.cell_get(row_of[id(op)], fid) is _MISSING]
    for op in ops:
        assert read(op, "label") == op.label
        assert read(op, "shape", None) == getattr(op, "shape", None)
        assert read(op, "modules", None) == getattr(op, "modules", None)
        assert read(op, "transformed_out_shape", None) == getattr(op, "transformed_out_shape", None)
        assert read(op, "no_such_field", None) is None
        assert read(op, "no_such_field", "fallback") == "fallback"
    for op in unset:
        assert read(op, "transformed_out_shape", None) is None
