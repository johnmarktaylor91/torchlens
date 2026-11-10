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
from torchlens._data_classes._layer_state import PENDING
from torchlens._data_classes.op import Op
from torchlens._postprocess import _column_passes_finalize, finalization
from torchlens._postprocess._layer_row_scan import RowScan
from torchlens._trace_core.op_store import _MISSING, PooledCell
from torchlens._utils._record_group_table import record_member

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


class _LoopBranch(nn.Module):
    """A shared Linear applied ``steps`` times, then a data-dependent branch."""

    def __init__(self, steps: int = 3) -> None:
        """Build the shared and head layers."""

        super().__init__()
        self.steps = steps
        self.fc = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the loop, the branch and the head."""

        for _ in range(self.steps):
            x = torch.relu(self.fc(x))
        if x.sum() > 0:
            x = x * 2
            x = x + 1
        else:
            x = x - 1
        return self.head(x)


class _PlainLoop(nn.Module):
    """One Linear applied four times at one shape, with no branch."""

    def __init__(self) -> None:
        """Build the shared layer."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the loop."""

        for _ in range(4):
            x = torch.relu(self.cell(x)) + 1
        return x


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
    "long_loop": lambda: (_LoopBranch(steps=40), torch.ones(2, 4)),
    "tied_shrinking": lambda: (_TiedShrinking(), torch.ones(4, 4)),
    "plain_loop": lambda: (_PlainLoop(), torch.ones(2, 4)),
}
#: Models with no conditional event, whose layers stay pending after the trace.
_PLAIN = ("tied_shrinking", "plain_loop")


def _canon(value: Any) -> str:
    """Return an address-free ``repr`` of ``value``."""

    return _ADDRESS.sub("0x", repr(value))


def _unpooled(cell: Any) -> Any:
    """Return the value a stored cell holds, reading through a shared pooled cell."""

    return cell.prototype if cell.__class__ is PooledCell else cell


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
            _canon(_unpooled(store.cell_get(object.__getattribute__(op, "_row"), fid)))
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

    column = _trace_through_pass("long_loop", isolated_patch, facade=False, count=True)
    facade = _trace_through_pass("long_loop", isolated_patch, facade=True, count=True)
    assert column["snapshot"] == facade["snapshot"]
    # Measured on torch 2.13 / py3.10: calls 12358 vs 16296, facade reads 931 vs 3184.
    assert column["calls"] * 20 < facade["calls"] * 17, (column["calls"], facade["calls"])
    assert column["facade_reads"] * 3 < facade["facade_reads"], (
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
    assert any(layer.control_flow.role_stacks != [[]] for layer in trace._layer_logs.values())
    model, x = _MODELS["tied_shrinking"]()
    trace = tl.trace(model, x)
    assert any("tied_parameter_notation" in op.annotations for op in trace._layer_list)
    assert any("varying_across_passes" in layer.annotations for layer in trace._layer_logs.values())


def test_column_reader_matches_getattr_semantics() -> None:
    """The column reader returns what ``getattr(op, name[, default])`` returns."""

    torch.manual_seed(0)
    model, x = _MODELS["loop_branch"]()
    trace = tl.trace(model, x)
    ops = list(trace._layer_list)  # held: a view's id is stable only while it is held
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
        assert read(op, "transformed_out_shape", None) == record_member(
            op, "transformed_out_shape", None
        )
        assert read(op, "no_such_field", None) is None
        assert read(op, "no_such_field", "fallback") == "fallback"
    for op in unset:
        assert read(op, "transformed_out_shape", None) is None


def _trace_plain(model_name: str = "tied_shrinking") -> Any:
    """Trace one of the models with no conditional event."""

    torch.manual_seed(0)
    model, x = _MODELS[model_name]()
    return tl.trace(model, x)


def _pending_labels(trace: Any) -> list[str]:
    """Return the labels of the layers whose state is not built yet."""

    logs = trace._layer_logs
    return [label for label in logs if PENDING in dict.__getitem__(logs, label)]


def test_layers_stay_pending_until_looked_up() -> None:
    """A finished trace holds unbuilt layers; a lookup builds that layer only."""

    trace = _trace_plain()
    pending = _pending_labels(trace)
    assert len(pending) > 1
    layer = trace._layer_logs[pending[0]]
    assert PENDING not in layer.__dict__
    assert _pending_labels(trace) == pending[1:]


def test_many_pass_layer_stays_pending() -> None:
    """A layer whose passes agree keeps its facts and is built on its first lookup."""

    trace = _trace_plain("plain_loop")
    logs = trace._layer_logs
    pending = {label: dict.__getitem__(logs, label) for label in _pending_labels(trace)}
    looped = [label for label, state in pending.items() if len(state["ops"]) == 4]
    assert looped
    layer = logs[looped[0]]
    assert layer.call_labels == [f"{looped[0]}:{number}" for number in (1, 2, 3, 4)]
    assert [op.pass_number for op in layer.ops.values()] == [1, 2, 3, 4]


@pytest.mark.parametrize("model_name", _PLAIN)
def test_first_lookup_order_does_not_change_the_layers(model_name: str) -> None:
    """Layers looked up first-to-last and last-to-first hold the same state."""

    forward, backward = _trace_plain(model_name), _trace_plain(model_name)
    labels = list(forward._layer_logs)
    for label in labels:
        forward._layer_logs[label]
    for label in reversed(labels):
        backward._layer_logs[label]
    assert _snapshot(forward) == _snapshot(backward)


@pytest.mark.parametrize("model_name", _PLAIN)
def test_first_lookup_matches_a_build_inside_the_step(
    model_name: str, isolated_patch: pytest.MonkeyPatch
) -> None:
    """A layer built on its first lookup equals the layer step 15.5 would have built."""

    real_run = RowScan.run

    def run_then_build_all(scan: RowScan) -> None:
        """Run the scan, then ask the step to build every layer itself."""

        real_run(scan)
        scan.eager = True

    isolated_patch.setattr(RowScan, "run", run_then_build_all)
    built_in_step = _trace_plain(model_name)
    isolated_patch.undo()
    assert not _pending_labels(built_in_step)
    on_lookup = _trace_plain(model_name)
    assert _pending_labels(on_lookup)
    assert _snapshot(on_lookup) == _snapshot(built_in_step)
