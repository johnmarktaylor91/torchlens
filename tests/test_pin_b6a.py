"""The trace holds no ``Buffer``: its doors build owned Buffers on demand.

A Buffer is a row facade, so the trace's buffer accessor (and each module's
scoped one) keeps the row of each address and builds the Buffer when read. A
Buffer handed out by a built trace keeps that trace, which is what makes the
chained read ``tl.trace(model, x).buffers["bn.running_mean"].final_value``
work, and dropping the last Buffer frees the trace by reference counting.
"""

from __future__ import annotations

import copy
import gc
import pickle
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._op_row_containers import OpSequence
from torchlens._data_classes.buffer import Buffer, BufferAccessor


class _Scale(nn.Module):
    """A module with exactly one buffer, so its address names one Buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("factor", torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _BufferNet(nn.Module):
    """Conv, batch norm (three buffers, two written in training) and a one-buffer module."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.scale = _Scale()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(x))).mean((2, 3))
        return self.fc(self.scale(x))


def _build() -> tl.Trace:
    """Trace ``_BufferNet`` from a fixed seed (each call gives an equal trace)."""

    torch.manual_seed(0)
    model = _BufferNet()
    return tl.trace(model, torch.randn(2, 1, 6, 6))


def _version_label() -> str:
    """Return the label of an op that is a version of a buffer (read from a held trace)."""

    trace = _build()
    return str(trace.buffers["scale.factor"].versions[0].label)


# Every door that reaches a Buffer, read as one expression from the trace.
_DOORS: dict[str, Callable[[tl.Trace], Any]] = {
    "buffers[address]": lambda t: t.buffers["bn.running_mean"],
    "buffers[index]": lambda t: t.buffers[1],
    "buffers[short name]": lambda t: t.buffers["running_var"],
    "buffers[pass-qualified]": lambda t: t.buffers["bn.running_mean:1"],
    "buffers[module address]": lambda t: t.buffers["scale"],
    "buffers[version op label]": lambda t: t.buffers[_LABEL[0]],
    "iter(buffers)": lambda t: next(iter(t.buffers)),
    "buffers.values()": lambda t: t.buffers.values()[2],
    "buffers.items()": lambda t: t.buffers.items()[0][1],
    "buffers.to list": lambda t: list(t.buffers)[-1],
}
_LABEL: list[str] = []


def _summary(value: Any, depth: int = 0) -> Any:
    """Return a comparable summary of an attribute value (records by label or address)."""

    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), value.detach().flatten().tolist())
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    label = getattr(value, "label", None)
    if isinstance(value, tl.Op) and isinstance(label, str):
        return ("op", label)
    if isinstance(value, Buffer):
        return ("buffer", value.address)
    if isinstance(value, tl.Trace):
        return ("trace", len(value.ops))
    if depth < 2 and isinstance(value, (list, tuple)):
        return [_summary(item, depth + 1) for item in value]
    if depth < 2 and isinstance(value, dict):
        return {str(key): _summary(item, depth + 1) for key, item in value.items()}
    return ("type", type(value).__name__)


def _read(record: Any, name: str) -> Any:
    """Read ``name`` from ``record``: a summary of the value, or the exception type."""

    try:
        value = getattr(record, name)
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__)
    if callable(value) and not hasattr(value, "__iter__"):
        return "callable"
    return _summary(value)


def _public_names(record: Any) -> list[str]:
    """Return every public attribute name of ``record``."""

    names = set(dir(type(record)))
    names.update(getattr(record, "__dict__", {}))
    return sorted(name for name in names if not name.startswith("_"))


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_chained_read_matches_the_held_trace_control(door: str) -> None:
    """``door(tl.trace(m, x))`` keeps its trace: every public attribute reads as with it held."""

    if not _LABEL:
        _LABEL.append(_version_label())
    reach = _DOORS[door]
    trace = _build()
    held_record = reach(trace)
    assert isinstance(held_record, Buffer)
    names = _public_names(held_record)
    control = {name: _read(held_record, name) for name in names}
    del held_record, trace
    gc.collect()

    chained = reach(_build())
    gc.collect()
    assert isinstance(chained, Buffer)
    assert isinstance(chained.trace, tl.Trace)
    assert {name: _read(chained, name) for name in names} == control
    assert control["final_value"][0] == "tensor"
    assert control["trace"][0] == "trace"
    assert isinstance(control["versions"], list)


def test_chained_reads_return_the_buffer_values() -> None:
    """The values, versions and write ops read through a chained expression are the live ones."""

    torch.manual_seed(0)
    model = _BufferNet()
    x = torch.randn(2, 1, 6, 6)
    expected = dict(model.named_buffers())
    final = tl.trace(model, x).buffers["scale.factor"].final_value
    assert torch.equal(final, expected["scale.factor"])
    versions = tl.trace(_BufferNet(), x).buffers["bn.running_mean"].versions
    assert isinstance(versions, OpSequence)
    assert all(isinstance(op.source_trace, tl.Trace) for op in versions)
    out = tl.trace(_BufferNet(), x).buffers["bn.running_mean"].versions[-1].out
    assert isinstance(out, torch.Tensor)
    writes = tl.trace(_BufferNet(), x).buffers["bn.running_var"].write_ops
    assert writes and all(isinstance(op.source_trace, tl.Trace) for op in writes)


def _module_buffers(trace: tl.Trace) -> Any:
    """Return the batch norm module's scoped buffer accessor."""

    return trace.modules["bn"].buffers


# What a door hands out, held while the trace variable is dropped.
_HELD: dict[str, Callable[[tl.Trace], Any]] = {
    "trace.buffers": lambda t: t.buffers,
    "a Buffer": lambda t: t.buffers["bn.running_var"],
    "module.buffers": _module_buffers,
    "a module's Buffer": lambda t: _module_buffers(t)["bn.num_batches_tracked"],
    "buffer.versions": lambda t: t.buffers["bn.running_mean"].versions,
}


@pytest.mark.parametrize("door", sorted(_HELD))
def test_a_held_door_keeps_the_trace_and_dropping_it_frees_the_trace(door: str) -> None:
    """What a buffer door hands out keeps the trace, and its drop frees it without the collector."""

    trace = _build()
    trace_ref = weakref.ref(trace)
    held = _HELD[door](trace)
    del trace
    gc.collect()
    assert trace_ref() is not None, f"{door} did not keep its trace"
    if isinstance(held, BufferAccessor):
        assert all(buffer.trace is trace_ref() for buffer in held)
    gc.disable()
    try:
        del held
        assert trace_ref() is None, f"the trace outlived {door}"
    finally:
        gc.enable()


def test_a_trace_nobody_holds_is_freed_after_its_buffers_were_read() -> None:
    """Reading every buffer door and dropping the results leaves no reference to the trace."""

    trace = _build()
    trace_ref = weakref.ref(trace)
    for buffer in trace.buffers:
        read = buffer.final_value, buffer.versions, buffer.trace
    read = trace.modules["bn"].buffers["bn.running_mean"].write_ops
    gc.disable()
    try:
        del trace, buffer, read
        assert trace_ref() is None, "a buffer door left the trace held"
    finally:
        gc.enable()


def test_one_buffer_is_one_object_across_doors() -> None:
    """Every door names the same Buffer object while one is held."""

    trace = _build()
    buffer = trace.buffers["bn.running_mean"]
    assert trace.buffers["bn.running_mean"] is buffer
    assert trace.buffers[list(trace.buffers.keys()).index("bn.running_mean")] is buffer
    assert trace.buffers["running_mean"] is buffer
    assert trace.modules["bn"].buffers["bn.running_mean"] is buffer
    assert trace.modules["self"].buffers["bn.running_mean"] is buffer
    assert [b for b in trace.buffers if b.address == "bn.running_mean"] == [buffer]
    assert trace.buffers["scale"] is trace.modules["scale"].buffers[0]


def test_the_trace_holds_no_buffer_object() -> None:
    """The stored accessor names rows; a Buffer read and dropped is gone, and read again equal."""

    trace = _build()
    stored = trace.__dict__["_buffer_accessor"]
    assert "_dict" not in stored.__dict__ and "_list" not in stored.__dict__
    first = trace.buffers["bn.running_mean"]
    first_ref = weakref.ref(first)
    address, final = first.address, first.final_value
    del first
    assert first_ref() is None
    again = trace.buffers["bn.running_mean"]
    assert again.address == address
    assert torch.equal(again.final_value, final)


def test_a_copied_or_restored_buffer_names_no_trace() -> None:
    """A Buffer copied or restored on its own names no trace; a restored one keeps none alive.

    A shallow or deep copy carries the version ops as a plain list, and those
    ops keep their trace, as they do without the pin.
    """

    trace = _build()
    trace_ref = weakref.ref(trace)
    buffer = trace.buffers["bn.running_mean"]
    restored = pickle.loads(pickle.dumps(buffer))
    copied = copy.copy(buffer)
    deep = copy.deepcopy(buffer)
    for other in (restored, copied, deep):
        assert other is not buffer
        assert other.trace is None
        assert other.address == buffer.address
        assert not hasattr(other, "_owner")
    assert type(buffer.__getstate__()["versions"]) is list
    del trace, buffer, copied, deep
    gc.collect()
    assert trace_ref() is None, "a restored Buffer kept the trace"
    assert restored.trace is None


def test_an_accessor_pickles_as_one_that_holds_its_buffers() -> None:
    """The keyed accessor's pickle is byte-identical to that of an accessor built from the records."""

    trace = _build()
    keyed = trace.buffers
    records = dict(keyed.items())
    direct = BufferAccessor(records, source_trace=trace)
    assert pickle.dumps(keyed) == pickle.dumps(direct)
    assert keyed.__getstate__().keys() == direct.__getstate__().keys()
    restored = pickle.loads(pickle.dumps(keyed))
    assert list(restored.keys()) == list(keyed.keys())
    assert all(buffer.trace is None for buffer in restored)


def test_a_fork_reads_its_own_buffers() -> None:
    """A fork's buffers name the fork; the parent's still name the parent."""

    trace = _build()
    fork = trace.fork()
    parent_buffer = trace.buffers["bn.running_mean"]
    fork_buffer = fork.buffers["bn.running_mean"]
    assert fork_buffer is not parent_buffer
    assert fork_buffer.trace is fork
    assert parent_buffer.trace is trace
    assert list(fork.buffers.keys()) == list(trace.buffers.keys())
    for address in trace.buffers.keys():
        assert torch.equal(fork.buffers[address].final_value, trace.buffers[address].final_value)
    assert [op.label for op in fork_buffer.versions] == [op.label for op in parent_buffer.versions]
    assert all(op.source_trace is fork for op in fork_buffer.versions)
    fork_ref = weakref.ref(fork)
    held = fork.buffers["bn.running_var"]
    del fork, fork_buffer
    gc.collect()
    assert fork_ref() is not None
    assert held.trace is fork_ref()
