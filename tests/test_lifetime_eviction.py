"""The lifetime of a trace's saved payloads: who keeps them, and what releases them.

A record you keep a reference to keeps its trace, and with it the saved
activations and arguments. That is what makes the chained read work:
``tl.trace(model, x)["relu_1"].out`` returns the tensor from the plain door,
the ops door and the layers door, although nothing else holds the trace.
Dropping the last record (or door) frees the trace by reference counting, and
``Trace.cleanup()`` is the explicit release while records are still held.

``Trace.cleanup()`` also deletes ``_save_budget_accountant`` with the rest of
the session state.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._record_group_table import record_member


class _StackNet(nn.Module):
    """Forward whose stack call saves a CONTAINER of tensors as an argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x + 1, x + 2]).sum(dim=0)


def _nested_arg_tensor_refs(trace) -> list[weakref.ref]:
    """Weakrefs to every tensor nested in retained arg containers."""

    refs: list[weakref.ref] = []

    def _collect(value) -> None:
        if isinstance(value, torch.Tensor):
            refs.append(weakref.ref(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)

    for op in trace.ops.values():
        for field_name in ("saved_args", "input_activations"):
            _collect(record_member(op, field_name, None))
    return refs


def _stack_trace_and_refs() -> tuple[tl.Trace, list[weakref.ref]]:
    """Capture ``_StackNet`` saving arguments; return the trace and its nested tensors."""

    log = tl.trace(
        _StackNet(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_args=True),
    )
    refs = _nested_arg_tensor_refs(log)
    assert refs, "expected retained arg containers holding tensors"
    return log, refs


def _holds_a_tensor(value) -> bool:
    """Whether ``value`` is a tensor or a builtin container holding one."""

    if isinstance(value, torch.Tensor):
        return True
    return isinstance(value, (list, tuple)) and any(_holds_a_tensor(item) for item in value)


def test_held_record_keeps_nested_payloads_until_it_is_dropped() -> None:
    """A held Op keeps its trace's saved arguments; dropping it frees them without the collector."""

    log, refs = _stack_trace_and_refs()
    trace_ref = weakref.ref(log)
    retained_op = log["stack"]
    del log
    gc.collect()
    assert trace_ref() is not None and retained_op.source_trace is trace_ref()
    assert all(ref() is not None for ref in refs)
    assert _holds_a_tensor(retained_op.call.saved_args)
    gc.collect()
    gc.disable()
    try:
        del retained_op
        survivors = [ref for ref in refs if ref() is not None]
        assert trace_ref() is None, "the trace outlived its last record"
        assert not survivors, f"{len(survivors)}/{len(refs)} nested arg tensors survived"
    finally:
        gc.enable()


def test_cleanup_releases_payloads_while_a_record_is_held() -> None:
    """``cleanup()`` is the explicit release: the payloads go although a record is still held."""

    log, refs = _stack_trace_and_refs()
    retained_op = log["stack"]
    log.cleanup()
    del log
    gc.collect()
    survivors = [ref for ref in refs if ref() is not None]
    assert not survivors, (
        f"{len(survivors)}/{len(refs)} nested arg tensors survived cleanup() "
        "through the retained Op"
    )
    assert retained_op is not None


def test_cleanup_husks_the_save_budget_accountant() -> None:
    """cleanup() must delete the accountant with the rest of the session state."""

    log = tl.trace(nn.Linear(4, 4), torch.randn(1, 4))
    assert "_save_budget_accountant" in log.__dict__
    log.cleanup()
    assert "_save_budget_accountant" not in log.__dict__
    assert "_predicate_lookback_candidates" not in log.__dict__


class _ConvReluNet(nn.Module):
    """Small conv net with one relu op to hold past its Trace."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.conv(x)).flatten(1))


def _relu_op(trace):
    return next(trace[label] for label in trace.op_labels if trace[label].func_name == "relu")


@pytest.mark.parametrize("save", [None, "relu"], ids=["save-all", "save-selected"])
def test_op_out_reads_after_the_trace_variable_is_dropped(save) -> None:
    """A held Op keeps its Trace, so ``.out`` and ``.source_trace`` still read."""

    kwargs = {} if save is None else {"save": tl.func(save)}
    trace = tl.trace(_ConvReluNet(), torch.randn(1, 1, 6, 6), **kwargs)
    op = _relu_op(trace)
    expected = op.out
    assert isinstance(expected, torch.Tensor)
    trace_ref = weakref.ref(trace)
    del trace
    gc.collect()
    assert op.out is expected
    assert op.source_trace is trace_ref()
    assert op.func_name == "relu"
    del op
    assert trace_ref() is None  # freed by reference counting: no collector run


def _relu_label() -> str:
    """Return the label that names the relu of ``_ConvReluNet`` at every door."""

    trace = tl.trace(_ConvReluNet(), torch.randn(1, 1, 6, 6))
    return _relu_op(trace).layer_label


@pytest.mark.parametrize("door", ["plain", "ops", "layers"])
def test_chained_read_returns_the_tensor_from_every_door(door) -> None:
    """``tl.trace(model, x)[label].out`` works in one expression at each door."""

    torch.manual_seed(0)
    model, x = _ConvReluNet(), torch.randn(1, 1, 6, 6)
    expected = torch.relu(model.conv(x))
    label = _relu_label()
    if door == "plain":
        out = tl.trace(model, x)[label].out
    elif door == "ops":
        out = tl.trace(model, x).ops[label].out
    else:
        out = tl.trace(model, x).layers[label].out
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("door", ["ops", "layers"])
def test_a_held_door_keeps_its_trace_and_dropping_it_frees_the_trace(door) -> None:
    """A door is a view like a record: it keeps the trace, by reference counting only."""

    trace = tl.trace(_ConvReluNet(), torch.randn(1, 1, 6, 6))
    trace_ref = weakref.ref(trace)
    held = trace.ops if door == "ops" else trace.layers
    assert (trace.ops if door == "ops" else trace.layers) is held
    del trace
    gc.collect()
    assert trace_ref() is not None
    assert [record.label for record in held]
    gc.disable()
    try:
        del held
        assert trace_ref() is None, "the trace outlived its last door"
    finally:
        gc.enable()


def test_identity_holds_while_a_record_is_held_and_no_longer() -> None:
    """The trace keeps no Op or Layer: a lookup returns the held one, or builds a new one."""

    trace = tl.trace(_ConvReluNet(), torch.randn(1, 1, 6, 6))
    label = _relu_op(trace).layer_label
    op, layer = trace.ops[label], trace.layers[label]
    assert trace.ops[label] is op and next(iter(trace.ops.values())) is not None
    assert trace.layers[label] is layer and trace[label] is layer
    op_ref, layer_ref = weakref.ref(op), weakref.ref(layer)
    del op, layer
    assert op_ref() is None and layer_ref() is None
    assert trace.ops[label].func_name == "relu" and trace.layers[label].func_name == "relu"
