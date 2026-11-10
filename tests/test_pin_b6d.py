"""A held ``GradFn`` or ``GradFnCall`` keeps its trace; a trace holds none of them.

The trace keeps the state of its backward records and builds a record on a
read. A record read from the trace holds the trace, so a read chained off a
capture in one expression returns what the same read returns with the trace
in a variable. A record copied or restored on its own names no trace.
"""

from __future__ import annotations

import copy
import gc
import pickle
import weakref
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from benchmarks.capture_core_counters import retained_after
from torchlens._data_classes import grad_fn_call as grad_fn_call_module
from torchlens._data_classes.backward_pass import BackwardPass
from torchlens._data_classes.grad_fn import GradFn
from torchlens._data_classes.grad_fn_call import GradFnCall

_KINDS = (GradFn, GradFnCall, BackwardPass)
_LAST: list[Any] = []

#: Integer attributes that say where a record sits in its trace (the same in every capture).
_POSITIONS = frozenset(
    {
        "backward_pass_number",
        "call_index",
        "order",
        "ordinal",
        "ordinal_index",
        "pass_number",
        "step_index",
        "type_index",
    }
)


class _Net(nn.Module):
    """A small model with a parameterised module, a buffer and a function call."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layers."""

        x = torch.relu(self.bn(self.conv(x))).mean((2, 3))
        return self.fc(x)


def _trace(passes: int = 1, read_between: bool = False) -> Any:
    """Capture ``_Net`` and ``passes`` backward passes; return the trace.

    With ``read_between`` the backward records are read after every pass, so
    each pass is projected on its own (two passes give two projections).
    """

    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 1, 6, 6, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
    for _ in range(passes):
        loss = trace[trace.output_layers[0]].out.square().mean()
        trace.backward(loss, retain_graph=True)
        if read_between:
            assert len(trace.autograd.grad_fn_calls) > 0
    _LAST[:] = [weakref.ref(trace)]
    return trace


def _two_epochs() -> Any:
    """Capture two backward passes, each projected on its own."""

    return _trace(passes=2, read_between=True)


def _census() -> Counter[str]:
    """Count the live backward records, the way the C4 census counts record kinds."""

    gc.collect()
    gc.collect()
    return Counter(type(obj).__name__ for obj in gc.get_objects() if type(obj) in _KINDS)


def _held(build: Callable[[], Any]) -> tuple[Any, dict[str, int]]:
    """Run ``build``; return its result and how many backward records it keeps alive."""

    before = _census()
    result = build()
    after = _census()
    return result, {kind.__name__: after[kind.__name__] - before[kind.__name__] for kind in _KINDS}


_NONE_HELD = {kind.__name__: 0 for kind in _KINDS}

#: Every door that hands out a ``GradFn`` or a ``GradFnCall``, as one expression on a trace.
DOORS: dict[str, Callable[[Any], Any]] = {
    "autograd.grad_fns[0]": lambda t: t.autograd.grad_fns[0],
    "autograd.grad_fns[op label]": lambda t: t.autograd.grad_fns["relu_1"],
    "autograd.grad_fns iteration": lambda t: next(iter(t.autograd.grad_fns)),
    "autograd.saved_grad_fns[0]": lambda t: t.autograd.saved_grad_fns[0],
    "autograd.grad_fn_calls[0]": lambda t: t.autograd.grad_fn_calls[0],
    "autograd.saved_grad_fn_calls[0]": lambda t: t.autograd.saved_grad_fn_calls[0],
    "op.autograd.grad_fn": lambda t: t["relu_1"].autograd.grad_fn,
    "layer.autograd.grad_fn": lambda t: t.layers["relu_1"].autograd.grad_fn,
    "layer.ops[0].autograd.grad_fn": lambda t: t.layers["relu_1"].ops[0].autograd.grad_fn,
    "module address, output op": lambda t: t["fc"].autograd.grad_fn,
    "module[layer].autograd.grad_fn": lambda t: t.modules["fc"]["linear_1"].autograd.grad_fn,
    "grad_fn.calls[0]": lambda t: t.autograd.grad_fns[0].calls[0],
    "grad_fn.calls.for_pass(1)": lambda t: t.autograd.grad_fns[0].calls.for_pass(1),
    "grad_fn.calls.values()": lambda t: t.autograd.grad_fns[0].calls.values()[0],
    "backward_passes[0].grad_fn_calls[0]": lambda t: t.autograd.backward_passes[0].grad_fn_calls[0],
    "last_backward_pass.grad_fn_calls[0]": lambda t: t.autograd.last_backward_pass.grad_fn_calls[0],
}


def _names(record: Any) -> list[str]:
    """Return the public attribute names of ``record``."""

    names = set(dir(type(record)))
    names.update(getattr(record, "__dict__", {}))
    return sorted(name for name in names if not name.startswith("_"))


def _outcome(record: Any, name: str) -> Any:
    """Describe what reading ``record.<name>`` gives, in a form two captures share."""

    try:
        value = getattr(record, name)
    except Exception as exc:  # noqa: BLE001 -- the outcome of the read is what is compared
        return f"raises {type(exc).__name__}"
    if isinstance(value, (str, bool, type(None))):
        return value
    if isinstance(value, int) and name in _POSITIONS:
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return (type(value).__name__, *value)
    if callable(value) and not hasattr(value, "__iter__"):
        return "callable"
    size = len(value) if hasattr(value, "__len__") and not isinstance(value, torch.Tensor) else None
    return f"{type(value).__name__} of {size}"


def _outcomes(record: Any, names: list[str]) -> dict[str, Any]:
    """Describe every read in ``names`` of ``record``."""

    return {name: _outcome(record, name) for name in names}


@pytest.mark.parametrize("door", sorted(DOORS))
def test_chained_read_returns_what_a_held_trace_returns(door: str) -> None:
    """A record read in one expression, with no variable holding the trace, reads the same."""

    reach = DOORS[door]
    held_trace = _trace()
    control = reach(held_trace)
    assert type(control) in (GradFn, GradFnCall)
    assert control.trace is held_trace
    names = _names(control)
    expected = _outcomes(control, names)
    del control, held_trace

    record = reach(_trace())

    trace = _LAST[0]()
    assert trace is not None, "the record keeps its trace"
    assert record.trace is trace
    del trace
    assert _outcomes(record, names) == expected


def test_a_door_hands_out_one_record_per_state() -> None:
    """Two doors to the same record return the identical object while it is held."""

    trace = _trace()
    grad_fn = trace.autograd.grad_fns["relu_1"]
    assert trace.autograd.grad_fns["relu_1"] is grad_fn
    assert trace["relu_1"].autograd.grad_fn is grad_fn
    assert trace.layers["relu_1"].autograd.grad_fn is grad_fn
    call = grad_fn.calls[0]
    assert grad_fn.calls[0] is call
    assert any(found is call for found in trace.autograd.grad_fn_calls)
    assert any(found is call for found in trace.autograd.backward_passes[0].grad_fn_calls)
    assert grad_fn.calls is grad_fn.calls


def test_a_record_written_through_one_door_reads_through_another() -> None:
    """A record is a view: an attribute set on it is there on the next read of the trace."""

    trace = _trace()
    trace.autograd.grad_fns[0].note = "kept"
    trace.autograd.grad_fn_calls[0].note = "kept too"
    assert trace.autograd.grad_fns[0].note == "kept"
    assert trace.autograd.grad_fn_calls[0].note == "kept too"


@pytest.mark.parametrize(
    "remake", [copy.copy, copy.deepcopy, lambda r: pickle.loads(pickle.dumps(r))]
)
def test_a_record_copied_or_restored_alone_names_no_trace(remake: Callable[[Any], Any]) -> None:
    """A copy of a record is detached: it has no trace and does not keep the original's."""

    trace = _trace()
    grad_fn = remake(trace.autograd.grad_fns[0])
    call = remake(trace.autograd.grad_fn_calls[0])
    assert grad_fn.trace is None
    assert grad_fn.handle is None
    assert call.trace is None
    assert call.ordinal_index == -1
    assert [inner.trace for inner in grad_fn.calls.values()] == [None] * len(grad_fn.calls)
    del trace
    assert _LAST[0]() is None, "a detached copy does not keep the trace it was copied from"
    assert grad_fn.trace is None and call.trace is None


@pytest.mark.parametrize("build", [_trace, _two_epochs], ids=["one_backward", "two_epochs"])
def test_a_trace_holds_no_backward_record(build: Callable[[], Any]) -> None:
    """C4 for the backward kinds: a captured trace nobody reads keeps no record object."""

    trace, retained = retained_after(build)
    assert retained["kinds"]["GradFn"] == 0
    del trace
    trace, held = _held(build)
    assert held == _NONE_HELD
    assert len(trace.autograd.grad_fns) > 0 and len(trace.autograd.grad_fn_calls) > 0


@pytest.mark.parametrize("build", [_trace, _two_epochs], ids=["one_backward", "two_epochs"])
def test_reading_every_door_leaves_no_record_behind(build: Callable[[], Any]) -> None:
    """Records built for a read are dropped with the reader's last reference."""

    def read() -> Any:
        trace = build()
        for reach in DOORS.values():
            record = reach(trace)
            assert record.trace is trace
            _outcomes(record, _names(record))
        assert [call.ordinal_index for call in trace.autograd.grad_fn_calls] == list(
            range(len(trace.autograd.grad_fn_calls))
        )
        return trace

    trace, held = _held(read)
    assert held == _NONE_HELD
    del trace


def _restored() -> Any:
    return pickle.loads(pickle.dumps(_trace()))


def _deep_copied() -> Any:
    return copy.deepcopy(_trace())


def _forked() -> Any:
    return _trace().fork()


def _forked_with_parent() -> Any:
    parent = _trace()
    return parent, parent.fork()


@pytest.mark.parametrize(
    "build",
    [_restored, _deep_copied, _forked, _forked_with_parent],
    ids=["pickled", "deepcopied", "forked", "forked_parent_held"],
)
def test_a_copy_of_a_trace_holds_no_backward_record(build: Callable[[], Any]) -> None:
    """A restored, copied or forked trace holds state too, and its records name it."""

    result, held = _held(build)
    assert held == _NONE_HELD
    trace = result[1] if isinstance(result, tuple) else result
    grad_fn = trace.autograd.grad_fns["relu_1"]
    assert grad_fn.trace is trace
    assert trace["relu_1"].autograd.grad_fn is grad_fn
    assert trace.layers["relu_1"].autograd.grad_fn is grad_fn
    call = grad_fn.calls[0]
    assert call.trace is trace
    assert call.ordinal_index >= 0
    assert trace.autograd.backward_passes[0].grad_fn_calls[0].trace is trace


def test_a_fork_reads_its_own_grad_fn_through_an_op() -> None:
    """A fork's op names the fork's ``GradFn``, never its parent's."""

    parent = _trace()
    fork = parent.fork()
    theirs = parent["relu_1"].autograd.grad_fn
    mine = fork["relu_1"].autograd.grad_fn
    assert mine is not theirs
    assert mine.trace is fork and theirs.trace is parent
    assert mine is fork.autograd.grad_fns["relu_1"]
    assert mine.label == theirs.label


def test_pickle_state_carries_plain_containers() -> None:
    """The saved state of a trace holds plain containers of records, as it always did."""

    trace = _trace()
    state = trace.__getstate__()
    logs = state["_grad_fn_logs"]
    assert type(logs).__name__ == "OrderedDict"
    assert all(type(record) is GradFn for record in logs.values())
    passes = state["_backward_pass_logs"]
    assert type(passes).__name__ == "OrderedDict"
    assert all(type(record) is BackwardPass for record in passes.values())
    record = next(iter(logs.values()))
    calls = record.calls.__getstate__()
    assert type(calls["_dict"]) is dict and type(calls["_list"]) is list
    assert all(type(call) is GradFnCall for call in calls["_list"])
    pass_state = dict(next(iter(passes.values())).__tl_state_items__())
    assert type(pass_state["grad_fn_calls"]) is list


def test_a_dropped_trace_is_freed_by_reference_counting() -> None:
    """With no record held, dropping the trace frees it at once, with the collector off."""

    gc.collect()
    gc.disable()
    try:
        trace = _trace()
        for reach in DOORS.values():
            reach(trace)
        assert trace.autograd.grad_fn_calls[0].ordinal_index == 0
        del trace
        assert _LAST[0]() is None
    finally:
        gc.enable()


def test_a_held_record_keeps_the_trace_until_it_is_dropped() -> None:
    """The record holds the trace one way: dropping the record frees the trace."""

    gc.collect()
    gc.disable()
    try:
        call = _trace().autograd.grad_fns[0].calls[0]
        assert call.trace is _LAST[0]()
        del call
        assert _LAST[0]() is None
    finally:
        gc.enable()


def test_ordinal_index_ignores_a_call_whose_id_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call that is not in the projection reads ``-1`` even when its ``id`` matches one that is.

    ``id`` values are reused once an object dies, so a cache keyed on ``id``
    alone can answer for the wrong call. The reuse is forced here: ``id`` as
    ``grad_fn_call`` sees it gives a foreign call the key of a real one.
    """

    trace = _trace()
    real = trace.autograd.grad_fn_calls[0]
    assert real.ordinal_index == 0
    foreign = pickle.loads(pickle.dumps(real))
    foreign._source_trace = trace

    def reused(obj: Any) -> int:
        if obj is foreign:
            return id(real)
        if obj is foreign.__dict__:
            return id(real.__dict__)
        return id(obj)

    monkeypatch.setattr(grad_fn_call_module, "id", reused, raising=False)
    assert foreign.ordinal_index == -1
    assert real.ordinal_index == 0
