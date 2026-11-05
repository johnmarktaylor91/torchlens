"""The sibling-view rebind of a view write refuses every shape it cannot prove.

A write through a view of a base rebases every other live view of that base;
capture moves a sibling's op onto the view node autograd regenerates for it over
``CopySlices``. These tests pin the refusals: a sibling a recorded op read before
the write, a sibling of another base, a custom ``autograd.Function`` view, a
sibling the write itself reads, a torch build without the rebind capabilities,
and the multi-output view ops (``chunk``, ``split``, ``unbind``) whose outputs
torch checks when their ``grad_fn`` is read after the base was written. Reading
that ``grad_fn`` during capture must never change the capture's outcome, under
the warnings-as-errors filters too. The last test pins the cost: one scan of the
recorded nodes per write, whatever the number of live siblings.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import TraceCaptureError
from torchlens._utils import _autograd_view_nodes, _torch_compat
from torchlens.backends.torch import _user_grad_fn, param_mutation
from torchlens.errors._validation import MetadataInvariantError
from torchlens.validation._invariants_backward_graph import _check_backward_graph_invariants


class _Write(nn.Module):
    """Run ``write`` on a projection of the input."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.write = write

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return what ``write`` returns for the projected input."""

        return self.write(self.lin(x))


def _input() -> torch.Tensor:
    return torch.randn(3, 8, generator=torch.Generator().manual_seed(0)).requires_grad_(True)


def _strict_trace(write: Callable[[torch.Tensor], torch.Tensor]) -> tl.Trace:
    """Trace and run a managed backward with every warning raised as an error."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trace = tl.trace(_Write(write), _input())
        trace.backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _quiet_trace(write: Callable[[torch.Tensor], torch.Tensor]) -> tl.Trace:
    """Trace and run a managed backward with warnings ignored."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_Write(write), _input())
        trace.backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _backward_verdict(write: Callable[[torch.Tensor], torch.Tensor]) -> object:
    """Return ``tl.validate(scope="backward")``'s verdict, or the invariant error's name."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return tl.validate(_Write(write), _input(), scope="backward")
        except MetadataInvariantError as error:
            return type(error).__name__


def _layers(trace: tl.Trace, func_names: tuple[str, ...]) -> list[tl.Layer]:
    return [layer for layer in trace._layer_list if layer.func_name in func_names]


def _writer(trace: tl.Trace) -> tl.Layer:
    (writer,) = _layers(trace, ("add_",))
    return writer


# --- multi-output view ops: torch checks their outputs on a stale grad_fn read ---

_MULTI_OUTPUT: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, ...]]] = {
    "chunk": lambda y: y.chunk(2, dim=1),
    "split": lambda y: y.split(4, dim=1),
    "unbind": lambda y: y.unbind(dim=1),
}


def _write_beside(kind: str, read_after: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Hold every output of a multi-output view op across a write through another view."""

    def write(y: torch.Tensor) -> torch.Tensor:
        outs = _MULTI_OUTPUT[kind](y)
        y[:, 6:7].add_(0.25)
        return outs[0] * 3.0 if read_after else y * 3.0

    return write


def _write_through_output(kind: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Write in place through one output of a multi-output view op, the others alive."""

    def write(y: torch.Tensor) -> torch.Tensor:
        outs = _MULTI_OUTPUT[kind](y)
        outs[0].add_(0.25)
        return outs[-1] * 3.0

    return write


def _eager_error(write: Callable[[torch.Tensor], torch.Tensor]) -> str | None:
    """Run the model eagerly; return the RuntimeError message it raises, or ``None``."""

    try:
        _Write(write)(_input())
    except RuntimeError as error:
        return str(error)
    return None


_MULTI_OUTPUT_NODES = {"SplitBackward0", "SplitWithSizesBackward0", "UnbindBackward0"}


@pytest.mark.parametrize("kind", sorted(_MULTI_OUTPUT))
def test_multi_output_siblings_unread_after_a_write_capture_under_warnings_as_errors(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outputs held but never read after the write: eager runs, so capture must too.

    Capture reads each overlapping output's ``grad_fn`` when it hooks the
    written base's aliases and when it proves the sibling rebind; torch refuses
    that read for a multi-output view of a written base. Both reads are guarded,
    so the capture neither raises nor warns, the outputs' op keeps its pre-write
    node, and the writer is bound to ``CopySlices``.
    """

    write = _write_beside(kind, read_after=False)
    assert _eager_error(write) is None
    calls = _spy_rebind(monkeypatch)
    trace = _strict_trace(write)
    _check_backward_graph_invariants(trace)
    assert _writer(trace).autograd.grad_fn.class_name == "CopySlices"
    multi = [
        label
        for before, _ in calls
        for label, node in before.items()
        if type(node).__name__ in _MULTI_OUTPUT_NODES
    ]
    assert multi
    assert all(after[label] is before[label] for before, after in calls for label in multi)
    assert _backward_verdict(write) is True


@pytest.mark.parametrize("kind", sorted(_MULTI_OUTPUT))
def test_multi_output_sibling_read_after_a_write_raises_as_eager_does(kind: str) -> None:
    """An output read after the write raises torch's own error, in eager and in capture."""

    write = _write_beside(kind, read_after=True)
    eager = _eager_error(write)
    assert eager is not None and "its base or another view of its base" in eager
    with pytest.raises(TraceCaptureError, match="its base or another view of its base") as info:
        _quiet_trace(write)
    assert isinstance(info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("kind", sorted(_MULTI_OUTPUT))
def test_write_through_a_multi_output_view_raises_as_eager_does(kind: str) -> None:
    """A write through one output raises torch's own error, in eager and in capture."""

    write = _write_through_output(kind)
    eager = _eager_error(write)
    assert eager is not None and "is being modified inplace" in eager
    with pytest.raises(TraceCaptureError, match="is being modified inplace") as info:
        _quiet_trace(write)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_a_warning_or_error_on_the_grad_fn_read_refuses_the_tensor() -> None:
    """A warning on the read is recorded, never shown or raised, and refuses like a raise."""

    class _WarnsOnRead:
        @property
        def grad_fn(self) -> object:
            warnings.warn("stale view", UserWarning, stacklevel=1)
            return object()

    class _RaisesOnRead:
        @property
        def grad_fn(self) -> object:
            raise RuntimeError("stale view")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _user_grad_fn.grad_fn_read_refused(_WarnsOnRead())
        assert _user_grad_fn.grad_fn_read_refused(_RaisesOnRead())
        assert not _user_grad_fn.grad_fn_read_refused(torch.ones(2, requires_grad=True) * 2.0)


def test_the_refusal_check_records_a_read_only_for_a_grad_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tensor autograd cannot regenerate a node for answers ``False`` with no recorded read.

    The check runs on every in-place live hook, so the ordinary path (a non-view
    or a view with no grad history) must not enter ``warnings.catch_warnings``;
    a view of a grad-requiring base still takes the recorded read.
    """

    entered: list[bool] = []
    real_catch = warnings.catch_warnings

    def counting_catch(*args: Any, **kwargs: Any) -> Any:
        entered.append(True)
        return real_catch(*args, **kwargs)

    monkeypatch.setattr(_user_grad_fn.warnings, "catch_warnings", counting_catch)
    base = torch.ones(4, 3)
    assert not _user_grad_fn.grad_fn_read_refused(base)
    assert not _user_grad_fn.grad_fn_read_refused(base[1:])
    assert not _user_grad_fn.grad_fn_read_refused(torch.ones(2, requires_grad=True) * 2.0)
    assert entered == []
    assert not _user_grad_fn.grad_fn_read_refused((torch.ones(4, requires_grad=True) * 2.0)[1:])
    assert entered == [True]


# --- proof refusals for one sibling ---


def _sibling_read_before_write(y: torch.Tensor) -> torch.Tensor:
    """A recorded op reads the sibling before the write; only the reader is used after."""

    sibling = y[:, 4:7]
    early = sibling * 2.0
    y[:, 5:6].add_(0.25)
    return early.sum() + y.sum()


def test_a_sibling_with_a_recorded_reader_keeps_its_node() -> None:
    """The reader keeps the sibling's pre-write node in the graph, so the op stays on it.

    The reader's ``MulBackward0`` points at the sibling's ``SliceBackward0``,
    which backward still visits. Moving the sibling's op onto the regenerated
    node (which no backward visits) would leave the reader's edge on a node no
    op owns and the sibling's op on an unvisited node.
    """

    trace = _quiet_trace(_sibling_read_before_write)
    _check_backward_graph_invariants(trace)
    sibling = _layers(trace, ("__getitem__",))[0]
    (reader,) = _layers(trace, ("__mul__", "mul"))
    assert sibling.autograd.grad_fn.class_name == "SliceBackward0"
    assert sibling.autograd.grad_fn.grad_fn_object_id in reader.autograd.grad_fn.next_grad_fn_ids
    assert trace._grad_fn_logs[sibling.autograd.grad_fn.grad_fn_object_id].has_op
    assert _backward_verdict(_sibling_read_before_write) is True


def _sibling_feeds_the_write(y: torch.Tensor) -> torch.Tensor:
    """The written value is a sibling view of the same base, never read again."""

    sibling = y[:, 4:5]
    y[:, 5:6].add_(sibling)
    return y * 3.0


def test_a_sibling_the_write_reads_keeps_its_node() -> None:
    """``CopySlices`` reads the sibling's pre-write node, so the sibling's op stays on it."""

    trace = _quiet_trace(_sibling_feeds_the_write)
    _check_backward_graph_invariants(trace)
    sibling = _layers(trace, ("__getitem__",))[0]
    writer = _writer(trace)
    assert writer.autograd.grad_fn.class_name == "CopySlices"
    assert sibling.autograd.grad_fn.class_name == "SliceBackward0"
    assert sibling.autograd.grad_fn.grad_fn_object_id in writer.autograd.grad_fn.next_grad_fn_ids
    assert _backward_verdict(_sibling_feeds_the_write) is True


def _sibling_of_another_base(y: torch.Tensor) -> torch.Tensor:
    """Two bases from one node (``var_mean``): write a view of one, read a view of the other."""

    var, mean = torch.var_mean(y, dim=1, keepdim=True)
    sibling = mean[0:2]
    var[1:2].add_(0.25)
    return sibling * 2.0 + var[0:2]


def test_a_view_of_another_base_keeps_its_node() -> None:
    """Both bases' chains reach the same ``VarMeanBackward0``; the other base's view stays."""

    trace = _quiet_trace(_sibling_of_another_base)
    _check_backward_graph_invariants(trace)
    sibling = _layers(trace, ("__getitem__",))[0]
    reader = _layers(trace, ("__mul__", "mul"))[0]
    assert sibling.autograd.grad_fn.class_name == "SliceBackward0"
    assert sibling.autograd.grad_fn.grad_fn_object_id in reader.autograd.grad_fn.next_grad_fn_ids
    assert _backward_verdict(_sibling_of_another_base) is True


def test_a_view_of_another_base_is_not_a_sibling() -> None:
    """The base check refuses a view of another output of the same node."""

    y = torch.randn(3, 8, requires_grad=True) * 1.0
    var, mean = torch.var_mean(y, dim=1, keepdim=True)
    receiver = var[1:2]
    assert not _user_grad_fn._is_other_grad_view_of(mean[0:2], receiver, var)
    assert _user_grad_fn._is_other_grad_view_of(var[0:1], receiver, var)


class _SliceFn(torch.autograd.Function):
    """A custom op whose output is a column view of its input."""

    @staticmethod
    def forward(ctx: Any, t: torch.Tensor) -> torch.Tensor:
        ctx.shape = t.shape
        return t[:, 4:7]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        full = grad.new_zeros(ctx.shape)
        full[:, 4:7] = grad
        return full


def _custom_function_sibling(y: torch.Tensor) -> torch.Tensor:
    """Hold a custom ``Function``'s view across a write through another view of its base."""

    sibling = _SliceFn.apply(y)
    y[:, 5:6].add_(0.25)
    del sibling
    return y * 3.0


def _spy_rebind(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Record the handle index before and after every view-write rebind."""

    rebind = param_mutation.rebind_rebased_view_producers
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def _spy(trace: tl.Trace, receiver: torch.Tensor) -> object:
        handles = trace.capture_events.grad_fn_handles_by_label_raw
        before = dict(handles)
        writer_node = rebind(trace, receiver)
        calls.append((before, dict(handles)))
        return writer_node

    monkeypatch.setattr(param_mutation, "rebind_rebased_view_producers", _spy)
    return calls


def test_a_custom_function_sibling_is_not_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom ``Function``'s view held across the write: capture runs and nothing moves to it.

    Eager runs this model (the custom view is never read after the write).
    Hooking the written base's aliases reads the custom view's ``grad_fn``, which
    torch refuses for a custom-Function view of a written base; the read is
    guarded, so capture completes under warnings-as-errors. No sibling proof
    returns a node and every handle the write changes moves only to ``CopySlices``.
    """

    calls = _spy_rebind(monkeypatch)
    proofs = _user_grad_fn._regenerated_sibling_node
    proven: list[object] = []

    def _spy_proof(*args: Any) -> Any:
        node = proofs(*args)
        proven.append(node)
        return node

    monkeypatch.setattr(_user_grad_fn, "_regenerated_sibling_node", _spy_proof)
    trace = _strict_trace(_custom_function_sibling)
    _check_backward_graph_invariants(trace)
    assert len(calls) == 1
    assert all(node is None for node in proven)
    before, after = calls[0]
    moved = [label for label in before if after[label] is not before[label]]
    assert all(type(after[label]).__name__ == "CopySlices" for label in moved)
    assert _writer(trace).autograd.grad_fn.class_name == "CopySlices"


# --- capability gaps: nothing is proven, so no sibling moves ---

_CAPABILITY_GAPS: dict[str, tuple[bool, bool]] = {
    "no_sequence_nr": (True, False),
    "no_schema_enumeration": (False, True),
    "neither": (True, True),
}


@pytest.fixture(params=sorted(_CAPABILITY_GAPS))
def capability_gap(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """Run as on a torch build without node sequence numbers, schema enumeration or both."""

    sequence_off, schemas_off = _CAPABILITY_GAPS[request.param]
    if sequence_off:
        monkeypatch.setattr(_torch_compat, "HAS_NODE_SEQUENCE_NR", False)
    if schemas_off:
        monkeypatch.setattr(_torch_compat, "get_jit_all_schemas", lambda: None)
    _autograd_view_nodes._aten_view_op_names.cache_clear()
    yield request.param
    _autograd_view_nodes._aten_view_op_names.cache_clear()


def _overlapping_sibling(y: torch.Tensor) -> torch.Tensor:
    sibling = y[:, 4:7]
    y[:, 5:6].add_(0.25)
    return sibling * 3.0


def _disjoint_sibling(y: torch.Tensor) -> torch.Tensor:
    sibling = y[0:1]
    y[2:3].add_(0.25)
    return sibling * 3.0


@pytest.mark.parametrize("write", [_overlapping_sibling, _disjoint_sibling])
def test_without_a_capability_no_sibling_moves(
    capability_gap: str,
    monkeypatch: pytest.MonkeyPatch,
    write: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A missing capability proves no view write, so every handle stays where it was."""

    calls = _spy_rebind(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.trace(_Write(write), _input())
    assert len(calls) == 1
    before, after = calls[0]
    assert after.keys() == before.keys()
    assert all(after[label] is before[label] for label in before)


# --- cost: one scan of the recorded nodes per write ---


def _columns_then_write(count: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Hold ``count`` column views across one write through another column."""

    def write(y: torch.Tensor) -> torch.Tensor:
        columns = [y[:, i : i + 1] for i in range(count)]
        y[:, 6:7].add_(0.25)
        return torch.cat(columns, dim=1) * 3.0

    return write


@pytest.mark.parametrize("count", [1, 4])
def test_a_write_scans_the_recorded_nodes_once_whatever_the_sibling_count(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader index is built once per write and every sibling's proof uses it."""

    index = _user_grad_fn._recorded_reader_index
    proofs = _user_grad_fn._regenerated_sibling_node
    counts = {"scans": 0, "proofs": 0}

    def _counted_index(*args: Any) -> Any:
        counts["scans"] += 1
        return index(*args)

    def _counted_proof(*args: Any) -> Any:
        counts["proofs"] += 1
        return proofs(*args)

    monkeypatch.setattr(_user_grad_fn, "_recorded_reader_index", _counted_index)
    monkeypatch.setattr(_user_grad_fn, "_regenerated_sibling_node", _counted_proof)
    trace = _quiet_trace(_columns_then_write(count))
    _check_backward_graph_invariants(trace)
    assert counts == {"scans": 1, "proofs": count}
