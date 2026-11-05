"""An in-place write through a view keeps every op's backward node in the recorded graph.

Autograd's ``rebase_history`` gives the base of a written view a new ``CopySlices``
node and drops the view's own node from the graph (for a view of a view, every
intermediate view node too). The op that produced the view recorded that dropped
node, so no backward pass ever reached it and the metadata invariant reported a
missing GradFn backpointer, inside an ``autograd.grad`` region and after a managed
backward alike. Capture now moves such an op onto the ``CopySlices`` node that took
its place. The invariant itself is unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import param_mutation
from torchlens.backends.torch._user_grad_fn import _orphaned_view_nodes
from torchlens.errors._validation import MetadataInvariantError
from torchlens.validation._invariants_backward_graph import _check_backward_graph_invariants


def _view_used(y: torch.Tensor) -> torch.Tensor:
    """Write a column view in place, then read the view."""

    return y[:, 5:6].add_(0.25)


def _view_of_view(y: torch.Tensor) -> torch.Tensor:
    """Write a view of a view in place."""

    return y[:, 2:8][:, 3:4].mul_(1.5)


def _chained(y: torch.Tensor) -> torch.Tensor:
    """Write the same view twice."""

    col = y[:, 5:6]
    col.add_(0.25)
    return col.mul_(3.0)


_WRITES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "view": _view_used,
    "view_of_view": _view_of_view,
    "chained": _chained,
}


class _GradRegion(nn.Module):
    """Write a column view of an intermediate inside an ``autograd.grad`` region."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Linear(1, 8)
        self.write = write

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an energy and its gradient with respect to ``x``."""

        x = x.detach().clone().requires_grad_(True)
        col = self.write(x * 2.0)
        energy = (self.emb(col) ** 2).sum() + x.sum() * 1e-3
        (grad,) = torch.autograd.grad(energy, x)
        return energy, grad


class _Plain(nn.Module):
    """The same write with no gradient region; the caller runs backward."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.emb = nn.Linear(1, 4)
        self.write = write

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a projection of the written view."""

        return self.emb(self.write(self.lin(x)))


def _input() -> torch.Tensor:
    return torch.randn(3, 8, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_view_write_inside_autograd_grad_validates(write: str) -> None:
    """Full validation (metadata included) passes for a view write in a grad region."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_GradRegion(_WRITES[write]), _input(), scope="forward")


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_view_write_backward_graph_keeps_view_ops(write: str) -> None:
    """After a managed backward every view op points at a node the pass visited."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_Plain(_WRITES[write]), _input().requires_grad_(True))
        trace.backward(trace[trace.output_layers[0]].out.sum())
    _check_backward_graph_invariants(trace)
    getitems = [layer for layer in trace._layer_list if layer.func_name == "__getitem__"]
    assert getitems
    for layer in getitems:
        assert layer.autograd.grad_fn is not None
        assert layer.autograd.grad_fn.class_name == "CopySlices"


def test_view_write_backward_validates() -> None:
    """Backward validation passes for a model that writes a view in place."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_Plain(_view_used), _input(), scope="backward")


def test_plain_view_read_keeps_its_own_node() -> None:
    """A view that is only read (never written) keeps the node its op produced."""

    class _ReadOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)[:, 5:6] * 3.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_ReadOnly(), _input().requires_grad_(True))
        trace.backward(trace[trace.output_layers[0]].out.sum())
    _check_backward_graph_invariants(trace)
    (getitem,) = [layer for layer in trace._layer_list if layer.func_name == "__getitem__"]
    assert getitem.autograd.grad_fn.class_name == "SliceBackward0"


class _ConsumedBeforeWrite(nn.Module):
    """Read a view, then write it in place; the earlier reader keeps the view's node."""

    def __init__(self, read_view_after: bool = False) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.read_view_after = read_view_after

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sum that reaches the view's node through the earlier reader."""

        y = self.lin(x)
        v = y[:, 5:6]
        a = v * 3.0
        v.add_(1.0)
        return a.sum() + (v.sum() if self.read_view_after else y.sum())


def _only_layer(trace: tl.Trace, func_names: tuple[str, ...]) -> tl.Layer:
    """Return the one layer whose function name is in ``func_names``."""

    (layer,) = [layer for layer in trace._layer_list if layer.func_name in func_names]
    return layer


def _traced_backward(model: nn.Module) -> tl.Trace:
    """Trace ``model`` on the shared input and run a managed backward from its output."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, _input().requires_grad_(True))
        trace.backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _assert_reader_edge_lands_on_view_op(trace: tl.Trace) -> None:
    """The view op keeps ``SliceBackward0`` and its reader's edge lands on it."""

    getitem = _only_layer(trace, ("__getitem__",))
    mul = _only_layer(trace, ("__mul__", "mul"))
    assert getitem.autograd.grad_fn.class_name == "SliceBackward0"
    mul_next_ids = mul.autograd.grad_fn.next_grad_fn_ids
    assert getitem.autograd.grad_fn.grad_fn_object_id in mul_next_ids
    assert all(trace._grad_fn_logs[next_id].has_op for next_id in mul_next_ids)


@pytest.mark.parametrize("read_view_after", [False, True], ids=["base_read", "view_read"])
def test_view_consumed_before_write_keeps_its_node(read_view_after: bool) -> None:
    """A view read before the write keeps its node; the reader's edge lands on its op.

    The reader's ``MulBackward0`` still points at the view's ``SliceBackward0``, so
    that node stays in the graph and the view op's recorded handle is already
    right. Rebinding it to the base's ``CopySlices`` would leave the reader's edge
    on a node no op owns while backward validation still passed.
    """

    _assert_reader_edge_lands_on_view_op(_traced_backward(_ConsumedBeforeWrite(read_view_after)))


def test_view_consumed_before_write_and_read_after_keeps_invariants() -> None:
    """With the written view read afterwards, the whole backward graph holds."""

    _check_backward_graph_invariants(_traced_backward(_ConsumedBeforeWrite(read_view_after=True)))


@pytest.mark.parametrize("read_view_after", [False, True], ids=["base_read", "view_read"])
def test_view_consumed_before_write_backward_validates(read_view_after: bool) -> None:
    """Backward validation passes when a view is read before its in-place write."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_ConsumedBeforeWrite(read_view_after), _input(), scope="backward")


class _TwoViewsWritten(nn.Module):
    """Take two column views, then write the first and the second in place."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sum over the second written view."""

        y = self.lin(x)
        first = y[:, 0:1]
        second = y[:, 5:6]
        first.add_(1.0)
        return second.mul_(2.0).sum()


def test_second_write_by_another_view_is_not_rebound() -> None:
    """A view taken before an earlier write of its base is not moved by its own write.

    The second view's node reads the base's node from before the first write, so
    the second write's ``CopySlices`` (which reads the first write's) did not take
    its place; the rebind proves nothing and leaves the record alone.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_TwoViewsWritten(), _input().requires_grad_(True))
    raw = trace.capture_events.grad_fn_handles_by_label_raw
    getitem_nodes = sorted(
        type(node).__name__ for key, node in raw.items() if "getitem" in str(key)
    )
    assert getitem_nodes == ["CopySlices", "SliceBackward0"]


def _write_then_handles(
    consume: Callable[[torch.Tensor, torch.Tensor], dict[str, object]],
) -> tuple[object, object, dict[str, object]]:
    """Write a column view of an intermediate; return its old node, ``CopySlices``, handles.

    ``consume(y, v)`` runs before the write and returns extra recorded handles.
    """

    x = torch.randn(3, 8, requires_grad=True)
    y = x * 2.0
    v = y[:, 5:6]
    handles: dict[str, object] = {"y": y.grad_fn, "v": v.grad_fn, **consume(y, v)}
    view_node = v.grad_fn
    v.add_(1.0)
    return view_node, y.grad_fn, handles


def test_orphaned_view_nodes_proves_a_plain_view_write() -> None:
    """The provable shape: the written view's node is the one ``CopySlices`` replaced."""

    view_node, copy_slices, handles = _write_then_handles(lambda y, v: {})
    assert _orphaned_view_nodes(view_node, copy_slices, handles) == [view_node]


@pytest.mark.parametrize(
    "case",
    ["reader_before_write", "newer_recorded_node", "non_view_start", "reaches_copy_slices"],
)
def test_orphaned_view_nodes_refuses_unprovable_chains(case: str) -> None:
    """Every shape the proof cannot establish returns ``[]`` (records stay unchanged)."""

    if case == "reader_before_write":
        view_node, copy_slices, handles = _write_then_handles(lambda y, v: {"a": (v * 3.0).grad_fn})
    else:
        view_node, copy_slices, handles = _write_then_handles(lambda y, v: {})
    if case == "newer_recorded_node":
        handles["later"] = (torch.ones(2, requires_grad=True) * 5.0).grad_fn
    start = {
        "non_view_start": handles["y"],
        "reaches_copy_slices": copy_slices,
    }.get(case, view_node)
    assert _orphaned_view_nodes(start, copy_slices, handles) == []


def test_orphaned_view_nodes_refuses_a_view_older_than_the_last_write() -> None:
    """A view taken before an earlier write meets the base lineage too far back."""

    x = torch.randn(3, 8, requires_grad=True)
    y = x * 2.0
    first, second = y[:, 0:1], y[:, 5:6]
    second_node = second.grad_fn
    first.add_(1.0)
    second.mul_(2.0)
    handles = {"y": y.grad_fn, "second": second_node}
    assert _orphaned_view_nodes(second_node, y.grad_fn, handles) == []


def test_a_wrong_handle_still_fails_the_backward_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation stays a tripwire: a view op bound to a node off the graph raises."""

    stray = (torch.ones(1, requires_grad=True) * 2.0).grad_fn

    def _bind_stray(trace: tl.Trace, receiver: torch.Tensor) -> None:
        handles = trace.capture_events.grad_fn_handles_by_label_raw
        trace.__dict__.setdefault("_backward_gradfn_refs", []).append(stray)
        for raw_label, handle in list(handles.items()):
            if type(handle).__name__ == "SliceBackward0":
                handles[raw_label] = stray

    monkeypatch.setattr(param_mutation, "rebind_rebased_view_producers", _bind_stray)
    trace = _traced_backward(_Plain(_view_used))
    with pytest.raises(MetadataInvariantError, match="getitem"):
        _check_backward_graph_invariants(trace)


def test_without_the_rebind_the_backward_invariant_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the rebind disabled, the view op's dropped node still raises."""

    monkeypatch.setattr(
        param_mutation, "rebind_rebased_view_producers", lambda trace, receiver: None
    )
    trace = _traced_backward(_Plain(_view_used))
    with pytest.raises(MetadataInvariantError, match="missing its GradFn backpointer"):
        _check_backward_graph_invariants(trace)
