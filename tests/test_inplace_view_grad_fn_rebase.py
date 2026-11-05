"""An in-place write through a view keeps every op's backward node in the recorded graph.

Autograd's ``rebase_history`` gives the base of a written view a new ``CopySlices``
node and drops the view's own node from the graph (for a view of a view, every
intermediate view node too). The op that produced the view recorded that dropped
node, so no backward pass ever reached it and the metadata invariant reported a
missing GradFn backpointer, inside an ``autograd.grad`` region and after a managed
backward alike. Capture now moves such an op onto the ``CopySlices`` node that took
its place. The backward-graph invariant's dead-branch carve-out is narrower too: it no
longer excuses an in-place op recorded on a view backward node while backward visited
a ``CopySlices`` that no op records.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _autograd_view_nodes, _torch_compat
from torchlens._utils._autograd_view_nodes import is_view_backward_class
from torchlens.backends.torch import param_mutation
from torchlens.backends.torch._user_grad_fn import (
    _MAX_VIEW_REBASE_DEPTH,
    _orphaned_view_nodes,
    _view_write_node,
)
from torchlens.errors._validation import MetadataInvariantError
from torchlens.validation._invariants_backward_domain import (
    _view_write_gradient_runs_through_copy_slices,
)
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


@pytest.mark.parametrize("read_view_after", [False, True], ids=["base_read", "view_read"])
def test_view_consumed_before_write_keeps_invariants(read_view_after: bool) -> None:
    """The whole backward graph holds whether the written base or the view is read after.

    ``base_read`` is the shape where the write's gradient reaches the loss only
    through the base's ``CopySlices``; the in-place op must be bound to it.
    """

    _check_backward_graph_invariants(_traced_backward(_ConsumedBeforeWrite(read_view_after)))


def _backward_verdict(model: nn.Module) -> object:
    """Return ``tl.validate(scope="backward")``'s verdict, or the invariant error's name."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return tl.validate(model, _input(), scope="backward")
        except MetadataInvariantError as error:
            return type(error).__name__


@pytest.mark.parametrize("read_view_after", [False, True], ids=["base_read", "view_read"])
def test_view_consumed_before_write_backward_validates(read_view_after: bool) -> None:
    """Backward validation passes because the in-place op is bound to the node backward uses.

    Backward validation runs the backward-graph invariant on its own capture.
    Before the writer was bound to ``CopySlices``, ``base_read`` failed it there
    too: the write through the view recorded the view's regenerated node, which
    backward never visits.
    """

    assert _backward_verdict(_ConsumedBeforeWrite(read_view_after)) is True


def test_writer_of_a_read_view_is_bound_to_copy_slices() -> None:
    """The base's reader after the write reaches the in-place op's own node.

    The view's reader before the write keeps the view's node, so the view op is
    not rebound; the in-place op records the base's ``CopySlices`` (the node that
    carries its gradient), and the base reader's edge lands on it.
    """

    trace = _traced_backward(_ConsumedBeforeWrite())
    writer = _only_layer(trace, ("add_",))
    base_sum = [layer for layer in trace._layer_list if layer.func_name == "sum"][-1]
    assert writer.autograd.grad_fn.class_name == "CopySlices"
    assert writer.autograd.grad_fn.grad_fn_object_id in base_sum.autograd.grad_fn.next_grad_fn_ids


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


def test_second_write_by_another_view_is_not_rebound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A view taken before an earlier write of its base is not moved by its own write.

    The second view's node reads the base's node from before the first write, so
    the second write's ``CopySlices`` (which reads the first write's) did not take
    its place; the rebind proves nothing and leaves the record alone. The handle
    index is read during capture, right after each rebind.
    """

    rebind = param_mutation.rebind_rebased_view_producers
    after_each_write: list[list[str]] = []

    writer_nodes: list[str] = []

    def _spy(trace: tl.Trace, receiver: torch.Tensor) -> object:
        writer_node = rebind(trace, receiver)
        handles = trace.capture_events.grad_fn_handles_by_label_raw.values()
        after_each_write.append([type(node).__name__ for node in handles])
        writer_nodes.append(type(writer_node).__name__)
        return writer_node

    monkeypatch.setattr(param_mutation, "rebind_rebased_view_producers", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.trace(_TwoViewsWritten(), _input().requires_grad_(True))
    first_write, second_write = after_each_write
    assert writer_nodes == ["CopySlices", "NoneType"]
    assert first_write.count("CopySlices") == 1
    # The first write's CopySlices: its view op and (after the label move) its writer.
    assert second_write.count("CopySlices") == 2
    assert second_write.count("SliceBackward0") == 1


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


def _assert_both_entry_points_reject(model_factory: Callable[[], nn.Module], label: str) -> None:
    """A managed backward's invariant and backward validation both reject the capture."""

    with pytest.raises(MetadataInvariantError, match=label):
        _check_backward_graph_invariants(_traced_backward(model_factory()))
    assert _backward_verdict(model_factory()) == "MetadataInvariantError"


def test_a_wrong_view_op_handle_fails_both_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation catches a view op bound to a real node of the graph that backward skips.

    The rebind runs, then the view op's handle is put back on the view's own
    pre-write node: a node of this forward's autograd graph (its edge reads the
    base's producer) that the write cut out of the backward walk. Everything else
    in the capture is right, so only validation can reject it, and both entry
    points do.
    """

    rebind = param_mutation.rebind_rebased_view_producers

    def _undo_onto_old_node(trace: tl.Trace, receiver: torch.Tensor) -> object:
        handles = trace.capture_events.grad_fn_handles_by_label_raw
        before = dict(handles)
        writer_node = rebind(trace, receiver)
        for raw_label, handle in list(handles.items()):
            if type(handle).__name__ == "CopySlices" and before.get(raw_label) is not handle:
                handles[raw_label] = before[raw_label]
        return writer_node

    monkeypatch.setattr(param_mutation, "rebind_rebased_view_producers", _undo_onto_old_node)
    _assert_both_entry_points_reject(lambda: _Plain(_view_used), "getitem")


def test_a_wrong_writer_handle_fails_both_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation catches the in-place op bound to the view's regenerated node.

    The regenerated node is a real node of the graph (it reads the base's
    ``CopySlices``), but backward never visits it when only the base is read
    after the write. Binding the writer there instead of ``CopySlices`` must fail
    the managed backward's invariant and backward validation alike.
    """

    rebind = param_mutation.rebind_rebased_view_producers

    def _regenerated_node(trace: tl.Trace, receiver: torch.Tensor) -> object:
        writer_node = rebind(trace, receiver)
        return receiver.grad_fn if writer_node is not None else None

    monkeypatch.setattr(param_mutation, "rebind_rebased_view_producers", _regenerated_node)
    _assert_both_entry_points_reject(_ConsumedBeforeWrite, "add_")


class _DeadViewWrite(nn.Module):
    """Read a view, write it in place, then read neither the view nor the base."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sum of the earlier reader only; the write is a dead branch."""

        y = self.lin(x)
        v = y[:, 5:6]
        a = v * 3.0
        v.add_(1.0)
        return a.sum()


@pytest.mark.parametrize("writer_bound", [True, False], ids=["bound", "unbound"])
def test_a_dead_view_write_is_still_excused(
    monkeypatch: pytest.MonkeyPatch, writer_bound: bool
) -> None:
    """A view write that feeds nothing stays outside the backward walk, bound or not.

    No ``CopySlices`` is visited when neither the base nor the view is read after
    the write, so the stricter view-write rule leaves the dead-branch exemption in
    place, with the writer on ``CopySlices`` or on its regenerated view node.
    """

    if not writer_bound:
        monkeypatch.setattr(
            param_mutation, "rebind_rebased_view_producers", lambda trace, receiver: None
        )
    _check_backward_graph_invariants(_traced_backward(_DeadViewWrite()))
    assert _backward_verdict(_DeadViewWrite()) is True


def test_view_write_node_needs_a_proven_chain() -> None:
    """``CopySlices`` is the writer's node for a proven chain, read before the write or not."""

    view_node, copy_slices, handles = _write_then_handles(lambda y, v: {})
    assert _view_write_node(view_node, copy_slices, handles) is copy_slices
    view_node, copy_slices, handles = _write_then_handles(lambda y, v: {"a": (v * 3.0).grad_fn})
    assert _view_write_node(view_node, copy_slices, handles) is copy_slices
    assert _view_write_node(handles["y"], copy_slices, handles) is None
    handles["later"] = (torch.ones(2, requires_grad=True) * 5.0).grad_fn
    assert _view_write_node(view_node, copy_slices, handles) is None


def _base_only(y: torch.Tensor) -> torch.Tensor:
    """Write a column view in place, then read only the base."""

    y[:, 5:6].add_(0.25)
    return y


def _base_before_and_after(y: torch.Tensor) -> torch.Tensor:
    """Read the base, write a column view in place, read the base again."""

    early = y * 2.0
    y[:, 5:6].add_(0.25)
    return y * early


def _intermediate_view(y: torch.Tensor) -> torch.Tensor:
    """Write a view of an intermediate view in place, then read the intermediate."""

    inner = y[:, 2:8]
    inner[:, 3:4].mul_(1.5)
    return inner


def _sibling_view(y: torch.Tensor) -> torch.Tensor:
    """Take a sibling view, write an overlapping column view, read the sibling."""

    sibling = y[:, 4:7]
    y[:, 5:6].add_(0.25)
    return sibling


_ALIAS_READS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "base_only": _base_only,
    "base_before_and_after": _base_before_and_after,
    "intermediate_view": _intermediate_view,
}


class _AliasRead(nn.Module):
    """Write through a view of an intermediate, then read another alias of it."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.write = write

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scaled copy of the alias the write function reads."""

        return self.write(self.lin(x)) * 3.0


@pytest.mark.parametrize("read", sorted(_ALIAS_READS))
def test_view_write_then_alias_read_keeps_invariants(read: str) -> None:
    """Reading the base or an intermediate view after a view write passes both checks.

    The write's gradient reaches the loss through the base's ``CopySlices`` (or a
    view regenerated over it), so the in-place op records ``CopySlices``.
    """

    trace = _traced_backward(_AliasRead(_ALIAS_READS[read]))
    _check_backward_graph_invariants(trace)
    writer = _only_layer(trace, ("add_", "mul_"))
    assert writer.autograd.grad_fn.class_name == "CopySlices"
    assert _backward_verdict(_AliasRead(_ALIAS_READS[read])) is True


def test_sibling_view_read_after_a_view_write_fails_backward_validation_closed() -> None:
    """An open capture gap stays rejected: the sibling view's op holds a node backward skips.

    ``sibling`` is taken before the write, so its op recorded the pre-write
    ``SliceBackward0``; reading ``sibling`` after the write goes through a view
    node autograd regenerates over ``CopySlices``, which no op records. Backward
    validation's capture (no base-label move) keeps the sibling op as the
    reader's parent and rejects the record. Binding the sibling op to the
    regenerated node is the postprocess rebind proposed by the view-write-2 lane.
    """

    assert _backward_verdict(_AliasRead(_sibling_view)) == "MetadataInvariantError"


@pytest.mark.parametrize(
    ("class_name", "is_view"),
    [
        ("SliceBackward0", True),
        ("AsStridedBackward0", True),
        ("MulBackward0", False),
        ("UnsafeViewBackward0", False),
        ("CopySlices", False),
        ("NoSuchBackward9", False),
    ],
)
def test_view_backward_classes_are_read_from_torch_schemas(class_name: str, is_view: bool) -> None:
    """View classes come from aten schema alias annotations; anything unresolvable is not one."""

    assert is_view_backward_class(class_name) is is_view


@pytest.mark.parametrize("depth", [3, _MAX_VIEW_REBASE_DEPTH + 1], ids=["short", "over_bound"])
def test_view_chain_walk_stops_at_its_depth_bound(depth: int) -> None:
    """A chain of views proves up to the bound; a longer one proves nothing."""

    y = torch.randn(depth + 4, 2, requires_grad=True) * 2.0
    base_node = y.grad_fn
    view = y
    for _ in range(depth):
        view = view[1:]
    view_node = view.grad_fn
    view.add_(1.0)
    chain = _orphaned_view_nodes(view_node, y.grad_fn, {"y": base_node, "v": view_node})
    assert len(chain) == (depth if depth <= _MAX_VIEW_REBASE_DEPTH else 0)


def test_without_the_rebind_the_backward_invariant_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the rebind disabled, the view op's dropped node still raises."""

    monkeypatch.setattr(
        param_mutation, "rebind_rebased_view_producers", lambda trace, receiver: None
    )
    trace = _traced_backward(_Plain(_view_used))
    with pytest.raises(MetadataInvariantError, match="missing its GradFn backpointer"):
        _check_backward_graph_invariants(trace)


# (sequence numbers off, schema enumeration off) for each capability gap.
_CAPABILITY_GAPS: dict[str, tuple[bool, bool]] = {
    "no_sequence_nr": (True, False),
    "no_schema_enumeration": (False, True),
    "neither": (True, True),
}


@pytest.fixture(params=sorted(_CAPABILITY_GAPS))
def capability_gap(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """Run the test as on a torch build that lacks one or both rebind capabilities.

    ``HAS_NODE_SEQUENCE_NR`` off makes every node's creation order unknown;
    ``get_jit_all_schemas`` answering ``None`` makes the set of view ops empty.
    The cached view-op set is cleared on entry and exit so no other test sees it.
    """

    sequence_off, schemas_off = _CAPABILITY_GAPS[request.param]
    if sequence_off:
        monkeypatch.setattr(_torch_compat, "HAS_NODE_SEQUENCE_NR", False)
    if schemas_off:
        monkeypatch.setattr(_torch_compat, "get_jit_all_schemas", lambda: None)
    _autograd_view_nodes._aten_view_op_names.cache_clear()
    yield request.param
    _autograd_view_nodes._aten_view_op_names.cache_clear()


@pytest.mark.parametrize("model_factory", [_ConsumedBeforeWrite, lambda: _Plain(_view_used)])
def test_without_a_capability_capture_binds_nothing(
    capability_gap: str,
    monkeypatch: pytest.MonkeyPatch,
    model_factory: Callable[[], nn.Module],
) -> None:
    """A missing capability proves no view write, so no record moves to ``CopySlices``."""

    rebind = param_mutation.rebind_rebased_view_producers
    writer_nodes: list[object] = []
    handle_classes: list[str] = []

    def _spy(trace: tl.Trace, receiver: torch.Tensor) -> object:
        writer_node = rebind(trace, receiver)
        writer_nodes.append(writer_node)
        handles = trace.capture_events.grad_fn_handles_by_label_raw.values()
        handle_classes.extend(type(node).__name__ for node in handles)
        return writer_node

    monkeypatch.setattr(param_mutation, "rebind_rebased_view_producers", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.trace(model_factory(), _input().requires_grad_(True))
    assert writer_nodes == [None]
    assert "CopySlices" not in handle_classes


# Backward validation's verdict on the base-read view write for each capability gap.
_GAP_VALIDATE_VERDICT: dict[str, object] = {
    # View classes still resolve, so the narrowed carve-out refuses the unbound writer.
    "no_sequence_nr": "MetadataInvariantError",
    # No view class resolves: the carve-out excuses the writer as it did before the
    # narrowing, and the managed backward alone rejects the capture.
    "no_schema_enumeration": True,
    "neither": True,
}


def test_without_a_capability_the_carve_out_outcome_is_pinned(capability_gap: str) -> None:
    """With a capability missing the writer stays unbound; each entry point's outcome is pinned.

    The managed backward moves the base's label to the in-place op, so the op
    feeds the loss and its missing backpointer raises whatever the carve-out
    says. Backward validation's capture keeps the historical topology, so there
    the narrowed carve-out decides: it needs the view-op set, and without schema
    enumeration it is off, as before the narrowing.
    """

    trace = _traced_backward(_ConsumedBeforeWrite())
    writer = _only_layer(trace, ("add_",))
    assert writer.grad_fn_class_name != "CopySlices"
    schemas_off = _CAPABILITY_GAPS[capability_gap][1]
    assert _view_write_gradient_runs_through_copy_slices(trace, writer) is not schemas_off
    with pytest.raises(MetadataInvariantError, match="add_"):
        _check_backward_graph_invariants(trace)
    assert _backward_verdict(_ConsumedBeforeWrite()) == _GAP_VALIDATE_VERDICT[capability_gap]


class _BufferWriteBesideDeadInplace(nn.Module):
    """Write into a view of a buffer that needs no grad, and leave a non-view in-place op dead.

    The buffer's ``CopySlices`` has no prior node, so the write is never bound
    and backward visits a ``CopySlices`` that no op records.
    """

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.side = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sum over the buffer and the projection; ``dead`` feeds nothing."""

        h = self.lin(x)
        buffer = torch.zeros(3, 8)
        buffer[:, 0:1].copy_(h[:, 0:1])
        dead = self.side(x)
        dead.mul_(2.0)
        return buffer.sum() + h.sum()


def test_an_unowned_copy_slices_leaves_a_dead_non_view_write_excused() -> None:
    """The narrowed carve-out still excuses a dead in-place op that is not a view write.

    Backward visits a ``CopySlices`` no op records (the buffer write), which is
    the condition the narrowing keys on; the dead ``mul_`` records a non-view
    node, so the carve-out excuses it as it always did, on both entry points.
    """

    trace = _traced_backward(_BufferWriteBesideDeadInplace())
    assert any(
        grad_fn.class_name == "CopySlices" and not grad_fn.has_op
        for grad_fn in trace._grad_fn_logs.values()
    )
    dead = _only_layer(trace, ("mul_",))
    assert dead.autograd.grad_fn is None
    assert _view_write_gradient_runs_through_copy_slices(trace, dead) is False
    _check_backward_graph_invariants(trace)
    assert _backward_verdict(_BufferWriteBesideDeadInplace()) is True


def test_sibling_view_read_gap_is_accepted_by_the_managed_backward() -> None:
    """Open gap, pinned on the managed entry point: it accepts what validation rejects.

    The managed capture moves the base's label to the in-place op, so the
    sibling view's op has no forward children and the dead-branch carve-out
    excuses its pre-write ``SliceBackward0``, which backward never visits (the
    read goes through a view node regenerated over ``CopySlices``). Backward
    validation rejects the same model (see the test above). The fix is the
    capture-side rebind of regenerated view nodes; when it lands, this test
    must change to assert that both entry points pass.
    """

    trace = _traced_backward(_AliasRead(_sibling_view))
    _check_backward_graph_invariants(trace)
    sibling = [layer for layer in trace._layer_list if layer.func_name == "__getitem__"][0]
    assert sibling.autograd.grad_fn is None
