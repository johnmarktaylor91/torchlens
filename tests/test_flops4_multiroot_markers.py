"""Multi-root backwards and shared-node markers in backward FLOP attribution.

Two defects, one regression test each, plus the agreement of the three readers:

* ``torch.autograd.backward([root1, root2])`` over two captures used to walk
  only the first root's graph in each capture, so the second root's nodes had
  no ``GradFn`` record anywhere and their work read unattributed.
* When two captures hook one node their marker pushes and pops did not nest,
  so a finished node's marker stayed on the stack and later calls were charged
  to it; the census matched the executing node, but the private ATen recorder
  and the checkpoint unpack evidence took the innermost marker.

Every expected FLOP count is the closed form ``2 * batch * in * out`` of the
weight gradient of a ``Linear`` whose input does not require grad.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

import torchlens as tl
from torchlens.backends.torch import backward as backward_module
from torchlens.backends.torch._aten_capture import _activate_aten_recording_for_tests
from torchlens.report import backward_status
from torchlens.validation.invariants import check_metadata_invariants

_OPTIONS = tl.options.CaptureOptions(backward_ready=True)
_MATMULS = ("mm", "addmm")


def _two_captures(
    *, armed: bool, first: nn.Module | None = None
) -> tuple[Any, Any, nn.Module, nn.Module]:
    """Trace ``Linear(4, 3)`` (or ``first``) on (5, 4) and ``Linear(3, 2)`` on (5, 3)."""

    torch.manual_seed(3)
    first = nn.Linear(4, 3) if first is None else first
    second = nn.Linear(3, 2)
    x1, x2 = torch.randn(5, 4), torch.randn(5, 3)
    kwargs: dict[str, Any] = {"capture": _OPTIONS, "save_mode": "reference"}
    if armed:
        with _activate_aten_recording_for_tests():
            trace1, trace2 = tl.trace(first, x1, **kwargs), tl.trace(second, x2, **kwargs)
    else:
        trace1, trace2 = tl.trace(first, x1, **kwargs), tl.trace(second, x2, **kwargs)
    return trace1, trace2, first, second


def _out(trace: Any) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def _graph_nodes(root: torch.Tensor) -> dict[int, Any]:
    """Every node reachable from ``root`` (raw torch state), by object id.

    The nodes are held, not just their ids: ``next_functions`` can mint a fresh
    wrapper per access (torch 2.7), so a wrapper held only by its id can be
    freed and its address reused by the next one, and the ids would no longer
    name the wrappers the capture later walks.
    """

    seen: dict[int, Any] = {}
    queue = [root.grad_fn]
    while queue:
        node = queue.pop()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = node
        queue.extend(next_fn for next_fn, _ in node.next_functions)
    return seen


def _single_pass(trace: Any) -> Any:
    status = backward_status(trace)
    (attribution,) = status.attributions
    rows = sum(row.flops for row in attribution.rows)
    assert rows + attribution.unattributed_flops == attribution.executed_flops
    return attribution


def _matmul_parents(trace: Any) -> list[Any]:
    """The ATen recorder's GradFn-call parent of every backward matmul row."""

    profile = trace._primitive_op_profile
    return [
        row.parent_grad_fn_call_ref
        for row in profile.primitive_ops
        if row.capture_phase == "backward" and row.operator in _MATMULS
    ]


@pytest.mark.parametrize("entry", ["backward", "grad"])
def test_multi_root_backward_walks_each_root_in_the_capture_it_reaches(entry: str) -> None:
    """Each capture walks every root whose graph reaches it, and only those."""

    trace1, trace2, first, second = _two_captures(armed=True)
    try:
        y1, y2 = _out(trace1), _out(trace2)
        root1, root2 = y1.sum(), y2.sum()
        graph1, graph2 = _graph_nodes(root1), _graph_nodes(root2)
        with _activate_aten_recording_for_tests():
            if entry == "backward":
                torch.autograd.backward([root1, root2])
            else:
                params = [*first.parameters(), *second.parameters()]
                torch.autograd.grad([root1, root2], params)
        assert set(trace1.grad_fn_logs) == graph1.keys()
        assert set(trace2.grad_fn_logs) == graph2.keys()
        cases = ((trace1, root1, 2 * 5 * 4 * 3), (trace2, root2, 2 * 5 * 3 * 2))
        for trace, root, closed_form in cases:
            (record,) = trace.autograd.backward_passes
            assert list(record.root_grad_fn_ids) == [id(root.grad_fn)]
            attribution = _single_pass(trace)
            assert attribution.executed_flops == closed_form
            assert attribution.unattributed_flops == 0
            (counted,) = [row for row in attribution.rows if row.flops]
            assert counted.flops == closed_form
            assert counted.op_label in {str(op.layer_label) for op in trace.layer_list}
            # The recorder never names a node its capture did not walk (no
            # unresolved primitive ownership survives).
            assert check_metadata_invariants(trace)
            named = [parent[0] for parent in _matmul_parents(trace) if parent is not None]
            assert all(label.startswith("addmm_back_") for label in named)
    finally:
        trace1.cleanup()
        trace2.cleanup()


def test_one_capture_with_two_roots_walks_both() -> None:
    """A single capture handed two roots records both roots' nodes."""

    torch.manual_seed(3)
    trace = tl.trace(nn.Linear(4, 3), torch.randn(5, 4), capture=_OPTIONS)
    try:
        y = _out(trace)
        root1, root2 = y.sum(), (y * 2).sum()
        expected = _graph_nodes(root1) | _graph_nodes(root2)
        torch.autograd.backward([root1, root2])
        assert set(trace.grad_fn_logs) == expected.keys()
        (record,) = trace.autograd.backward_passes
        assert list(record.root_grad_fn_ids) == [id(root1.grad_fn), id(root2.grad_fn)]
        assert len(record.root_meta) == 2
        attribution = _single_pass(trace)
        # One addmm node runs once on the summed gradient: 2*5*4*3.
        assert attribution.executed_flops == 2 * 5 * 4 * 3
        assert attribution.unattributed_flops == 0
        assert check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_shared_node_markers_never_outlive_their_node() -> None:
    """Two captures hook every node of one root; a third root no capture owns runs after.

    The untraced Linear(3,2)'s weight gradient (60 FLOPs) ran on no walked node,
    so every reader leaves it unowned; a finished shared node's marker used to
    absorb it in the ATen recorder.
    """

    torch.manual_seed(3)
    untraced = nn.Linear(3, 2)
    z = untraced(torch.randn(5, 3)).sum()  # built first: its nodes run after the shared ones
    trace1, trace2, _, _ = _two_captures(armed=True)
    try:
        with _activate_aten_recording_for_tests():
            torch.autograd.backward([_out(trace1).sum() + _out(trace2).sum(), z])
        for trace in (trace1, trace2):
            attribution = _single_pass(trace)
            assert attribution.unattributed_flops == 60
            assert [op for op in attribution.unattributed_ops if op[1]] == [("aten.mm.default", 60)]
        parents = [parent for trace in (trace1, trace2) for parent in _matmul_parents(trace)]
        assert None in parents, "the untraced root's matmul was charged to a walked node"
        named = [parent[0] for parent in parents if parent is not None]
        assert named and all(label.startswith("addmm_back_") for label in named)
    finally:
        trace1.cleanup()
        trace2.cleanup()


class _Checkpointed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.a, self.b = nn.Linear(4, 6), nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint.checkpoint(
            lambda h: self.b(torch.relu(self.a(h))), x, use_reentrant=False
        )


def _census_refs(
    trace: Any, pass_index: int, *, own_ops_only: bool = True
) -> set[tuple[str, int, int]]:
    """Label-keyed GradFn-call refs the census recorded on nodes this trace walked.

    ``own_ops_only`` keeps the nodes whose forward op this trace recorded.
    """

    record = trace.autograd.backward_passes[pass_index - 1]
    refs = set()
    for op in record.dispatched_ops:
        ref = op.grad_fn_call_ref
        grad_fn = None if ref is None else trace.grad_fn_logs.get(ref[0])
        if grad_fn is not None and (grad_fn.has_op or not own_ops_only):
            refs.add((grad_fn.label, ref[1], ref[2]))
    return refs


def _unpack_refs(trace: Any, pass_index: int) -> set[tuple[str, int, int]]:
    state = backward_module._CHECKPOINT_TOKEN_STATE[trace]
    refs = set()
    for record in state["tokens"].values():
        for point in record["unpack_evidence"]:
            bracket = point["fire_bracket"]
            if point["pass_index"] == pass_index and bracket is not None:
                refs.add((trace.grad_fn_logs[bracket[0]].label, bracket[1], bracket[2]))
    return refs


def test_three_readers_charge_each_capture_its_own_marker() -> None:
    """A node hooked by two captures whose pass numbers differ.

    The checkpointed capture's second pass is shared with a second capture
    whose first pass it is. For each capture the census, the ATen recorder and
    the unpack evidence name that capture's own call (call 2 of pass 2 for the
    checkpointed one, call 1 of pass 1 for the other), never the other
    capture's marker for the same node.
    """

    trace1, trace2, _, _ = _two_captures(armed=True, first=_Checkpointed())
    try:
        with _activate_aten_recording_for_tests():
            y1, y2 = _out(trace1), _out(trace2)
            y1.sum().backward(retain_graph=True)
            (y1.sum() + y2.sum()).backward()
        shared = {trace1: 2, trace2: 1}
        recorded_any = False
        for trace, pass_index in shared.items():
            census = _census_refs(trace, pass_index)
            assert census
            # Nodes the trace's own forward ops own: its own call of this pass.
            assert {(call, pass_) for _, call, pass_ in census} == {(pass_index, pass_index)}
            recorded = {
                parent
                for parent in _matmul_parents(trace)
                if parent is not None and parent[2] == pass_index
            }
            recorded_any = recorded_any or bool(recorded)
            assert recorded <= _census_refs(trace, pass_index, own_ops_only=False)
            assert check_metadata_invariants(trace)
        assert recorded_any, "the armed recorder saw no backward matmul of the shared pass"
        unpacked = _unpack_refs(trace1, 2)
        assert unpacked
        assert {(call, pass_) for _, call, pass_ in unpacked} == {(2, 2)}
        assert unpacked <= _census_refs(trace1, 2, own_ops_only=False)
        assert backward_status(trace1).attributions[1].unattributed_flops == 0
    finally:
        trace1.cleanup()
        trace2.cleanup()


def test_marker_pops_nest_whatever_order_the_posthooks_run() -> None:
    """Prehooks push t1 then t2 on one node; posthooks pop t1 then t2: nothing is left."""

    from torchlens.backends.torch import _backward_markers

    token = _backward_markers._ACTIVE_BACKWARD_GRAD_FN_REFS.set(())
    try:
        first = _backward_markers.begin_grad_fn_marker(7, 1, 1, 101)
        second = _backward_markers.begin_grad_fn_marker(7, 1, 1, 202)
        _backward_markers.end_grad_fn_marker(first)
        assert _backward_markers._ACTIVE_BACKWARD_GRAD_FN_REFS.get() == (second,)
        _backward_markers.end_grad_fn_marker(second)
        assert _backward_markers._ACTIVE_BACKWARD_GRAD_FN_REFS.get() == ()
    finally:
        _backward_markers._ACTIVE_BACKWARD_GRAD_FN_REFS.reset(token)
