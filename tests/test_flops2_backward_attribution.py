"""Executed-backward FLOPs attributed to GradFn calls and forward ops.

The bar, fixed before the attribution existed: for every captured backward the
per-row backward FLOPs plus the unattributed bucket equal the pass's executed
count exactly (integers), per pass and in total, on every matmul-family and
zoo case of ``tests/fixtures/flops1_oracle_cases.json``. Recompute labels under
activation checkpointing split a row's count and never move a FLOP between
rows.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

import torchlens as tl
from tests.test_flops1_backward_dispatch import (
    MATMUL_CASES,
    MV_DISCLOSED,
    ZOO_BUILDERS,
    ZOO_CASES,
    _build_matmul_case,
    _leaf,
    _traced_backward,
)
from torchlens.report import backward_status
from torchlens.report._cost_backward import recompute_observation


def _assert_conserved(trace: Any) -> Any:
    """Rows plus the unattributed bucket equal the executed count, per pass and in total."""

    status = backward_status(trace)
    assert status.state == "backward_observed"
    assert status.executed_flops is not None, status.reason
    assert len(status.attributions) == len(trace.autograd.backward_passes)
    total = 0
    for attribution, (pass_index, pass_count) in zip(
        status.attributions, status.executed_flops_per_pass, strict=True
    ):
        assert attribution.pass_index == pass_index
        rows = sum(row.flops for row in attribution.rows)
        assert rows + attribution.unattributed_flops == attribution.executed_flops == pass_count
        assert all(isinstance(row.flops, int) for row in attribution.rows)
        total += pass_count
    assert total == status.executed_flops
    assert status.unattributed_flops == sum(a.unattributed_flops for a in status.attributions)
    return status


def _counted_rows_name_their_forward_op(trace: Any, status: Any) -> None:
    """Every row with counted work names a forward op the trace holds, or a node without one."""

    labels = {str(op.layer_label) for op in trace.layer_list}
    for attribution in status.attributions:
        for row in attribution.rows:
            if row.flops and row.op_label is not None:
                assert row.op_label in labels, (row, sorted(labels))


@pytest.mark.parametrize("case", sorted(MATMUL_CASES))
def test_matmul_family_attribution_conserves_the_count(case: str) -> None:
    """All 29 matmul-family cases: rows + unattributed == executed, exactly."""

    row = MATMUL_CASES[case]
    trace = _traced_backward(*_build_matmul_case(case))
    try:
        status = _assert_conserved(trace)
        expected = MV_DISCLOSED.get(case, row["fcm_bwd"])
        assert status.executed_flops == expected
        _counted_rows_name_their_forward_op(trace, status)
        # The engine's seed gradient runs under no recorded node: unattributed, named.
        names = [name for a in status.attributions for name, _flops in a.unattributed_ops]
        assert "aten.ones_like.default" in names
    finally:
        trace.cleanup()


@pytest.mark.parametrize("case", sorted(ZOO_BUILDERS))
def test_zoo_attribution_conserves_the_count(case: str) -> None:
    """The seven single-pass zoo cases: rows + unattributed == executed == closed form."""

    trace = _traced_backward(*ZOO_BUILDERS[case]())
    try:
        status = _assert_conserved(trace)
        assert status.executed_flops == ZOO_CASES[case]["closed_bwd"]
        _counted_rows_name_their_forward_op(trace, status)
    finally:
        trace.cleanup()


class _SecondOrder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(_leaf((6, 3), True, 7).detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


def test_zoo_second_order_attribution_conserves_each_pass() -> None:
    """create_graph grad (pass 1) and the double backward (pass 2), each conserved."""

    row = ZOO_CASES["second_order"]
    model = _SecondOrder()
    trace = tl.trace(
        model, _leaf((4, 6), False, 8), capture=tl.options.CaptureOptions(backward_ready=True)
    )
    try:
        y = trace[trace.output_layers[0]].out
        (grad,) = torch.autograd.grad((y**2).sum(), model.w, create_graph=True)
        (grad**2).sum().backward()
        status = _assert_conserved(trace)
        assert status.executed_flops_per_pass == (
            (1, row["closed_grad_phase"]),
            (2, row["closed_double_backward"]),
        )
        assert [a.pass_index for a in status.attributions] == [1, 2]
    finally:
        trace.cleanup()


def test_two_traces_in_one_backward_each_keep_their_own_nodes() -> None:
    """A backward shared by two captures: each view charges only its own nodes."""

    torch.manual_seed(3)
    first, second = nn.Linear(4, 3), nn.Linear(3, 2)
    x1, x2 = _leaf((5, 4), False, 21), _leaf((5, 3), False, 22)
    options = tl.options.CaptureOptions(backward_ready=True)
    trace1 = tl.trace(first, x1, capture=options)
    trace2 = tl.trace(second, x2, capture=options)
    try:
        y1 = trace1[trace1.output_layers[0]].out
        y2 = trace2[trace2.output_layers[0]].out
        (y1.sum() + y2.sum()).backward()
        status1, status2 = _assert_conserved(trace1), _assert_conserved(trace2)
        # Closed forms: dW of Linear(4,3) on (5,4) = 2*5*4*3; of Linear(3,2) on (5,3) = 2*5*3*2.
        assert status1.executed_flops == 2 * 5 * 4 * 3
        assert status2.executed_flops == 2 * 5 * 3 * 2
        for trace, status in ((trace1, status1), (trace2, status2)):
            (attribution,) = status.attributions
            assert attribution.other_trace_ops > 0
            assert "another trace's nodes" in status.reason
            (counted,) = [row for row in attribution.rows if row.flops]
            assert counted.op_label in {str(op.layer_label) for op in trace.layer_list}
    finally:
        trace1.cleanup()
        trace2.cleanup()


class _Segment(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.a, self.b = nn.Linear(4, 6), nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class _Checkpointed(nn.Module):
    """A Linear-ReLU-Linear segment, checkpointed or not, under a Linear head."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.seg = _Segment()
        torch.manual_seed(6)
        self.head = nn.Linear(3, 2)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            hidden = self.seg(x)
        else:
            hidden = checkpoint.checkpoint(self.seg, x, use_reentrant=self.mode == "reentrant")
        return self.head(hidden)


#: Closed form of the recomputed segment's forward on a (5, 4) input:
#: Linear(4, 6) is 2*5*4*6 and Linear(6, 3) is 2*5*6*3 (the ReLU is not counted).
SEGMENT_FORWARD = 2 * 5 * 4 * 6 + 2 * 5 * 6 * 3


def _checkpointed_trace(mode: str) -> Any:
    model = _Checkpointed(mode)
    x = _leaf((5, 4), True, 9)
    # Early stop off: the non-reentrant replay runs the whole segment, as reentrant does.
    with checkpoint.set_checkpoint_early_stop(False):
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
        out = trace[trace.output_layers[0]].out
        trace.log_backward(out.sum())
    return trace


def _rows_by_label(status: Any) -> dict[str, int]:
    return {
        row.grad_fn_label: row.flops - row.recompute_flops
        for attribution in status.attributions
        for row in attribution.rows
    }


@pytest.mark.parametrize("mode", ["non_reentrant", "reentrant"])
def test_checkpoint_recompute_is_labeled_and_never_moves_a_flop(mode: str) -> None:
    """Recompute FLOPs equal the segment's closed form; the twin has none."""

    trace = _checkpointed_trace(mode)
    twin = _checkpointed_trace("none")
    try:
        status, twin_status = _assert_conserved(trace), _assert_conserved(twin)
        assert recompute_observation(trace) == "observed"
        assert recompute_observation(twin) == "none_observed"
        assert status.recompute_flops == SEGMENT_FORWARD
        assert twin_status.recompute_flops == 0
        assert [a.recompute for a in twin_status.attributions] == ["none_observed"]
        assert [a.recompute for a in status.attributions] == ["labeled"]
        assert status.executed_flops - status.recompute_flops == twin_status.executed_flops
        assert "recomputed forward under checkpointing" in status.status_line
        if mode == "non_reentrant":
            # Same nodes as the twin: removing the recompute share leaves every row unchanged.
            assert _rows_by_label(status) == _rows_by_label(twin_status)
        else:
            (row,) = [r for a in status.attributions for r in a.rows if r.recompute_flops]
            assert row.op_label is None and row.recompute_flops == SEGMENT_FORWARD
        forward = {
            str(op.layer_label): op.flops_forward
            for op in trace.layer_list
            if op.func_name == "linear"
        }
        twin_forward = {
            str(op.layer_label): op.flops_forward
            for op in twin.layer_list
            if op.func_name == "linear"
        }
        assert forward == twin_forward
    finally:
        trace.cleanup()
        twin.cleanup()


def test_recompute_under_create_graph_is_unknown_not_guessed() -> None:
    """A create_graph pass that checkpoints cannot separate replay from backward."""

    model = _Checkpointed("non_reentrant")
    x = _leaf((5, 4), True, 9)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
    try:
        out = trace[trace.output_layers[0]].out
        torch.autograd.grad(out.sum(), x, create_graph=True)
        status = _assert_conserved(trace)
        assert status.recompute_flops is None
        assert [a.recompute for a in status.attributions] == ["unknown"]
        assert "recompute share unknown" in status.status_line
    finally:
        trace.cleanup()


def test_backward_pass_reads_its_own_attribution() -> None:
    """``BackwardPass.flops_attribution`` is the status's per-pass record."""

    trace = _traced_backward(*ZOO_BUILDERS["partial_requires_grad"]())
    try:
        (record,) = trace.autograd.backward_passes
        attribution = record.flops_attribution
        assert attribution == backward_status(trace).attributions[0]
        assert attribution.source == "census"
        by_op = attribution.flops_by_op()
        attributed = sum(value for value in by_op.values() if value)
        assert attributed + attribution.unattributed_flops == attribution.executed_flops
        assert "BackwardFlopsAttribution(pass 1" in repr(attribution)
    finally:
        trace.cleanup()


class _RaiseInBackward(torch.autograd.Function):
    """Identity forward whose backward raises partway through the pass."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("backward refused on purpose")


class _RaisesMidPass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(8)
        self.first, self.second = nn.Linear(4, 6), nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(_RaiseInBackward.apply(self.first(x)))


def test_a_backward_that_raises_reads_not_run_never_a_number() -> None:
    """Calls dispatched before the raise are kept as evidence; the pass has no count."""

    model = _RaisesMidPass()
    trace = tl.trace(
        model, _leaf((5, 4), False, 4), capture=tl.options.CaptureOptions(backward_ready=True)
    )
    try:
        out = trace[trace.output_layers[0]].out
        with pytest.raises(RuntimeError, match="refused on purpose"):
            trace.log_backward(out.sum())
        (record,) = trace.autograd.backward_passes
        assert record.status != "ok"
        assert record.dispatched_ops  # the second Linear's backward ran before the raise
        status = backward_status(trace)
        assert status.state == "backward_observed"
        assert status.executed_flops is None and status.executed_evidence == "unknown"
        assert status.executed_flops_per_pass == ((1, None),)
        assert "not run to completion" in status.reason
        assert "UNKNOWN" in status.status_line
        assert record.flops_attribution.executed_flops is None
    finally:
        trace.cleanup()


def test_without_the_node_resolver_the_count_reads_unknown_never_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HAS_CURRENT_AUTOGRAD_NODE`` False: no call is charged to a node and the count is unknown."""

    from torchlens.backends.torch import _backward_dispatch
    from torchlens.utils import _torch_compat

    monkeypatch.setattr(_torch_compat, "HAS_CURRENT_AUTOGRAD_NODE", False)
    monkeypatch.setattr(_backward_dispatch, "get_current_autograd_node_fn", lambda: None)
    assert _torch_compat.get_torch_capability_snapshot()["HAS_CURRENT_AUTOGRAD_NODE"] is False
    model = nn.Linear(4, 3)
    trace = tl.trace(
        model, _leaf((5, 4), False, 4), capture=tl.options.CaptureOptions(backward_ready=True)
    )
    try:
        trace.log_backward(trace[trace.output_layers[0]].out.sum())
        (record,) = trace.autograd.backward_passes
        assert record.status == "ok" and record.dispatched_ops
        assert all(op.grad_fn_call_ref is None for op in record.dispatched_ops)
        attribution = record.flops_attribution
        assert attribution.executed_flops is None and not attribution.rows
        status = backward_status(trace)
        assert status.state == "backward_observed"
        assert status.executed_flops is None and status.executed_evidence == "unknown"
        assert "HAS_CURRENT_AUTOGRAD_NODE is False" in status.reason
        assert "UNKNOWN" in status.status_line
        tree = tl.report.cost_tree(trace)
        assert tree.backward_total is None
        assert "executed backward: unknown" in str(tree)
    finally:
        trace.cleanup()


def test_backward_guide_attribution_example_runs() -> None:
    """The runnable example in docs/backward.md executes as written."""

    from pathlib import Path

    page = (Path(__file__).parent.parent / "docs" / "backward.md").read_text(encoding="utf-8")
    marker = "<!-- flops2-attribution-example -->\n```python\n"
    start = page.index(marker) + len(marker)
    source = page[start : page.index("```", start)]
    exec(compile(source, "docs/backward.md", "exec"), {"__name__": "docs_backward_example"})


class _SplitOneUsed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(11)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        used, _unused = torch.split(self.lin(x), 2, dim=1)
        return used.sum()


class _SvdDropS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(12)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, _s, v = torch.svd(self.lin(x))
        return u.sum() + v.sum()


class _EigValuesUnused(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(13)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _w, v = torch.linalg.eig(self.lin(x))
        # |v| is invariant to each eigenvector's arbitrary phase, so the loss is well defined.
        return v.abs().sum()


#: Which multi-output op each model calls, and which of its outputs is unused.
_UNUSED_OUTPUT_CASES = {
    "split": (_SplitOneUsed, "split", 1),
    "svd": (_SvdDropS, "svd", 1),
    "eig": (_EigValuesUnused, "eig", 0),
}


@pytest.mark.parametrize("case", sorted(_UNUSED_OUTPUT_CASES))
def test_an_unused_output_gets_no_gradient_and_backward_runs(case: str) -> None:
    """A tensor hook handed None records nothing; the used outputs match plain autograd."""

    builder, func_name, unused_index = _UNUSED_OUTPUT_CASES[case]
    x = _leaf((4, 4), False, 14)
    plain = builder()
    plain(x).backward()
    expected = {name: param.grad.clone() for name, param in plain.named_parameters()}

    model = builder()
    trace = tl.trace(
        model, x, capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True)
    )
    try:
        trace.log_backward(trace[trace.output_layers[0]].out)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.grad, expected[name])
        outputs = [op for op in trace.layer_list if op.func_name == func_name]
        assert outputs, [op.func_name for op in trace.layer_list]
        unused = [op for op in outputs if op.multi_output_index == unused_index]
        assert unused and not any(op.has_grad for op in unused)
        used = [op for op in outputs if op.multi_output_index != unused_index]
        assert used and all(op.has_grad for op in used)
        assert backward_status(trace).executed_flops is not None
    finally:
        trace.cleanup()
