"""Autograd calls on a live trace: reference saves stay graph-connected, and
repeated ``torch.autograd.grad`` calls fold new grad-fn nodes into the backward
projection with flat per-call work instead of rebuilding it every call."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import Op
from torchlens.backends.torch import backward as backward_mod
from torchlens.options import CaptureOptions


class _ResidualBlock(nn.Module):
    """One residual MLP block: ``x + relu(linear(x))``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.linear(x))


class _ResidualNet(nn.Module):
    """Three residual blocks, so ``blocks.0`` feeds ``blocks.2`` through ``blocks.1``."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ResidualBlock() for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _model_and_input() -> tuple[_ResidualNet, torch.Tensor]:
    torch.manual_seed(0)
    return _ResidualNet().eval(), torch.randn(2, 8)


@pytest.mark.parametrize("backward_ready", [False, True])
def test_reference_save_keeps_residual_outputs_graph_connected(backward_ready: bool) -> None:
    """A reference save of a residual block output is the live tensor, not a clone.

    The residual ``__add__`` is not a mutating op, so reference mode must not
    fall back to a copy: the saved block output keeps its own autograd node and
    ``torch.autograd.grad`` between two saved outputs matches the model's.
    """

    model, x = _model_and_input()
    trace = tl.trace(
        model, x, capture=CaptureOptions(save_mode="reference", backward_ready=backward_ready)
    )
    early = trace["blocks.0"].out
    late = trace["blocks.2"].out
    assert early.grad_fn is not None
    assert type(early.grad_fn).__name__ != "CloneBackward0"
    assert type(late.grad_fn).__name__ != "CloneBackward0"

    (captured,) = torch.autograd.grad(late.sum(), early)
    hidden = model.blocks[0](x)
    (expected,) = torch.autograd.grad(model.blocks[2](model.blocks[1](hidden)).sum(), hidden)
    torch.testing.assert_close(captured, expected)


def _projection_snapshot(trace: tl.Trace) -> dict[str, Any]:
    """Return every backward projection field a growth fold writes, by value."""

    grad_fns = {
        object_id: (
            record.label,
            record.type_index,
            record.ordinal_index,
            record.step_index,
            record.order,
            record.op_label,
            record.creator_object_id,
            record.differentiates,
            record.origin_backward_pass,
            list(record.children),
            list(record.parents),
            list(record.siblings),
            list(record.co_parents),
            record.module_address,
            list(record.modules),
            record.module_membership_source,
            sorted(record.calls),
        )
        for object_id, record in trace._grad_fn_logs.items()
    }
    passes = {
        pass_index: (record.order, record.status, len(record.grad_fn_calls))
        for pass_index, record in trace._backward_pass_logs.items()
    }
    return {
        "grad_fns": grad_fns,
        "passes": passes,
        "order": list(trace.autograd.grad_fn_order),
        "num_passes": len(trace.autograd.backward_passes),
    }


def _per_element_grads(trace: tl.Trace, calls: int) -> None:
    """Take one ``torch.autograd.grad`` per output element, like a saliency loop."""

    early = trace["blocks.0"].out
    flat = trace["blocks.2"].out.reshape(-1)
    for index in range(calls):
        torch.autograd.grad(flat[index], early, retain_graph=True)


def test_per_element_grad_calls_fold_new_nodes_like_a_scratch_rebuild() -> None:
    """Each call adds one indexing node; the fold appends it and equals a rebuild."""

    model, x = _model_and_input()
    trace = tl.trace(model, x)
    rebuilds: list[int] = []
    real_impl = backward_mod._materialize_backward_projections_impl

    def counting_impl(trace_arg: tl.Trace, events: list, stream: object = None) -> None:
        rebuilds.append(len(events))
        real_impl(trace_arg, events, stream=stream)

    with mock.patch.object(backward_mod, "_materialize_backward_projections_impl", counting_impl):
        _per_element_grads(trace, 6)
        folded = _projection_snapshot(trace)

    assert folded["num_passes"] == 6
    assert len(rebuilds) == 1, f"calls after the first must fold, not rebuild: {rebuilds}"
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)
    assert _projection_snapshot(trace) == folded


def test_per_element_grad_call_work_stays_flat() -> None:
    """Op attribute reads per ``autograd.grad`` call stay constant as calls accumulate.

    Before the growth fold every call rebuilt the projection and resolved each
    grad-fn's op by scanning every op, so the reads grew with the calls taken.
    """

    model, x = _model_and_input()
    trace = tl.trace(model, x)
    early = trace["blocks.0"].out
    flat = trace["blocks.2"].out.reshape(-1)
    torch.autograd.grad(flat[0], early, retain_graph=True)

    reads = [0]
    real_getattribute = Op.__dict__["__getattribute__"]

    def counting_getattribute(self: Op, name: str) -> Any:
        reads[0] += 1
        return real_getattribute(self, name)

    per_call: list[int] = []
    passes: list[int] = []
    with mock.patch.object(Op, "__getattribute__", counting_getattribute):
        for index in range(1, 9):
            reads[0] = 0
            torch.autograd.grad(flat[index], early, retain_graph=True)
            per_call.append(reads[0])
            passes.append(len(trace._backward_pass_logs))

    assert passes == list(range(2, 10)), "one BackwardPass per call"
    assert len(set(per_call)) == 1, f"op attribute reads per call grew: {per_call}"
