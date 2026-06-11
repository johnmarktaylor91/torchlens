"""P3 backward payload retention and per-pass gradient access tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _PayloadModel(nn.Module):
    """Small model with reusable backward payloads."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.fc(x)).sum()


def _trace_with_two_backward_passes() -> tl.Trace:
    """Capture a tiny model and run two retained backward passes.

    Returns
    -------
    tl.Trace
        Trace with saved op gradient records.
    """

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads=True)
    loss = trace[trace.output_layers[0]].out
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss, retain_graph=True)
    return trace


def test_op_grads_are_local_dense_per_pass_records() -> None:
    """Op.grads stores one local-dense record per saved backward pass."""

    trace = _trace_with_two_backward_passes()
    try:
        op = next(op for op in trace.saved_grad_ops if len(op.grads) == 2)
        assert [record.ordinal for record in op.grads] == [1, 2]
        assert [record.backward_pass_index for record in op.grads] == [1, 2]
        assert torch.equal(op.grads.for_pass(1).grad, op.grad_for(bwd=1))
        assert torch.equal(op.grads.for_pass(2).grad, op.grad_for(bwd=2))
    finally:
        trace.cleanup()


def test_save_grads_is_the_trace_side_public_surface() -> None:
    """Trace accepts save_grads and rejects removed gradient-save aliases."""

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads=True)
    try:
        assert trace.save_gradients is True
        assert trace.gradients_to_save == "all"
    finally:
        trace.cleanup()

    with pytest.raises(TypeError):
        tl.trace(model, x, layers_to_save="all", save_gradients=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        tl.trace(model, x, layers_to_save="all", gradients_to_save="all")  # type: ignore[call-arg]


def test_op_grad_is_loud_when_multiple_passes_are_saved() -> None:
    """Plain Op.grad raises when several per-pass gradients are retained."""

    trace = _trace_with_two_backward_passes()
    try:
        op = next(op for op in trace.saved_grad_ops if len(op.grads) == 2)
        with pytest.raises(ValueError, match=r"use op\.grads\[\.\.\.\] / op\.grad_for\(bwd=k\)"):
            _ = op.grad
    finally:
        trace.cleanup()


def test_to_pandas_represents_ambiguous_grad_without_raising() -> None:
    """Trace.to_pandas exposes ambiguous per-pass grads as None plus a count."""

    trace = _trace_with_two_backward_passes()
    try:
        frame = trace.to_pandas()
        grad_rows = frame[frame["num_saved_grads"] > 1]
        assert not grad_rows.empty
        assert grad_rows["grad"].isna().all()
    finally:
        trace.cleanup()


def test_total_backward_memory_counts_unique_saved_payload_refs() -> None:
    """Trace.total_backward_memory is derived from saved OpGradObserved payload refs."""

    trace = _trace_with_two_backward_passes()
    try:
        expected = sum(
            int(record.memory)
            for op in trace.saved_grad_ops
            for record in op.grads
            if record.grad is not None
        )
        assert trace.total_backward_memory == expected
        assert trace.total_gradient_memory == expected
    finally:
        trace.cleanup()


def test_param_grads_capture_accumulategrad_increments() -> None:
    """Param.grads records accumulating backward increments while Param.grad stays live."""

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads=True)
    try:
        loss = trace[trace.output_layers[0]].out
        trace.log_backward(loss)
        param = trace.params["fc.weight"]
        assert param.grad is model.fc.weight.grad
        assert len(param.grads) == 1
        assert param.grads.for_pass(1).grad is not None
        assert torch.equal(param.grads.for_pass(1).grad, model.fc.weight.grad)
    finally:
        trace.cleanup()
