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
        assert trace.save_grads is True
        assert not hasattr(trace, "save_gradients")
        assert not hasattr(trace, "gradients_to_save")
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


def test_log_backward_save_grads_override_widens_and_narrows_payloads() -> None:
    """Per-trigger save_grads overrides decide payload retention at hook fire time."""

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads=False)
    try:
        loss = trace[trace.output_layers[0]].out
        trace.log_backward(loss, retain_graph=True, save_grads=True)
        assert any(record.grad is not None for op in trace.ops for record in op.grads)

        trace.log_backward(loss, retain_graph=True, save_grads=None)
        second_pass_records = [
            record for op in trace.ops for record in op.grads if record.backward_pass_index == 2
        ]
        assert second_pass_records
        assert all(record.grad is None for record in second_pass_records)
        assert [pass_record.save_grads_policy for pass_record in trace.backward_passes] == [
            "True",
            "None",
        ]
    finally:
        trace.cleanup()


def test_save_grads_predicate_can_select_backward_pass() -> None:
    """Trace-side save_grads predicates can match the current backward pass."""

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads=tl.in_backward_pass(2))
    try:
        loss = trace[trace.output_layers[0]].out
        trace.log_backward(loss, retain_graph=True)
        trace.log_backward(loss, retain_graph=True)
        first_pass_records = [
            record for op in trace.ops for record in op.grads if record.backward_pass_index == 1
        ]
        second_pass_records = [
            record for op in trace.ops for record in op.grads if record.backward_pass_index == 2
        ]
        assert first_pass_records
        assert second_pass_records
        assert all(record.grad is None for record in first_pass_records)
        assert any(record.grad is not None for record in second_pass_records)
    finally:
        trace.cleanup()


def test_grad_transform_is_projected_per_backward_pass() -> None:
    """Per-pass gradient records retain transformed payloads from the event stream."""

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        layers_to_save="all",
        save_grads=True,
        grad_transform=lambda grad: grad.mean(),
    )
    try:
        loss = trace[trace.output_layers[0]].out
        trace.log_backward(loss, retain_graph=True)
        trace.log_backward(loss, retain_graph=True)
        op = next(op for op in trace.saved_grad_ops if len(op.grads) == 2)
        assert all(record.transformed_grad is not None for record in op.grads)
        assert all(record.transformed_grad_shape == () for record in op.grads)
        assert torch.equal(op.grads.for_pass(1).transformed_grad, op.grads[0].grad.mean())
        assert torch.equal(op.grads.for_pass(2).transformed_grad, op.grads[1].grad.mean())
    finally:
        trace.cleanup()


def test_save_raw_gradients_false_keeps_transformed_per_pass_payload() -> None:
    """Raw-disabled grad capture stores transformed per-pass payloads only."""

    torch.manual_seed(0)
    model = _PayloadModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        layers_to_save="all",
        save_grads=True,
        grad_transform=lambda grad: grad.mean(),
        save_raw_gradients=False,
    )
    try:
        loss = trace[trace.output_layers[0]].out
        trace.log_backward(loss)
        op = next(op for op in trace.saved_grad_ops if len(op.grads) == 1)
        record = op.grads.for_pass(1)
        assert record.grad is None
        assert record.transformed_grad is not None
        assert op.grad is None
        assert op.transformed_grad is not None
        expected_backward_memory = sum(
            int(record.transformed_gradient_memory or 0)
            for saved_op in trace.saved_grad_ops
            for record in saved_op.grads
            if record.transformed_grad is not None
        )
        expected_gradient_memory = sum(
            int(record.memory)
            for saved_op in trace.saved_grad_ops
            for record in saved_op.grads
            if record.is_saved
        )
        assert trace.total_backward_memory == expected_backward_memory
        assert trace.total_gradient_memory == expected_gradient_memory
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
