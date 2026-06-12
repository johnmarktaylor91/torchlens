"""P2 backward projection, label, accessor, and invariant tests."""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants


class _TwoPassModel(nn.Module):
    """Small model with a reusable backward graph."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 1)

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

        return self.fc2(torch.relu(self.fc1(x)))


def _captured_two_pass_trace() -> tl.Trace:
    """Capture one model and run two backward passes over the same graph.

    Returns
    -------
    tl.Trace
        Trace with two projected backward passes.
    """

    torch.manual_seed(0)
    model = _TwoPassModel()
    x = torch.randn(3, 4, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads="all")
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss, retain_graph=True)
    return trace


def test_backward_passes_and_grad_fn_calls_are_event_projections() -> None:
    """BackwardPass and GradFnCall projections expose global pass membership."""

    trace = _captured_two_pass_trace()
    try:
        assert trace.num_backward_passes == 2
        backward_passes = trace.backward_passes
        assert [backward_pass.pass_index for backward_pass in backward_passes] == [1, 2]
        assert backward_passes[0] is backward_passes.for_pass(1)
        assert trace.last_backward_pass.pass_index == 2
        assert backward_passes.for_pass(2).trigger == "backward"

        reused_grad_fn = next(grad_fn for grad_fn in trace.grad_fns if len(grad_fn.calls) == 2)
        assert [call.ordinal for call in reused_grad_fn.calls.values()] == [1, 2]
        assert [call.backward_pass_index for call in reused_grad_fn.calls.values()] == [1, 2]
        assert reused_grad_fn.calls[0].call_label == f"{reused_grad_fn.label}:1"
        assert reused_grad_fn.calls.for_pass(2).call_label == f"{reused_grad_fn.label}:2"
        assert trace.grad_fn_calls[f"{reused_grad_fn.label}:2"].backward_pass_index == 2
    finally:
        trace.cleanup()


def test_backward_labels_are_native_discovery_order_for_paired_nodes() -> None:
    """Paired GradFn labels use backward discovery numbering, not forward labels."""

    trace = _captured_two_pass_trace()
    try:
        labels = [grad_fn.label for grad_fn in trace.grad_fns]
        assert all(re.fullmatch(r"[a-z0-9_]+_back_[1-9]\d*_[1-9]\d*", label) for label in labels)
        paired = [grad_fn for grad_fn in trace.grad_fns if grad_fn.has_op]
        assert paired
        assert all(grad_fn.op_label not in grad_fn.label for grad_fn in paired)
        assert [grad_fn.step_index for grad_fn in trace.grad_fns] == list(
            range(1, len(trace.grad_fns) + 1)
        )
    finally:
        trace.cleanup()


def test_backward_projection_structural_invariants_trip_on_corruption() -> None:
    """Backward structural invariants reject non-dense local call ordinals."""

    trace = _captured_two_pass_trace()
    try:
        victim = next(grad_fn for grad_fn in trace.grad_fns if len(grad_fn.calls) == 2)
        trace._capture_events.backward_events.clear()
        call = victim.calls._dict.pop(2)
        victim.calls._dict[3] = call

        with pytest.raises(MetadataInvariantError, match="non-dense local call ordinals"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_higher_order_autograd_grad_records_creator_order() -> None:
    """create_graph autograd.grad records creator attribution and pass order."""

    class HigherOrderModel(nn.Module):
        """Tiny nonlinear model with differentiable first gradients."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.tanh(x) ** 3).sum()

    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    trace = tl.trace(HigherOrderModel(), x, save_grads="all")
    try:
        loss = trace[trace.output_layers[0]].out
        first_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(first_grad.sum(), x, retain_graph=True)

        assert [backward_pass.order for backward_pass in trace.backward_passes] == [1, 2]
        higher_order_grad_fns = [
            grad_fn
            for grad_fn in trace.grad_fns
            if grad_fn.origin_backward_pass == 1 and grad_fn.creator_object_id is not None
        ]
        assert higher_order_grad_fns
        assert {grad_fn.order for grad_fn in higher_order_grad_fns} == {2}
        assert all(grad_fn.differentiates is not None for grad_fn in higher_order_grad_fns)
        assert all(
            backward_pass.order_attribution_coverage == 1.0
            for backward_pass in trace.backward_passes
        )
    finally:
        trace.cleanup()


def test_higher_order_nodes_reach_third_order() -> None:
    """Repeated differentiable grads attribute newly-created nodes past order two."""

    class ThirdOrderModel(nn.Module):
        """Tiny nonlinear model whose derivatives remain differentiable."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.sin(x) * torch.exp(x)).sum()

    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    trace = tl.trace(ThirdOrderModel(), x, save_grads="all")
    try:
        grad = trace[trace.output_layers[0]].out
        for _order_index in range(2):
            grad = torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)[0]

        created_in_second_pass = [
            grad_fn
            for grad_fn in trace.grad_fns
            if grad_fn.origin_backward_pass == 2 and grad_fn.creator_object_id is not None
        ]
        assert created_in_second_pass
        assert 3 in {grad_fn.order for grad_fn in created_in_second_pass}
        assert max(grad_fn.order for grad_fn in trace.grad_fns if grad_fn.order is not None) >= 3
        assert all(
            backward_pass.order_attribution_coverage == 1.0
            for backward_pass in trace.backward_passes
        )
    finally:
        trace.cleanup()


def test_higher_order_induction_stress_reaches_expected_depth() -> None:
    """Higher-order creator attribution grows inductively across repeated grads."""

    class InductionModel(nn.Module):
        """Tiny nonlinear model with nonzero repeated derivatives."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.sin(x) * torch.exp(x)).sum()

    num_grad_passes = 4
    torch.manual_seed(0)
    x = torch.randn(2, requires_grad=True)
    trace = tl.trace(InductionModel(), x, save_grads="all")
    try:
        grad = trace[trace.output_layers[0]].out
        for _pass_index in range(num_grad_passes):
            grad = torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)[0]

        resolved_orders = {grad_fn.order for grad_fn in trace.grad_fns if grad_fn.order is not None}
        assert set(range(1, num_grad_passes + 2)).issubset(resolved_orders)
        assert max(resolved_orders) == num_grad_passes + 1
        assert all(
            grad_fn.order == trace.grad_fn_logs[grad_fn.creator_object_id].order + 1
            for grad_fn in trace.grad_fns
            if grad_fn.creator_object_id is not None and grad_fn.order is not None
        )
    finally:
        trace.cleanup()


def test_higher_order_mixed_order_pass_records_reused_and_created_nodes() -> None:
    """A differentiable higher-order pass may fire reused and newly-created orders."""

    class MixedOrderModel(nn.Module):
        """Tiny nonlinear model that refires first-order nodes during higher-order grads."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.tanh(x) ** 4).sum()

    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    trace = tl.trace(MixedOrderModel(), x, save_grads="all")
    try:
        grad = trace[trace.output_layers[0]].out
        grad = torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)

        label_to_order = {grad_fn.label: grad_fn.order for grad_fn in trace.grad_fns}
        second_pass_orders = {
            label_to_order[call.label] for call in trace.backward_passes.for_pass(2).grad_fn_calls
        }
        assert {1, 2}.issubset(second_pass_orders)
        assert any(
            grad_fn.origin_backward_pass == 2 and grad_fn.order == 3 for grad_fn in trace.grad_fns
        )
    finally:
        trace.cleanup()
