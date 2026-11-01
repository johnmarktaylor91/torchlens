"""P2 backward projection, label, accessor, and invariant tests."""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.grad_fn import GradFn
from torchlens.backends.torch.backward import _assign_grad_fn_orders
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
    trace = tl.trace(
        model, x, capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all")
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss, retain_graph=True)
    return trace


def test_backward_passes_and_grad_fn_calls_are_event_projections() -> None:
    """BackwardPass and GradFnCall projections expose global pass membership."""

    trace = _captured_two_pass_trace()
    try:
        assert trace.num_backward_passes == 2
        backward_passes = trace.autograd.backward_passes
        assert [backward_pass.pass_index for backward_pass in backward_passes] == [1, 2]
        assert backward_passes[0] is backward_passes.for_pass(1)
        assert trace.autograd.last_backward_pass.pass_index == 2
        assert backward_passes.for_pass(2).trigger == "backward"

        reused_grad_fn = next(
            grad_fn for grad_fn in trace.autograd.grad_fns if len(grad_fn.calls) == 2
        )
        assert [call.ordinal for call in reused_grad_fn.calls.values()] == [1, 2]
        assert [call.backward_pass_index for call in reused_grad_fn.calls.values()] == [1, 2]
        assert reused_grad_fn.calls[0].call_label == f"{reused_grad_fn.label}:1"
        assert reused_grad_fn.calls.for_pass(2).call_label == f"{reused_grad_fn.label}:2"
        assert trace.autograd.grad_fn_calls[f"{reused_grad_fn.label}:2"].backward_pass_index == 2
    finally:
        trace.cleanup()


def test_backward_labels_are_native_discovery_order_for_paired_nodes() -> None:
    """Paired GradFn labels use backward discovery numbering, not forward labels."""

    trace = _captured_two_pass_trace()
    try:
        labels = [grad_fn.label for grad_fn in trace.autograd.grad_fns]
        assert all(re.fullmatch(r"[a-z0-9_]+_back_[1-9]\d*_[1-9]\d*", label) for label in labels)
        paired = [grad_fn for grad_fn in trace.autograd.grad_fns if grad_fn.has_op]
        assert paired
        assert all(grad_fn.op_label not in grad_fn.label for grad_fn in paired)
        assert [grad_fn.step_index for grad_fn in trace.autograd.grad_fns] == list(
            range(1, len(trace.autograd.grad_fns) + 1)
        )
    finally:
        trace.cleanup()


def test_backward_projection_structural_invariants_trip_on_corruption() -> None:
    """Backward structural invariants reject non-dense local call ordinals."""

    trace = _captured_two_pass_trace()
    try:
        victim = next(grad_fn for grad_fn in trace.autograd.grad_fns if len(grad_fn.calls) == 2)
        # Corrupt only the projected calls dict: deleting journal events instead
        # would trip the deletion-visible seq invariant before this pin fires.
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
    trace = tl.trace(HigherOrderModel(), x, capture=tl.options.CaptureOptions(save_grads="all"))
    try:
        loss = trace[trace.output_layers[0]].out
        first_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(first_grad.sum(), x, retain_graph=True)

        assert [backward_pass.order for backward_pass in trace.autograd.backward_passes] == [1, 2]
        higher_order_grad_fns = [
            grad_fn
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.origin_backward_pass == 1 and grad_fn.creator_object_id is not None
        ]
        assert higher_order_grad_fns
        assert {grad_fn.order for grad_fn in higher_order_grad_fns} == {2}
        assert all(grad_fn.differentiates is not None for grad_fn in higher_order_grad_fns)
        assert all(
            backward_pass.order_attribution_coverage == 1.0
            for backward_pass in trace.autograd.backward_passes
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
    trace = tl.trace(ThirdOrderModel(), x, capture=tl.options.CaptureOptions(save_grads="all"))
    try:
        grad = trace[trace.output_layers[0]].out
        for _order_index in range(2):
            grad = torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)[0]

        created_in_second_pass = [
            grad_fn
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.origin_backward_pass == 2 and grad_fn.creator_object_id is not None
        ]
        assert created_in_second_pass
        assert 3 in {grad_fn.order for grad_fn in created_in_second_pass}
        assert (
            max(grad_fn.order for grad_fn in trace.autograd.grad_fns if grad_fn.order is not None)
            >= 3
        )
        assert all(
            backward_pass.order_attribution_coverage == 1.0
            for backward_pass in trace.autograd.backward_passes
        )
    finally:
        trace.cleanup()


def test_third_order_scalar_chain_records_root_based_pass_order() -> None:
    """Third-order autograd.grad pass order follows the root creator pass."""

    class ScalarPowerModel(nn.Module):
        """Scalar parameter model with a nonzero third derivative."""

        def __init__(self) -> None:
            """Initialize the scalar parameter."""

            super().__init__()
            self.w = nn.Parameter(torch.tensor(2.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar fourth-power output."""

            return (self.w * x).pow(4)

    model = ScalarPowerModel()
    x = torch.tensor(3.0)
    trace = tl.trace(
        model, x, capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True)
    )
    try:
        output = trace[trace.output_layers[0]].out
        grad = torch.autograd.grad(output, model.w, create_graph=True, retain_graph=True)[0]
        grad = torch.autograd.grad(grad, model.w, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(grad, model.w, retain_graph=True)

        assert [backward_pass.order for backward_pass in trace.autograd.backward_passes] == [
            1,
            2,
            3,
        ]
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
    trace = tl.trace(InductionModel(), x, capture=tl.options.CaptureOptions(save_grads="all"))
    try:
        grad = trace[trace.output_layers[0]].out
        for _pass_index in range(num_grad_passes):
            grad = torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)[0]

        assert [backward_pass.order for backward_pass in trace.autograd.backward_passes] == list(
            range(1, num_grad_passes + 1)
        )
        resolved_orders = {
            grad_fn.order for grad_fn in trace.autograd.grad_fns if grad_fn.order is not None
        }
        assert set(range(1, num_grad_passes + 2)).issubset(resolved_orders)
        assert max(resolved_orders) == num_grad_passes + 1
        assert all(
            grad_fn.order == trace.grad_fn_logs[grad_fn.creator_object_id].order + 1
            for grad_fn in trace.autograd.grad_fns
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
    trace = tl.trace(MixedOrderModel(), x, capture=tl.options.CaptureOptions(save_grads="all"))
    try:
        grad = trace[trace.output_layers[0]].out
        grad = torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(grad.sum(), x, create_graph=True, retain_graph=True)

        label_to_order = {grad_fn.label: grad_fn.order for grad_fn in trace.autograd.grad_fns}
        second_pass_orders = {
            label_to_order[call.label]
            for call in trace.autograd.backward_passes.for_pass(2).grad_fn_calls
        }
        assert {1, 2}.issubset(second_pass_orders)
        assert any(
            grad_fn.origin_backward_pass == 2 and grad_fn.order == 3
            for grad_fn in trace.autograd.grad_fns
        )
    finally:
        trace.cleanup()


def _order_record(
    object_id: int,
    *,
    creator_object_id: int | None = None,
    next_ids: tuple[int, ...] = (),
    has_op: bool = False,
) -> GradFn:
    """Build a minimal GradFn record for order-assignment tests."""

    return GradFn(
        grad_fn_object_id=object_id,
        class_name="Node",
        class_qualname="torch.autograd.Node",
        is_custom=False,
        label=f"node_{object_id}",
        type="node",
        type_index=object_id,
        ordinal_index=object_id,
        has_op=has_op,
        creator_object_id=creator_object_id,
        next_grad_fn_ids=list(next_ids),
    )


def test_assign_grad_fn_orders_reaches_a_fixed_point_through_inferred_creators() -> None:
    """A creation of an inferred-order node sits one order above the inferred order.

    Chain: ``root`` (forward, order 1) creates ``first`` (order 2); ``bridge`` is a
    creator-less, op-less node feeding ``first`` (inferred to 2); ``bridge`` creates
    ``second``, which must be 3, and ``second`` creates ``third``, which must be 4.
    Before the fix the creator rule ran once, before inference, and left ``second``
    at 2 (the base ``backward_graph_invariants`` failure of AGT-1485).
    """

    root, first, bridge, second, third = 1, 2, 3, 4, 5
    records = {
        # Discovery order puts creations after their creators, and the bridge
        # (discovered at the second pass's root walk) after the first pass's nodes.
        root: _order_record(root),
        first: _order_record(first, creator_object_id=root),
        bridge: _order_record(bridge, next_ids=(first,)),
        second: _order_record(second, creator_object_id=bridge),
        third: _order_record(third, creator_object_id=second),
    }
    _assign_grad_fn_orders(records)
    assert {r.label: r.order for r in records.values()} == {
        "node_1": 1,
        "node_2": 2,
        "node_3": 2,
        "node_4": 3,
        "node_5": 4,
    }


def test_assign_grad_fn_orders_keeps_paired_forward_nodes_at_order_one() -> None:
    """Inference never lifts a node paired with a forward op, so its creations stay at 2."""

    root, first, paired, creation = 1, 2, 3, 4
    records = {
        root: _order_record(root),
        first: _order_record(first, creator_object_id=root),
        paired: _order_record(paired, next_ids=(first,), has_op=True),
        creation: _order_record(creation, creator_object_id=paired),
    }
    _assign_grad_fn_orders(records)
    assert [r.order for r in records.values()] == [1, 2, 1, 2]


def test_forward_op_on_a_first_order_gradient_keeps_creator_orders_consistent() -> None:
    """Differentiating a forward op applied to a gradient keeps every order at creator + 1.

    ``penalty = (g ** 2).sum()`` applies forward ops to the first-order gradient ``g``;
    their grad_fn nodes are inferred into order 2, and the nodes the second pass creates
    by differentiating them must be order 3, or ``check_metadata_invariants`` fails on
    ``backward_graph_invariants`` (AGT-1485).
    """

    class TanhSquare(nn.Module):
        """Scalar nonlinear model with nonzero higher derivatives."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return torch.tanh(x).pow(2).sum()

    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    trace = tl.trace(TanhSquare(), x, capture=tl.options.CaptureOptions(save_grads="all"))
    try:
        out = trace[trace.output_layers[0]].out
        (grad,) = torch.autograd.grad(out, x, create_graph=True, retain_graph=True)
        penalty = (grad**2).sum()
        torch.autograd.grad(penalty, x, create_graph=True, retain_graph=True)

        check_metadata_invariants(trace)
        inferred_ids = {
            grad_fn.grad_fn_object_id
            for grad_fn in trace.grad_fns
            if grad_fn.creator_object_id is None and grad_fn.order == 2
        }
        assert inferred_ids, "the penalty's forward nodes were not inferred into order 2"
        creations = [
            grad_fn for grad_fn in trace.grad_fns if grad_fn.creator_object_id in inferred_ids
        ]
        assert creations
        assert {grad_fn.order for grad_fn in creations} == {3}
        assert all(
            grad_fn.order == trace.grad_fn_logs[grad_fn.creator_object_id].order + 1
            for grad_fn in trace.grad_fns
            if grad_fn.creator_object_id is not None and grad_fn.order is not None
        )
    finally:
        trace.cleanup()


def test_backward_epoch_publication_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed epoch adoption publishes nothing, and the retry converges.

    Review finding 2: revision/watermark were published and the epoch
    list replaced BEFORE the three ``adopt_rows`` calls, so a mid-adoption
    failure left the facade holding the complete projection, the core epoch
    holding only ``grad_fn``, and the published revision suppressing retry
    (permanent facade/store divergence).
    """

    import torchlens._trace_core.record_rows as record_rows

    class TinyGrad(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).sum()

    torch.manual_seed(0)
    x = torch.randn(1, 3, requires_grad=True)
    trace = tl.trace(
        TinyGrad(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
    )
    try:
        out = trace[trace.output_layers[0]].out
        out.backward(retain_graph=True)
        _ = trace.autograd.grad_fns  # materialize the baseline projection + epoch

        real_adopt = record_rows.adopt_rows
        adopt_calls: list[str] = []

        def failing_adopt(store_registry: dict, kind: str, records: object) -> None:
            adopt_calls.append(kind)
            if kind == "grad_fn_call":
                raise RuntimeError("injected epoch adoption failure")
            real_adopt(store_registry, kind, records)

        # A second backward invalidates the projection; the pass-end
        # re-materialization hits the injected mid-adoption failure.
        monkeypatch.setattr(record_rows, "adopt_rows", failing_adopt)
        with pytest.raises(RuntimeError, match="injected epoch adoption failure"):
            out.backward()
        assert "grad_fn_call" in adopt_calls

        core = trace.__dict__["_trace_core"]
        # Nothing published: no partial epoch, no revision to suppress retry.
        assert core.backward_epochs == []
        assert trace._backward_projection_revision is None

        # Retry (without the injection) fully converges: one epoch holding
        # all three kinds, stamps matching the facade.
        monkeypatch.setattr(record_rows, "adopt_rows", real_adopt)
        grad_fns = trace.autograd.grad_fns
        assert len(grad_fns) > 0
        assert len(trace.autograd.backward_passes) == 2
        assert len(core.backward_epochs) == 1
        epoch = core.backward_epochs[0]
        assert set(epoch.stores) == {"grad_fn", "grad_fn_call", "backward_pass"}
        assert epoch.revision == trace._backward_projection_revision
        assert epoch.watermark == trace._backward_projection_event_count
    finally:
        trace.cleanup()


def test_save_works_after_two_differentiable_backward_passes(tmp_path) -> None:
    """grind-r6 b5 R45 (sol HIGH root, probe-proven): Trace.save after 2x create_graph.

    The higher-order rewalk parked its label type counter in
    ``trace.__dict__["_backward_grad_fn_type_counter"]`` -- an undeclared
    private field with no _PORTABLE_STATE_SPEC row, so ``Trace.save`` raised
    ArtifactError after two differentiable grad passes. The counter is
    now rewalk-scoped scratch and never touches the trace.
    """

    model = nn.Linear(4, 2)
    x = torch.randn(1, 4, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    try:
        trace.log_backward(trace[-1].out.sum(), retain_graph=True, create_graph=True)
        trace.log_backward(trace[-1].out.sum(), create_graph=True)
        assert "_backward_grad_fn_type_counter" not in trace.__dict__
        trace.save(str(tmp_path / "two_pass.tlspec"))
    finally:
        trace.cleanup()
