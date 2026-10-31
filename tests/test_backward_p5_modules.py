"""P5 backward module-containment tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants


class _ModuleContainmentModel(nn.Module):
    """Small model with paired, intervening, and parameter backward nodes."""

    def __init__(self) -> None:
        """Initialize child modules."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two module-owned operations separated by a functional op."""

        hidden = self.fc1(x)
        activated = torch.relu(hidden)
        return self.fc2(activated)


def _trace_with_backward(layers_to_save: str | list[str] = "all") -> tl.Trace:
    """Capture the containment fixture and run one backward pass.

    Parameters
    ----------
    layers_to_save:
        Forward activation save policy passed to ``tl.trace``.

    Returns
    -------
    tl.Trace
        Trace with projected backward records.
    """

    torch.manual_seed(0)
    model = _ModuleContainmentModel()
    x = torch.randn(2, 4, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save=layers_to_save, save_grads="all"),
    )
    loss = trace[trace.output_layers[0]].out.square().mean()
    trace.log_backward(loss, retain_graph=True)
    return trace


def test_backward_module_containment_sources_are_projected() -> None:
    """Backward GradFns expose paired, inferred, and post-forward module sources."""

    trace = _trace_with_backward()
    try:
        paired_by_module = {
            grad_fn.module_address
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.module_membership_source == "paired"
        }
        assert {"fc1", "fc2"}.issubset(paired_by_module)

        accumulate_modules = {
            grad_fn.module_address
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.class_name == "AccumulateGrad"
            and grad_fn.module_membership_source == "paired"
        }
        assert {"fc1", "fc2"}.issubset(accumulate_modules)

        inferred = [
            grad_fn
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.module_membership_source == "inferred"
        ]
        assert inferred
        assert {grad_fn.module_address for grad_fn in inferred} <= {"fc1", "fc2"}

        post_forward = [
            grad_fn
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.class_name in {"MeanBackward0", "PowBackward0"}
        ]
        assert post_forward
        assert all(grad_fn.module_membership_source is None for grad_fn in post_forward)
        assert all(grad_fn.module_address is None for grad_fn in post_forward)
    finally:
        trace.cleanup()


def test_backward_module_pairing_matches_selective_capture() -> None:
    """Two-pass capture keeps module containment parity with full-save capture."""

    full_trace = _trace_with_backward("all")
    selective_trace = _trace_with_backward([-1])
    try:
        full_pairs = {
            (grad_fn.class_name, grad_fn.module_address)
            for grad_fn in full_trace.autograd.grad_fns
            if grad_fn.module_membership_source == "paired"
        }
        selective_pairs = {
            (grad_fn.class_name, grad_fn.module_address)
            for grad_fn in selective_trace.autograd.grad_fns
            if grad_fn.module_membership_source == "paired"
        }
        assert full_pairs == selective_pairs
    finally:
        full_trace.cleanup()
        selective_trace.cleanup()


def test_recurrent_tanh_grad_fns_pair_directly_before_output_boundary() -> None:
    """Shared output grad_fn ids prefer compute ops and keep paired source metadata."""

    class TanhRNN(nn.Module):
        """Three-step tanh RNN with a shared linear module."""

        def __init__(self) -> None:
            """Initialize the recurrent projection."""

            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run three recurrent tanh iterations."""

            hidden = x
            for _iteration in range(3):
                hidden = torch.tanh(self.lin(hidden))
            return hidden

    torch.manual_seed(0)
    trace = tl.trace(
        TanhRNN(),
        torch.randn(2, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(
            layers_to_save="all", save_grads=True, backward_ready=True
        ),
    )
    try:
        trace[trace.output_layers[0]].out.sum().backward()

        tanh_grad_fns = [
            grad_fn for grad_fn in trace.autograd.grad_fns if grad_fn.class_name == "TanhBackward0"
        ]
        addmm_grad_fns = [
            grad_fn for grad_fn in trace.autograd.grad_fns if grad_fn.class_name == "AddmmBackward0"
        ]
        assert len(tanh_grad_fns) == 3
        assert len(addmm_grad_fns) == 3
        assert all(grad_fn.op_label == "tanh_1_2" for grad_fn in tanh_grad_fns)
        assert all(grad_fn.module_membership_source == "paired" for grad_fn in tanh_grad_fns)
        assert all(not grad_fn.op_label.startswith("output_") for grad_fn in tanh_grad_fns)
        assert all(grad_fn.op_label == "linear_1_1" for grad_fn in addmm_grad_fns)
        assert all(grad_fn.module_membership_source == "paired" for grad_fn in addmm_grad_fns)
    finally:
        trace.cleanup()


def test_backward_module_containment_invariant_rejects_bad_source() -> None:
    """Backward metadata invariants reject invalid module-containment flags."""

    trace = _trace_with_backward()
    try:
        victim = next(
            grad_fn
            for grad_fn in trace.autograd.grad_fns
            if grad_fn.module_membership_source is not None
        )
        victim.module_membership_source = "guessed"
        with pytest.raises(MetadataInvariantError, match="invalid module_membership_source"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_multi_pass_module_grad_fns_keep_layer_label_and_hooks_see_their_own_pass() -> None:
    """A module called twice: GradFns name the layer; live hooks see each pass's own op.

    ``GradFn.op_label`` is a layer label (the backward-graph invariant and the
    load path assume it), so both passes' ``AddmmBackward0`` nodes name the one
    multi-pass ``linear`` layer. The live backward hook context describes the
    exact pass each node differentiates, resolved from the pass whose recorded
    ``grad_fn_object_id`` is that node's identity, never from the multi-pass
    ``Layer`` (whose per-pass ``label`` refuses).
    """

    class SharedLinearTwice(nn.Module):
        """One linear module called at one call site twice (rows 26-27)."""

        def __init__(self) -> None:
            """Initialize the shared projection."""

            super().__init__()
            self.lin = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the shared projection twice around a ReLU."""

            for _call in range(2):
                x = torch.relu(self.lin(x))
            return x

    torch.manual_seed(0)
    trace = tl.trace(
        SharedLinearTwice(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    seen: list[tuple[str, Any, Any]] = []

    def _probe(grad: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Record the GradFn and the pass its hook context describes."""

        seen.append(
            (
                hook.kwargs["grad_fn_handle"].label,
                hook.layer_log["label"],
                hook.layer_log["pass_index"],
            )
        )
        return grad

    try:
        linear = next(layer for layer in trace.layer_logs.values() if layer.layer_type == "linear")
        assert linear.num_passes == 2
        trace.attach_hooks(
            tl.grad_fn(type="AddmmBackward0"), tl.bwd_hook(_probe), confirm_mutation=True
        )
        trace[trace.output_layers[0]].out.sum().backward()

        addmm = {
            gf.label: gf for gf in trace.autograd.grad_fns if gf.class_name == "AddmmBackward0"
        }
        assert len(addmm) == 2
        assert {gf.op_label for gf in addmm.values()} == {linear.layer_label}
        assert all(
            gf.op_label is None or gf.op_label in trace.layer_labels
            for gf in trace.autograd.grad_fns
        )

        # One record per grad_input slot; every slot of one node sees the same pass.
        by_node = {grad_fn_label: (label, pass_index) for grad_fn_label, label, pass_index in seen}
        assert len(set(seen)) == len(by_node)
        assert set(by_node) == set(addmm)
        assert sorted(by_node.values()) == [
            (f"{linear.layer_label}:1", 1),
            (f"{linear.layer_label}:2", 2),
        ]
        for grad_fn_label, (pass_label, _pass_index) in by_node.items():
            assert trace[pass_label].grad_fn_object_id == addmm[grad_fn_label].grad_fn_object_id
    finally:
        trace.cleanup()
