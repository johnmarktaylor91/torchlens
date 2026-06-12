"""P5 backward module-containment tests."""

from __future__ import annotations

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
    trace = tl.trace(model, x, layers_to_save=layers_to_save, save_grads="all")
    loss = trace[trace.output_layers[0]].out.square().mean()
    trace.log_backward(loss, retain_graph=True)
    return trace


def test_backward_module_containment_sources_are_projected() -> None:
    """Backward GradFns expose paired, inferred, and post-forward module sources."""

    trace = _trace_with_backward()
    try:
        paired_by_module = {
            grad_fn.module_address
            for grad_fn in trace.grad_fns
            if grad_fn.module_membership_source == "paired"
        }
        assert {"fc1", "fc2"}.issubset(paired_by_module)

        accumulate_modules = {
            grad_fn.module_address
            for grad_fn in trace.grad_fns
            if grad_fn.class_name == "AccumulateGrad"
            and grad_fn.module_membership_source == "paired"
        }
        assert {"fc1", "fc2"}.issubset(accumulate_modules)

        inferred = [
            grad_fn for grad_fn in trace.grad_fns if grad_fn.module_membership_source == "inferred"
        ]
        assert inferred
        assert {grad_fn.module_address for grad_fn in inferred} <= {"fc1", "fc2"}

        post_forward = [
            grad_fn
            for grad_fn in trace.grad_fns
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
            for grad_fn in full_trace.grad_fns
            if grad_fn.module_membership_source == "paired"
        }
        selective_pairs = {
            (grad_fn.class_name, grad_fn.module_address)
            for grad_fn in selective_trace.grad_fns
            if grad_fn.module_membership_source == "paired"
        }
        assert full_pairs == selective_pairs
    finally:
        full_trace.cleanup()
        selective_trace.cleanup()


def test_backward_module_containment_invariant_rejects_bad_source() -> None:
    """Backward metadata invariants reject invalid module-containment flags."""

    trace = _trace_with_backward()
    try:
        victim = next(
            grad_fn for grad_fn in trace.grad_fns if grad_fn.module_membership_source is not None
        )
        victim.module_membership_source = "guessed"
        with pytest.raises(MetadataInvariantError, match="invalid module_membership_source"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()
