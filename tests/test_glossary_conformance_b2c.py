"""Focused coverage for glossary-conformance phase B2c structural fields."""

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _B2cModel(nn.Module):
    """Model docstring captured on Trace.class_docstring."""

    def __init__(self) -> None:
        """Initialize B2c model modules."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Run the B2c forward pass."""

        return torch.relu(self.linear(x)) * scale


def _make_trace(**kwargs: Any) -> tl.Trace:
    """Create a small B2c trace."""

    model = _B2cModel()
    x = torch.randn(2, 3, requires_grad=True)
    return tl.trace(model, x, **kwargs)


@pytest.mark.smoke
def test_trace_parent_and_root_trace_resolve_fork_lineage() -> None:
    """Trace parent/root properties resolve the legacy parent_run weakrefs."""

    root = _make_trace(layers_to_save="all")
    child = root.fork("child")
    grandchild = child.fork("grandchild")

    assert root.parent_trace is None
    assert root.root_trace is None
    assert child.parent_trace is root
    assert child.root_trace is root
    assert grandchild.parent_trace is child
    assert grandchild.root_trace is root


@pytest.mark.smoke
def test_trace_source_introspection_fields_are_captured() -> None:
    """Trace exposes source model docstrings and signatures."""

    trace = _make_trace(layers_to_save="all")

    assert trace.class_docstring == _B2cModel.__doc__
    assert trace.init_signature == "(self) -> None"
    assert trace.init_docstring == _B2cModel.__init__.__doc__
    assert trace.forward_signature == "(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor"
    assert trace.forward_docstring == _B2cModel.forward.__doc__


@pytest.mark.smoke
def test_grad_fn_calls_accessor_and_call_savedness() -> None:
    """GradFn.calls replaces ops and GradFnCall.is_saved mirrors saved grad payloads."""

    trace = _make_trace(layers_to_save="all", save_grads=True)
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    grad_fn = next(record for record in trace.grad_fns if record.num_calls)
    grad_fn_call = grad_fn.calls[0]

    assert not hasattr(grad_fn, "ops")
    assert trace.grad_fn_calls[grad_fn_call.call_label] is grad_fn_call
    assert grad_fn_call.is_saved is (
        grad_fn_call.grad_inputs is not None or grad_fn_call.grad_outputs is not None
    )


@pytest.mark.smoke
def test_module_boundary_fields_are_populated_from_calls() -> None:
    """Module aggregate input/output fields are derived from ModuleCall boundaries."""

    trace = _make_trace(layers_to_save="all")
    module = trace.modules["linear"]
    call = module.calls[0]

    assert module.input_ops == call.input_ops
    assert module.input_layers == call.input_layers
    assert module.output_ops == call.output_ops
    assert module.output_layers == call.output_layers
    assert module.output_structure == call.output_structure


@pytest.mark.smoke
def test_trace_layers_to_save_public_view() -> None:
    """Trace.layers_to_save exposes saved Op labels instead of raw indexes."""

    all_trace = _make_trace(layers_to_save="all")
    selected_label = all_trace.output_layers[0]
    selected_trace = _make_trace(layers_to_save=[selected_label])

    assert all_trace.layers_to_save == "all"
    assert selected_trace.layers_to_save
    assert all(isinstance(label, str) for label in selected_trace.layers_to_save)
    assert all(label in selected_trace.op_labels for label in selected_trace.layers_to_save)
