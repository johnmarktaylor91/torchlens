"""Phase 1 capture-unification regression tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import example_models
import torchlens as tl
from torchlens.validation import validate_forward_pass


class SmallCNN(nn.Module):
    """Small convolutional model for event-only materialization coverage."""

    def __init__(self) -> None:
        """Initialize convolutional layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional forward pass.

        Parameters
        ----------
        x
            Input image batch.

        Returns
        -------
        torch.Tensor
            Flattened pooled activations.
        """

        x = self.conv(x)
        x = torch.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def _assert_trace_structure(trace: tl.Trace) -> None:
    """Assert core graph and tensor metadata are structurally complete.

    Parameters
    ----------
    trace
        Materialized trace to inspect.

    Returns
    -------
    None
        Raises assertions on malformed traces.
    """

    assert trace.op_labels
    assert trace.input_layers
    assert trace.output_layers
    assert trace.modules["self"].address == "self"
    for op in trace.ops:
        assert op.label
        assert op.layer_label
        assert op.shape is None or isinstance(op.shape, tuple)
        assert op.dtype is None or str(op.dtype)
        for parent in op.parents:
            assert parent in trace.layer_dict_all_keys
            assert op.layer_label in trace[parent].children or op.label in trace[parent].children
        for child in op.children:
            assert child in trace.layer_dict_all_keys
            assert op.layer_label in trace[child].parents or op.label in trace[child].parents


@pytest.mark.smoke
def test_phase1_event_materializer_small_cnn() -> None:
    """Trace a CNN and validate graph, metadata, modules, and replay."""

    torch.manual_seed(10)
    model = SmallCNN().eval()
    x = torch.randn(2, 3, 8, 8)
    trace = tl.trace(model, x)

    _assert_trace_structure(trace)
    assert "conv" in trace.modules
    assert "norm" in trace.modules
    assert any(op.layer_type == "conv2d" for op in trace.ops)
    assert validate_forward_pass(model, x)


@pytest.mark.smoke
def test_phase1_event_materializer_recurrent_params() -> None:
    """Trace a recurrent parameter model and validate pass grouping."""

    torch.manual_seed(11)
    model = example_models.RecurrentParamsSimple().eval()
    x = torch.rand(5, 5)
    trace = tl.trace(model, x)

    _assert_trace_structure(trace)
    recurrent_layers = [layer for layer in trace.layers if len(layer.ops) > 1]
    assert recurrent_layers
    assert any(op.recurrent_ops for op in trace.ops)
    assert validate_forward_pass(model, x)


@pytest.mark.smoke
def test_phase1_event_materializer_buffers_and_conditionals() -> None:
    """Trace buffer and conditional models through event materialization."""

    torch.manual_seed(12)
    buffer_model = example_models.BufferModel().eval()
    buffer_input = torch.rand(12, 12)
    buffer_trace = tl.trace(buffer_model, buffer_input)
    _assert_trace_structure(buffer_trace)
    assert buffer_trace.buffer_layers
    assert validate_forward_pass(buffer_model, buffer_input)

    conditional_model = example_models.ConditionalBranching().eval()
    conditional_input = torch.ones(2, 3, 8, 8)
    conditional_trace = tl.trace(conditional_model, conditional_input, save_code_context=True)
    _assert_trace_structure(conditional_trace)
    assert conditional_trace.has_conditional_branching
    assert validate_forward_pass(conditional_model, conditional_input)
