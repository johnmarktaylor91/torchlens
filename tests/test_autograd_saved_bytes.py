"""Tests for per-op autograd saved tensor memory accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.ops import _get_autograd_saved_stats_for_tensor
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.op import Op
from torchlens.data_classes.trace import Trace
from torchlens.options import CaptureOptions


class TinySequentialModel(nn.Module):
    """Small model with parameterized and out ops."""

    def __init__(self) -> None:
        """Initialize the sequential test layers."""
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the sequential layers."""
        return self.layers(x)


class TinyAddModel(nn.Module):
    """Small model that exposes an add operation."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors."""
        return x + y


class SaveInputFunction(torch.autograd.Function):
    """Custom autograd function that saves its input for backward."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Save the input tensor and return a scaled output."""
        ctx.save_for_backward(x)
        return x * 2

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        """Return a simple grad using the saved input."""
        (saved_x,) = ctx.saved_tensors
        return grad_output * saved_x


def _log_sequential(requires_grad: bool = True) -> Trace:
    """Return a logged pass through the tiny sequential model."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    x = torch.randn(4, 10, requires_grad=requires_grad)
    return tl.trace(model, x, capture=CaptureOptions(layers_to_save="all", random_seed=0))


def _non_source_ops(trace: Trace) -> list[Op]:
    """Return operation logs, excluding synthetic source and output nodes."""
    return [
        layer for layer in trace.layer_list if layer.layer_type not in {"input", "buffer", "output"}
    ]


def _single_layer_log_for_pass(trace: Trace, pass_log: Op) -> Layer:
    """Return the aggregate Layer for a pass log."""
    return trace.layer_logs[pass_log.layer_label]


def _sum_layer_autograd_bytes(trace: Trace) -> Optional[int]:
    """Sum non-None layer-level autograd byte values."""
    values = [
        layer.autograd_memory
        for layer in trace.layer_logs.values()
        if layer.autograd_memory is not None
    ]
    if not values:
        return None
    return sum(values)


@pytest.mark.smoke
def test_autograd_memory_basic_shape_model() -> None:
    """Linear and ReLU ops should report autograd saved tensor memory."""
    trace = _log_sequential(requires_grad=True)

    linear_ops = [layer for layer in trace.layer_list if layer.layer_type == "linear"]
    relu_ops = [layer for layer in trace.layer_list if layer.layer_type == "relu"]

    assert linear_ops
    assert relu_ops
    assert all(layer.autograd_memory is not None for layer in linear_ops)
    assert all(layer.autograd_memory > 0 for layer in linear_ops)
    assert all(layer.num_autograd_tensors is not None for layer in linear_ops)
    assert all(layer.num_autograd_tensors > 0 for layer in linear_ops)
    assert relu_ops[0].autograd_memory is not None
    assert relu_ops[0].autograd_memory >= 0
    assert relu_ops[0].num_autograd_tensors is not None
    assert relu_ops[0].num_autograd_tensors >= 0


def test_add_op_reports_zero_autograd_memory() -> None:
    """Add should have a grad_fn_handle but save no tensors for backward."""
    model = TinyAddModel()
    x = torch.ones(2, 3, requires_grad=True)
    y = torch.ones(2, 3, requires_grad=True)
    trace = tl.trace(model, (x, y), layers_to_save="all")
    add_pass = next(layer for layer in trace.layer_list if layer.layer_type == "add")

    assert add_pass.grad_fn_object_id is not None
    assert add_pass.autograd_memory == 0
    assert add_pass.num_autograd_tensors == 0
    assert trace.layer_logs[add_pass.layer_label].autograd_memory == 0
    assert trace.total_autograd_memory == 0


def test_no_grad_sets_autograd_saved_fields_to_none() -> None:
    """torch.no_grad should produce None autograd saved fields at every level."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    x = torch.randn(4, 10, requires_grad=True)

    with torch.no_grad():
        trace = tl.trace(model, x, layers_to_save="all", random_seed=0)

    assert all(layer.autograd_memory is None for layer in trace.layer_list)
    assert all(layer.num_autograd_tensors is None for layer in trace.layer_list)
    assert all(layer.autograd_memory is None for layer in trace.layer_logs.values())
    assert all(layer.num_autograd_tensors is None for layer in trace.layer_logs.values())
    assert trace.total_autograd_memory is None


def test_requires_grad_false_sets_autograd_saved_fields_to_none() -> None:
    """Inputs without requires_grad should not create grad_fn_handle-backed saved fields."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(4, 10, requires_grad=False)
    trace = tl.trace(model, x, layers_to_save="all", random_seed=0, backward_ready=True)

    assert all(layer.autograd_memory is None for layer in _non_source_ops(trace))
    assert all(layer.num_autograd_tensors is None for layer in _non_source_ops(trace))
    assert all(layer.autograd_memory is None for layer in trace.layer_logs.values())
    assert trace.total_autograd_memory is None


def test_layer_log_autograd_saved_rollup_matches_pass_values() -> None:
    """Layer values should equal the sum of their pass-level values."""
    trace = _log_sequential(requires_grad=True)

    for pass_log in _non_source_ops(trace):
        layer_log = _single_layer_log_for_pass(trace, pass_log)
        assert layer_log.autograd_memory == pass_log.autograd_memory
        assert layer_log.num_autograd_tensors == pass_log.num_autograd_tensors


def test_trace_autograd_saved_rollup_matches_layer_values() -> None:
    """Trace total should equal the sum of non-None layer-level values."""
    trace = _log_sequential(requires_grad=True)

    assert trace.total_autograd_memory == _sum_layer_autograd_bytes(trace)


def test_custom_autograd_function_saved_tensor_bytes() -> None:
    """Custom autograd.Function saved_tensors should be measured by introspection."""
    x = torch.randn(2, 3, requires_grad=True)
    output = SaveInputFunction.apply(x)
    expected_bytes = x.numel() * x.element_size()

    autograd_memory, num_autograd_tensors = _get_autograd_saved_stats_for_tensor(output)

    assert autograd_memory == expected_bytes
    assert num_autograd_tensors == 1


def test_autograd_saved_fields_roundtrip_through_bundle_save_load(tmp_path: Path) -> None:
    """Autograd saved byte fields should survive portable bundle save/load."""
    trace = _log_sequential(requires_grad=True)
    bundle_path = tmp_path / "autograd_memory.tl"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)

    assert loaded.total_autograd_memory == trace.total_autograd_memory
    for original_layer, loaded_layer in zip(trace.layer_list, loaded.layer_list):
        assert loaded_layer.autograd_memory == original_layer.autograd_memory
        assert loaded_layer.num_autograd_tensors == original_layer.num_autograd_tensors
    for label, original_layer in trace.layer_logs.items():
        loaded_layer = loaded.layer_logs[label]
        assert loaded_layer.autograd_memory == original_layer.autograd_memory
        assert loaded_layer.num_autograd_tensors == original_layer.num_autograd_tensors
