"""Tests for per-op autograd saved tensor memory accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.output_tensors import _get_autograd_saved_stats_for_tensor
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ModelLog


class TinySequentialModel(nn.Module):
    """Small model with parameterized and activation ops."""

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
        """Return a simple gradient using the saved input."""
        (saved_x,) = ctx.saved_tensors
        return grad_output * saved_x


def _log_sequential(requires_grad: bool = True) -> ModelLog:
    """Return a logged pass through the tiny sequential model."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    x = torch.randn(4, 10, requires_grad=requires_grad)
    return tl.log_forward_pass(model, x, layers_to_save="all", random_seed=0)


def _non_source_passes(model_log: ModelLog) -> list[LayerPassLog]:
    """Return operation logs, excluding synthetic source and output nodes."""
    return [
        layer
        for layer in model_log.layer_list
        if layer.layer_type not in {"input", "buffer", "output"}
    ]


def _single_layer_log_for_pass(model_log: ModelLog, pass_log: LayerPassLog) -> LayerLog:
    """Return the aggregate LayerLog for a pass log."""
    return model_log.layer_logs[pass_log.layer_label_no_pass]


def _sum_layer_autograd_bytes(model_log: ModelLog) -> Optional[int]:
    """Sum non-None layer-level autograd byte values."""
    values = [
        layer.autograd_saved_bytes
        for layer in model_log.layer_logs.values()
        if layer.autograd_saved_bytes is not None
    ]
    if not values:
        return None
    return sum(values)


@pytest.mark.smoke
def test_autograd_saved_bytes_basic_shape_model() -> None:
    """Linear and ReLU ops should report autograd saved tensor memory."""
    model_log = _log_sequential(requires_grad=True)

    linear_passes = [layer for layer in model_log.layer_list if layer.layer_type == "linear"]
    relu_passes = [layer for layer in model_log.layer_list if layer.layer_type == "relu"]

    assert linear_passes
    assert relu_passes
    assert all(layer.autograd_saved_bytes is not None for layer in linear_passes)
    assert all(layer.autograd_saved_bytes > 0 for layer in linear_passes)
    assert all(layer.autograd_saved_tensor_count is not None for layer in linear_passes)
    assert all(layer.autograd_saved_tensor_count > 0 for layer in linear_passes)
    assert relu_passes[0].autograd_saved_bytes is not None
    assert relu_passes[0].autograd_saved_bytes >= 0
    assert relu_passes[0].autograd_saved_tensor_count is not None
    assert relu_passes[0].autograd_saved_tensor_count >= 0


def test_add_op_reports_zero_autograd_saved_bytes() -> None:
    """Add should have a grad_fn but save no tensors for backward."""
    model = TinyAddModel()
    x = torch.ones(2, 3, requires_grad=True)
    y = torch.ones(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(model, (x, y), layers_to_save="all")
    add_pass = next(layer for layer in model_log.layer_list if layer.layer_type == "add")

    assert add_pass.grad_fn_id is not None
    assert add_pass.autograd_saved_bytes == 0
    assert add_pass.autograd_saved_tensor_count == 0
    assert model_log.layer_logs[add_pass.layer_label_no_pass].autograd_saved_bytes == 0
    assert model_log.total_autograd_saved_bytes == 0


def test_no_grad_sets_autograd_saved_fields_to_none() -> None:
    """torch.no_grad should produce None autograd saved fields at every level."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    x = torch.randn(4, 10, requires_grad=True)

    with torch.no_grad():
        model_log = tl.log_forward_pass(model, x, layers_to_save="all", random_seed=0)

    assert all(layer.autograd_saved_bytes is None for layer in model_log.layer_list)
    assert all(layer.autograd_saved_tensor_count is None for layer in model_log.layer_list)
    assert all(layer.autograd_saved_bytes is None for layer in model_log.layer_logs.values())
    assert all(layer.autograd_saved_tensor_count is None for layer in model_log.layer_logs.values())
    assert model_log.total_autograd_saved_bytes is None


def test_requires_grad_false_sets_autograd_saved_fields_to_none() -> None:
    """Inputs without requires_grad should not create grad_fn-backed saved fields."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(4, 10, requires_grad=False)
    model_log = tl.log_forward_pass(model, x, layers_to_save="all", random_seed=0, train_mode=True)

    assert all(layer.autograd_saved_bytes is None for layer in _non_source_passes(model_log))
    assert all(layer.autograd_saved_tensor_count is None for layer in _non_source_passes(model_log))
    assert all(layer.autograd_saved_bytes is None for layer in model_log.layer_logs.values())
    assert model_log.total_autograd_saved_bytes is None


def test_layer_log_autograd_saved_rollup_matches_pass_values() -> None:
    """LayerLog values should equal the sum of their pass-level values."""
    model_log = _log_sequential(requires_grad=True)

    for pass_log in _non_source_passes(model_log):
        layer_log = _single_layer_log_for_pass(model_log, pass_log)
        assert layer_log.autograd_saved_bytes == pass_log.autograd_saved_bytes
        assert layer_log.autograd_saved_tensor_count == pass_log.autograd_saved_tensor_count


def test_model_log_autograd_saved_rollup_matches_layer_values() -> None:
    """ModelLog total should equal the sum of non-None layer-level values."""
    model_log = _log_sequential(requires_grad=True)

    assert model_log.total_autograd_saved_bytes == _sum_layer_autograd_bytes(model_log)


def test_custom_autograd_function_saved_tensor_bytes() -> None:
    """Custom autograd.Function saved_tensors should be measured by introspection."""
    x = torch.randn(2, 3, requires_grad=True)
    output = SaveInputFunction.apply(x)
    expected_bytes = x.numel() * x.element_size()

    autograd_saved_bytes, autograd_saved_tensor_count = _get_autograd_saved_stats_for_tensor(output)

    assert autograd_saved_bytes == expected_bytes
    assert autograd_saved_tensor_count == 1


def test_autograd_saved_fields_roundtrip_through_bundle_save_load(tmp_path: Path) -> None:
    """Autograd saved byte fields should survive portable bundle save/load."""
    model_log = _log_sequential(requires_grad=True)
    bundle_path = tmp_path / "autograd_saved_bytes.tl"

    tl.save(model_log, bundle_path)
    loaded = tl.load(bundle_path)

    assert loaded.total_autograd_saved_bytes == model_log.total_autograd_saved_bytes
    for original_layer, loaded_layer in zip(model_log.layer_list, loaded.layer_list):
        assert loaded_layer.autograd_saved_bytes == original_layer.autograd_saved_bytes
        assert (
            loaded_layer.autograd_saved_tensor_count == original_layer.autograd_saved_tensor_count
        )
    for label, original_layer in model_log.layer_logs.items():
        loaded_layer = loaded.layer_logs[label]
        assert loaded_layer.autograd_saved_bytes == original_layer.autograd_saved_bytes
        assert (
            loaded_layer.autograd_saved_tensor_count == original_layer.autograd_saved_tensor_count
        )
