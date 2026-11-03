"""Tests for per-op autograd saved tensor memory accounting."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._ops_autograd import _get_autograd_saved_stats_for_tensor


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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_groups.py.
def test_no_grad_sets_autograd_saved_fields_to_none() -> None:
    """torch.no_grad should produce None autograd saved fields at every level."""
    torch.manual_seed(0)
    model = TinySequentialModel()
    x = torch.randn(4, 10, requires_grad=True)

    with torch.no_grad():
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=0), save="all")

    assert all(layer.autograd_memory is None for layer in trace._layer_list)
    assert all(layer.num_autograd_tensors is None for layer in trace._layer_list)
    assert all(layer.autograd_memory is None for layer in trace._layer_logs.values())
    assert all(layer.num_autograd_tensors is None for layer in trace._layer_logs.values())
    assert trace.cost.total_autograd_memory is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_ops_autograd.py, campaign mutant kills.
def test_custom_autograd_function_saved_tensor_bytes() -> None:
    """Custom autograd.Function saved_tensors should be measured by introspection."""
    x = torch.randn(2, 3, requires_grad=True)
    output = SaveInputFunction.apply(x)
    expected_bytes = x.numel() * x.element_size()

    autograd_memory, num_autograd_tensors = _get_autograd_saved_stats_for_tensor(output)

    assert autograd_memory == expected_bytes
    assert num_autograd_tensors == 1
