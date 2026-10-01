"""Regression tests for gradients reaching user input leaves after capture."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

import torchlens as tl


class _InputGradientModel(nn.Module):
    """Small deterministic network with a nontrivial input gradient."""

    def __init__(self) -> None:
        """Initialize two affine stages and an elementwise nonlinearity."""
        super().__init__()
        self.first = nn.Linear(4, 5)
        self.second = nn.Linear(5, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the fixed operation sequence used by both gradient runs."""
        return self.second(torch.tanh(self.first(inputs)))


@pytest.mark.smoke
def test_backward_ready_input_leaf_gradient_is_bit_exact_to_eager() -> None:
    """Backward-ready capture preserves the exact eager input-leaf gradient."""
    torch.manual_seed(314159)
    eager_model = _InputGradientModel()
    traced_model = copy.deepcopy(eager_model)
    input_values = torch.randn(2, 4)
    eager_input = input_values.clone().requires_grad_(True)
    traced_input = input_values.clone().requires_grad_(True)

    eager_model(eager_input).sum().backward()
    trace = tl.trace(
        traced_model, traced_input, capture=tl.options.CaptureOptions(backward_ready=True)
    )
    traced_output = trace.output_ops[0].out
    assert isinstance(traced_output, torch.Tensor)
    traced_output.sum().backward()

    assert eager_input.grad is not None
    assert traced_input.grad is not None
    assert eager_input.grad.dtype == traced_input.grad.dtype
    assert eager_input.grad.device == traced_input.grad.device
    assert torch.equal(traced_input.grad, eager_input.grad)
