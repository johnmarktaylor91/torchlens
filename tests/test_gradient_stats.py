"""Gradient streaming statistics tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class TinyClassifier(nn.Module):
    """Small model for gradient aggregation tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.net(x)


def test_aggregate_grad_requires_loss_fn_raises_typeerror() -> None:
    """Gradient aggregation requires a loss function."""

    model = TinyClassifier()
    batches = [(torch.ones(2, 3), torch.ones(2, 1))]

    with pytest.raises(TypeError, match="loss_fn"):
        tl.aggregate(model, batches, {"relu": tl.stats.Mean()}, target="grad")


def test_aggregate_grad_basic_norms() -> None:
    """Aggregate a basic gradient norm statistic."""

    model = TinyClassifier()
    batches = [
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.zeros(2, 3), torch.zeros(2, 1)),
    ]

    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return MSE loss."""

        return torch.nn.functional.mse_loss(output, target)

    result = tl.aggregate(
        model,
        batches,
        {"relu": tl.stats.Norm(name="norm")},
        target="grad",
        loss_fn=loss_fn,
    )

    assert "relu" in result
    assert result["relu"] >= 0.0
