"""Gradient streaming statistics tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning


class TinyClassifier(nn.Module):
    """Small model for gradient aggregation tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.net(x)


class TwoLinearClassifier(nn.Module):
    """Small model with two linear layers for ambiguity tests."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.fc2(torch.relu(self.fc1(x)))


def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean-squared error for gradient aggregation tests.

    Parameters
    ----------
    output:
        Model predictions.
    target:
        Target tensor.

    Returns
    -------
    torch.Tensor
        Scalar mean-squared error.
    """

    return torch.nn.functional.mse_loss(output, target)


def test_aggregate_grad_requires_loss_fn_raises_typeerror() -> None:
    """Gradient aggregation requires a loss function."""

    model = TinyClassifier()
    batches = [(torch.ones(2, 3), torch.ones(2, 1))]

    with pytest.raises(TypeError, match="loss_fn"):
        tl.aggregate(model, batches, {"relu": tl.stats.Mean()}, target="grad")


def test_aggregate_grad_basic_norms() -> None:
    """Aggregate a gradient norm and match an explicit per-batch reference."""

    model = TinyClassifier()
    batches = [
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.zeros(2, 3), torch.zeros(2, 1)),
    ]

    result = tl.aggregate(
        model,
        batches,
        {"relu": tl.stats.Norm(name="norm")},
        target="grad",
        loss_fn=_mse_loss,
    )

    expected_norms: list[float] = []
    for inputs, targets in batches:
        trace = tl.trace(
            model,
            inputs,
            capture=tl.options.CaptureOptions(layers_to_save="all", save_grads=True),
        )
        loss = _mse_loss(trace[trace.output_layers[-1]].out, targets)
        trace.log_backward(loss)
        grad = next(layer.grad for layer in trace.layer_list if layer.layer_type == "relu")
        expected_norms.append(float(torch.linalg.vector_norm(grad.reshape(-1)).item()))
        trace.cleanup()

    assert result["relu"] == pytest.approx(sum(expected_norms) / len(expected_norms))


def test_aggregate_grad_warns_on_ambiguous_saved_selector_and_uses_first_match() -> None:
    """Ambiguous gradient selectors should warn and keep first-match semantics."""

    model = TwoLinearClassifier()
    batch = (torch.arange(6, dtype=torch.float32).reshape(2, 3), torch.ones(2, 1))
    trace = tl.trace(
        model,
        batch[0],
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads=tl.func("linear")),
    )
    loss = _mse_loss(trace[trace.output_layers[-1]].out, batch[1])
    trace.log_backward(loss)
    matches = [
        layer
        for layer in trace.layer_list
        if "linear" in str(layer.layer_label) and layer.has_grad and layer.grad is not None
    ]
    expected = float(torch.linalg.vector_norm(matches[0].grad.reshape(-1)).item())
    trace.cleanup()

    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        result = tl.aggregate(
            model,
            [batch],
            {"linear": tl.stats.Norm(name="norm")},
            target="grad",
            loss_fn=_mse_loss,
        )

    assert result["linear"] == pytest.approx(expected)
