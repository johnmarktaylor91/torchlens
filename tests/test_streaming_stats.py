"""Phase 4 streaming statistics tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning


def _manual_covariance(matrix: torch.Tensor) -> torch.Tensor:
    """Return the sample covariance of a row-wise feature matrix.

    Parameters
    ----------
    matrix:
        Two-dimensional tensor with observations in rows.

    Returns
    -------
    torch.Tensor
        Sample covariance matrix in ``float64``.
    """

    centered = matrix.to(dtype=torch.float64) - matrix.to(dtype=torch.float64).mean(
        dim=0, keepdim=True
    )
    return centered.T @ centered / (matrix.shape[0] - 1)


class _TwoLinearOutputs(torch.nn.Module):
    """Tiny model with two saved linear sites for ambiguity tests."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.fc2(torch.relu(self.fc1(x)))


def test_aggregate_warns_on_ambiguous_saved_output_selector_and_uses_first_match() -> None:
    """Ambiguous substring matches should warn and preserve first-match behavior."""

    model = _TwoLinearOutputs()
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    trace = tl.trace(model, batch, save=tl.func("linear"))
    matches = [
        layer
        for layer in trace._layer_list
        if "linear" in str(layer.layer_label) and layer.has_saved_activation
    ]
    expected = float(matches[0].out.detach().to(dtype=torch.float64).mean().item())
    trace.cleanup()

    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        result = tl.stats.aggregate(model, [batch], {"linear": tl.stats.Mean()})

    assert result["linear"] == pytest.approx(expected)


def test_pca_rejects_feature_dimension_changes() -> None:
    """PCA should inherit the covariance feature-width guard."""

    stat = tl.stats.PCA(2)
    stat.update(torch.ones(2, 3))

    with pytest.raises(ValueError, match="feature dimensions cannot change"):
        stat.update(torch.ones(2, 4))
