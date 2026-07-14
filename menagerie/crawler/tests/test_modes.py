"""Tests for meaningful modes and divergence classification."""

from __future__ import annotations

import torch

from menagerie.crawler.constants import RunMode
from menagerie.crawler.modes import (
    classify_train_eval_divergence,
    detect_meaningful_modes,
)


class _BatchNormModel(torch.nn.Module):
    """Fixture with statistical mode behavior."""

    def __init__(self) -> None:
        """Construct one BatchNorm layer."""

        super().__init__()
        self.norm = torch.nn.BatchNorm1d(3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply mode-sensitive normalization.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        return self.norm(value)


class _ShapeBranchModel(torch.nn.Module):
    """Fixture with structural train/eval behavior."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return mode-dependent output shapes.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Full tensor in train mode and one column in eval mode.
        """

        return value if self.training else value[:, :1]


def test_modes_classify_none_statistical_and_structural() -> None:
    """Captured outputs distinguish equality, value drift, and shape drift."""

    value = torch.tensor([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]])
    assert classify_train_eval_divergence(value, value.clone()).classification == "none"

    statistical = _BatchNormModel()
    statistical.train()
    train_output = statistical(value)
    statistical.eval()
    eval_output = statistical(value)
    assert classify_train_eval_divergence(train_output, eval_output).classification == "statistical"
    assert detect_meaningful_modes(statistical) == (RunMode.TRAIN, RunMode.EVAL)

    structural = _ShapeBranchModel()
    structural.train()
    train_output = structural(value)
    structural.eval()
    eval_output = structural(value)
    assert classify_train_eval_divergence(train_output, eval_output).classification == "structural"
    assert detect_meaningful_modes(structural) == (RunMode.TRAIN, RunMode.EVAL)
