"""Append-aware backward validator regression tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import AppendStateValidationWarning


class _ValidatorAppendModel(nn.Module):
    """Small model for backward validator append tests."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.linear(x)


def test_validate_backward_on_stacked_trace_warns() -> None:
    """Existing appended traces warn and are treated as authoritative."""

    model = _ValidatorAppendModel().eval()
    trace = tl.trace(model, torch.randn(1, 3), intervention_ready=True)
    trace.rerun(model, torch.randn(1, 3), append=True)

    with pytest.warns(AppendStateValidationWarning, match="stacked appended trace"):
        assert tl.validate_backward_pass(trace, torch.randn(1, 3)) is True


def test_validate_backward_pass_fresh_capture_only_still_works() -> None:
    """The normal validator path still captures and validates a fresh trace."""

    model = _ValidatorAppendModel().eval()

    assert tl.validate_backward_pass(model, torch.randn(2, 3), validate_metadata=False) is True
