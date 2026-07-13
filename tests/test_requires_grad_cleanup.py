"""Regression coverage for requires-grad cleanup after failed captures."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class UserBaseException(BaseException):
    """User-defined non-``Exception`` failure used by capture cleanup tests."""


class FrozenParameterFailureModel(nn.Module):
    """Linear model with frozen parameters that can fail after a logged operation."""

    def __init__(self, failure_type: type[BaseException]) -> None:
        """Initialize frozen parameters and the exception to raise once.

        Parameters
        ----------
        failure_type
            Exception type raised by the first forward call.
        """

        super().__init__()
        self.linear = nn.Linear(2, 2)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.failure_type: type[BaseException] | None = failure_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer, then optionally raise the configured failure.

        Parameters
        ----------
        x
            Input tensor for the linear layer.

        Returns
        -------
        torch.Tensor
            Linear-layer output when failure has been disabled.
        """

        output = self.linear(x)
        if self.failure_type is not None:
            raise self.failure_type("expected forward failure")
        return output


@pytest.mark.parametrize("failure_type", [RuntimeError, UserBaseException, KeyboardInterrupt])
def test_failed_trace_restores_frozen_parameter_grad_flags(
    failure_type: type[BaseException],
) -> None:
    """Failed captures restore frozen parameters before a later successful trace.

    Parameters
    ----------
    failure_type
        Exception class raised from the model's first forward pass.
    """

    model = FrozenParameterFailureModel(failure_type)
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]

    with pytest.raises(failure_type, match="expected forward failure"):
        tl.trace(model, torch.ones(1, 2))

    assert [parameter.requires_grad for parameter in model.parameters()] == original_requires_grad

    model.failure_type = None
    tl.trace(model, torch.ones(1, 2))

    assert [parameter.requires_grad for parameter in model.parameters()] == original_requires_grad
