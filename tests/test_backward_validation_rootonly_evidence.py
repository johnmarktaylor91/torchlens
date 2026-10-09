"""Backward validation of a model with no child module calls says why it is unverifiable.

The module-output gradient check (on by default) excludes the root module call
by design: its output gradient is the loss gradient itself. A model whose
forward calls no child module therefore has zero eligible module outputs, and
the check's exact acceptance rule (``covered_count > 0``) fails closed. That
verdict stays ``False``; what these tests pin is that it is disclosed like every
other unverifiable backward census, not returned as a bare ``False`` that reads
as a capture bug.
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _RootOnly(nn.Module):
    """Scale the input by a parameter with no child module call."""

    def __init__(self) -> None:
        """Register the parameter."""

        super().__init__()
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x * w``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The scaled input.
        """

        return x * self.w


class _Scale(nn.Module):
    """Child module scaling its input by a parameter."""

    def __init__(self) -> None:
        """Register the parameter."""

        super().__init__()
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x * w``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The scaled input.
        """

        return x * self.w


class _DeepcopyBufferWithChild(nn.Module):
    """Deep-copy a buffer, then route it through a child module call."""

    def __init__(self) -> None:
        """Register the buffer and the child module."""

        super().__init__()
        self.register_buffer("buf", torch.arange(4.0))
        self.scale = _Scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``scale(x * deepcopy(buf)) + buf``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The child module's output plus the buffer.
        """

        return self.scale(x * copy.deepcopy(self.buf)) + self.buf


def test_root_only_model_backward_verdict_is_disclosed_unverifiable() -> None:
    """No eligible module output: still ``False``, now with the reason."""

    torch.manual_seed(0)
    with pytest.warns(RuntimeWarning, match="no module-output gradient to compare"):
        verdict = tl.validate(_RootOnly(), torch.randn(4, requires_grad=True), scope="backward")
    assert verdict is False


def test_root_only_model_parameter_gradients_alone_validate() -> None:
    """The parameter-gradient census passes when module-output evidence is not requested."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*no module-output gradient to compare.*")
        verdict = tl.validate(
            _RootOnly(),
            torch.randn(4, requires_grad=True),
            scope="backward",
            validate_layer_grads=False,
        )
    assert verdict is True


def test_deepcopied_buffer_validates_backward_with_a_child_module() -> None:
    """A deep copy of a buffer has no effect on backward validation."""

    torch.manual_seed(0)
    verdict = tl.validate(
        _DeepcopyBufferWithChild(), torch.randn(4, requires_grad=True), scope="backward"
    )
    assert verdict is True
