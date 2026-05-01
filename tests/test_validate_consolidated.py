"""Phase 5a tests for consolidated validation."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.validation.consolidated import InterventionValidationReport


class TinyModel(torch.nn.Module):
    """Small deterministic model for validation tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.linear(x)


@pytest.fixture
def model_and_input() -> tuple[TinyModel, torch.Tensor]:
    """Return a deterministic model/input pair.

    Returns
    -------
    tuple[TinyModel, torch.Tensor]
        Model and input tensor.
    """

    torch.manual_seed(0)
    model = TinyModel()
    torch.manual_seed(1)
    return model, torch.randn(1, 3)


def test_validate_forward_and_saved_scopes(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Consolidated forward and saved scopes should return booleans."""

    model, x = model_and_input
    assert isinstance(tl.validate(model, x, scope="forward", random_seed=42), bool)
    assert isinstance(tl.validate(model, x, scope="saved", random_seed=42), bool)


def test_validate_backward_scope(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Consolidated backward scope should call the backward validator."""

    model, x = model_and_input

    def loss_fn(output: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss.

        Parameters
        ----------
        output:
            Model output.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        return output.sum()

    assert tl.validate(model, x, scope="backward", loss_fn=loss_fn)


def test_validate_intervention_scope_returns_report(
    model_and_input: tuple[TinyModel, torch.Tensor],
) -> None:
    """Intervention scope should return a truthy five-axis report."""

    model, x = model_and_input
    report = tl.validate(model, x, scope="intervention", random_seed=42)
    assert isinstance(report, InterventionValidationReport)
    assert bool(report) is report.passed
    assert set(report.details) == {
        "invariance",
        "specificity",
        "completeness",
        "consistency",
        "locality",
    }


def test_scope_keyword_visibility(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Backward-only keywords should fail outside backward scope."""

    model, x = model_and_input
    with pytest.raises(TypeError, match="loss_fn only valid for scope='backward'"):
        tl.validate(model, x, scope="forward", loss_fn=lambda output: output.sum())


def test_legacy_validator_positionals(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Deprecated wrappers should preserve legacy positional argument binding."""

    model, x = model_and_input

    def loss_fn(output: Any) -> torch.Tensor:
        """Return a scalar loss.

        Parameters
        ----------
        output:
            Model output.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        return output.sum()

    with pytest.warns(DeprecationWarning):
        forward_result = tl.validate_forward_pass(model, x, None, 42)
    with pytest.warns(DeprecationWarning):
        saved_result = tl.validate_saved_activations(model, x, None, 42)
    with pytest.warns(DeprecationWarning):
        backward_result = tl.validate_backward_pass(model, x, None, loss_fn)

    assert isinstance(forward_result, bool)
    assert isinstance(saved_result, bool)
    assert isinstance(backward_result, bool)
