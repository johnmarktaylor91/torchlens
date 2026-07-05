"""Regression tests for validation exemption hardening."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.validation import core
from torchlens.validation import backward as backward_validation
from torchlens.validation.exemptions import (
    _binary_extrema_nonperturbed_arg_dominates,
    perturbed_layer_at_structural_position,
)


def _fake_layer(**kwargs: Any) -> Any:
    """Build a minimal layer-like object for exemption unit tests.

    Parameters
    ----------
    **kwargs:
        Attributes to install on the fake object.

    Returns
    -------
    Any
        Layer-like namespace.
    """

    return SimpleNamespace(**kwargs)


def test_structural_arg_exemption_uses_parent_position_not_equal_value() -> None:
    """Identical tensor values must not prove a parent occupies a structural slot."""

    layer = _fake_layer(
        saved_args=(torch.tensor([1, 2]), torch.tensor([1, 2])),
        parent_arg_positions={"args": {1: "index_parent"}, "kwargs": {}},
    )

    assert not perturbed_layer_at_structural_position(
        None,  # type: ignore[arg-type]
        layer,
        ["value_parent"],
        {1},
    )
    assert perturbed_layer_at_structural_position(
        None,  # type: ignore[arg-type]
        layer,
        ["index_parent"],
        {1},
    )


def test_full_is_not_inplace_rng_arg_logging_exemption() -> None:
    """A deterministic ``full`` parent must not use the in-place RNG carve-out."""

    parent = _fake_layer(
        layer_label="full_1_1",
        label="full_1_1:1",
        func_name="full",
        out=torch.full((2,), 3.0),
        out_versions_by_child={},
    )
    child = _fake_layer(
        layer_label="add_1_2",
        label="add_1_2:1",
        parent_arg_positions={"args": {0: "full_1_1"}, "kwargs": {}},
        parents=["full_1_1"],
    )
    trace = {"full_1_1": parent}

    assert not core._check_arglocs_correct_for_arg(  # noqa: SLF001
        trace,  # type: ignore[arg-type]
        child,
        parent,
        "args",
        0,
        torch.zeros(2),
    )


def test_binary_extrema_requires_actual_nonperturbed_dominance() -> None:
    """Equal output alone must not exempt binary extrema perturbation."""

    layer = _fake_layer(parent_arg_positions={"args": {0: "lhs", 1: "rhs"}, "kwargs": {}})
    args = (torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))

    assert not _binary_extrema_nonperturbed_arg_dominates("maximum", args, layer, ["lhs"])
    assert not _binary_extrema_nonperturbed_arg_dominates("maximum", args, layer, ["rhs"])


def test_magnitude_ratio_shortcut_removed_from_posthoc_exemptions() -> None:
    """The old ``other_mag / perturbed_mag > 100`` predicate must stay removed."""

    source = inspect.getsource(core.posthoc_perturb_check)

    assert "other_mag" not in source
    assert "perturbed_mag" not in source


class DetachedParamModel(nn.Module):
    """Model whose parameter is deliberately disconnected from the loss."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an output disconnected from parameters.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor depending only on input.
        """

        return x * 2


class OneHotModel(nn.Module):
    """Model that consumes integer class indices through ``one_hot``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return float one-hot encodings.

        Parameters
        ----------
        x:
            Integer class-index tensor.

        Returns
        -------
        torch.Tensor
            One-hot tensor.
        """

        return torch.nn.functional.one_hot(x, num_classes=4).float()


def test_backward_validation_zero_param_grads_is_not_pass() -> None:
    """Backward validation must not pass when no parameter grads are checked."""

    model = DetachedParamModel()

    assert not backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 3),
        random_seed=5,
        validate_metadata=False,
    )


def test_one_hot_index_perturbation_uses_valid_alternate_class() -> None:
    """One-hot index validation should perturb within ``num_classes``."""

    trace = tl.trace(
        OneHotModel(),
        torch.tensor([1]),
        layers_to_save="all",
        save_arg_values=True,
    )

    assert trace.validate_forward_pass([torch.tensor([[0.0, 1.0, 0.0, 0.0]])])
