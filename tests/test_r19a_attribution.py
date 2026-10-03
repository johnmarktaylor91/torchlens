"""Regression tests for r19a attribution honesty/robustness fixes.

Covers:
- W2A3-05: attribution restores exact per-module train/eval state (not the root flag).
- W2A3-06: attribution computes correctly under an ambient ``torch.no_grad()``.

Owner-reserved wrong-math findings (W2A3-01/02/03/13/17) are NOT fixed here; their
evidence lives in the private R19A triage report.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchlens.attribution import (
    input_x_grad,
    integrated_gradients,
    layer_attribution,
    saliency,
    smoothgrad,
)


class _MixedModeNet(nn.Module):
    """Model whose submodules carry an intentionally-mixed train/eval configuration."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.frozen_bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(self.frozen_bn(x))


def _set_mixed_mode(model: _MixedModeNet) -> tuple[bool, bool, bool]:
    """Put root+lin in train and frozen_bn in eval; return the snapshot."""

    model.train()
    model.lin.train()
    model.frozen_bn.eval()
    return (model.training, model.lin.training, model.frozen_bn.training)


def _first_tensor(values) -> torch.Tensor:
    """Extract a representative tensor from an attribution value tree."""

    if isinstance(values, torch.Tensor):
        return values
    if isinstance(values, (tuple, list)):
        for item in values:
            if item is not None:
                return _first_tensor(item)
    if isinstance(values, dict):
        for item in values.values():
            if item is not None:
                return _first_tensor(item)
    raise AssertionError("no tensor found in attribution values")


# --------------------------------------------------------------------------- #
# W2A3-05: per-module train/eval state must be restored exactly.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda m, x: saliency(m, x, target=0), id="saliency"),
        pytest.param(lambda m, x: input_x_grad(m, x, target=0), id="input_x_grad"),
        pytest.param(
            lambda m, x: integrated_gradients(m, x, target=0, n_steps=3),
            id="integrated_gradients",
        ),
        pytest.param(
            lambda m, x: smoothgrad(m, x, target=0, n_samples=2, seed=0),
            id="smoothgrad",
        ),
        pytest.param(
            lambda m, x: layer_attribution(m, x, target=0, layer="lin"),
            id="layer_attribution",
        ),
    ],
)
def test_mixed_train_eval_state_is_restored_exactly(run) -> None:
    """Every attribution method must leave mixed per-module modes untouched."""

    torch.manual_seed(0)
    model = _MixedModeNet()
    before = _set_mixed_mode(model)
    assert before == (True, True, False)

    x = torch.randn(8, 4)
    run(model, x)

    after = (model.training, model.lin.training, model.frozen_bn.training)
    assert after == before, f"attribution clobbered per-module modes: {before} -> {after}"


def test_eval_root_with_training_submodule_is_restored() -> None:
    """The reverse mix (eval root, one training submodule) is also preserved."""

    torch.manual_seed(0)
    model = _MixedModeNet()
    model.eval()
    model.lin.train()
    before = (model.training, model.lin.training, model.frozen_bn.training)
    assert before == (False, True, False)

    saliency(model, torch.randn(8, 4), target=0)

    after = (model.training, model.lin.training, model.frozen_bn.training)
    assert after == before


# --------------------------------------------------------------------------- #
# W2A3-06: attribution must compute under an ambient torch.no_grad().
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda m, x: saliency(m, x, target=0), id="saliency"),
        pytest.param(lambda m, x: input_x_grad(m, x, target=0), id="input_x_grad"),
        pytest.param(
            lambda m, x: integrated_gradients(m, x, target=0, n_steps=3),
            id="integrated_gradients",
        ),
        pytest.param(
            lambda m, x: layer_attribution(m, x, target=0, layer="lin"),
            id="layer_attribution",
        ),
    ],
)
def test_attribution_works_under_ambient_no_grad(run) -> None:
    """Ambient no_grad must not defeat attribution; values match the normal path."""

    torch.manual_seed(0)
    model = _MixedModeNet()
    model.eval()
    x = torch.randn(8, 4)

    reference = run(model, x)
    with torch.no_grad():
        under_no_grad = run(model, x)

    ref_tensor = _first_tensor(reference.values)
    got_tensor = _first_tensor(under_no_grad.values)
    assert torch.allclose(ref_tensor, got_tensor)


def test_no_grad_does_not_leak_grad_state() -> None:
    """Running attribution under no_grad must not turn grad back on for the caller."""

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    x = torch.randn(8, 4)
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        saliency(model, x, target=0)
        assert not torch.is_grad_enabled()
