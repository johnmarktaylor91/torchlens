"""The completeness backstop counts a stale torch call inside a wrapped submodule.

A torch function bound before TorchLens wraps torch (a module-global
``from torch import f`` alias, as in legacy ``torch.nn.utils.weight_norm``) reaches
the dispatcher with no torch-function wrapper. At the top level the witness records
``unowned_dispatch``; inside a wrapped submodule the innermost token is the module's
forward token, so it records ``owner_not_captured``. Both are the same silent drop,
and validation must fail on both. ``torch.equal`` / ``torch.allclose`` deciding a
branch is owned by its own wrapper and must keep validating.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.user_funcs import _validate_forward_pass_torch

_WEIGHT_NORM_MODULE = sys.modules.get("torch.nn.utils.weight_norm")


def _legacy_weight_norm(module: nn.Module) -> nn.Module:
    """Apply the deprecated hook-based weight_norm without its FutureWarning."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.utils.weight_norm(module)


def _raw(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the original torch callable behind an installed TorchLens wrapper."""

    wrap_torch()
    return _state._decorated_to_orig.get(id(func), func)


class _NestedWeightNorm(nn.Module):
    """Legacy weight-normed conv inside a ``Sequential``; optionally a stale alias."""

    def __init__(self, stale_alias: bool) -> None:
        super().__init__()
        self.stale_alias = stale_alias
        assert _WEIGHT_NORM_MODULE is not None
        self.raw_weight_norm = _raw(_WEIGHT_NORM_MODULE._weight_norm) if stale_alias else None
        self.block = nn.Sequential(_legacy_weight_norm(nn.Conv1d(4, 4, 3, padding=1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block, with the module alias pre-wrap when ``stale_alias`` is set."""

        if not self.stale_alias:
            return self.block(x) * 2.0
        assert _WEIGHT_NORM_MODULE is not None
        wrapped = _WEIGHT_NORM_MODULE._weight_norm
        # Reproduce the untracked alias: the pre-hook calls the original function.
        _WEIGHT_NORM_MODULE._weight_norm = self.raw_weight_norm
        try:
            return self.block(x) * 2.0
        finally:
            _WEIGHT_NORM_MODULE._weight_norm = wrapped


class _EqualBranch(nn.Module):
    """Pure-read ``torch.equal`` / ``torch.allclose`` control flow."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a pure read of the linear output."""

        y = self.lin(x)
        if torch.equal(y, y) and torch.allclose(y, y):
            return y + 1
        return y - 1


class _NestedEqualBranch(nn.Module):
    """``_EqualBranch`` one level down, under a module-forward token."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(_EqualBranch())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested block."""

        return self.block(x)


@pytest.fixture
def _restore_alias() -> Iterator[None]:
    """Leave the weight_norm module alias as the test found it."""

    if _WEIGHT_NORM_MODULE is None:
        pytest.skip("torch.nn.utils.weight_norm is unavailable")
    before = _WEIGHT_NORM_MODULE._weight_norm
    yield
    _WEIGHT_NORM_MODULE._weight_norm = before


def _validate(model: nn.Module) -> bool:
    torch.manual_seed(0)
    return _validate_forward_pass_torch(
        model, [torch.randn(1, 4, 8)], {}, random_seed=0, validate_metadata=True
    )


@pytest.mark.usefixtures("_restore_alias")
def test_stale_alias_under_module_forward_fails_completeness() -> None:
    """A dropped ``_weight_norm`` inside a ``Sequential`` fails ``bfs_completeness``."""

    assert not _validate(_NestedWeightNorm(stale_alias=True))
    failure = tl.validation.last_validation_failure()
    assert failure is not None
    assert "bfs_completeness" in failure.summary()


@pytest.mark.usefixtures("_restore_alias")
def test_tracked_alias_under_module_forward_validates() -> None:
    """With the alias tracked the op is captured and the same model validates."""

    assert _validate(_NestedWeightNorm(stale_alias=False)), tl.validation.last_validation_failure()


@pytest.mark.parametrize("model_cls", [_EqualBranch, _NestedEqualBranch])
def test_equal_allclose_control_flow_still_validates(model_cls: type[nn.Module]) -> None:
    """Wrapper-owned pure reads stay benign at any nesting depth."""

    torch.manual_seed(0)
    passed = _validate_forward_pass_torch(
        model_cls(), [torch.randn(1, 4, 8)], {}, random_seed=0, validate_metadata=True
    )
    assert passed, tl.validation.last_validation_failure()
