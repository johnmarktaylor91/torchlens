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
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.user_funcs import _validate_forward_pass_torch
from torchlens.validation._completeness_backstop import completeness_backstop_counts

_WEIGHT_NORM_MODULE = sys.modules.get("torch.nn.utils.weight_norm")


def _legacy_weight_norm(module: nn.Module) -> nn.Module:
    """Apply the deprecated hook-based weight_norm without its FutureWarning."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.utils.weight_norm(module)


def _raw(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the original torch callable behind an installed wrapper, in an opaque holder.

    Capture preparation rebinds pristine torch functions held directly on a
    model, so a bare original would no longer escape; the custom callable
    object is a holder it never rebinds, which keeps the escape these
    completeness tripwires must catch.
    """

    wrap_torch()
    return OpaqueCallable(_state._decorated_to_orig.get(id(func), func))


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
# The un-captured weight reaches conv1d with no recorded parent; that disclosure is expected.
@pytest.mark.filterwarnings("ignore:TorchLens found tensor arguments with no graph:UserWarning")
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


# Census-level unit checks: minimal witness records, no capture needed.


def _backstop_op(func_call_id: int) -> SimpleNamespace:
    """Build a minimal captured-op stand-in for the backstop census."""

    return SimpleNamespace(func_call_id=func_call_id)


def _backstop_dec(owner_func_call_id: int) -> dict[str, Any]:
    """Build a minimal accounted decomposition record."""

    return {
        "owner_func_call_id": owner_func_call_id,
        "capture_accounted": True,
        "in_replacement_hook": False,
    }


def _backstop_diag(
    *,
    reason: str,
    owner_wrapper: str,
    in_replacement_hook: bool = False,
    mutates: bool = False,
) -> dict[str, Any]:
    """Build a minimal unaccounted-dispatch witness diagnostic record."""

    return {
        "reason": reason,
        "owner_wrapper": owner_wrapper,
        "in_replacement_hook": in_replacement_hook,
        "mutates": mutates,
        "state_view_accessor": False,
    }


def _backstop_trace(
    layer_list: list[SimpleNamespace],
    decompositions: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> SimpleNamespace:
    """Wrap census inputs in a trace-shaped object for the backstop helper."""

    return SimpleNamespace(
        layer_list=layer_list,
        completeness_decompositions=decompositions,
        completeness_diagnostics=diagnostics,
    )


@pytest.mark.parametrize("module_token", ["module_forward:exhaustive", "module_forward:predicate"])
def test_completeness_backstop_counts_module_forward_owned_drop(module_token: str) -> None:
    """A dispatch owned only by a module-forward token is a drop, like an unowned one.

    Inside a wrapped submodule the innermost token is the module's, so a stale pre-wrap
    torch reference there records ``owner_not_captured`` instead of ``unowned_dispatch``.
    Nesting must not change the verdict; ``torch_func:*``-owned pure reads stay benign.
    """

    ops = [_backstop_op(1), _backstop_op(2)]
    decs = [_backstop_dec(1), _backstop_dec(2)]

    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            ops, decs, [_backstop_diag(reason="owner_not_captured", owner_wrapper=module_token)]
        )
    )
    assert (dispatch, captured) == (3, 2)

    # Mutating and module-forward-owned: counted once, not twice.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            ops,
            decs,
            [_backstop_diag(reason="owner_not_captured", owner_wrapper=module_token, mutates=True)],
        )
    )
    assert (dispatch, captured) == (3, 2)

    # Inside a genuine replacement hook it is construction, as for unowned dispatches.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            ops,
            decs,
            [
                _backstop_diag(
                    reason="owner_not_captured",
                    owner_wrapper=module_token,
                    in_replacement_hook=True,
                )
            ],
        )
    )
    assert dispatch == captured

    # A torch-function-owned pure read (equal/allclose control flow) stays benign,
    # and so does a user forward-hook token (not a module-forward token).
    for benign_owner in (
        "torch_func:equal:logged",
        "torch_func:allclose:logged",
        "module_forward_hook:user",
    ):
        dispatch, captured = completeness_backstop_counts(
            _backstop_trace(
                ops, decs, [_backstop_diag(reason="owner_not_captured", owner_wrapper=benign_owner)]
            )
        )
        assert dispatch == captured, benign_owner


@pytest.mark.usefixtures("_restore_alias")
def test_attribute_held_original_alias_is_rebound_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stale-alias case with the bare original, as it was first written.

    Capture preparation rebinds the held ``_weight_norm`` original to its
    wrapper, so the alias the forward installs is tracked: no escape, and
    ``bfs_completeness`` passes.
    """

    def bare(func: Callable[..., Any]) -> Callable[..., Any]:
        wrap_torch()
        return _state._decorated_to_orig.get(id(func), func)

    monkeypatch.setitem(globals(), "_raw", bare)
    model = _NestedWeightNorm(stale_alias=True)
    assert not isinstance(model.raw_weight_norm, OpaqueCallable)
    assert _validate(model), tl.validation.last_validation_failure()
