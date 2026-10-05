"""Legacy ``torch.nn.utils.weight_norm`` records ``_weight_norm`` in every capture.

``torch/nn/utils/weight_norm.py`` binds ``from torch import _weight_norm`` at
``import torch`` time, and its ``WeightNorm`` forward pre-hook calls that module
global on every forward. Before the alias was tracked in
``_TORCH_SUBMODULE_ALIAS_TARGETS`` the wrapped ``torch._weight_norm`` never
reached it: ``tl.trace`` recorded the op only through the mode rescue re-run
(``capture_verified=False``), the validation capture (no rescue) dropped it, and
validation failed ``bfs_completeness`` for a direct child while a nested one
passed with the op missing.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.user_funcs import _validate_forward_pass_torch


def _legacy_weight_norm(module: nn.Module) -> nn.Module:
    """Apply the deprecated hook-based weight_norm without its FutureWarning."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return nn.utils.weight_norm(module)


class _DirectChild(nn.Module):
    """Weight-normed conv as a direct child of the traced model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = _legacy_weight_norm(nn.Conv1d(4, 4, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve and scale."""

        return self.conv(x) * 2.0


class _Nested(nn.Module):
    """Weight-normed convs inside a ``Sequential`` block."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _legacy_weight_norm(nn.Conv1d(4, 4, 3, padding=1)),
            nn.ReLU(),
            _legacy_weight_norm(nn.Conv1d(4, 4, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.block(x) + x


@pytest.mark.parametrize(("model_cls", "expected_ops"), [(_DirectChild, 1), (_Nested, 2)])
def test_trace_records_weight_norm_without_rescue(model_cls: type[nn.Module], expected_ops: int):
    """The primary capture sees every ``_weight_norm`` call; no rescue re-run."""

    torch.manual_seed(0)
    trace = tl.trace(model_cls(), torch.randn(1, 4, 8))
    func_names = [layer.func_name for layer in trace.layers]
    assert func_names.count("_weight_norm") == expected_ops
    assert trace.capture_verified is not False, trace.capture_verification_reason


def test_weight_norm_module_alias_is_wrapped_while_installed():
    """After a capture the module global is the same wrapper as ``torch._weight_norm``."""

    # ``from torch.nn.utils import weight_norm`` yields the function that shadows
    # the submodule; the alias lives on the submodule itself.
    weight_norm_module = sys.modules["torch.nn.utils.weight_norm"]
    tl.trace(_DirectChild(), torch.randn(1, 4, 8))
    assert weight_norm_module._weight_norm is torch._weight_norm
    assert weight_norm_module.norm_except_dim is torch.norm_except_dim


@pytest.mark.parametrize("model_cls", [_DirectChild, _Nested])
def test_validation_passes_with_weight_norm_captured(model_cls: type[nn.Module]):
    """Validation passes because the op is captured, for any nesting."""

    torch.manual_seed(0)
    passed = _validate_forward_pass_torch(
        model_cls(), [torch.randn(1, 4, 8)], {}, random_seed=0, validate_metadata=True
    )
    assert passed, tl.validation.last_validation_failure()


def test_unwrap_restores_the_weight_norm_module_alias():
    """``unwrap_torch`` puts the original builtins back on the submodule too."""

    from torchlens.backends.torch import wrappers

    weight_norm_module = sys.modules["torch.nn.utils.weight_norm"]
    tl.trace(_DirectChild(), torch.randn(1, 4, 8))
    try:
        wrappers.unwrap_torch()
        for name in ("_weight_norm", "norm_except_dim"):
            restored = getattr(weight_norm_module, name)
            assert isinstance(restored, types.BuiltinFunctionType), (name, restored)
            assert restored is getattr(torch, name)
    finally:
        wrappers.wrap_torch()
    assert weight_norm_module._weight_norm is torch._weight_norm


def test_shadowed_namespace_resolution_is_narrow():
    """Only a non-module walk result with an imported same-named submodule is redirected."""

    from torchlens.utils._torch_compat import get_optional_torch_namespace

    # The shadowed case: the package attribute is the function, the roster means the module.
    assert callable(torch.nn.utils.weight_norm)
    assert (
        get_optional_torch_namespace("torch.nn.utils.weight_norm")
        is sys.modules["torch.nn.utils.weight_norm"]
    )
    # A non-module namespace with no same-named submodule keeps the attribute walk.
    assert "torch.Tensor" not in sys.modules
    assert get_optional_torch_namespace("torch.Tensor") is torch.Tensor
    # Ordinary modules and missing names are unchanged.
    assert get_optional_torch_namespace("torch.nn.functional") is torch.nn.functional
    assert get_optional_torch_namespace("torch.no_such_namespace_xyz") is None
