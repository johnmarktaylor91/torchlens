"""Stale module-global torch references are patched at wrap time, so validation sees the op.

A model module that ran ``from torch import bmm`` (or
``from torch.nn.functional import linear``) before TorchLens first wrapped torch
holds the pristine callable. ``tl.trace`` recovered it through the rescue
re-run, but the validation capture runs mode-free with no rescue, so the op
vanished from the graph and completeness failed (Menagerie rung 4: f-s2vc
``bmm``, f-cltr ``linear``, f-soundstream ``cat``). The wrap-time module sweep
now patches every wrapped torch callable, not only the protocol-invisible belt
members, and ``unwrap_torch()`` reverses it.
"""

from __future__ import annotations

import sys
import types
import warnings
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import belt
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.validation import last_validation_failure

_MODULE_SOURCE = """
class StaleBmm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        h = self.lin(x)
        return bmm(h, h.transpose(1, 2)).relu()
"""


@pytest.fixture()
def stale_module() -> Iterator[types.ModuleType]:
    """A defining module whose ``bmm`` global was bound while torch was unwrapped."""

    unwrap_torch()
    module = types.ModuleType("_tl_stale_bmm_global")
    module.__dict__.update({"bmm": torch.bmm, "nn": nn, "torch": torch})
    exec(_MODULE_SOURCE, module.__dict__)  # noqa: S102 - throwaway defining module
    sys.modules[module.__name__] = module
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)
        wrap_torch()


@pytest.mark.smoke
def test_stale_module_global_validates_and_captures_the_op(
    stale_module: types.ModuleType,
) -> None:
    """The validation capture records ``bmm`` and validates without a rescue."""

    pristine = stale_module.bmm
    torch.manual_seed(0)
    model = stale_module.StaleBmm().eval()
    x = torch.randn(2, 3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True, last_validation_failure()
        trace = tl.trace(model, x)
    assert "bmm" in [op.func_name for op in trace.ops]
    assert trace.rescue_rerun is None
    assert stale_module.bmm is _state._orig_to_decorated[id(pristine)]


@pytest.mark.smoke
def test_unwrap_restores_the_swept_global(stale_module: types.ModuleType) -> None:
    """``unwrap_torch()`` puts the pristine callable back; a rewrap patches it again."""

    pristine = stale_module.bmm
    wrap_torch()
    assert stale_module.bmm is not pristine
    unwrap_torch()
    assert stale_module.bmm is pristine
    wrap_torch()
    assert stale_module.bmm is _state._orig_to_decorated[id(pristine)]


@pytest.mark.smoke
def test_sweep_never_replaces_a_class_or_a_foreign_callable() -> None:
    """Only exact wrapped torch callables are patched; ``Tensor`` stays a type."""

    wrap_torch()
    module = types.ModuleType("_tl_stale_sweep_negatives")
    module.__dict__.update({"Tensor": torch.Tensor, "helper": len, "torch": torch})
    sys.modules[module.__name__] = module
    try:
        belt.sweep_stale_belt_references()
        assert module.Tensor is torch.Tensor
        assert module.helper is len
    finally:
        sys.modules.pop(module.__name__, None)
