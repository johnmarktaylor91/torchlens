"""An indexing op whose slice bound is a 0-d tensor is captured.

``x[i:i + 1, :length] = v`` with ``length`` a 0-d tensor makes Python convert
the bound through ``Tensor.__index__`` INSIDE the ``__setitem__`` call. The
``__index__`` wrapper used to take the enclosing call's bottom-level barcode,
so the ``__setitem__`` op was never logged and completeness failed (Menagerie
rung 4: f-tm2t, f-cg-detr). ``__index__`` is now barcode-transparent, like
``as_subclass``.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


class _FlipPrefix(nn.Module):
    """Reverse each row's first ``lens[i]`` steps in place (the tm2t shape)."""

    def __init__(self, tensor_bounds: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.tensor_bounds = tensor_bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.lin(x).clone()
        lens = torch.tensor([3, 5]) if self.tensor_bounds else [3, 5]
        for i, length in enumerate(lens):
            b[i : i + 1, :length] = torch.flip(b[i : i + 1, :length].clone(), dims=[1])
        return b


class _ZeroTail(nn.Module):
    """``frame[i][vlen[i]:] *= 0.`` with ``vlen`` a tensor (the cg-detr shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame = self.lin(x).clone()
        vlen = torch.tensor([2, 3])
        for i in range(2):
            frame[i][vlen[i] :] *= 0.0
        return frame


@pytest.mark.smoke
@pytest.mark.parametrize("tensor_bounds", [True, False], ids=["tensor_bound", "int_bound"])
def test_slice_bound_setitem_validates(tensor_bounds: bool) -> None:
    torch.manual_seed(0)
    model = _FlipPrefix(tensor_bounds).eval()
    x = torch.randn(2, 5, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True, last_validation_failure()
        trace = tl.trace(model, x)
    assert [op.func_name for op in trace.ops].count("__setitem__") == 2


@pytest.mark.smoke
def test_tensor_bound_augmented_slice_assignment_validates() -> None:
    torch.manual_seed(0)
    model = _ZeroTail().eval()
    x = torch.randn(2, 5, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True, last_validation_failure()
