"""Regression tests for r18bauto.

Item 2 (landed): ``tl.record`` must enforce the same unsupported-tensor-variant guard that
``tl.trace`` enforces (meta / sparse / symbolic-shape inputs). Before this fix ``record`` skipped
the guard and crashed deep in the capture pipeline with an opaque torch error instead of raising a
clean ``UnsupportedTensorVariantError`` up front.

Item 1 (A3-05 double-construction reuse) is a deferred patch request routed to the bridge/hf.py
owner; it is documented in the private R18BAUTO triage report and not tested here because
its fix lives in an off-lease file.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._robustness import UnsupportedTensorVariantError


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_record_rejects_meta_tensor_like_trace() -> None:
    model = _AddOne()
    # trace already rejects; record must match.
    with pytest.raises(UnsupportedTensorVariantError):
        tl.trace(model, torch.randn(4, device="meta"))
    with pytest.raises(UnsupportedTensorVariantError):
        tl.record(model, torch.randn(4, device="meta"), save=tl.func("add"))


def test_record_rejects_sparse_tensor_like_trace() -> None:
    model = _AddOne()
    with pytest.raises(UnsupportedTensorVariantError):
        tl.trace(model, torch.randn(4).to_sparse())
    with pytest.raises(UnsupportedTensorVariantError):
        tl.record(model, torch.randn(4).to_sparse(), save=tl.func("add"))


def test_record_rejects_sparse_in_keyword_input() -> None:
    """Guard walks keyword inputs too (sibling boundary trace() covers)."""

    class _AddKw(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    model = _AddKw()
    with pytest.raises(UnsupportedTensorVariantError):
        tl.record(model, [], input_kwargs={"x": torch.randn(4).to_sparse()}, save=tl.func("add"))


def test_record_accepts_dense_cpu_input() -> None:
    """Control: a normal dense CPU input still records without error."""
    model = _AddOne()
    rec = tl.record(model, torch.randn(4), save=tl.func("add"))
    assert rec is not None
