"""Regression tests for predicate exceptions in fastlog capture."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import PredicateError, RecordContext


def test_predicate_exception_propagates() -> None:
    """A fail-fast predicate exception surfaces through fastlog.record."""

    def keep_op(ctx: RecordContext) -> bool:
        """Raise from the first operation predicate invocation."""

        raise RuntimeError(f"stop at {ctx.label}")

    with pytest.raises(RuntimeError, match="stop at"):
        tl.fastlog.record(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            torch.ones(1, 3),
            keep_op=keep_op,
            on_predicate_error="fail-fast",
        )


def test_predicate_exception_accumulate_preserved() -> None:
    """RAM auto mode still reports accumulated predicate failures."""

    def keep_op(ctx: RecordContext) -> bool:
        """Raise from operation predicate invocations."""

        raise RuntimeError(f"bad predicate at {ctx.label}")

    with pytest.raises(PredicateError, match="fastlog predicate failed"):
        tl.fastlog.record(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            torch.ones(1, 3),
            keep_op=keep_op,
        )
