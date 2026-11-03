"""Predicate exception mode tests for fastlog."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PredicateError
from torchlens.recording import RecordContext


class ExceptionModel(nn.Module):
    """Model with several operation events."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.layers(x)


def _raising_predicate(ctx: RecordContext) -> bool:
    """Raise for every operation predicate call."""

    raise RuntimeError(f"bad predicate at {ctx.label}")


def test_max_predicate_failures_tracks_overflow_separately() -> None:
    """Overflow failures are counted separately from retained failures."""

    with pytest.raises(PredicateError) as exc_info:
        tl.record(
            ExceptionModel(),
            torch.ones(1, 3),
            save=_raising_predicate,
            on_predicate_error="accumulate",
            max_predicate_failures=0,
        )

    assert exc_info.value.failures == []
    assert exc_info.value.overflow == exc_info.value.total_count
