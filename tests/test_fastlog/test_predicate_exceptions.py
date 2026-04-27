"""Predicate exception mode tests for fastlog."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import PredicateError, RecordContext


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


def test_accumulate_ram_raises_at_end_with_capped_failures() -> None:
    """RAM accumulate mode collects failures up to max_predicate_failures."""

    with pytest.raises(PredicateError) as exc_info:
        tl.fastlog.record(
            ExceptionModel(),
            torch.ones(1, 3),
            keep_op=_raising_predicate,
            on_predicate_error="accumulate",
            max_predicate_failures=2,
        )

    assert len(exc_info.value.failures) == 2
    assert exc_info.value.overflow > 0
    assert exc_info.value.total_count > 2


def test_fail_fast_raises_on_first_failure() -> None:
    """Fail-fast mode raises immediately on the first predicate failure."""

    calls = 0

    def keep_op(ctx: RecordContext) -> bool:
        """Count and then fail."""

        nonlocal calls
        calls += 1
        raise PredicateError("first failure", ctx=ctx)

    with pytest.raises(PredicateError, match="first failure"):
        tl.fastlog.record(
            ExceptionModel(),
            torch.ones(1, 3),
            keep_op=keep_op,
            on_predicate_error="fail-fast",
        )

    assert calls == 1


def test_auto_maps_to_accumulate_for_ram() -> None:
    """Auto predicate error mode accumulates failures for RAM-only sessions."""

    with pytest.raises(PredicateError) as exc_info:
        tl.fastlog.record(
            ExceptionModel(),
            torch.ones(1, 3),
            keep_op=_raising_predicate,
            on_predicate_error="auto",
            max_predicate_failures=1,
        )

    assert len(exc_info.value.failures) == 1
    assert exc_info.value.total_count > 1


def test_auto_maps_to_fail_fast_for_disk(tmp_path: Path) -> None:
    """Auto predicate error mode fails fast for disk-backed sessions."""

    calls = 0

    def keep_op(ctx: RecordContext) -> bool:
        """Count and then fail."""

        nonlocal calls
        calls += 1
        raise PredicateError("disk failure", ctx=ctx)

    with pytest.raises(PredicateError, match="disk failure"):
        tl.fastlog.record(
            ExceptionModel(),
            torch.ones(1, 3),
            keep_op=keep_op,
            on_predicate_error="auto",
            streaming=tl.StreamingOptions(
                bundle_path=tmp_path / "disk.tlfast",
                retain_in_memory=False,
            ),
        )

    assert calls == 1


def test_max_predicate_failures_tracks_overflow_separately() -> None:
    """Overflow failures are counted separately from retained failures."""

    with pytest.raises(PredicateError) as exc_info:
        tl.fastlog.record(
            ExceptionModel(),
            torch.ones(1, 3),
            keep_op=_raising_predicate,
            on_predicate_error="accumulate",
            max_predicate_failures=0,
        )

    assert exc_info.value.failures == []
    assert exc_info.value.overflow == exc_info.value.total_count
