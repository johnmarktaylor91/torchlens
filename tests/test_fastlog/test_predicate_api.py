"""Comprehensive predicate API contract tests for fastlog."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import CaptureSpec, PredicateError, RecordContext


class PredicateApiModel(nn.Module):
    """Small model for predicate API tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one differentiable operation."""

        return torch.relu(x + 1)


def _make_generator() -> Generator[bool, None, None]:
    """Return a generator object for invalid-return validation."""

    yield True


def test_predicate_accepts_record_context_signature() -> None:
    """Predicates receive a RecordContext instance."""

    seen: list[RecordContext] = []

    def keep_op(ctx: RecordContext) -> bool:
        """Keep operation events and record the predicate argument."""

        seen.append(ctx)
        return ctx.kind == "op"

    recording = tl.fastlog.record(PredicateApiModel(), torch.ones(2), keep_op=keep_op)

    assert len(recording) > 0
    assert seen
    assert all(isinstance(ctx, RecordContext) for ctx in seen)


@pytest.mark.parametrize(
    "result",
    [True, False, None, CaptureSpec(save_activation=True, save_metadata=False)],
)
def test_predicate_accepts_supported_return_types(result: bool | CaptureSpec | None) -> None:
    """Bool, CaptureSpec, and None predicate returns are accepted."""

    def keep_op(ctx: RecordContext) -> bool | CaptureSpec | None:
        """Return the parametrized decision for operation events."""

        return result if ctx.kind == "op" else False

    recording = tl.fastlog.record(
        PredicateApiModel(),
        torch.ones(2),
        keep_op=keep_op,
        default_op=True,
    )

    assert recording.n_passes == 1


@pytest.mark.parametrize(
    "bad_result",
    [torch.tensor(True), "yes", 1, _make_generator()],
)
def test_predicate_rejects_invalid_return_types_with_event_context(bad_result: Any) -> None:
    """Unsupported predicate returns raise PredicateError with event context."""

    def keep_op(ctx: RecordContext) -> Any:
        """Return an invalid predicate value for operation events."""

        return bad_result if ctx.kind == "op" else False

    with pytest.raises(PredicateError) as exc_info:
        tl.fastlog.record(PredicateApiModel(), torch.ones(2), keep_op=keep_op)

    assert exc_info.value.ctx is not None or exc_info.value.failures
    if exc_info.value.ctx is not None:
        assert exc_info.value.ctx.kind == "op"
        assert exc_info.value.result is bad_result
