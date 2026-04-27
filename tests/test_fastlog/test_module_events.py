"""Comprehensive module-event behavior tests for fastlog."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._state as torchlens_state
import torchlens.fastlog._state as fastlog_state
from torchlens.fastlog import PredicateError, RecordContext


class RootLinear(nn.Linear):
    """A root-only linear model."""


class SharedModule(nn.Module):
    """Model that calls one shared child twice."""

    def __init__(self) -> None:
        """Initialize the shared module."""

        super().__init__()
        self.shared = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same child twice."""

        return self.shared(self.shared(x))


class IdentitySequence(nn.Module):
    """Sequential model containing an identity pass-through."""

    def __init__(self) -> None:
        """Initialize the sequence."""

        super().__init__()
        self.sequence = nn.Sequential(nn.Linear(3, 3), nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the sequence."""

        return self.sequence(x)


class RaisingModel(nn.Module):
    """Model that raises from forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise a controlled exception."""

        raise RuntimeError("forward exploded")


def test_root_only_linear_emits_root_enter_exit() -> None:
    """A single root nn.Linear still emits root enter and exit events."""

    recording = tl.fastlog.record(
        RootLinear(3, 2),
        torch.ones(1, 3),
        keep_module=lambda ctx: ctx.module_address == "",
    )

    assert [record.ctx.kind for record in recording] == ["module_enter", "module_exit"]


def test_shared_module_pass_counter_increments_for_two_calls() -> None:
    """Shared module pass counters increment for repeated calls in one forward."""

    recording = tl.fastlog.record(
        SharedModule(),
        torch.ones(1, 3),
        keep_module=lambda ctx: ctx.module_address == "shared",
    )
    child_events = [record.ctx for record in recording if record.ctx.module_address == "shared"]

    assert [ctx.kind for ctx in child_events] == [
        "module_enter",
        "module_exit",
        "module_enter",
        "module_exit",
    ]
    assert [ctx.module_pass_index for ctx in child_events] == [1, 1, 2, 2]


def test_identity_sequence_pass_through_tensor_is_visible() -> None:
    """Identity modules emit events and pass-through tensors remain visible."""

    recording = tl.fastlog.record(
        IdentitySequence(),
        torch.ones(1, 3),
        keep_module=lambda ctx: bool(ctx.module_address and "1" in ctx.module_address),
        default_op=False,
    )

    assert any(record.ctx.module_type == "Identity" for record in recording)


def test_forward_exception_cleans_recording_stack() -> None:
    """Forward exceptions clean global TorchLens and fastlog state."""

    with pytest.raises(RuntimeError, match="forward exploded"):
        tl.fastlog.record(RaisingModel(), torch.ones(1), default_module=True)

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_model_log is None
    assert fastlog_state._active_recording_state is None


def test_predicate_exception_cleans_stack_and_respects_fail_fast() -> None:
    """Fail-fast predicate exceptions clean global state immediately."""

    def keep_op(ctx: RecordContext) -> bool:
        """Raise for operation predicates."""

        raise PredicateError("predicate exploded", ctx=ctx)

    with pytest.raises(PredicateError, match="predicate exploded"):
        tl.fastlog.record(
            SharedModule(),
            torch.ones(1, 3),
            keep_op=keep_op,
            on_predicate_error="fail-fast",
        )

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_model_log is None
    assert fastlog_state._active_recording_state is None
