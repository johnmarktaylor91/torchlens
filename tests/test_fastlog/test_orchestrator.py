"""Tests for the internal fastlog predicate orchestrator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._capture.projections as fastlog_state
import torchlens._state as torchlens_state
from torchlens.recording.exceptions import PredicateError
from torchlens.recording.options import RecordingOptions


class RootOnly(nn.Module):
    """Model with operations only at the root module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a root-only operation."""

        return x + 1


class SharedModule(nn.Module):
    """Model that calls one child module twice."""

    def __init__(self) -> None:
        """Initialize the shared child."""

        super().__init__()
        self.shared = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same module twice."""

        return self.shared(self.shared(x))


class IdentityModel(nn.Module):
    """Model containing an Identity pass-through module."""

    def __init__(self) -> None:
        """Initialize the identity module."""

        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the identity output."""

        return self.identity(x)


class RaisingModel(nn.Module):
    """Model whose forward method always raises."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise during forward."""

        raise RuntimeError("forward failed")


def _options() -> RecordingOptions:
    """Return options that keep all module events and operation payloads."""

    return RecordingOptions(
        keep_op=lambda ctx: ctx.kind == "op",
        default_op=False,
        default_module=False,
        include_source_events=True,
    )


def test_shared_module_called_twice_has_balanced_events() -> None:
    """A shared child module called twice emits balanced enter/exit events."""

    recording = tl.record(
        SharedModule(),
        torch.ones(1, 3),
        save=_options().keep_op,
        recording=tl.options.RecordingOptions(default_module=True, include_source_events=True),
    )
    child_events = [
        record.ctx
        for record in recording
        if record.ctx.address == "shared" and record.ctx.kind.startswith("module")
    ]

    assert [ctx.kind for ctx in child_events] == [
        "module_enter",
        "module_exit",
        "module_enter",
        "module_exit",
    ]
    assert [ctx.module_pass_number for ctx in child_events] == [1, 1, 2, 2]


def test_predicate_exception_cleans_logging_state() -> None:
    """Predicate exceptions leave global logging and recording state inactive."""

    def bad_halt(ctx) -> bool:
        """Raise from the halt predicate on every event."""

        raise RuntimeError(f"bad predicate for {ctx.kind}")

    with pytest.raises(PredicateError, match="fastlog predicate failed"):
        tl.record(
            RootOnly(),
            torch.ones(1),
            stop_after=bad_halt,
            recording=tl.options.RecordingOptions(default_module=True, default_op=False),
        )

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_trace is None
    assert fastlog_state._active_recording_state is None
