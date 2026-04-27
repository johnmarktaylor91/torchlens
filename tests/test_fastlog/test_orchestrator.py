"""Tests for the internal fastlog predicate orchestrator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens._state as torchlens_state
import torchlens.fastlog._state as fastlog_state
from torchlens.fastlog._orchestrator import _run_predicate_pass
from torchlens.fastlog.exceptions import PredicateError
from torchlens.fastlog.options import RecordingOptions


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
        keep_module=lambda ctx: True,
        default_op=False,
        default_module=False,
        include_source_events=True,
    )


def test_root_only_model_emits_root_events() -> None:
    """Root-only models still get synthetic root enter and exit events."""

    output, recording = _run_predicate_pass(RootOnly(), torch.ones(1), None, _options())
    kinds = [record.ctx.kind for record in recording]

    assert torch.equal(output, torch.tensor([2.0]))
    assert kinds.count("module_enter") == 1
    assert kinds.count("module_exit") == 1
    assert recording.records[0].ctx.label == "root:enter:1"
    assert recording.records[-1].ctx.label == "root:exit:1"


def test_shared_module_called_twice_has_balanced_events() -> None:
    """A shared child module called twice emits balanced enter/exit events."""

    _, recording = _run_predicate_pass(SharedModule(), torch.ones(1, 3), None, _options())
    child_events = [
        record.ctx
        for record in recording
        if record.ctx.module_address == "shared" and record.ctx.kind.startswith("module")
    ]

    assert [ctx.kind for ctx in child_events] == [
        "module_enter",
        "module_exit",
        "module_enter",
        "module_exit",
    ]
    assert [ctx.module_pass_index for ctx in child_events] == [1, 1, 2, 2]


def test_identity_passthrough_has_module_events() -> None:
    """Identity modules are represented even when their tensor passes through."""

    output, recording = _run_predicate_pass(IdentityModel(), torch.ones(1), None, _options())
    labels = [record.ctx.module_address for record in recording]

    assert torch.equal(output, torch.ones(1))
    assert "identity" in labels


def test_forward_exception_cleans_logging_state() -> None:
    """Forward exceptions leave global logging and recording state inactive."""

    with pytest.raises(RuntimeError, match="forward failed"):
        _run_predicate_pass(RaisingModel(), torch.ones(1), None, _options())

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_model_log is None
    assert fastlog_state._active_recording_state is None


def test_predicate_exception_cleans_logging_state() -> None:
    """Predicate exceptions leave global logging and recording state inactive."""

    def bad_keep_module(ctx) -> bool:
        """Raise from the module predicate."""

        raise RuntimeError(f"bad predicate for {ctx.kind}")

    with pytest.raises(PredicateError, match="fastlog predicate failed"):
        _run_predicate_pass(
            RootOnly(),
            torch.ones(1),
            None,
            RecordingOptions(keep_module=bad_keep_module, default_op=False),
        )

    assert torchlens_state._logging_enabled is False
    assert torchlens_state._active_model_log is None
    assert fastlog_state._active_recording_state is None
