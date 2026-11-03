"""Phase 4 regression tests: save/load schema versioning and the tlspec-6 floor."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION, ArtifactTooOldError


def _simple_model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    """Build a deterministic model and input pair."""

    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    return model, torch.randn(2, 8)


def test_io_format_version_and_floor() -> None:
    """The coordinated tlspec v11 bump stamps 11 and reads only stamp 11."""

    assert TLSPEC_VERSION == 11
    assert MIN_TLSPEC_VERSION == 11


def test_pre_floor_module_call_state_refuses_typed() -> None:
    """A v5 ModuleCall state refuses as a torchlens 2.x file."""

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    call = trace.module_calls["0:1"]
    state = call.__getstate__()
    state["tlspec_version"] = 5

    restored = type(call).__new__(type(call))
    with pytest.raises(ArtifactTooOldError, match="saved by torchlens 2.x"):
        restored.__setstate__(state)
