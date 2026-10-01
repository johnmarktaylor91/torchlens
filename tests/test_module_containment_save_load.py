"""Phase 4 regression tests: save/load schema versioning and the 2.33 floor."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION, ArtifactVersionBelowFloorError


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
    """The coordinated tlspec v9 bump stamps 9; the 2.33 rehydration floor stays 6."""

    assert TLSPEC_VERSION == 9
    assert MIN_TLSPEC_VERSION == 6


def test_pre_floor_module_call_state_refuses_typed() -> None:
    """A v5 ModuleCall state refuses at the 2.33 rehydration floor."""

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    call = trace.module_calls["0:1"]
    state = call.__getstate__()
    state["tlspec_version"] = 5

    restored = type(call).__new__(type(call))
    with pytest.raises(ArtifactVersionBelowFloorError, match="torchlens 2.33"):
        restored.__setstate__(state)


def test_round_trip_save_load_preserves_module_containment(tmp_path: Path) -> None:
    """Save a current trace, load it, and confirm module containment survives."""

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    bundle_path = tmp_path / "test_bundle.tlspec"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)

    op_with_modules = next((op for op in loaded.layer_list if getattr(op, "modules", [])), None)
    assert op_with_modules is not None, "loaded trace should preserve module containment"
    assert isinstance(op_with_modules.modules, tuple)


def test_legacy_thread_field_pickle_refuses_typed() -> None:
    """A v2 Op state with thread-replay fields refuses instead of resurrecting.

    Pre-floor behavior dropped the legacy thread fields with a one-per-process
    ``DeprecationWarning``; the 2.33 floor replaces that resurrection path with
    a typed refusal that names the floor.
    """

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    op = next(iter(trace.layer_list))
    state = op.__getstate__()
    state["tlspec_version"] = 2
    state["_module_boundary_thread_output"] = [("+", "fake_module", 1)]
    state["_module_boundary_threads_inputs"] = {"fake_label": []}
    state["module_entry_exit_threads_inputs"] = {"old_alias": []}

    decoded = pickle.loads(pickle.dumps(state))
    with pytest.raises(ArtifactVersionBelowFloorError, match="tlspec_version=2"):
        type(op).__setstate__(op, decoded)
