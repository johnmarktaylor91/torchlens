"""Phase 4 regression tests: save/load schema migration."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
import torchlens as tl

from torchlens._io import IO_FORMAT_VERSION, reset_legacy_thread_warning


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Generator[None, None, None]:
    """Reset the legacy-warning once flag around each test."""

    reset_legacy_thread_warning()
    yield
    reset_legacy_thread_warning()


def _simple_model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    """Build a deterministic model and input pair."""

    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    return model, torch.randn(2, 8)


def test_io_format_version_is_four() -> None:
    """M8 bumps ``IO_FORMAT_VERSION`` from 3 to 4."""

    assert IO_FORMAT_VERSION == 4


def test_round_trip_save_load_preserves_module_containment(tmp_path: Path) -> None:
    """Save a v3 trace, load it, and confirm module containment survives."""

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    bundle_path = tmp_path / "test_bundle.tlspec"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)

    op_with_modules = next((op for op in loaded.layer_list if getattr(op, "modules", [])), None)
    assert op_with_modules is not None, "loaded trace should preserve module containment"
    assert isinstance(op_with_modules.modules, list)


def test_legacy_pickle_load_drops_thread_fields() -> None:
    """Load a v2 Op state and confirm legacy thread fields are dropped."""

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    op = next(iter(trace.layer_list))
    state = op.__getstate__()
    state["io_format_version"] = 2
    state["_module_boundary_thread_output"] = [("+", "fake_module", 1)]
    state["_module_boundary_threads_inputs"] = {"fake_label": []}
    state["module_entry_exit_threads_inputs"] = {"old_alias": []}

    decoded = pickle.loads(pickle.dumps(state))
    with pytest.warns(DeprecationWarning, match="legacy thread-replay fields"):
        type(op).__setstate__(op, decoded)

    for attr in (
        "_module_boundary_thread_output",
        "_module_boundary_threads_inputs",
        "module_entry_exit_threads_inputs",
    ):
        assert attr not in op.__dict__, f"{attr} should be dropped on legacy-pickle load"


def test_legacy_pickle_load_emits_one_deprecation_warning() -> None:
    """Loading multiple legacy OpLogs emits one deprecation warning per process."""

    model, x = _simple_model_and_input()
    trace = tl.trace(model, x)
    op = next(iter(trace.layer_list))
    legacy_pickles = []
    for _ in range(3):
        state = op.__getstate__()
        state["io_format_version"] = 2
        state["_module_boundary_thread_output"] = [("+", "x", 1)]
        legacy_pickles.append((op, pickle.dumps(state)))

    reset_legacy_thread_warning()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for op, pickled in legacy_pickles:
            decoded = pickle.loads(pickled)
            type(op).__setstate__(op, decoded)

    ours = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and "legacy thread-replay fields" in str(warning.message)
    ]
    assert len(ours) == 1, (
        f"expected exactly 1 legacy-thread DeprecationWarning per process, got {len(ours)}"
    )
