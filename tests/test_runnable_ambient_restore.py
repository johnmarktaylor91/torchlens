"""Runnable replay leaves every global its ambient context touches unchanged.

``trace.run()`` applies the capture's recorded ambient context (matmul
precision, deterministic mode, cuDNN and SDPA switches, ...) for the run and
must hand the caller back exactly the globals it found, on success and on
error. On torch >= 2.9 the legacy precision setters also write torch's
``fp32_precision`` fields, which re-applying the legacy snapshot used to leave
changed ('none' -> 'ieee').
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable_state_context import _ambient_execution_context_restored
from torchlens.utils import _torch_compat


def _globals() -> dict[str, Any]:
    """Every process global the ambient context reads or writes."""

    return {
        "legacy": _torch_compat.snapshot_ambient_execution_context(),
        "fp32_precision": _torch_compat.snapshot_fp32_precision_controls(),
    }


@pytest.fixture
def unset_fp32_precision_children() -> Iterator[None]:
    """Put every per-backend ``fp32_precision`` child at torch's 'none' default.

    'none' (inherit from the root) is the state a process that never touched
    the precision controls is in, and the one the legacy setters overwrite.
    """

    fields = _torch_compat.snapshot_fp32_precision_controls()
    _torch_compat.restore_fp32_precision_controls(
        {path: ("none" if path else value) for path, value in fields.items()}
    )
    try:
        yield
    finally:
        _torch_compat.restore_fp32_precision_controls(fields)


@pytest.mark.usefixtures("unset_fp32_precision_children")
def test_loaded_runnable_run_leaves_globals_unchanged(tmp_path: Path) -> None:
    """A loaded sparse run restores the caller's exact globals afterwards."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x)
    path = tmp_path / "ambient.tlspec"
    tl.save(trace, path, level="runnable", include_weights=True)
    loaded = tl.load(path)

    before = _globals()
    loaded.run(inputs=x, seed=0)
    assert _globals() == before


@pytest.mark.usefixtures("unset_fp32_precision_children")
def test_ambient_context_restores_globals_when_the_run_raises() -> None:
    """The restore runs on the error path too, fp32_precision fields included."""

    recorded = SimpleNamespace(**_torch_compat.snapshot_ambient_execution_context())
    recorded.default_device = None
    recorded.float32_matmul_precision = (
        "high" if torch.get_float32_matmul_precision() != "high" else "medium"
    )
    before = _globals()
    with pytest.raises(RuntimeError, match="raised inside the run"):
        with _ambient_execution_context_restored(recorded):
            assert torch.get_float32_matmul_precision() == recorded.float32_matmul_precision
            raise RuntimeError("raised inside the run")
    assert _globals() == before


def test_fp32_precision_snapshot_round_trips() -> None:
    """Snapshot then restore is exact even after a legacy setter rewrote the fields."""

    snapshot = _torch_compat.snapshot_fp32_precision_controls()
    if not _torch_compat.HAS_FP32_PRECISION_CONTROLS:
        assert snapshot == {}
        return
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        torch.set_float32_matmul_precision(previous)
    finally:
        _torch_compat.restore_fp32_precision_controls(snapshot)
    assert _torch_compat.snapshot_fp32_precision_controls() == snapshot
