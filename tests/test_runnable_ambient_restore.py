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
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from torchlens._runnable.state_context import _ambient_execution_context_restored
from torchlens._utils import _torch_compat


def _globals() -> dict[str, Any]:
    """Every process global the ambient context reads or writes."""

    return {
        "legacy": _torch_compat.snapshot_ambient_execution_context(),
        "fp32_precision": _torch_compat.snapshot_fp32_precision_controls(),
    }


@pytest.fixture
def unset_fp32_precision_children() -> Iterator[None]:
    """Put the oneDNN matmul ``fp32_precision`` control at torch's 'none' default.

    'none' (inherit from the parent) is the state of a process that never
    touched the precision controls, and the one the legacy matmul setter
    overwrites. Only this control is reset: forcing the cuDNN children too
    would build a mixed conv/RNN state the legacy ``allow_tf32`` getter refuses.
    """

    fields = _torch_compat.snapshot_fp32_precision_controls()
    if "mkldnn.matmul" in fields:
        _torch_compat.restore_fp32_precision_controls({**fields, "mkldnn.matmul": "none"})
    try:
        yield
    finally:
        _torch_compat.restore_fp32_precision_controls(fields)


# Kept: sole branch coverage of _runnable/state_context.py.
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
