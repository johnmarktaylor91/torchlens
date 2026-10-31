"""Pre-floor .tlspec artifacts refuse typed; current restores stay clean."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens._ir.workspaces import (
    LEGACY_TRACE_BUILD_STATE_KEYS,
    ModuleCaptureWorkspace,
    RawGraphWorkspace,
    WrapperRuntimeWorkspace,
)

V3_GOLDEN = Path(__file__).parent / "golden" / "io_v3_sample.tlspec"
_DROPPED_CAPTURE_FIELDS = (
    "_raw_layer_dict",
    "_raw_layer_labels_list",
    "_layer_counter",
)


def test_load_pre_floor_golden_refuses_typed() -> None:
    """The checked-in pre-floor golden bundle refuses with the floor named."""

    from torchlens.errors import ArtifactTooOldError

    with pytest.raises(ArtifactTooOldError, match="torchlens 2.31.0"):
        tl.load(str(V3_GOLDEN))


def test_save_load_roundtrip_v4(tmp_path: Path) -> None:
    """Save a v4 Trace, reload it, and confirm capture scratch stays absent."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))
    path = tmp_path / "roundtrip.tlspec"

    trace.save(path)
    loaded = tl.load(path)

    for field_name in _DROPPED_CAPTURE_FIELDS:
        assert field_name not in loaded.__dict__


def test_plain_restore_drops_legacy_flat_and_nested_build_scratch() -> None:
    """Never resurrect legacy capture scratch while restoring a finished Trace."""

    source = tl.trace(nn.Sequential(nn.Linear(2, 2), nn.ReLU()), torch.randn(1, 2))
    restored = Trace.__new__(Trace)
    try:
        state = source.__getstate__()
        state.update({field_name: object() for field_name in LEGACY_TRACE_BUILD_STATE_KEYS})
        state["_build_state"] = object()
        state["_raw_graph_ws"] = RawGraphWorkspace()
        state["_module_capture_ws"] = ModuleCaptureWorkspace()
        state["_wrapper_runtime_ws"] = WrapperRuntimeWorkspace()
        restored.__setstate__(state)

        assert "_build_state" not in restored.__dict__
        assert "_raw_graph_ws" not in restored.__dict__
        assert "_module_capture_ws" not in restored.__dict__
        assert "_wrapper_runtime_ws" not in restored.__dict__
        assert not LEGACY_TRACE_BUILD_STATE_KEYS.intersection(restored.__dict__)
    finally:
        source.cleanup()
        restored.cleanup()
