"""v3-format .tlspec artifacts must still load cleanly under v4."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import torchlens as tl

V3_GOLDEN = Path(__file__).parent / "golden" / "io_v3_sample.tlspec"
_DROPPED_CAPTURE_FIELDS = (
    "_raw_layer_dict",
    "_raw_layer_labels_list",
    "_layer_counter",
)


def test_load_v3_artifact_into_v4_runtime() -> None:
    """Load a v3 bundle and confirm loader-only normalization strips scratch fields."""

    log = tl.load(str(V3_GOLDEN))

    assert log.num_ops > 0
    assert hasattr(log, "layer_logs")
    assert hasattr(log, "input_layers")

    for field_name in _DROPPED_CAPTURE_FIELDS:
        assert field_name not in log.__dict__, (
            f"v3-loaded Trace carries legacy field {field_name!r}; normalizer failed"
        )


def test_save_load_roundtrip_v4(tmp_path: Path) -> None:
    """Save a v4 Trace, reload it, and confirm capture scratch stays absent."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))
    path = tmp_path / "roundtrip.tlspec"

    tl.save(trace, path)
    loaded = tl.load(path)

    for field_name in _DROPPED_CAPTURE_FIELDS:
        assert field_name not in loaded.__dict__
