"""tvscope B1 + composition row 1: trace-side provenance population + survival.

The B1 stamp: any capture that APPLIED an input transform carries an honest
provenance record (never a guess); bridge stamps override it; captures
without a transform keep the historical ``None``. Composition row 1: the
record survives save/load field-for-field, and the derived status is stable
across the round trip.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.preprocessing as pp
from torchlens.data_classes.trace import ResolvedPreprocessing

pytestmark = [pytest.mark.smoke]


class _Tiny(nn.Module):
    """Two-layer module for cheap captures."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One linear + relu."""

        return torch.relu(self.proj(x))


def test_transform_capture_stamps_user_transform_record() -> None:
    """A plain trace with transform= carries honest unknown provenance."""

    log = tl.trace(
        _Tiny().eval(),
        [1.0] * 8,
        capture=tl.options.CaptureOptions(
            transform=lambda x: torch.tensor([x], dtype=torch.float32)
        ),
    )
    record = log.input_preprocessor
    assert record is not None
    assert record.source == "user_transform"
    assert record.verified is False
    assert record.status == pp.STATUS_UNKNOWN
    assert "0x" not in record.identifier or "0x<scrubbed>" in record.identifier


def test_plain_tensor_capture_keeps_none() -> None:
    """No transform applied -> no provenance invented (the field stays None)."""

    log = tl.trace(_Tiny().eval(), torch.randn(2, 8))
    assert log.input_preprocessor is None


def test_record_survives_save_load_with_stable_status(tmp_path) -> None:
    """Composition row 1: source/verified/config survive field-for-field."""

    log = tl.trace(
        _Tiny().eval(),
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(layers_to_save="none"),
    )
    log.input_preprocessor = ResolvedPreprocessing(
        source="torchvision_weights",
        identifier="ResNet50_Weights.IMAGENET1K_V2",
        verified=True,
        config={"mean": [0.485, 0.456, 0.406], "resolution_method": "model_metadata"},
        description="torchvision preset: resize=232 crop=224",
    )
    path = tmp_path / "trace.tlspec"
    tl.save(log, str(path))
    loaded = tl.load(str(path))
    record = loaded.input_preprocessor
    assert record is not None
    assert record.source == "torchvision_weights"
    assert record.verified is True
    assert record.config["mean"] == [0.485, 0.456, 0.406]
    assert record.description == "torchvision preset: resize=232 crop=224"
    assert record.status == pp.STATUS_AUTHORITATIVE


def test_unverified_fallback_status_survives_save_load(tmp_path) -> None:
    """The demoted fallback stays unverified_fallback after a round trip."""

    log = tl.trace(
        _Tiny().eval(),
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(layers_to_save="none"),
    )
    log.input_preprocessor = ResolvedPreprocessing(
        source="imagenet_default",
        identifier="ImageNet-default-resize256-crop224",
        verified=False,
        config={},
        description="fallback",
    )
    path = tmp_path / "trace_fallback.tlspec"
    tl.save(log, str(path))
    loaded = tl.load(str(path))
    assert loaded.input_preprocessor.status == pp.STATUS_UNVERIFIED_FALLBACK
