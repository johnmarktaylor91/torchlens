"""Smoke tests for fastlog RAM, disk, and recovery storage paths."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import CaptureSpec
from torchlens.fastlog.exceptions import PredicateError, RecordingConfigError
from torchlens.fastlog.types import RecordContext


class TinyModel(nn.Module):
    """Small model for storage smoke tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x).relu()


class DeepLinearModel(nn.Module):
    """Stack of linear layers for disk-path depth regressions."""

    def __init__(self, depth: int) -> None:
        """Initialize a deterministic linear stack."""

        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layer = nn.Linear(3, 3)
            with torch.no_grad():
                layer.weight.copy_(torch.eye(3))
                layer.bias.zero_()
            layers.extend((layer, nn.ReLU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full depth stack."""

        return self.layers(x)


def test_ram_only_roundtrip_records_in_memory() -> None:
    """RAM-only recording returns an in-memory Recording."""

    recording = tl.fastlog.record(TinyModel(), torch.ones(1, 3), default_op=True)

    assert recording.bundle_path is None
    assert recording.recovered is False
    assert len(recording) > 0
    assert recording.n_records == len(recording)
    assert any(record.ram_payload is not None for record in recording)


def test_lazy_recording_n_records_is_current_before_records_access() -> None:
    """`n_records` should match the retained records projection."""

    recording = tl.record(TinyModel(), torch.ones(1, 3), save=tl.func("linear"))

    assert recording.n_records == 1
    assert len(recording.records) == 1


def test_disk_only_roundtrip_loads_bundle(tmp_path: Path) -> None:
    """Disk-only recording writes a fastlog bundle that load can read."""

    bundle_path = tmp_path / "recording.tlfast"
    recording = tl.fastlog.record(
        TinyModel(),
        torch.ones(1, 3),
        default_op=True,
        streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )
    loaded = tl.fastlog.load(bundle_path)

    assert recording.bundle_path == bundle_path
    assert (bundle_path / "manifest.json").exists()
    assert (bundle_path / "metadata.json").exists()
    assert loaded.recovered is False
    assert len(loaded) == len(recording)
    assert loaded.n_records == len(recording)
    assert loaded.records[0].metadata["blob_id"] == recording.records[0].metadata["blob_id"]


@pytest.mark.parametrize(("depth", "ceiling_seconds"), [(4, 5.0), (12, 5.0)])
def test_disk_only_deep_stack_stays_linear_and_round_trips(
    tmp_path: Path,
    depth: int,
    ceiling_seconds: float,
) -> None:
    """Disk-mode fastlog should stay under a bounded wall-clock ceiling at depth."""

    bundle_path = tmp_path / f"deep_{depth}.tlfast"
    model = DeepLinearModel(depth)
    inputs = torch.ones(1, 3)

    start = time.perf_counter()
    recording = tl.fastlog.record(
        model,
        inputs,
        save=lambda ctx: ctx.kind == "op",
        streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )
    elapsed = time.perf_counter() - start
    loaded = tl.fastlog.load(bundle_path)

    assert elapsed < ceiling_seconds
    assert len(loaded.records) == len(recording.records)
    assert [record.ctx.label for record in loaded.records] == [
        record.ctx.label for record in recording.records
    ]
    assert loaded.recovery_warnings == []


def test_keep_grad_disk_only_default_raises_at_construction(tmp_path: Path) -> None:
    """Static keep_grad defaults are rejected before any bundle is created."""

    bundle_path = tmp_path / "recording.tlfast"

    with pytest.raises(RecordingConfigError, match="keep_grad=True"):
        tl.fastlog.record(
            TinyModel(),
            torch.ones(1, 3),
            default_op=CaptureSpec(keep_grad=True),
            streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
        )

    assert not bundle_path.exists()


def test_keep_grad_disk_only_predicate_raises_at_runtime(tmp_path: Path) -> None:
    """Runtime keep_grad predicate decisions are rejected before writing a record."""

    bundle_path = tmp_path / "recording.tlfast"

    def keep_first(ctx: RecordContext) -> CaptureSpec | bool:
        """Request keep_grad for the first operation only."""

        return CaptureSpec(keep_grad=True) if ctx.kind == "op" else False

    with pytest.raises(PredicateError, match="disk-only"):
        tl.fastlog.record(
            TinyModel(),
            torch.ones(1, 3),
            save=keep_first,
            streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
        )


def test_keep_grad_integer_dtype_rejected_per_record() -> None:
    """Integer tensors cannot be retained with keep_grad=True."""

    class IntegerModel(nn.Module):
        """Model producing integer output."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Produce an integer tensor."""

            return (x + 1).to(torch.int64)

    with pytest.raises(PredicateError, match="integer or bool"):
        tl.fastlog.record(
            IntegerModel(),
            torch.ones(1),
            default_op=CaptureSpec(keep_grad=True),
        )


def test_crash_recovery_from_partial_bundle(tmp_path: Path) -> None:
    """Recover skips a partial bundle's missing blob and reports a warning."""

    bundle_path = tmp_path / "partial.tlfast"
    tl.fastlog.record(
        TinyModel(),
        torch.ones(1, 3),
        default_op=True,
        streaming=tl.options.StreamingOptions(bundle_path=bundle_path, retain_in_memory=False),
    )
    partial_path = tmp_path / "partial_unfinalized.tlfast"
    bundle_path.rename(partial_path)
    (partial_path / "manifest.json").unlink()
    first_blob = next((partial_path / "blobs").glob("*.safetensors"))
    first_blob.unlink()

    recovered = tl.fastlog.recover(partial_path)

    assert recovered.recovered is True
    assert any("missing blob" in warning for warning in recovered.recovery_warnings)
