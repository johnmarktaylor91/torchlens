"""Runnable payloads of a 3.0 artifact bind on load and refuse when the chunk is tampered."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import read_manifest
from torch import nn

import torchlens as tl
from torchlens._io._format3_runnable_bind import (
    read_runnable_payload,
    runnable_labels,
    runnable_payload_index,
)
from torchlens._io.format_errors import ArtifactError

pytestmark = pytest.mark.heavy


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x))) * self.scale


def _save(root: Path, **save_kwargs: Any) -> tuple[nn.Module, Path]:
    torch.manual_seed(0)
    model = _Tiny().eval()
    trace = tl.trace(model, torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True))
    try:
        trace.save(root, level="runnable", **save_kwargs)
    finally:
        trace.cleanup()
    return model, root


def test_weights_bind_to_their_state_names_after_load(tmp_path: Path) -> None:
    """Every ``runnable_weight_`` blob resolves to a sorted state name and its stored tensor."""

    model, root = _save(tmp_path / "weights.tlspec", include_weights=True)
    loaded = tl.load(root)
    descriptor = loaded._runnable.descriptor
    manifest = read_manifest(root)
    index = runnable_payload_index(manifest, descriptor)
    weights = {ref.label: ref for ref in index.tensors if ref.kind == "runnable_weight"}
    expected = model.state_dict()
    assert set(weights) == set(expected)
    for name, ref in weights.items():
        tensor = read_runnable_payload(root / ref.relative_path, ref, "cpu")
        assert torch.equal(tensor, expected[name]), name
    labels = runnable_labels(descriptor)
    assert all(
        blob_id in labels for blob_id in manifest["tensors"] if blob_id.startswith("runnable_")
    )


def test_tampered_runnable_chunk_refuses_at_load(tmp_path: Path) -> None:
    """A flipped byte in a runnable chunk refuses with the integrity code before any binding."""

    _model, root = _save(tmp_path / "tamper.tlspec", include_weights=True)
    manifest = read_manifest(root)
    chunk = next(
        root / manifest["tensors"][blob_id]["blob"]
        for blob_id in sorted(manifest["tensors"])
        if blob_id.startswith("runnable_weight_")
    )
    data = bytearray(chunk.read_bytes())
    data[-1] ^= 0xFF
    chunk.write_bytes(bytes(data))
    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    assert info.value.fields["code"] == "bundle_metadata_integrity_refused"


def test_descriptor_declares_every_runnable_payload(tmp_path: Path) -> None:
    """The run descriptor restored on load names a slot for every runnable blob the manifest lists."""

    _model, root = _save(tmp_path / "declared.tlspec", include_weights=True)
    loaded = tl.load(root)
    assert loaded._runnable is not None and loaded._runnable.descriptor is not None
    manifest = read_manifest(root)
    runnable_ids = [blob_id for blob_id in manifest["tensors"] if blob_id.startswith("runnable_")]
    assert runnable_ids
    assert set(runnable_ids) <= set(runnable_labels(loaded._runnable.descriptor))
