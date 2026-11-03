"""Smoke tests for fastlog RAM, disk, and recovery storage paths."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


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


def test_disk_only_roundtrip_loads_bundle(tmp_path: Path) -> None:
    """Disk-only recording writes a fastlog bundle that load can read."""

    bundle_path = tmp_path / "recording.tlfast"
    recording = tl.record(
        TinyModel(),
        torch.ones(1, 3),
        storage=tl.options.StorageOptions(path=bundle_path, retain_in_memory=False),
        recording=tl.options.RecordingOptions(default_op=True),
    )
    loaded = tl.load(bundle_path)

    assert recording.path == bundle_path
    assert (bundle_path / "manifest.json").exists()
    assert (bundle_path / "metadata.json").exists()
    assert loaded.is_recovered is False
    assert len(loaded) == len(recording)
    assert loaded.n_records == len(recording)
    assert loaded.records[0].metadata["blob_id"] == recording.records[0].metadata["blob_id"]
