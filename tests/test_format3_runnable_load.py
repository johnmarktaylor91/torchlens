"""The runnable component (contract section 7) refuses at load when a family file or the
descriptor under ``runnable/`` is tampered: the file index digests them, so the integrity check
fires before any binding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import read_manifest
from torch import nn

import torchlens as tl
from torchlens._io.format_errors import ArtifactError

pytestmark = pytest.mark.heavy

_INTEGRITY = "bundle_metadata_integrity_refused"


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


def _runnable_files(root: Path) -> list[str]:
    manifest = read_manifest(root)
    listed = sorted(path for path in manifest["files"] if path.startswith("runnable/"))
    assert listed, "a runnable save lists its component files in the manifest"
    assert "runnable/descriptor.json" in listed
    return listed


def _flip_last_byte(target: Path) -> None:
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))


def test_runnable_component_lists_its_files_and_loads(tmp_path: Path) -> None:
    """An untouched runnable save lists ``runnable/`` in the file index and loads."""

    _model, root = _save(tmp_path / "clean.tlspec", include_weights=True)
    listed = _runnable_files(root)
    assert any(path.endswith(".safetensors") for path in listed), listed
    loaded = tl.load(root)
    assert loaded._runnable is not None and loaded._runnable.descriptor is not None


def test_tampered_runnable_family_refuses_at_load(tmp_path: Path) -> None:
    """A flipped byte in a family file refuses with the integrity code before any binding."""

    _model, root = _save(tmp_path / "tamper.tlspec", include_weights=True)
    family = next(path for path in _runnable_files(root) if path.endswith(".safetensors"))
    _flip_last_byte(root / family)
    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    assert info.value.fields["code"] == _INTEGRITY


def test_tampered_runnable_descriptor_refuses_at_load(tmp_path: Path) -> None:
    """An edited ``runnable/descriptor.json`` refuses with the integrity code."""

    _model, root = _save(tmp_path / "descriptor.tlspec", include_weights=True)
    target = root / "runnable" / "descriptor.json"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ArtifactError) as info:
        tl.load(root)
    assert info.value.fields["code"] == _INTEGRITY
