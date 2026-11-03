"""A file an earlier major release saved refuses at ``tl.load`` with one clear error.

A major release carries no reader for the field names and package paths an earlier
major release wrote, so the load door refuses such a file up front: the message names
the saving release and the two ways out (the earlier release reads it, or capture again);
a file torchlens 2.x saved takes the one 2.x refusal every other 2.x door raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ArtifactTooOldError


def _saved_trace(tmp_path: Path) -> tuple[Path, str]:
    """Save a small trace and return its path and the release stamped in its manifest."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Sequential(nn.Linear(3, 2), nn.ReLU()), torch.randn(1, 3))
    path = tmp_path / "trace.tlspec"
    trace.save(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return path, str(manifest["torchlens_version"])


def _restamp(path: Path, saved: str) -> None:
    """Rewrite the release stamp in a saved trace's root manifest."""

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["torchlens_version"] = saved
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_file_saved_by_torchlens_2_refuses_at_load_under_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file stamped by torchlens 2.x takes the one 2.x refusal under torchlens 3."""

    path, _ = _saved_trace(tmp_path)
    _restamp(path, "2.36.3")
    monkeypatch.setattr(tl, "__version__", "3.0.0")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)

    message = str(excinfo.value)
    assert "saved by torchlens 2.x" in message
    assert "load it with torchlens 2.36.x" in message
    assert "capture the model again" in message
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"] == "torchlens_version=2.36.3"
    assert excinfo.value.fields["path"] == str(path)


def test_file_saved_by_an_earlier_major_release_refuses_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A major older than 2 refuses with the same error class, naming the saver."""

    path, _ = _saved_trace(tmp_path)
    _restamp(path, "1.4.0")
    monkeypatch.setattr(tl, "__version__", "3.0.0")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)

    message = str(excinfo.value)
    assert "saved by torchlens 1.4.0" in message
    assert "torchlens<3" in message
    assert "capture the model again" in message
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"] == "torchlens_version=1.4.0"
    assert excinfo.value.fields["path"] == str(path)
