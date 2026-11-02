"""A file an earlier major release saved refuses at ``tl.load`` with one clear error.

A major release carries no reader for the field names and package paths an earlier
major release wrote, so the load door refuses such a file up front: the message names
the saving release and the two ways out (the earlier release reads it, or capture again).
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


def test_file_saved_by_an_earlier_major_release_refuses_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading under the next major release raises one typed error naming the saver."""

    path, saved = _saved_trace(tmp_path)
    next_major = int(saved.split(".", 1)[0]) + 1
    monkeypatch.setattr(tl, "__version__", f"{next_major}.0.0")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)

    message = str(excinfo.value)
    assert f"saved by torchlens {saved}" in message
    assert f"torchlens<{next_major}" in message
    assert "capture the model again" in message
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"] == f"torchlens_version={saved}"
    assert excinfo.value.fields["path"] == str(path)


def test_file_saved_by_the_running_major_release_loads(tmp_path: Path) -> None:
    """The same file loads under the release line that saved it."""

    path, _ = _saved_trace(tmp_path)

    loaded = tl.load(path)

    assert type(loaded).__name__ == "Trace"
