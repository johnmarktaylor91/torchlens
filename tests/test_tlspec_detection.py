"""Tests for TorchLens ``.tlspec`` format detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from torchlens.io import detect_tlspec_format


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON object to one path.

    Parameters
    ----------
    path:
        Destination path.
    data:
        JSON-serializable object.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("manifest", "spec", "expected"),
    [
        ({"format": "torchlens.saved"}, None, "unknown"),
        ({"tlspec_version": 1, "kind": "intervention"}, None, "pre_3"),
        ({"kind": "intervention", "format_version": "1"}, {"format_version": "1"}, "pre_3"),
        # R73: losing either discriminator must still refuse as 2.x, not dispatch
        # into an intervention reader or escape through a missing spec.json.
        ({"kind": "intervention", "format_version": "1"}, None, "pre_3"),
        ({"format_version": "1"}, {"format_version": "1"}, "pre_3"),
        ({"tlspec_version": 2}, None, "pre_3"),
        ({"io_format_version": 2}, None, "pre_3"),
        ({"n_activation_blobs": 1}, None, "pre_3"),
        ({"torchlens_version": "2.36.0"}, None, "pre_3"),
        ({}, None, "unknown"),
        ({"application": "unrelated"}, None, "unknown"),
    ],
)
def test_detect_tlspec_format_ordering(
    tmp_path: Path,
    manifest: dict[str, Any],
    spec: dict[str, Any] | None,
    expected: str,
) -> None:
    """Detection distinguishes the current format, 2.x markers, and unrelated JSON."""

    tlspec_path = tmp_path / "sample.tlspec"
    tlspec_path.mkdir()
    _write_json(tlspec_path / "manifest.json", manifest)
    if spec is not None:
        _write_json(tlspec_path / "spec.json", spec)

    assert detect_tlspec_format(tlspec_path) == expected


@pytest.mark.parametrize("marker", ["metadata.pkl", "bundle.json", "spec.json"])
def test_detect_root_only_2x_markers(tmp_path: Path, marker: str) -> None:
    """Recognize a 2.x root file without opening pickle content.

    Parameters
    ----------
    tmp_path:
        Temporary artifact directory.
    marker:
        Root member written by a 2.x artifact family.
    """

    (tmp_path / marker).write_text(
        json.dumps({"format_version": "1"}) if marker == "spec.json" else "not decoded",
        encoding="utf-8",
    )
    assert detect_tlspec_format(tmp_path) == "pre_3"


def test_detect_empty_directory(tmp_path: Path) -> None:
    """An empty directory has no artifact markers.

    Parameters
    ----------
    tmp_path:
        Empty temporary directory.
    """

    assert detect_tlspec_format(tmp_path) == "unknown"


def test_detect_current_save(tmp_path: Path) -> None:
    """A real current save detects as v3, even alongside a 2.x marker.

    Parameters
    ----------
    tmp_path:
        Parent of the saved artifact.
    """

    from _format3_envelope import save_tiny

    path = tmp_path / "current.tlspec"
    trace = save_tiny(path)
    assert detect_tlspec_format(path) == "v3"
    (path / "metadata.pkl").write_bytes(b"not opened")
    assert detect_tlspec_format(path) == "v3"
    trace.cleanup()


@pytest.mark.parametrize("marker", ["metadata.pkl", "bundle.json", "spec.json", "manifest.json"])
def test_pre3_marker_refuses_at_load_and_validation(tmp_path: Path, marker: str) -> None:
    """Both public doors give the same marker-specific 2.x refusal.

    Parameters
    ----------
    tmp_path:
        Temporary artifact root.
    marker:
        Root marker identifying a 2.x artifact.
    """

    import torchlens as tl
    from torchlens.errors import ArtifactTooOldError
    from torchlens.validation import validate_tlspec

    content = {"format_version": "1"} if marker == "spec.json" else {"kind": "trace"}
    (tmp_path / marker).write_text(json.dumps(content), encoding="utf-8")
    errors = []
    for door in (tl.load, validate_tlspec):
        with pytest.raises(ArtifactTooOldError) as excinfo:
            door(tmp_path)
        errors.append(excinfo.value)
    assert errors[0].fields == errors[1].fields
    assert errors[0].fields["code"] == "artifact_version_below_floor"
    assert errors[0].fields["path"] == str(tmp_path)
    assert errors[0].fields["observed"]


@pytest.mark.parametrize("marker", ["metadata.pkl", "bundle.json", "spec.json"])
def test_detect_rejects_symlinked_2x_markers(tmp_path: Path, marker: str) -> None:
    """A 2.x marker cannot cause detection to follow a metadata symlink.

    Parameters
    ----------
    tmp_path:
        Temporary parent of the artifact and external file.
    marker:
        Metadata member to replace with a symlink.
    """

    from torchlens.errors import ArtifactError

    outside = tmp_path / "outside.json"
    outside.write_text('{"format_version": "1"}', encoding="utf-8")
    root = tmp_path / "artifact"
    root.mkdir()
    (root / marker).symlink_to(outside)
    with pytest.raises(ArtifactError) as excinfo:
        detect_tlspec_format(root)
    assert excinfo.value.fields["code"] == "load_path_symlink_rejected"
