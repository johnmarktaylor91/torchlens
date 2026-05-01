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


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("manifest", "spec", "expected"),
    [
        ({"tlspec_version": 1, "kind": "intervention"}, None, "v2.0_unified"),
        ({"kind": "intervention", "format_version": "1"}, None, "v2.16_intervention_with_kind"),
        ({"format_version": "1"}, {"format_version": "1"}, "v2.16_intervention"),
        ({"io_format_version": 2}, None, "v2.16_modellog_portable"),
        ({}, None, "unknown"),
    ],
)
def test_detect_tlspec_format_ordering(
    tmp_path: Path,
    manifest: dict[str, Any],
    spec: dict[str, Any] | None,
    expected: str,
) -> None:
    """Detection should follow Phase 11.0's first-match-wins ordering."""

    tlspec_path = tmp_path / "sample.tlspec"
    tlspec_path.mkdir()
    _write_json(tlspec_path / "manifest.json", manifest)
    if spec is not None:
        _write_json(tlspec_path / "spec.json", spec)

    assert detect_tlspec_format(tlspec_path) == expected
