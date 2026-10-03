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
        (
            {"kind": "intervention", "format_version": "1"},
            {"format_version": "1"},
            "v2.16_intervention_with_kind",
        ),
        # A kind-bearing manifest WITHOUT spec.json is not an intervention
        # artifact (every v2.16 intervention save writes spec.json); inferring
        # it from `kind` alone misrouted a unified manifest with a deleted
        # tlspec_version into the intervention loader, which died on the
        # absent spec.json untyped (R73).
        ({"kind": "intervention", "format_version": "1"}, None, "unknown"),
        ({"format_version": "1"}, {"format_version": "1"}, "v2.16_intervention"),
        # `tlspec_version` is itself a tlspec-schema-only marker: the genuine
        # pre-tlspec v2.16 ModelLog format never carried it (it used
        # `io_format_version`/`n_activation_blobs` instead, covered below). A
        # manifest with `tlspec_version` but no `kind` is an older-but-still-
        # modern unified manifest (or one whose `kind` was lost), not the
        # legacy release -- it must route through v2.0_unified so the real
        # numeric tlspec_version floor check applies, never the unrelated
        # legacy-format refusal (R73 fast-tier fuzz finding, 2026-10: this
        # exact shape -- a fresh manifest with `kind` deleted -- misclassified
        # as the legacy format and raised a false "pre-tlspec ModelLog
        # format" ArtifactVersionBelowFloorError).
        ({"tlspec_version": 2}, None, "v2.0_unified"),
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
