"""Tests for trusted roster snapshotting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.intake import (
    IntakeError,
    create_intake_snapshot,
    legacy_requires_fidelity_audit,
    load_intake_snapshot,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write compact fixture JSONL.

    Parameters
    ----------
    path:
        Fixture path.
    rows:
        JSON object rows.
    """

    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_intake_is_idempotent_and_preserves_stable_ids(tmp_path: Path) -> None:
    """Re-running byte-identical intake preserves IDs and creates no new snapshot."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    stable_ids = tmp_path / "stable.jsonl"
    _write_jsonl(
        master,
        [{"name": "TinyNet", "zoo": "fixtures", "variant": "", "recipe": {"code": "bad"}}],
    )
    _write_jsonl(deferred, [{"name": "LaterNet", "zoo": "fixtures", "variant": "small"}])
    _write_jsonl(
        stable_ids,
        [{"name": "TinyNet", "zoo": "fixtures", "variant": "", "stable_id": "m7"}],
    )

    first = create_intake_snapshot(
        master, deferred, tmp_path / "snapshots", stable_ids_path=stable_ids
    )
    second = create_intake_snapshot(
        master, deferred, tmp_path / "snapshots", stable_ids_path=stable_ids
    )

    assert first.created is True
    assert second.created is False
    assert first.snapshot_id == second.snapshot_id
    assert [item.stable_id for item in first.items] == [item.stable_id for item in second.items]
    assert {item.name: item.stable_id for item in first.items}["TinyNet"] == "m7"
    assert load_intake_snapshot(first.root).items == first.items


def test_all_legacy_audit_classes_survive_snapshot_reload(tmp_path: Path) -> None:
    """Classic, faithful, and slop risks remain canonical after a clean reload."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": "ClassicNet", "zoo": "unregistered-classics-pytorch", "variant": ""},
            {
                "name": "FaithfulNet",
                "zoo": "fixtures",
                "variant": "",
                "notes": "faithful verified implementation",
            },
            {
                "name": "SlopNet",
                "zoo": "fixtures",
                "variant": "",
                "flags": ["legacy-known-slop"],
            },
        ],
    )
    _write_jsonl(deferred, [])

    snapshot = create_intake_snapshot(master, deferred, tmp_path / "snapshots")
    loaded = load_intake_snapshot(snapshot.root)
    flags_by_name = {item.name: set(item.preserved_legacy_flags) for item in loaded.items}

    assert "legacy-classic-requires-fidelity-audit" in flags_by_name["ClassicNet"]
    assert "legacy-fidelity-claim" in flags_by_name["FaithfulNet"]
    assert "legacy-slop-requires-fidelity-audit" in flags_by_name["SlopNet"]
    assert all(legacy_requires_fidelity_audit(flags) for flags in flags_by_name.values())


def test_load_rejects_changed_manifest_source_without_reconstruction(tmp_path: Path) -> None:
    """Snapshot loading always verifies declared source bytes, even with no model artifacts."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(master, [{"name": "TinyNet", "zoo": "fixtures", "variant": "base"}])
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "snapshots")
    (snapshot.root / "sources" / "master_catalog.jsonl").write_bytes(b"changed\n")

    with pytest.raises(IntakeError, match="source digest changed"):
        load_intake_snapshot(snapshot.root)


def test_load_rejects_self_consistent_forged_snapshot_id(tmp_path: Path) -> None:
    """A rewritten display ID cannot replace the digest-derived intake identity."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(master, [{"name": "TinyNet", "zoo": "fixtures", "variant": "base"}])
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "snapshots")
    manifest_path = snapshot.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = "intake-forged-display-id"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IntakeError, match="snapshot_id is not derived"):
        load_intake_snapshot(snapshot.root)


def test_load_accepts_renamed_snapshot_directory(tmp_path: Path) -> None:
    """Directory basenames remain non-authoritative for an unchanged snapshot."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(master, [{"name": "TinyNet", "zoo": "fixtures", "variant": "base"}])
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "snapshots")
    renamed = tmp_path / "renamed-intake-directory"
    snapshot.root.rename(renamed)

    loaded = load_intake_snapshot(renamed)

    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.root == renamed
