"""Tests for trusted roster snapshotting."""

from __future__ import annotations

import json
from pathlib import Path

from menagerie.crawler.intake import create_intake_snapshot, load_intake_snapshot


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
