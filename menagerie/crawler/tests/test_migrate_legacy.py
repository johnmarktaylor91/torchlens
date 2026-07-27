"""Tests for total-authority legacy migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.migrate_legacy import LegacyModuleHint, migrate_legacy


def _write_row(path: Path, row: Mapping[str, object]) -> None:
    """Write one JSONL fixture row.

    Parameters
    ----------
    path:
        Fixture path.
    row:
        Fixture row.
    """

    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_migration_retains_only_hashes_and_risk_flags(tmp_path: Path) -> None:
    """Legacy runs/fidelity/recipe claims become explicitly untrusted hints."""

    row = {
        "name": "ClaimedNet",
        "zoo": "fixtures",
        "variant": "",
        "recipe": {"type": "statement", "code": "model = fake()"},
        "notes": "faithful verified trace runs",
        "flags": ["known-slop"],
        "source_url": None,
    }
    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_row(master, row)
    deferred.write_text("", encoding="utf-8")
    module = tmp_path / "claimed.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "snapshots")

    migrated = migrate_legacy(
        snapshot,
        [row],
        classic_modules=[
            LegacyModuleHint(("ClaimedNet", "fixtures", ""), module, ("classic-audit",))
        ],
    )[0].to_dict()

    assert migrated["workflow_state"] == "UNTRIAGED"
    assert migrated["intake"]["legacy_claims_untrusted"] is True
    assert migrated["intake"]["legacy_recipe_sha256"].startswith("sha256:")
    assert migrated["intake"]["legacy_module_sha256"].startswith("sha256:")
    assert "model = fake()" not in json.dumps(migrated)
    assert migrated["inherited_claims"] == {
        "runs": False,
        "rung": None,
        "source_verified": False,
        "fidelity": None,
        "recipe_accepted": False,
    }
    flags = set(migrated["intake"]["preserved_legacy_flags"])
    assert {"known-slop", "legacy-opaque-recipe", "legacy-fidelity-claim", "classic-audit"} <= flags
