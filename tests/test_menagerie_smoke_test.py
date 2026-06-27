"""Tests for menagerie smoke-test orchestration helpers."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from menagerie import ledger, smoke_gate, smoke_test
from menagerie.catalog import CatalogRow, write_catalog
from menagerie.cluster_runner import GIANT_REGISTRY


def _verification_run(stable_id: str) -> ledger.VerificationRun:
    """Build one minimal passing verification run for an isolated ledger.

    Parameters
    ----------
    stable_id:
        Stable ID for the synthetic row.

    Returns
    -------
    ledger.VerificationRun
        A passing forward-scope run.
    """

    return ledger.VerificationRun(
        stable_id=stable_id,
        recipe_revision_sha256="recipe",
        name="n",
        zoo="z",
        variant="",
        scope="forward",
        status="passed",
        forward_pass=1,
        backward_pass=None,
        backward_na_reason=None,
        metadata_ok=1,
        n_ops=3,
        graph_shape_hash="shape",
        svg_sha256=None,
        torchlens_version="2",
        torch_version="2",
        python_version="3",
        device_requested="cpu",
        device_actual="cpu",
        env_hash="env",
        lock_hash="lock",
        torchlens_source_hash="src",
        input_scale=1.0,
        runner_host="host",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_sec=1.0,
    )


def _seed_production_ledger(db_path: Path, stable_ids: list[str]) -> None:
    """Create an isolated production-like ledger with the given rows.

    Parameters
    ----------
    db_path:
        Destination ledger path.
    stable_ids:
        Stable IDs to append.
    """

    conn = ledger.connect(db_path)
    try:
        for stable_id in stable_ids:
            ledger.append_verification_run(conn, _verification_run(stable_id))
    finally:
        conn.close()


def _write_production_snapshot(out_dir: Path, production_db: Path, smoke_start: str) -> None:
    """Write a ``production_snapshot_before.json`` like the smoke producer.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    production_db:
        Production verification ledger to fingerprint.
    smoke_start:
        Smoke start timestamp recorded in the snapshot.
    """

    payload = {
        "smoke_start": smoke_start,
        "stable_ids": [],
        "verification_db": smoke_test.snapshot_verification_content(production_db),
    }
    (out_dir / "production_snapshot_before.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _case(
    stable_id: str, include: str = "default", synthetic: bool = False
) -> smoke_test.SmokeCase:
    """Build one smoke case fixture.

    Parameters
    ----------
    stable_id:
        Stable ID.
    include:
        Inclusion group.
    synthetic:
        Whether the row is synthetic.

    Returns
    -------
    smoke_test.SmokeCase
        Smoke case.
    """

    return smoke_test.SmokeCase(
        {
            "stable_id": stable_id,
            "include": include,
            "synthetic": synthetic,
            "expected_env": "base",
            "expected_status": "validated",
        }
    )


def _row() -> CatalogRow:
    """Build a compact catalog row fixture."""

    return CatalogRow(
        model_id=1,
        display_index=1,
        stable_id="m1",
        name="UnitNet",
        variant="",
        family="unit",
        family_normalized="unit",
        domain="unit",
        zoo="unit",
        constructor_call="torch.nn.Identity()",
        input_shape="(1,)",
        input_dtype="float32",
        era="2026",
        verified=True,
        notes="",
        source="test",
        recipe_revision_sha256="recipe-a",
    )


def test_select_cases_respects_optional_groups() -> None:
    """Case selection honors all-islands, heavy, and no-cluster flags."""

    cases = [
        _case("m1"),
        _case("m2", "all_islands"),
        _case("m3", "heavy"),
        _case("m4", "cluster"),
    ]

    default = smoke_test.select_cases(
        cases, all_islands=False, with_heavy_giant=False, no_cluster=False
    )
    extended = smoke_test.select_cases(
        cases, all_islands=True, with_heavy_giant=True, no_cluster=True
    )

    assert [case.stable_id for case in default] == ["m1", "m4"]
    assert [case.stable_id for case in extended] == ["m1", "m2", "m3"]


def test_select_cases_drops_cluster_runner_under_no_cluster() -> None:
    """--no-cluster drops any case that expects a remote cluster runner.

    A genuine force_cluster giant is tagged include="heavy" (opt-in via
    --with-heavy-giant) and expected_runner="cluster". Under --no-cluster the
    run uses --runner local, so such a case must be dropped rather than routed
    local against a remote expectation.
    """

    forced_id = next(sid for sid, entry in GIANT_REGISTRY.items() if entry.force_cluster)
    forced = smoke_test.SmokeCase(
        {
            "stable_id": forced_id,
            "include": "heavy",
            "synthetic": False,
            "expected_env": "base",
            "expected_runner": "cluster",
            "expected_status": "validated",
        }
    )
    cases = [_case("m1"), forced]

    with_giant = smoke_test.select_cases(
        cases, all_islands=False, with_heavy_giant=True, no_cluster=False
    )
    no_cluster = smoke_test.select_cases(
        cases, all_islands=False, with_heavy_giant=True, no_cluster=True
    )

    assert [case.stable_id for case in with_giant] == ["m1", forced_id]
    assert [case.stable_id for case in no_cluster] == ["m1"]


def test_smoke_manifest_cluster_cases_are_force_cluster_giants() -> None:
    """Every committed cluster-runner expectation follows the true routing policy.

    A model is expected remote IFF it genuinely force-routes to the shared
    cluster (force_cluster=True). A stale cluster tag on a model that fits
    locally would demand a forbidden remote dispatch.
    """

    cases = smoke_test.load_cases(smoke_test.DEFAULT_SMOKE_MANIFEST)
    cluster_ids = [
        case.stable_id for case in cases if case.payload.get("expected_runner") == "cluster"
    ]
    assert cluster_ids, "smoke manifest must retain at least one forced-cluster case"
    for stable_id in cluster_ids:
        entry = GIANT_REGISTRY.get(stable_id)
        assert entry is not None, f"{stable_id} missing from GIANT_REGISTRY"
        assert entry.force_cluster, f"{stable_id} expected remote but force_cluster=False"


def test_smoke_gate_rejects_stale_cluster_expectation(tmp_path: Path) -> None:
    """The gate fails loudly on a cluster expectation for a local-routing model."""

    fitting_giant = next(sid for sid, entry in GIANT_REGISTRY.items() if not entry.force_cluster)
    cases = [{"stable_id": fitting_giant, "expected_runner": "cluster"}]
    with pytest.raises(RuntimeError, match="stale cluster expectation"):
        smoke_gate._assert_cluster(cases, tmp_path)  # noqa: SLF001


def test_insert_synthetic_rows_adds_smoke_catalog_rows(tmp_path: Path) -> None:
    """Synthetic smoke rows are injected only into the copied catalog DB."""

    catalog_db = tmp_path / "catalog.db"
    write_catalog([_row()], canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)
    smoke_test._insert_synthetic_rows(  # noqa: SLF001
        catalog_db,
        [_case("smoke_exc_1", synthetic=True)],
    )

    with sqlite3.connect(catalog_db) as connection:
        row = connection.execute(
            "SELECT stable_id, name, source FROM models WHERE stable_id = 'smoke_exc_1'"
        ).fetchone()

    assert tuple(row) == ("smoke_exc_1", "smoke_exc_1", "smoke")


def test_smoke_manifest_jsonl_is_valid() -> None:
    """The committed smoke manifest parses as JSONL."""

    cases = smoke_test.load_cases(smoke_test.DEFAULT_SMOKE_MANIFEST)

    assert cases
    assert len({case.stable_id for case in cases}) == len(cases)
    assert all(json.dumps(case.payload) for case in cases)


def test_snapshot_verification_content_fields(tmp_path: Path) -> None:
    """The content fingerprint reports count, max rowid, and a row hash."""

    db = tmp_path / "verification.db"
    _seed_production_ledger(db, ["a", "b"])

    fingerprint = smoke_test.snapshot_verification_content(db)

    assert fingerprint["exists"] is True
    assert fingerprint["runs_count"] == 2
    assert fingerprint["max_rowid"] == 2
    assert isinstance(fingerprint["content_sha256"], str)


def test_snapshot_verification_content_missing(tmp_path: Path) -> None:
    """An absent ledger yields a null content fingerprint."""

    fingerprint = smoke_test.snapshot_verification_content(tmp_path / "absent.db")

    assert fingerprint["exists"] is False
    assert fingerprint["runs_count"] is None
    assert fingerprint["max_rowid"] is None
    assert fingerprint["content_sha256"] is None


def test_production_unchanged_passes_under_wal_churn(tmp_path: Path) -> None:
    """A WAL checkpoint + mtime bump (identical content) must NOT trip the gate.

    This is the FALSE-POSITIVE regression: the smoke only READS the production
    ledger, but a SQLite WAL checkpoint rewrites the file bytes (and mtime/size)
    with no logical change. The byte fingerprint flagged that as "changed"; the
    content fingerprint must pass.
    """

    production_db = tmp_path / "verification.db"
    _seed_production_ledger(production_db, ["a", "b"])
    out_dir = tmp_path / "smoke_out"
    out_dir.mkdir()
    _write_production_snapshot(out_dir, production_db, smoke_start="2026-06-01T00:00:00+00:00")

    # Perturb the FILE bytes without changing content: full WAL checkpoint
    # (rewrites/truncates the file) plus an explicit mtime bump.
    with sqlite3.connect(production_db) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("SELECT COUNT(*) FROM verification_runs").fetchone()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    stat = production_db.stat()
    os.utime(production_db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    # Must pass: no logical row change occurred.
    smoke_gate._assert_production_unchanged([], out_dir, production_db)  # noqa: SLF001


def test_production_unchanged_fails_on_injected_row(tmp_path: Path) -> None:
    """A fresh production row (real pollution) MUST trip the gate."""

    production_db = tmp_path / "verification.db"
    _seed_production_ledger(production_db, ["a", "b"])
    out_dir = tmp_path / "smoke_out"
    out_dir.mkdir()
    _write_production_snapshot(out_dir, production_db, smoke_start="2026-06-01T00:00:00+00:00")

    # Simulate pollution: a smoke run leaked a row into the production ledger.
    _seed_production_ledger(production_db, ["smoke_leak"])

    with pytest.raises(RuntimeError, match="production verification.db changed"):
        smoke_gate._assert_production_unchanged([], out_dir, production_db)  # noqa: SLF001


def test_production_unchanged_fails_on_fresh_smoke_stable_id(tmp_path: Path) -> None:
    """The independent fresh-smoke-rows witness still catches a stamped leak.

    Even if a leak somehow matched count/rowid (it cannot, given append-only),
    the direct query over smoke stable IDs stamped at/after the smoke start
    remains a second, independent tripwire. Here we make the content fingerprint
    match by snapshotting AFTER the leak, and confirm the row query still fails.
    """

    production_db = tmp_path / "verification.db"
    _seed_production_ledger(production_db, ["a", "b"])
    # Leak a smoke row, THEN snapshot -> content fingerprint matches current.
    _seed_production_ledger(production_db, ["smoke_case_1"])
    out_dir = tmp_path / "smoke_out"
    out_dir.mkdir()
    _write_production_snapshot(out_dir, production_db, smoke_start="2026-01-01T00:00:00+00:00")

    cases = [{"stable_id": "smoke_case_1", "expected_status": "validated"}]
    with pytest.raises(RuntimeError, match="fresh smoke rows"):
        smoke_gate._assert_production_unchanged(cases, out_dir, production_db)  # noqa: SLF001
