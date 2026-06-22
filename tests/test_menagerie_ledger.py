"""Tests for the menagerie verification ledger."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from menagerie.catalog import CatalogRow
from menagerie.ledger import (
    VerificationRun,
    append_verification_run,
    connect,
    seed_from_legacy,
    verified_count,
)


def _run(**overrides: object) -> VerificationRun:
    """Build a compact verification run fixture.

    Parameters
    ----------
    overrides:
        Field overrides for the default run.

    Returns
    -------
    VerificationRun
        Test verification run.
    """

    data = {
        "stable_id": "m1",
        "recipe_revision_sha256": "recipe-a",
        "name": "ToyNet",
        "zoo": "unit-zoo",
        "variant": "",
        "scope": "forward",
        "status": "passed",
        "forward_pass": 1,
        "backward_pass": None,
        "backward_na_reason": None,
        "metadata_ok": 1,
        "n_ops": 4,
        "graph_shape_hash": "shape-a",
        "svg_sha256": None,
        "torchlens_version": "tl-current",
        "torch_version": "torch-test",
        "python_version": "py-test",
        "device_requested": "cpu",
        "device_actual": "cpu",
        "env_hash": None,
        "runner_host": "unit-host",
        "started_at": "2026-06-22T00:00:00+00:00",
        "finished_at": "2026-06-22T00:00:01+00:00",
        "duration_sec": 1.0,
        "error_class": None,
        "error_message": None,
        "run_id": "run-a",
    }
    data.update(overrides)
    return VerificationRun(**data)  # type: ignore[arg-type]


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    overrides:
        Field overrides for the default row.

    Returns
    -------
    CatalogRow
        Test catalog row.
    """

    data = {
        "model_id": 1,
        "display_index": 1,
        "stable_id": "m1",
        "name": "ToyNet",
        "variant": "",
        "family": "toy",
        "family_normalized": "toy",
        "domain": "vision",
        "zoo": "unit-zoo",
        "constructor_call": "torch.nn.Linear(4, 2)",
        "input_shape": "(1, 4)",
        "input_dtype": "float32",
        "era": "2024",
        "verified": True,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "recipe-a",
    }
    data.update(overrides)
    return CatalogRow(**data)


def test_append_round_trip_and_verified_count(tmp_path: Path) -> None:
    """Appending a full passing current-version row makes the honest count rise."""

    conn = connect(tmp_path / "verification.db")

    run_id = append_verification_run(conn, _run())
    stored = conn.execute(
        "SELECT run_id, stable_id, n_ops, graph_shape_hash FROM verification_runs"
    ).fetchone()

    assert run_id == "run-a"
    assert dict(stored) == {
        "run_id": "run-a",
        "stable_id": "m1",
        "n_ops": 4,
        "graph_shape_hash": "shape-a",
    }
    assert verified_count(conn, "tl-current") == 1


def test_verified_count_rises_after_real_torchlens_trace(tmp_path: Path) -> None:
    """A fresh ledger count rises only after a real validation-like trace row."""

    import torch
    import torchlens as tl

    conn = connect(tmp_path / "verification.db")
    model = torch.nn.Linear(4, 2).eval()
    with torch.no_grad():
        trace = tl.trace(model, torch.ones(1, 4), inference_only=True)
    n_ops = int(getattr(trace, "num_ops", 0) or len(getattr(trace, "layer_logs", {}) or {}))
    graph_shape_hash = str(getattr(trace, "graph_shape_hash", "") or "")

    assert verified_count(conn, "tl-current") == 0

    append_verification_run(
        conn,
        _run(
            run_id="run-real",
            n_ops=n_ops,
            graph_shape_hash=graph_shape_hash,
            torchlens_version="tl-current",
        ),
    )

    assert verified_count(conn, "tl-current") == 1


def test_update_and_delete_are_rejected_by_trigger(tmp_path: Path) -> None:
    """The database rejects mutation of ledger rows."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run())

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE verification_runs SET status='failed'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM verification_runs")


def test_current_verification_returns_latest_forward_per_recipe(tmp_path: Path) -> None:
    """The current view keeps the latest forward row for each stable recipe."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            run_id="run-old",
            status="failed",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            finished_at="2026-06-22T00:00:01+00:00",
        ),
    )
    append_verification_run(
        conn,
        _run(run_id="run-new", finished_at="2026-06-22T00:00:02+00:00"),
    )

    current = conn.execute("SELECT run_id, status FROM current_verification").fetchone()

    assert dict(current) == {"run_id": "run-new", "status": "passed"}


def test_legacy_seed_and_failure_n_ops_null_do_not_count(tmp_path: Path) -> None:
    """Legacy seeds and failure rows are excluded from the current-version count."""

    conn = connect(tmp_path / "verification.db")
    assert seed_from_legacy(conn, [_row(), _row(stable_id="m2", verified=False)]) == 1
    append_verification_run(
        conn,
        _run(
            run_id="run-fail",
            stable_id="m3",
            status="failed",
            forward_pass=0,
            metadata_ok=1,
            n_ops=None,
            graph_shape_hash="shape-fail",
            error_class="RuntimeError",
            error_message="boom",
        ),
    )

    rows = conn.execute("SELECT torchlens_version, n_ops FROM verification_runs").fetchall()

    assert ("legacy-unknown", None) in [(row["torchlens_version"], row["n_ops"]) for row in rows]
    assert verified_count(conn, "tl-current") == 0


def test_failed_run_with_n_ops_is_rejected(tmp_path: Path) -> None:
    """Failure rows cannot smuggle a zero or concrete op count."""

    conn = connect(tmp_path / "verification.db")

    with pytest.raises(ValueError, match="n_ops must be NULL"):
        append_verification_run(
            conn,
            _run(status="failed", forward_pass=0, n_ops=0, graph_shape_hash=None),
        )


def test_concurrent_appends_from_two_threads_both_land(tmp_path: Path) -> None:
    """WAL and busy timeout allow concurrent ledger appends."""

    db_path = tmp_path / "verification.db"
    connect(db_path).close()

    def append_one(index: int) -> str:
        """Append one run in a separate connection.

        Parameters
        ----------
        index:
            Unique run index.

        Returns
        -------
        str
            Inserted run ID.
        """

        conn = connect(db_path)
        return append_verification_run(
            conn,
            _run(
                run_id=f"run-{index}",
                stable_id=f"m{index}",
                finished_at=f"2026-06-22T00:00:0{index}+00:00",
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        run_ids = sorted(executor.map(append_one, [1, 2]))

    conn = connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0]

    assert run_ids == ["run-1", "run-2"]
    assert count == 2
