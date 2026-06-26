"""Tests for the menagerie verification ledger."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from menagerie.catalog import CatalogRow
from menagerie.ledger import (
    CASCADE_ARTIFACT_ERROR_CLASS,
    ENV_VERIFICATION_DB,
    LEGACY_UNKNOWN,
    VerificationRun,
    VerificationTarget,
    append_verification_run,
    base_env_hash,
    base_lock_hash,
    cascade_suppressed_count,
    cascade_suppressed_stable_ids,
    connect,
    seed_from_legacy,
    torchlens_source_hash,
    verified_count,
)


def _cascade_artifact(**overrides: object) -> VerificationRun:
    """Build a synthesized pre-fix batch-cascade artifact row.

    Mirrors the exact signature of the 290 job-6014456 rows: status=failed,
    ``CASCADE_ARTIFACT_ERROR_CLASS``, zero duration, started==finished, NULL
    n_ops/graph_shape_hash, written on a non-cluster orchestrator host.

    Parameters
    ----------
    overrides:
        Field overrides for the default artifact.

    Returns
    -------
    VerificationRun
        Cascade-artifact verification run.
    """

    data: dict[str, object] = {
        "status": "failed",
        "forward_pass": None,
        "metadata_ok": None,
        "n_ops": None,
        "graph_shape_hash": None,
        "svg_sha256": None,
        "duration_sec": 0.0,
        "started_at": "2026-06-26T08:37:54+00:00",
        "finished_at": "2026-06-26T08:37:54+00:00",
        "runner_host": "zmachine",
        "error_class": CASCADE_ARTIFACT_ERROR_CLASS,
        "error_message": (
            "cluster job failed; sbatch --wait returncode=1; Submitted batch job 6014456"
        ),
    }
    data.update(overrides)
    return _run(**data)


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
        "env_hash": base_env_hash(),
        "lock_hash": base_lock_hash(),
        "torchlens_source_hash": torchlens_source_hash(),
        "input_scale": 1.0,
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


def test_connect_resolves_verification_db_env_at_connect_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default ledger connection honors the env override lazily."""

    ledger_db = tmp_path / "smoke" / "verification.db"
    monkeypatch.setenv(ENV_VERIFICATION_DB, str(ledger_db))

    with connect() as conn:
        append_verification_run(conn, _run())

    assert ledger_db.exists()
    with sqlite3.connect(ledger_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0]
    assert count == 1


def test_append_round_trip_and_verified_count(tmp_path: Path) -> None:
    """Appending a full passing current-version row makes the honest count rise."""

    conn = connect(tmp_path / "verification.db")

    run_id = append_verification_run(conn, _run())
    stored = conn.execute(
        """
        SELECT run_id, stable_id, n_ops, graph_shape_hash, lock_hash,
               torchlens_source_hash, input_scale
        FROM verification_runs
        """
    ).fetchone()

    assert run_id == "run-a"
    assert dict(stored) == {
        "run_id": "run-a",
        "stable_id": "m1",
        "n_ops": 4,
        "graph_shape_hash": "shape-a",
        "lock_hash": base_lock_hash(),
        "torchlens_source_hash": torchlens_source_hash(),
        "input_scale": 1.0,
    }
    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}) == 1


def test_identity_columns_migrate_existing_ledger(tmp_path: Path) -> None:
    """Legacy ledgers gain identity columns with non-current sentinels."""

    db_path = tmp_path / "verification.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE verification_runs(
            run_id TEXT PRIMARY KEY,
            stable_id TEXT NOT NULL,
            recipe_revision_sha256 TEXT NOT NULL,
            name TEXT NOT NULL,
            zoo TEXT NOT NULL,
            variant TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL CHECK(scope IN ('forward','backward')),
            status TEXT NOT NULL CHECK(status IN ('passed','failed')),
            forward_pass INTEGER,
            backward_pass INTEGER,
            backward_na_reason TEXT,
            metadata_ok INTEGER,
            n_ops INTEGER,
            graph_shape_hash TEXT,
            svg_sha256 TEXT,
            torchlens_version TEXT NOT NULL,
            torch_version TEXT NOT NULL,
            python_version TEXT NOT NULL,
            device_requested TEXT NOT NULL,
            device_actual TEXT,
            env_hash TEXT,
            runner_host TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            error_class TEXT,
            error_message TEXT
        );
        INSERT INTO verification_runs(
            run_id, stable_id, recipe_revision_sha256, name, zoo, scope, status,
            forward_pass, metadata_ok, n_ops, graph_shape_hash, torchlens_version,
            torch_version, python_version, device_requested, started_at, finished_at,
            duration_sec
        )
        VALUES (
            'legacy-run', 'm1', 'recipe-a', 'ToyNet', 'unit-zoo', 'forward', 'passed',
            1, 1, 4, 'shape-a', 'tl-current', 'torch-test', 'py-test', 'cpu',
            '2026-06-22T00:00:00+00:00', '2026-06-22T00:00:01+00:00', 1.0
        );
        """
    )
    conn.close()

    migrated = connect(db_path)
    row = migrated.execute(
        """
        SELECT lock_hash, torchlens_source_hash, input_scale
        FROM verification_runs
        WHERE run_id = 'legacy-run'
        """
    ).fetchone()

    assert dict(row) == {
        "lock_hash": "legacy-unknown",
        "torchlens_source_hash": "legacy-unknown",
        "input_scale": None,
    }


def test_status_constraint_migration_preserves_identity_columns(tmp_path: Path) -> None:
    """Re-running the status migration preserves real identity tuple values."""

    db_path = tmp_path / "verification.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE verification_runs(
            run_id TEXT PRIMARY KEY,
            stable_id TEXT NOT NULL,
            recipe_revision_sha256 TEXT NOT NULL,
            name TEXT NOT NULL,
            zoo TEXT NOT NULL,
            variant TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL CHECK(scope IN ('forward','backward')),
            status TEXT NOT NULL CHECK(status IN ('passed','failed')),
            forward_pass INTEGER,
            backward_pass INTEGER,
            backward_na_reason TEXT,
            metadata_ok INTEGER,
            n_ops INTEGER,
            graph_shape_hash TEXT,
            svg_sha256 TEXT,
            torchlens_version TEXT NOT NULL,
            torch_version TEXT NOT NULL,
            python_version TEXT NOT NULL,
            device_requested TEXT NOT NULL,
            device_actual TEXT,
            env_hash TEXT,
            lock_hash TEXT NOT NULL DEFAULT 'legacy-unknown',
            torchlens_source_hash TEXT NOT NULL DEFAULT 'legacy-unknown',
            input_scale REAL,
            runner_host TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            peak_rss_mb INTEGER,
            error_class TEXT,
            error_message TEXT
        );
        INSERT INTO verification_runs(
            run_id, stable_id, recipe_revision_sha256, name, zoo, scope, status,
            forward_pass, metadata_ok, n_ops, graph_shape_hash, torchlens_version,
            torch_version, python_version, device_requested, env_hash, lock_hash,
            torchlens_source_hash, input_scale, started_at, finished_at, duration_sec
        )
        VALUES (
            'real-identity-run', 'm1', 'recipe-a', 'ToyNet', 'unit-zoo', 'forward', 'passed',
            1, 1, 4, 'shape-a', 'tl-current', 'torch-test', 'py-test', 'cpu',
            'real-env', 'real-lock', 'real-source', 0.00001,
            '2026-06-22T00:00:00+00:00', '2026-06-22T00:00:01+00:00', 1.0
        );
        """
    )
    conn.close()

    migrated = connect(db_path)
    row = migrated.execute(
        """
        SELECT env_hash, lock_hash, torchlens_source_hash, input_scale
        FROM verification_runs
        WHERE run_id = 'real-identity-run'
        """
    ).fetchone()

    assert dict(row) == {
        "env_hash": "real-env",
        "lock_hash": "real-lock",
        "torchlens_source_hash": "real-source",
        "input_scale": 0.00001,
    }


def test_new_terminal_statuses_round_trip(tmp_path: Path) -> None:
    """Island terminal statuses are accepted and stored by the ledger."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            run_id="run-install-failed",
            status="install_failed",
            forward_pass=None,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            error_class="InstallError",
            error_message="install failed",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="run-env-unavailable",
            stable_id="m2",
            status="env_unavailable",
            forward_pass=None,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            error_class="EnvUnavailable",
            error_message="disk floor",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="run-oom",
            stable_id="m3",
            status="oom",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            error_class="failed:oom",
            error_message="oom-kill",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="run-native-crash",
            stable_id="m4",
            status="native_crash",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            error_class="failed:native_crash",
            error_message="segfault",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="run-killed",
            stable_id="m5",
            status="killed",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            error_class="failed:killed",
            error_message="sigkill",
        ),
    )

    statuses = {
        str(row["status"])
        for row in conn.execute("SELECT status FROM verification_runs").fetchall()
    }

    assert statuses == {"env_unavailable", "install_failed", "killed", "native_crash", "oom"}


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

    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}) == 0

    append_verification_run(
        conn,
        _run(
            run_id="run-real",
            n_ops=n_ops,
            graph_shape_hash=graph_shape_hash,
            torchlens_version="tl-current",
        ),
    )

    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}) == 1


def test_update_and_delete_are_rejected_by_trigger(tmp_path: Path) -> None:
    """The database rejects mutation of ledger rows."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run())

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE verification_runs SET status='failed'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM verification_runs")


def test_current_verification_returns_latest_row_per_stable_id(tmp_path: Path) -> None:
    """The current view keeps the latest row for each stable model."""

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
    assert verified_count(conn, "tl-current", {"m1": "recipe-a", "m3": "recipe-a"}) == 0


def test_failed_run_with_n_ops_is_rejected(tmp_path: Path) -> None:
    """Failure rows cannot smuggle a zero or concrete op count."""

    conn = connect(tmp_path / "verification.db")

    with pytest.raises(ValueError, match="n_ops must be NULL"):
        append_verification_run(
            conn,
            _run(status="failed", forward_pass=0, n_ops=0, graph_shape_hash=None),
        )


def test_verified_count_requires_catalog_current_recipe_revision(tmp_path: Path) -> None:
    """A stale recipe pass does not certify a repaired catalog recipe."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(run_id="run-recipe-a", recipe_revision_sha256="recipe-a"))

    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}) == 1
    assert verified_count(conn, "tl-current", {"m1": "recipe-b"}) == 0


def test_verified_count_requires_full_identity_tuple(tmp_path: Path) -> None:
    """Only the latest row can satisfy the full verified identity tuple."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            run_id="stale-source",
            torchlens_source_hash="stale-source",
            finished_at="2026-06-22T00:00:01+00:00",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="stale-env",
            env_hash="stale-env",
            finished_at="2026-06-22T00:00:02+00:00",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="stale-lock",
            lock_hash="stale-lock",
            finished_at="2026-06-22T00:00:03+00:00",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="wrong-device",
            device_requested="cuda",
            finished_at="2026-06-22T00:00:04+00:00",
        ),
    )
    append_verification_run(
        conn,
        _run(
            run_id="legacy-source",
            torchlens_source_hash="legacy-unknown",
            finished_at="2026-06-22T00:00:05+00:00",
        ),
    )

    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}) == 0

    append_verification_run(
        conn,
        _run(
            run_id="current-recipe-b",
            recipe_revision_sha256="recipe-b",
            graph_shape_hash="shape-b",
            finished_at="2026-06-22T00:00:06+00:00",
        ),
    )

    assert verified_count(conn, "tl-current", {"m1": "recipe-b"}) == 1

    append_verification_run(
        conn,
        _run(
            run_id="newer-recipe-a-fail",
            recipe_revision_sha256="recipe-a",
            status="failed",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            error_class="RuntimeError",
            error_message="regressed",
            finished_at="2026-06-22T00:00:07+00:00",
        ),
    )

    assert verified_count(conn, "tl-current", {"m1": "recipe-b"}) == 0


def test_verified_count_excludes_half_migrated_legacy_identity(tmp_path: Path) -> None:
    """A row with either identity field still legacy-unknown is not headline-counted."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            run_id="half-migrated",
            torchlens_source_hash=LEGACY_UNKNOWN,
            lock_hash="lock-a",
        ),
    )
    targets = {
        "m1": VerificationTarget(
            recipe_revision_sha256="recipe-a",
            torchlens_source_hash=LEGACY_UNKNOWN,
            env_hash=base_env_hash(),
            lock_hash="lock-a",
            device_requested="cpu",
            scope="forward",
        )
    }

    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}, targets) == 0


def test_verified_count_rejects_newer_failure_at_same_identity(tmp_path: Path) -> None:
    """An old pass does not verify a model after the latest matching identity failed."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _run(run_id="old-pass"))
    append_verification_run(
        conn,
        _run(
            run_id="new-fail",
            status="failed",
            forward_pass=0,
            metadata_ok=0,
            n_ops=None,
            graph_shape_hash=None,
            error_class="RuntimeError",
            error_message="regressed",
            finished_at="2026-06-22T00:00:02+00:00",
        ),
    )

    assert verified_count(conn, "tl-current", {"m1": "recipe-a"}) == 0


def test_forward_backward_scope_round_trips(tmp_path: Path) -> None:
    """Forward-only and forward-plus-backward rows are distinct ledger scopes."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(
        conn,
        _run(
            run_id="forward-backward",
            scope="forward+backward",
            backward_pass=1,
        ),
    )

    row = conn.execute(
        "SELECT scope, forward_pass, backward_pass FROM verification_runs"
    ).fetchone()

    assert dict(row) == {"scope": "forward+backward", "forward_pass": 1, "backward_pass": 1}


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


def test_cascade_artifact_does_not_mask_real_prior_outcome(tmp_path: Path) -> None:
    """A synthesized cascade row never depresses a model's honest current status.

    The standard ``current_verification`` view shows the (later) cascade artifact
    as the model's current row, but ``current_verification_real`` skips it and
    surfaces the real earlier failure. Suppression is an HONESTY fix: it shows the
    model's true outcome, it does not flip anything to passed.
    """

    conn = connect(tmp_path / "verification.db")
    # Real earlier outcome: an honest replay failure (a tripwire firing).
    append_verification_run(
        conn,
        _run(
            run_id="run-real",
            stable_id="m1",
            status="failed",
            forward_pass=0,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            duration_sec=12.0,
            started_at="2026-06-25T00:00:00+00:00",
            finished_at="2026-06-25T00:00:12+00:00",
            error_class="failed:replay",
            error_message="False",
        ),
    )
    # Later synthesized cascade artifact (would otherwise be the rn=1 current row).
    append_verification_run(conn, _cascade_artifact(run_id="run-cascade", stable_id="m1"))

    standard = conn.execute(
        "SELECT status, error_class FROM current_verification WHERE stable_id = 'm1'"
    ).fetchone()
    assert standard["error_class"] == CASCADE_ARTIFACT_ERROR_CLASS

    real = conn.execute(
        "SELECT status, error_class FROM current_verification_real WHERE stable_id = 'm1'"
    ).fetchone()
    assert real["status"] == "failed"
    assert real["error_class"] == "failed:replay"


def test_cascade_only_model_falls_out_never_promoted(tmp_path: Path) -> None:
    """A model whose ONLY current row is a cascade artifact has no honest row.

    Critically it is NOT promoted to passed -- suppression removes the synthesized
    failure from the honest denominator, it never invents a success.
    """

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _cascade_artifact(run_id="run-cascade", stable_id="m1"))

    assert (
        conn.execute("SELECT COUNT(*) FROM current_verification WHERE stable_id = 'm1'").fetchone()[
            0
        ]
        == 1
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM current_verification_real WHERE stable_id = 'm1'"
        ).fetchone()[0]
        == 0
    )


def test_cascade_suppression_never_touches_genuine_failures(tmp_path: Path) -> None:
    """Real failures that share the marker error_class but ran a trace are kept.

    Defense-in-depth: the structural clauses (zero duration, NULL n_ops) mean a
    genuine failure is never suppressed even if it carried the marker string, so
    the tripwire stays armed. The honest per-model post-fix status
    ``failed:cluster_task_failed`` is also untouched.
    """

    conn = connect(tmp_path / "verification.db")
    # Honest post-fix per-model cluster failure: different error_class, real duration.
    append_verification_run(
        conn,
        _run(
            run_id="run-task",
            stable_id="m1",
            status="failed",
            forward_pass=0,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            duration_sec=5.0,
            started_at="2026-06-26T00:00:00+00:00",
            finished_at="2026-06-26T00:00:05+00:00",
            error_class="failed:cluster_task_failed",
            error_message="task .err tail",
        ),
    )
    # A row carrying the marker string but with a REAL trace duration: NOT an
    # artifact (the structural clauses exclude it), so it must stay visible.
    append_verification_run(
        conn,
        _run(
            run_id="run-marker-real",
            stable_id="m2",
            status="failed",
            forward_pass=0,
            metadata_ok=None,
            n_ops=None,
            graph_shape_hash=None,
            duration_sec=9.0,
            started_at="2026-06-26T00:00:00+00:00",
            finished_at="2026-06-26T00:00:09+00:00",
            error_class=CASCADE_ARTIFACT_ERROR_CLASS,
            error_message="real failure that happens to carry the marker",
        ),
    )

    m1 = conn.execute(
        "SELECT error_class FROM current_verification_real WHERE stable_id = 'm1'"
    ).fetchone()
    assert m1["error_class"] == "failed:cluster_task_failed"

    m2 = conn.execute(
        "SELECT error_class FROM current_verification_real WHERE stable_id = 'm2'"
    ).fetchone()
    assert m2 is not None
    assert m2["error_class"] == CASCADE_ARTIFACT_ERROR_CLASS

    assert cascade_suppressed_count(conn) == 0
    assert cascade_suppressed_stable_ids(conn) == []


def test_cascade_suppressed_helpers_report_masked_models(tmp_path: Path) -> None:
    """The suppression helpers enumerate exactly the masked models."""

    conn = connect(tmp_path / "verification.db")
    append_verification_run(conn, _cascade_artifact(run_id="run-c1", stable_id="m1"))
    append_verification_run(conn, _cascade_artifact(run_id="run-c2", stable_id="m2"))
    # A clean passing model is never reported.
    append_verification_run(conn, _run(run_id="run-ok", stable_id="m3"))

    assert cascade_suppressed_stable_ids(conn) == ["m1", "m2"]
    assert cascade_suppressed_count(conn) == 2
