"""Append-only SQLite verification ledger for the model menagerie."""

from __future__ import annotations

import platform
import os
import socket
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:
    from menagerie.catalog import CatalogRow


DATA_DIR = Path(__file__).resolve().parent / "data"
VERIFICATION_DB = DATA_DIR / "verification.db"
ENV_VERIFICATION_DB = "TORCHLENS_MENAGERIE_VERIFICATION_DB"
BUSY_TIMEOUT_MS = 30_000
LEGACY_UNKNOWN = "legacy-unknown"
BASE_LOCK_HASH = "base-no-lock"

# Frozen pre-fix batch-cascade artifact marker. A SLURM-array partial failure
# (some tasks validated, some failed) once cascaded the batch-level
# ``sbatch --wait`` returncode onto EVERY dispatched model, synthesizing rows
# with this ``error_class`` on the orchestrator host (job 6014456 stamped ~290
# such rows on 2026-06-26). The de-cascade fix (8e9dd386) removed every code path
# that writes this ``error_class``: current code attributes per-model and uses
# ``failed:cluster_task_failed`` for genuinely-missing task artifacts. So any row
# carrying ``failed:cluster_job_failed`` is, by definition, a frozen pre-fix
# cascade artifact -- a synthesized row whose REAL per-model outcome must be
# re-derived (re-attributed or re-run), never trusted as the model's status.
#
# These rows are append-only (the triggers forbid UPDATE/DELETE), so they cannot
# be removed; instead the ``current_verification_real`` view skips them when
# resolving each model's current row, so a frozen cascade artifact does not
# depress the headline while its real outcome is re-derived. Suppression NEVER
# promotes anything to ``passed`` -- a model whose only current row is a cascade
# artifact simply has no real current row (it falls out of the honest view).
CASCADE_ARTIFACT_ERROR_CLASS = "failed:cluster_job_failed"

# Full synthesized-artifact signature, used as defense-in-depth so suppression can
# never reach a genuine failure even if the marker string were ever reused. A
# real cascade artifact is status=failed, zero-duration, started==finished, with
# NULL n_ops / graph_shape_hash (it never ran a trace). All 290 known artifacts
# match every clause; a genuine ``failed:replay`` / ``failed:exception`` row has a
# real duration and is excluded by these structural clauses, not just the marker.
CASCADE_ARTIFACT_PREDICATE = (
    f"error_class = '{CASCADE_ARTIFACT_ERROR_CLASS}'\n"
    "          AND status = 'failed'\n"
    "          AND duration_sec = 0.0\n"
    "          AND started_at = finished_at\n"
    "          AND n_ops IS NULL\n"
    "          AND graph_shape_hash IS NULL"
)
Scope = Literal["forward", "forward+backward", "backward"]
Status = Literal[
    "passed",
    "failed",
    "skipped",
    "timeout",
    "not_applicable",
    "deferred",
    "error",
    "install_failed",
    "env_unavailable",
    "oom",
    "native_crash",
    "killed",
]
STATUS_VALUES: tuple[str, ...] = (
    "passed",
    "failed",
    "skipped",
    "timeout",
    "not_applicable",
    "deferred",
    "error",
    "install_failed",
    "env_unavailable",
    "oom",
    "native_crash",
    "killed",
)


@dataclass(frozen=True)
class VerificationTarget:
    """Current verification identity target for one catalog row.

    Parameters
    ----------
    recipe_revision_sha256:
        Current catalog recipe revision.
    torchlens_source_hash:
        Current TorchLens source hash.
    env_hash:
        Current assigned validation environment hash.
    lock_hash:
        Current assigned validation environment lock hash.
    device_requested:
        Requested device policy for the run.
    scope:
        Verification scope.
    """

    recipe_revision_sha256: str
    torchlens_source_hash: str
    env_hash: str
    lock_hash: str
    device_requested: str
    scope: Scope = "forward"


@dataclass(frozen=True)
class VerificationRun:
    """One append-only verification ledger row.

    Parameters
    ----------
    stable_id:
        Opaque durable model identity.
    recipe_revision_sha256:
        Frozen recipe fingerprint for the row's current construction recipe.
    name:
        Catalog display name.
    zoo:
        Source model zoo or library.
    variant:
        Optional natural-key discriminator.
    scope:
        Verification scope, ``"forward"``, ``"forward+backward"``, or ``"backward"``.
    status:
        Verification status.
    forward_pass:
        ``1`` when forward validation passed, ``0`` when it failed, otherwise ``None``.
    backward_pass:
        ``1`` when backward validation passed, ``0`` when attempted and failed, otherwise ``None``.
    backward_na_reason:
        Reason backward verification was not applicable.
    metadata_ok:
        ``1`` when TorchLens metadata validation passed, ``0`` when it failed, otherwise ``None``.
    n_ops:
        Real traced op count on pass; ``None`` on failure or legacy seed rows.
    graph_shape_hash:
        Trace graph shape hash.
    svg_sha256:
        SHA-256 of a rendered SVG artifact.
    torchlens_version:
        TorchLens version used for the run.
    torch_version:
        PyTorch version used for the run.
    python_version:
        Python version used for the run.
    device_requested:
        Requested execution device.
    device_actual:
        Actual execution device.
    env_hash:
        Optional environment hash.
    lock_hash:
        Validation environment lock hash.
    torchlens_source_hash:
        Hash of the editable TorchLens source used for validation.
    input_scale:
        Input scaling factor used for validation; ``None`` for legacy rows where
        the value was not recorded.
    runner_host:
        Hostname or runner identifier.
    started_at:
        ISO-8601 UTC start timestamp.
    finished_at:
        ISO-8601 UTC finish timestamp.
    duration_sec:
        Run duration in seconds.
    peak_rss_mb:
        Peak resident set size observed by the isolated validation worker, in MB.
    error_class:
        Exception class name, when available.
    error_message:
        Exception message, skip reason, or audit note.
    run_id:
        UUID primary key. A new UUID is generated when omitted.
    """

    stable_id: str
    recipe_revision_sha256: str
    name: str
    zoo: str
    variant: str
    scope: Scope
    status: Status
    forward_pass: int | None
    backward_pass: int | None
    backward_na_reason: str | None
    metadata_ok: int | None
    n_ops: int | None
    graph_shape_hash: str | None
    svg_sha256: str | None
    torchlens_version: str
    torch_version: str
    python_version: str
    device_requested: str
    device_actual: str | None
    env_hash: str | None
    lock_hash: str
    torchlens_source_hash: str
    input_scale: float | None
    runner_host: str | None
    started_at: str
    finished_at: str
    duration_sec: float
    peak_rss_mb: int | None = None
    error_class: str | None = None
    error_message: str | None = None
    run_id: str = ""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns
    -------
    str
        Current UTC timestamp with second precision.
    """

    return datetime.now(UTC).isoformat(timespec="seconds")


def python_version() -> str:
    """Return the running Python version string.

    Returns
    -------
    str
        Python implementation and version.
    """

    return f"{platform.python_implementation()} {platform.python_version()}"


def runner_host() -> str:
    """Return the local runner host name.

    Returns
    -------
    str
        Host name, or an empty string if unavailable.
    """

    return socket.gethostname()


def _package_version(package: str, fallback: str) -> str:
    """Return an installed package version with a safe fallback.

    Parameters
    ----------
    package:
        Distribution package name.
    fallback:
        Version string to return when distribution metadata is unavailable.

    Returns
    -------
    str
        Package version.
    """

    try:
        return version(package)
    except PackageNotFoundError:
        return fallback


def base_env_hash() -> str:
    """Return the deterministic base validation environment hash.

    Returns
    -------
    str
        SHA-256 hash of the sorted base environment identity.
    """

    import torch
    import torchlens as tl

    identity = {
        "python": python_version(),
        "torch": str(torch.__version__),
        "torchlens": _package_version("torchlens", str(getattr(tl, "__version__", "unknown"))),
    }
    payload = "\n".join(f"{key}={identity[key]}" for key in sorted(identity))
    return sha256(payload.encode("utf-8")).hexdigest()


def base_lock_hash() -> str:
    """Return the lock-hash sentinel for the mutable base validation environment.

    Returns
    -------
    str
        Base-environment lock-hash sentinel.
    """

    return BASE_LOCK_HASH


def torchlens_source_hash(repo_root: Path | None = None) -> str:
    """Return a reproducibility hash for the local TorchLens checkout.

    The hash includes the Git HEAD, tracked source diff, and untracked source
    files under ``torchlens`` and ``menagerie``. This makes editable-checkout
    validation rows stale when either committed or uncommitted source changes.

    Parameters
    ----------
    repo_root:
        Repository root. Defaults to the parent of the ``menagerie`` package.

    Returns
    -------
    str
        SHA-256 source hash, or ``"legacy-unknown"`` when Git metadata cannot be read.
    """

    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "--", "torchlens", "menagerie"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "torchlens",
                "menagerie",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return LEGACY_UNKNOWN
    digest = sha256()
    digest.update(head.encode("utf-8"))
    digest.update(b"\0")
    digest.update(diff.encode("utf-8"))
    for relative_path in sorted(untracked):
        path = root / relative_path
        if not path.is_file():
            continue
        digest.update(b"\0")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve_verification_db(db_path: Path | None = None) -> Path:
    """Resolve the active verification ledger path at connection time.

    Parameters
    ----------
    db_path:
        Explicit SQLite database path. When omitted, the
        ``TORCHLENS_MENAGERIE_VERIFICATION_DB`` environment override is honored
        before falling back to the committed production ledger.

    Returns
    -------
    pathlib.Path
        Verification ledger path.
    """

    if db_path is not None:
        return Path(db_path)
    override = os.environ.get(ENV_VERIFICATION_DB)
    if override:
        return Path(override).expanduser()
    return VERIFICATION_DB


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open and initialize a verification ledger connection.

    Parameters
    ----------
    db_path:
        SQLite database path. Defaults to the resolved verification ledger,
        honoring ``TORCHLENS_MENAGERIE_VERIFICATION_DB`` at connect time.

    Returns
    -------
    sqlite3.Connection
        Initialized SQLite connection with WAL and busy timeout enabled.
    """

    resolved_db_path = _resolve_verification_db(db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved_db_path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    conn.row_factory = sqlite3.Row
    configure_connection(conn)
    initialize(conn)
    return conn


def configure_connection(conn: sqlite3.Connection) -> None:
    """Configure SQLite durability and concurrency settings.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")


def initialize(conn: sqlite3.Connection) -> None:
    """Create the verification ledger schema if needed.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS verification_runs(
            run_id TEXT PRIMARY KEY,
            stable_id TEXT NOT NULL,
            recipe_revision_sha256 TEXT NOT NULL,
            name TEXT NOT NULL,
            zoo TEXT NOT NULL,
            variant TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL CHECK(scope IN ('forward','forward+backward','backward')),
            status TEXT NOT NULL CHECK(
                status IN (
                    'passed',
                    'failed',
                    'skipped',
                    'timeout',
                    'not_applicable',
                    'deferred',
                    'error',
                    'install_failed',
                    'env_unavailable',
                    'oom',
                    'native_crash',
                    'killed'
                )
            ),
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

        CREATE INDEX IF NOT EXISTS idx_vr_stable_id ON verification_runs(stable_id);
        CREATE INDEX IF NOT EXISTS idx_vr_recipe_revision
            ON verification_runs(recipe_revision_sha256);
        CREATE INDEX IF NOT EXISTS idx_vr_scope_status ON verification_runs(scope, status);
        CREATE INDEX IF NOT EXISTS idx_vr_finished_at ON verification_runs(finished_at);
        CREATE INDEX IF NOT EXISTS idx_vr_torchlens_version
            ON verification_runs(torchlens_version);

        CREATE TRIGGER IF NOT EXISTS verification_runs_no_update
        BEFORE UPDATE ON verification_runs
        BEGIN
            SELECT RAISE(ABORT, 'verification_runs is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS verification_runs_no_delete
        BEFORE DELETE ON verification_runs
        BEGIN
            SELECT RAISE(ABORT, 'verification_runs is append-only');
        END;
        """
    )
    _migrate_peak_rss_column(conn)
    _migrate_identity_columns(conn)
    _migrate_status_constraint(conn)
    _create_current_verification_view(conn)
    _create_current_verification_real_view(conn)


def _create_current_verification_view(conn: sqlite3.Connection) -> None:
    """Create the canonical latest verification row view.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    conn.executescript(
        """
        DROP VIEW IF EXISTS current_verification;
        CREATE VIEW current_verification AS
        WITH ranked AS (
            SELECT
                verification_runs.*,
                ROW_NUMBER() OVER (
                    PARTITION BY stable_id
                    ORDER BY finished_at DESC, run_id DESC
                ) AS rn
            FROM verification_runs
        )
        SELECT
            run_id,
            stable_id,
            recipe_revision_sha256,
            name,
            zoo,
            variant,
            scope,
            status,
            forward_pass,
            backward_pass,
            backward_na_reason,
            metadata_ok,
            n_ops,
            graph_shape_hash,
            svg_sha256,
            torchlens_version,
            torch_version,
            python_version,
            device_requested,
            device_actual,
            env_hash,
            lock_hash,
            torchlens_source_hash,
            input_scale,
            runner_host,
            started_at,
            finished_at,
            duration_sec,
            peak_rss_mb,
            error_class,
            error_message
        FROM ranked
        WHERE rn = 1;
        """
    )


def _create_current_verification_real_view(conn: sqlite3.Connection) -> None:
    """Create the cascade-suppressed current verification view.

    This is the HONEST-DENOMINATOR sibling of ``current_verification``. It ranks
    rows identically (latest ``finished_at``, then ``run_id``, per ``stable_id``)
    but EXCLUDES frozen pre-fix batch-cascade artifact rows
    (``CASCADE_ARTIFACT_ERROR_CLASS`` with the full synthesized signature) before
    selecting the current row. A synthesized cascade row therefore never masks a
    model's real prior outcome, and a model whose ONLY current row is a cascade
    artifact simply has no row here (it falls out -- it is never promoted to a
    pass). Real failures (``failed:replay`` / ``failed:cluster_task_failed`` /
    ``failed:exception`` / ``timeout`` / ...) are untouched: the structural clauses
    (zero duration, NULL ``n_ops``/``graph_shape_hash``) exclude every genuine
    failure, so the tripwire stays armed.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    conn.executescript(
        f"""
        DROP VIEW IF EXISTS current_verification_real;
        CREATE VIEW current_verification_real AS
        WITH ranked AS (
            SELECT
                verification_runs.*,
                ROW_NUMBER() OVER (
                    PARTITION BY stable_id
                    ORDER BY finished_at DESC, run_id DESC
                ) AS rn
            FROM verification_runs
            WHERE NOT (
                {CASCADE_ARTIFACT_PREDICATE}
            )
        )
        SELECT
            run_id,
            stable_id,
            recipe_revision_sha256,
            name,
            zoo,
            variant,
            scope,
            status,
            forward_pass,
            backward_pass,
            backward_na_reason,
            metadata_ok,
            n_ops,
            graph_shape_hash,
            svg_sha256,
            torchlens_version,
            torch_version,
            python_version,
            device_requested,
            device_actual,
            env_hash,
            lock_hash,
            torchlens_source_hash,
            input_scale,
            runner_host,
            started_at,
            finished_at,
            duration_sec,
            peak_rss_mb,
            error_class,
            error_message
        FROM ranked
        WHERE rn = 1;
        """
    )


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return whether a SQLite table has a column.

    Parameters
    ----------
    conn:
        SQLite connection.
    table:
        Table name.
    column:
        Column name.

    Returns
    -------
    bool
        ``True`` when the column exists.
    """

    return any(str(row["name"]) == column for row in conn.execute(f"PRAGMA table_info({table})"))


def _migrate_peak_rss_column(conn: sqlite3.Connection) -> None:
    """Add the nullable peak RSS column to legacy verification ledgers.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    if _has_column(conn, "verification_runs", "peak_rss_mb"):
        return
    conn.execute("ALTER TABLE verification_runs ADD COLUMN peak_rss_mb INTEGER")
    initialize(conn)


def _migrate_identity_columns(conn: sqlite3.Connection) -> None:
    """Add identity/provenance columns to legacy verification ledgers.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    migrated = False
    if not _has_column(conn, "verification_runs", "lock_hash"):
        conn.execute(
            "ALTER TABLE verification_runs "
            f"ADD COLUMN lock_hash TEXT NOT NULL DEFAULT '{LEGACY_UNKNOWN}'"
        )
        migrated = True
    if not _has_column(conn, "verification_runs", "torchlens_source_hash"):
        conn.execute(
            "ALTER TABLE verification_runs "
            f"ADD COLUMN torchlens_source_hash TEXT NOT NULL DEFAULT '{LEGACY_UNKNOWN}'"
        )
        migrated = True
    if not _has_column(conn, "verification_runs", "input_scale"):
        conn.execute("ALTER TABLE verification_runs ADD COLUMN input_scale REAL")
        migrated = True
    if migrated:
        initialize(conn)


def _migrate_status_constraint(conn: sqlite3.Connection) -> None:
    """Recreate legacy ledgers whose status or scope checks lack current values.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'verification_runs'
        """
    ).fetchone()
    if row is None:
        return
    table_sql = str(row["sql"] if isinstance(row, sqlite3.Row) else row[0])
    if (
        all(f"'{status}'" in table_sql for status in STATUS_VALUES)
        and "'forward+backward'" in table_sql
    ):
        return
    existing_columns = {
        str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
        for row in conn.execute("PRAGMA table_info(verification_runs)").fetchall()
    }
    lock_hash_expr = (
        "lock_hash" if "lock_hash" in existing_columns else f"'{LEGACY_UNKNOWN}' AS lock_hash"
    )
    source_hash_expr = (
        "torchlens_source_hash"
        if "torchlens_source_hash" in existing_columns
        else f"'{LEGACY_UNKNOWN}' AS torchlens_source_hash"
    )
    input_scale_expr = "input_scale" if "input_scale" in existing_columns else "NULL AS input_scale"
    peak_rss_column = "peak_rss_mb" if "peak_rss_mb" in existing_columns else "NULL AS peak_rss_mb"
    conn.executescript(
        f"""
        DROP VIEW IF EXISTS current_verification;
        DROP TRIGGER IF EXISTS verification_runs_no_update;
        DROP TRIGGER IF EXISTS verification_runs_no_delete;

        ALTER TABLE verification_runs RENAME TO verification_runs_legacy_status_check;

        CREATE TABLE verification_runs(
            run_id TEXT PRIMARY KEY,
            stable_id TEXT NOT NULL,
            recipe_revision_sha256 TEXT NOT NULL,
            name TEXT NOT NULL,
            zoo TEXT NOT NULL,
            variant TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL CHECK(scope IN ('forward','forward+backward','backward')),
            status TEXT NOT NULL CHECK(
                status IN (
                    'passed',
                    'failed',
                    'skipped',
                    'timeout',
                    'not_applicable',
                    'deferred',
                    'error',
                    'install_failed',
                    'env_unavailable',
                    'oom',
                    'native_crash',
                    'killed'
                )
            ),
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
            run_id,
            stable_id,
            recipe_revision_sha256,
            name,
            zoo,
            variant,
            scope,
            status,
            forward_pass,
            backward_pass,
            backward_na_reason,
            metadata_ok,
            n_ops,
            graph_shape_hash,
            svg_sha256,
            torchlens_version,
            torch_version,
            python_version,
            device_requested,
            device_actual,
            env_hash,
            lock_hash,
            torchlens_source_hash,
            input_scale,
            runner_host,
            started_at,
            finished_at,
            duration_sec,
            peak_rss_mb,
            error_class,
            error_message
        )
        SELECT
            run_id,
            stable_id,
            recipe_revision_sha256,
            name,
            zoo,
            variant,
            scope,
            status,
            forward_pass,
            backward_pass,
            backward_na_reason,
            metadata_ok,
            n_ops,
            graph_shape_hash,
            svg_sha256,
            torchlens_version,
            torch_version,
            python_version,
            device_requested,
            device_actual,
            env_hash,
            {lock_hash_expr},
            {source_hash_expr},
            {input_scale_expr},
            runner_host,
            started_at,
            finished_at,
            duration_sec,
            {peak_rss_column},
            error_class,
            error_message
        FROM verification_runs_legacy_status_check;

        DROP TABLE verification_runs_legacy_status_check;

        CREATE INDEX IF NOT EXISTS idx_vr_stable_id ON verification_runs(stable_id);
        CREATE INDEX IF NOT EXISTS idx_vr_recipe_revision
            ON verification_runs(recipe_revision_sha256);
        CREATE INDEX IF NOT EXISTS idx_vr_scope_status ON verification_runs(scope, status);
        CREATE INDEX IF NOT EXISTS idx_vr_finished_at ON verification_runs(finished_at);
        CREATE INDEX IF NOT EXISTS idx_vr_torchlens_version
            ON verification_runs(torchlens_version);
        """
    )
    initialize(conn)


def append_verification_run(conn: sqlite3.Connection, run: VerificationRun) -> str:
    """Insert one verification run into the append-only ledger.

    Parameters
    ----------
    conn:
        Initialized SQLite connection.
    run:
        Verification run payload.

    Returns
    -------
    str
        Inserted run ID.
    """

    if run.status != "passed" and run.n_ops is not None:
        raise ValueError("n_ops must be NULL unless the run passed")
    run_id = run.run_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO verification_runs(
            run_id,
            stable_id,
            recipe_revision_sha256,
            name,
            zoo,
            variant,
            scope,
            status,
            forward_pass,
            backward_pass,
            backward_na_reason,
            metadata_ok,
            n_ops,
            graph_shape_hash,
            svg_sha256,
            torchlens_version,
            torch_version,
            python_version,
            device_requested,
            device_actual,
            env_hash,
            lock_hash,
            torchlens_source_hash,
            input_scale,
            runner_host,
            started_at,
            finished_at,
            duration_sec,
            peak_rss_mb,
            error_class,
            error_message
        ) VALUES (
            :run_id,
            :stable_id,
            :recipe_revision_sha256,
            :name,
            :zoo,
            :variant,
            :scope,
            :status,
            :forward_pass,
            :backward_pass,
            :backward_na_reason,
            :metadata_ok,
            :n_ops,
            :graph_shape_hash,
            :svg_sha256,
            :torchlens_version,
            :torch_version,
            :python_version,
            :device_requested,
            :device_actual,
            :env_hash,
            :lock_hash,
            :torchlens_source_hash,
            :input_scale,
            :runner_host,
            :started_at,
            :finished_at,
            :duration_sec,
            :peak_rss_mb,
            :error_class,
            :error_message
        )
        """,
        {
            **run.__dict__,
            "run_id": run_id,
        },
    )
    return run_id


def _default_verification_targets(
    current_revisions: Mapping[str, str],
    source_hash: str | None = None,
    env_hash_value: str | None = None,
    lock_hash_value: str | None = None,
    device_requested: str = "cpu",
    scope: Scope = "forward",
) -> dict[str, VerificationTarget]:
    """Build same-environment verification targets from current recipe revisions.

    Parameters
    ----------
    current_revisions:
        Catalog mapping from stable ID to the current recipe revision hash.
    source_hash:
        Source hash to require. Defaults to the current checkout hash.
    env_hash_value:
        Environment hash to require. Defaults to the current base environment hash.
    lock_hash_value:
        Lock hash to require. Defaults to the base no-lock sentinel.
    device_requested:
        Requested device policy to require.
    scope:
        Verification scope to require.

    Returns
    -------
    dict[str, VerificationTarget]
        Verification targets keyed by stable ID.
    """

    resolved_source = source_hash or torchlens_source_hash()
    resolved_env = env_hash_value or base_env_hash()
    resolved_lock = lock_hash_value or base_lock_hash()
    return {
        stable_id: VerificationTarget(
            recipe_revision_sha256=recipe_revision,
            torchlens_source_hash=resolved_source,
            env_hash=resolved_env,
            lock_hash=resolved_lock,
            device_requested=device_requested,
            scope=scope,
        )
        for stable_id, recipe_revision in current_revisions.items()
    }


def _load_verification_targets(
    conn: sqlite3.Connection, current_targets: Mapping[str, VerificationTarget]
) -> None:
    """Load current verification targets into a temporary SQLite table.

    Parameters
    ----------
    conn:
        Initialized SQLite connection.
    current_targets:
        Current verification targets keyed by stable ID.
    """

    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS temp_current_verification_targets(
            stable_id TEXT PRIMARY KEY,
            recipe_revision_sha256 TEXT NOT NULL,
            torchlens_source_hash TEXT NOT NULL,
            env_hash TEXT NOT NULL,
            lock_hash TEXT NOT NULL,
            device_requested TEXT NOT NULL,
            scope TEXT NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM temp_current_verification_targets")
    conn.executemany(
        """
        INSERT INTO temp_current_verification_targets(
            stable_id,
            recipe_revision_sha256,
            torchlens_source_hash,
            env_hash,
            lock_hash,
            device_requested,
            scope
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                stable_id,
                target.recipe_revision_sha256,
                target.torchlens_source_hash,
                target.env_hash,
                target.lock_hash,
                target.device_requested,
                target.scope,
            )
            for stable_id, target in current_targets.items()
        ),
    )


def verified_count(
    conn: sqlite3.Connection,
    torchlens_version: str,
    current_revisions: Mapping[str, str],
    current_targets: Mapping[str, VerificationTarget] | None = None,
) -> int:
    """Return the current-recipe, current-version verified architecture count.

    Parameters
    ----------
    conn:
        Initialized SQLite connection.
    torchlens_version:
        TorchLens version that must match current verification rows.
    current_revisions:
        Catalog mapping from stable ID to the current recipe revision hash.
    current_targets:
        Optional full identity targets keyed by stable ID. When omitted, the
        current base environment identity is used for all revisions.

    Returns
    -------
    int
        Count of distinct stable IDs satisfying the full honesty predicate.
    """

    targets = (
        dict(current_targets)
        if current_targets is not None
        else _default_verification_targets(current_revisions)
    )
    if not targets:
        return 0
    _load_verification_targets(conn, targets)
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT current_verification.stable_id)
        FROM current_verification
        JOIN temp_current_verification_targets AS target
          ON target.stable_id = current_verification.stable_id
         AND target.recipe_revision_sha256 = current_verification.recipe_revision_sha256
         AND target.torchlens_source_hash = current_verification.torchlens_source_hash
         AND target.env_hash = current_verification.env_hash
         AND target.lock_hash = current_verification.lock_hash
         AND target.device_requested = current_verification.device_requested
         AND target.scope = current_verification.scope
        WHERE current_verification.status = 'passed'
          AND current_verification.forward_pass = 1
          AND current_verification.metadata_ok = 1
          AND current_verification.n_ops IS NOT NULL
          AND current_verification.graph_shape_hash IS NOT NULL
          AND current_verification.torchlens_version = ?
          AND NOT (
              current_verification.torchlens_source_hash = ?
              OR current_verification.lock_hash = ?
          )
        """,
        (torchlens_version, LEGACY_UNKNOWN, LEGACY_UNKNOWN),
    ).fetchone()
    return int(row[0])


def cascade_suppressed_stable_ids(conn: sqlite3.Connection) -> list[str]:
    """Return stable IDs whose current row is a suppressed cascade artifact.

    A stable ID is suppressed when its latest row in ``current_verification`` is a
    frozen pre-fix batch-cascade artifact, so ``current_verification_real`` shows a
    different (or no) row for it. These are exactly the models whose REAL outcome
    is pending re-derivation (re-attribution or re-run); the honest view hides the
    synthesized failure rather than depressing the headline with it.

    Parameters
    ----------
    conn:
        Initialized SQLite connection.

    Returns
    -------
    list[str]
        Sorted stable IDs currently masked by a cascade artifact.
    """

    rows = conn.execute(
        f"""
        SELECT stable_id
        FROM current_verification
        WHERE {CASCADE_ARTIFACT_PREDICATE}
        ORDER BY stable_id
        """
    ).fetchall()
    return [str(row["stable_id"]) for row in rows]


def cascade_suppressed_count(conn: sqlite3.Connection) -> int:
    """Return how many models are currently masked by a cascade artifact.

    Parameters
    ----------
    conn:
        Initialized SQLite connection.

    Returns
    -------
    int
        Count of stable IDs whose current row is a suppressed cascade artifact.
    """

    return len(cascade_suppressed_stable_ids(conn))


def seed_from_legacy(conn: sqlite3.Connection, rows: list[CatalogRow]) -> int:
    """Seed audit-history rows from legacy catalog verification flags.

    Parameters
    ----------
    conn:
        Initialized SQLite connection.
    rows:
        Catalog rows to inspect.

    Returns
    -------
    int
        Number of legacy seed rows inserted.
    """

    inserted = 0
    for row in rows:
        if not row.verified:
            continue
        now = utc_now()
        append_verification_run(
            conn,
            VerificationRun(
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                name=row.name,
                zoo=row.zoo,
                variant=row.variant,
                scope="forward",
                status="passed",
                forward_pass=1,
                backward_pass=None,
                backward_na_reason=None,
                metadata_ok=None,
                n_ops=None,
                graph_shape_hash=None,
                svg_sha256=None,
                torchlens_version="legacy-unknown",
                torch_version="legacy-unknown",
                python_version=sys.version.split()[0],
                device_requested="legacy-unknown",
                device_actual=None,
                env_hash=None,
                lock_hash=LEGACY_UNKNOWN,
                torchlens_source_hash=LEGACY_UNKNOWN,
                input_scale=None,
                runner_host="legacy-seed",
                started_at=now,
                finished_at=now,
                duration_sec=0.0,
                error_class=None,
                error_message="legacy seed from catalog verified bool; not re-run",
            ),
        )
        inserted += 1
    return inserted
