"""Append-only SQLite verification ledger for the model menagerie."""

from __future__ import annotations

import platform
import socket
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from menagerie.catalog import CatalogRow


DATA_DIR = Path(__file__).resolve().parent / "data"
VERIFICATION_DB = DATA_DIR / "verification.db"
BUSY_TIMEOUT_MS = 30_000
Scope = Literal["forward", "backward"]
Status = Literal[
    "passed",
    "failed",
    "skipped",
    "timeout",
    "not_applicable",
    "deferred",
    "error",
]
VERIFIED_COUNT_SQL = """
SELECT COUNT(DISTINCT stable_id)
FROM current_verification
WHERE forward_pass = 1
  AND metadata_ok = 1
  AND n_ops IS NOT NULL
  AND graph_shape_hash IS NOT NULL
  AND torchlens_version = :torchlens_version
"""


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
        Verification scope, ``"forward"`` or ``"backward"``.
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
    runner_host:
        Hostname or runner identifier.
    started_at:
        ISO-8601 UTC start timestamp.
    finished_at:
        ISO-8601 UTC finish timestamp.
    duration_sec:
        Run duration in seconds.
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
    runner_host: str | None
    started_at: str
    finished_at: str
    duration_sec: float
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


def connect(db_path: Path = VERIFICATION_DB) -> sqlite3.Connection:
    """Open and initialize a verification ledger connection.

    Parameters
    ----------
    db_path:
        SQLite database path.

    Returns
    -------
    sqlite3.Connection
        Initialized SQLite connection with WAL and busy timeout enabled.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
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
            scope TEXT NOT NULL CHECK(scope IN ('forward','backward')),
            status TEXT NOT NULL CHECK(
                status IN (
                    'passed',
                    'failed',
                    'skipped',
                    'timeout',
                    'not_applicable',
                    'deferred',
                    'error'
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
            runner_host TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_sec REAL NOT NULL,
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

        DROP VIEW IF EXISTS current_verification;
        CREATE VIEW current_verification AS
        WITH ranked AS (
            SELECT
                verification_runs.*,
                ROW_NUMBER() OVER (
                    PARTITION BY stable_id, recipe_revision_sha256
                    ORDER BY finished_at DESC, run_id DESC
                ) AS rn
            FROM verification_runs
            WHERE scope = 'forward'
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
            runner_host,
            started_at,
            finished_at,
            duration_sec,
            error_class,
            error_message
        FROM ranked
        WHERE rn = 1;

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
            runner_host,
            started_at,
            finished_at,
            duration_sec,
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
            :runner_host,
            :started_at,
            :finished_at,
            :duration_sec,
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


def verified_count(
    conn: sqlite3.Connection,
    torchlens_version: str,
    current_revisions: dict[str, str],
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

    Returns
    -------
    int
        Count of distinct stable IDs satisfying the full honesty predicate.
    """

    if not current_revisions:
        return 0
    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS temp_current_catalog_revisions(
            stable_id TEXT PRIMARY KEY,
            recipe_revision_sha256 TEXT NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM temp_current_catalog_revisions")
    conn.executemany(
        """
        INSERT INTO temp_current_catalog_revisions(stable_id, recipe_revision_sha256)
        VALUES (?, ?)
        """,
        current_revisions.items(),
    )
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT current_verification.stable_id)
        FROM current_verification
        JOIN temp_current_catalog_revisions
          ON temp_current_catalog_revisions.stable_id = current_verification.stable_id
         AND temp_current_catalog_revisions.recipe_revision_sha256 =
             current_verification.recipe_revision_sha256
        WHERE current_verification.forward_pass = 1
          AND current_verification.metadata_ok = 1
          AND current_verification.n_ops IS NOT NULL
          AND current_verification.graph_shape_hash IS NOT NULL
          AND current_verification.torchlens_version = ?
        """,
        (torchlens_version,),
    ).fetchone()
    return int(row[0])


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
