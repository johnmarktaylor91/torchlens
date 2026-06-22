"""Structured discovery-sweep provenance for the menagerie."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from menagerie.ledger import VERIFICATION_DB, connect as connect_ledger, utc_now


@dataclass(frozen=True)
class SweepProvenance:
    """One structured discovery or harvest sweep record.

    Parameters
    ----------
    sweep_id:
        Stable sweep identifier.
    date:
        Sweep date as an ISO date string.
    agent:
        Human or automation agent responsible for the sweep.
    sources:
        Source hubs, logs, or venue axes consulted.
    selection_rule:
        Rule used to decide what should enter the catalog.
    families_considered:
        Families or count considered by the sweep.
    families_added:
        Families or count added by the sweep.
    families_rejected:
        Families or count rejected as duplicates or out of scope.
    git_commit:
        Commit associated with the sweep, when known.
    notes:
        Additional notes, including raw historical prose when imported.
    created_at:
        Timestamp when the structured row was written.
    """

    sweep_id: str
    date: str
    agent: str
    sources: list[str]
    selection_rule: str
    families_considered: list[str] | int | str
    families_added: list[str] | int | str
    families_rejected: list[str] | int | str
    git_commit: str
    notes: str
    created_at: str


def initialize(conn: sqlite3.Connection) -> None:
    """Create the sweep provenance table if needed.

    Parameters
    ----------
    conn:
        SQLite connection.
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sweep_provenance(
            sweep_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            agent TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            selection_rule TEXT NOT NULL,
            families_considered_json TEXT NOT NULL,
            families_added_json TEXT NOT NULL,
            families_rejected_json TEXT NOT NULL,
            git_commit TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sweep_provenance_date ON sweep_provenance(date)")


def record_sweep(
    sweep_id: str,
    date: str,
    agent: str,
    sources: Sequence[str],
    selection_rule: str,
    families_considered: Sequence[str] | int | str,
    families_added: Sequence[str] | int | str,
    families_rejected: Sequence[str] | int | str,
    git_commit: str,
    notes: str,
    db_path: Path = VERIFICATION_DB,
) -> SweepProvenance:
    """Record or replace one structured discovery sweep row.

    Parameters
    ----------
    sweep_id:
        Stable sweep identifier.
    date:
        Sweep date as an ISO date string.
    agent:
        Human or automation agent responsible for the sweep.
    sources:
        Source hubs, logs, or venue axes consulted.
    selection_rule:
        Rule used to decide what should enter the catalog.
    families_considered:
        Families or count considered by the sweep.
    families_added:
        Families or count added by the sweep.
    families_rejected:
        Families or count rejected as duplicates or out of scope.
    git_commit:
        Commit associated with the sweep, when known.
    notes:
        Additional notes, including raw historical prose when imported.
    db_path:
        Verification database path that owns the provenance table.

    Returns
    -------
    SweepProvenance
        Stored sweep row.
    """

    created_at = utc_now()
    row = SweepProvenance(
        sweep_id=sweep_id,
        date=date,
        agent=agent,
        sources=list(sources),
        selection_rule=selection_rule,
        families_considered=_jsonable_value(families_considered),
        families_added=_jsonable_value(families_added),
        families_rejected=_jsonable_value(families_rejected),
        git_commit=git_commit,
        notes=notes,
        created_at=created_at,
    )
    with connect_ledger(db_path) as conn:
        initialize(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO sweep_provenance(
                sweep_id,
                date,
                agent,
                sources_json,
                selection_rule,
                families_considered_json,
                families_added_json,
                families_rejected_json,
                git_commit,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.sweep_id,
                row.date,
                row.agent,
                json.dumps(row.sources, sort_keys=True),
                row.selection_rule,
                json.dumps(row.families_considered, sort_keys=True),
                json.dumps(row.families_added, sort_keys=True),
                json.dumps(row.families_rejected, sort_keys=True),
                row.git_commit,
                row.notes,
                row.created_at,
            ),
        )
    return row


def read_sweeps(db_path: Path = VERIFICATION_DB) -> list[SweepProvenance]:
    """Read sweep provenance rows ordered by date and identifier.

    Parameters
    ----------
    db_path:
        Verification database path that owns the provenance table.

    Returns
    -------
    list[SweepProvenance]
        Structured sweep provenance rows.
    """

    with connect_ledger(db_path) as conn:
        initialize(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM sweep_provenance
            ORDER BY date, sweep_id
            """
        ).fetchall()
    return [_row_to_sweep(row) for row in rows]


def _jsonable_value(value: Sequence[str] | int | str) -> list[str] | int | str:
    """Normalize provenance count/list values for JSON storage.

    Parameters
    ----------
    value:
        Count, label, or string sequence.

    Returns
    -------
    list[str] | int | str
        JSON-serializable normalized value.
    """

    if isinstance(value, str | int):
        return value
    return list(value)


def _row_to_sweep(row: sqlite3.Row) -> SweepProvenance:
    """Convert a SQLite row to a provenance dataclass.

    Parameters
    ----------
    row:
        SQLite result row.

    Returns
    -------
    SweepProvenance
        Parsed provenance row.
    """

    return SweepProvenance(
        sweep_id=str(row["sweep_id"]),
        date=str(row["date"]),
        agent=str(row["agent"]),
        sources=list(json.loads(str(row["sources_json"]))),
        selection_rule=str(row["selection_rule"]),
        families_considered=json.loads(str(row["families_considered_json"])),
        families_added=json.loads(str(row["families_added_json"])),
        families_rejected=json.loads(str(row["families_rejected_json"])),
        git_commit=str(row["git_commit"]),
        notes=str(row["notes"]),
        created_at=str(row["created_at"]),
    )
