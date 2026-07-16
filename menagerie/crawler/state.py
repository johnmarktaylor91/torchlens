"""Disposable SQLite state rebuilt only from intake and canonical ledgers."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.models import JsonObject, LedgerPaths, RebuildSummary
from menagerie.crawler.recordio import LedgerCorruptionError, scan_jsonl

IntakeSource = Union[Path, str, Iterable[str], Iterable[Mapping[str, Any]]]


def _load_intake(source: IntakeSource) -> dict[str, JsonObject]:
    """Load a stable-ID keyed intake snapshot from a path or iterable.

    Parameters
    ----------
    source:
        JSON/JSONL path, stable-ID iterable, or intake-row iterable.

    Returns
    -------
    dict[str, dict[str, Any]]
        Stable-ID keyed intake payloads.
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".jsonl":
            rows: Iterable[Any] = scan_jsonl(path, validate=False)
        else:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            rows = loaded if isinstance(loaded, list) else loaded.get("items", [])
    else:
        rows = source
    result: dict[str, JsonObject] = {}
    for row in rows:
        payload = {"stable_id": row} if isinstance(row, str) else dict(row)
        stable_id = payload.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("each intake row must have a nonempty stable_id")
        if stable_id in result and result[stable_id] != payload:
            raise ValueError(f"conflicting intake rows for {stable_id}")
        result[stable_id] = payload
    return result


def _select_current(records: Iterable[Mapping[str, Any]]) -> dict[str, JsonObject]:
    """Validate revision parentage and select current model revisions.

    Parameters
    ----------
    records:
        Model revisions in canonical ledger order.

    Returns
    -------
    dict[str, dict[str, Any]]
        Highest valid revision for each stable ID.

    Raises
    ------
    LedgerCorruptionError
        If revision sequence/parentage forks or points stale.
    """

    current: dict[str, JsonObject] = {}
    seen_revisions: dict[str, JsonObject] = {}
    for record_mapping in records:
        record = dict(record_mapping)
        stable_id = str(record["stable_id"])
        revision = str(record["record_revision"])
        if revision in seen_revisions and seen_revisions[revision] != record:
            raise LedgerCorruptionError(f"conflicting model revision {revision}")
        previous = current.get(stable_id)
        expected_parent = None if previous is None else previous["record_revision"]
        if record["parent_revision"] != expected_parent:
            raise LedgerCorruptionError(
                f"stale/forked parent for {stable_id}: expected {expected_parent!r}, "
                f"received {record['parent_revision']!r}"
            )
        if previous is not None and record["record_seq"] <= previous["record_seq"]:
            raise LedgerCorruptionError(f"non-increasing record_seq for {stable_id}")
        seen_revisions[revision] = record
        current[stable_id] = record
    return current


def rebuild_state(
    database_path: Union[str, Path],
    intake: IntakeSource,
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
) -> RebuildSummary:
    """Atomically rebuild the disposable SQLite queue/current-state view.

    Parameters
    ----------
    database_path:
        Destination SQLite path.
    intake:
        Trusted intake snapshot or stable-ID iterable.
    ledgers:
        Canonical model/attempt/gate JSONL paths.
    context:
        Mandatory active authority used by the shared currency projection.

    Returns
    -------
    RebuildSummary
        Counts inserted into the rebuilt database.
    """

    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.rebuild")
    temporary.unlink(missing_ok=True)
    intake_rows = _load_intake(intake)
    model_records = scan_jsonl(ledgers.models)
    attempt_records = scan_jsonl(ledgers.attempts)
    gate_records = scan_jsonl(ledgers.gates)
    from menagerie.crawler.reducer import project_dependency_current  # noqa: PLC0415

    current = project_dependency_current(
        ledgers,
        context=context,
        model_records=model_records,
        attempt_records=attempt_records,
        gate_records=gate_records,
    ).current_records
    summary = RebuildSummary(
        intake_count=len(intake_rows),
        model_revision_count=len(model_records),
        attempt_count=len(attempt_records),
        gate_count=len(gate_records),
        current_count=len(current),
    )
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE intake (
                stable_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE model_revisions (
                record_revision TEXT PRIMARY KEY,
                stable_id TEXT NOT NULL,
                record_seq INTEGER NOT NULL,
                parent_revision TEXT,
                status_code TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(stable_id, record_seq)
            );
            CREATE TABLE current_models (
                stable_id TEXT PRIMARY KEY,
                record_revision TEXT NOT NULL,
                record_seq INTEGER NOT NULL,
                status_code TEXT NOT NULL,
                authored_metadata_state TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                stable_id TEXT,
                ledger_seq INTEGER NOT NULL UNIQUE,
                stage TEXT NOT NULL,
                result TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE gates (
                gate_id TEXT PRIMARY KEY,
                ledger_seq INTEGER NOT NULL UNIQUE,
                gate_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE gate_items (
                gate_id TEXT NOT NULL,
                stable_id TEXT NOT NULL,
                work_id TEXT NOT NULL,
                verdict TEXT NOT NULL,
                vet_identity TEXT NOT NULL,
                fidelity_identity TEXT,
                PRIMARY KEY(gate_id, stable_id, work_id)
            );
            CREATE VIEW terminal_funnel AS
            SELECT status_code, COUNT(*) AS model_count
            FROM current_models GROUP BY status_code;
            CREATE VIEW queue AS
            SELECT i.stable_id,
                   CASE WHEN c.stable_id IS NULL THEN 'UNTRIAGED' ELSE 'TERMINAL' END AS workflow_state
            FROM intake AS i LEFT JOIN current_models AS c USING (stable_id);
            """
        )
        connection.executemany(
            "INSERT INTO intake(stable_id, payload_json) VALUES (?, ?)",
            [
                (stable_id, canonical_json_bytes(payload).decode("utf-8"))
                for stable_id, payload in sorted(intake_rows.items())
            ],
        )
        for record in model_records:
            connection.execute(
                "INSERT INTO model_revisions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["record_revision"],
                    record["stable_id"],
                    record["record_seq"],
                    record["parent_revision"],
                    record["status"]["code"],
                    canonical_json_bytes(record).decode("utf-8"),
                ),
            )
        for stable_id, record in sorted(current.items()):
            connection.execute(
                "INSERT INTO current_models VALUES (?, ?, ?, ?, ?, ?)",
                (
                    stable_id,
                    record["record_revision"],
                    record["record_seq"],
                    record["status"]["code"],
                    record["authored_metadata_state"],
                    canonical_json_bytes(record).decode("utf-8"),
                ),
            )
        for attempt in attempt_records:
            connection.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    attempt["attempt_id"],
                    attempt["stable_id"],
                    attempt["ledger_seq"],
                    attempt["stage"],
                    attempt["result"],
                    canonical_json_bytes(attempt).decode("utf-8"),
                ),
            )
        for gate in gate_records:
            connection.execute(
                "INSERT INTO gates VALUES (?, ?, ?, ?)",
                (
                    gate["gate_id"],
                    gate["ledger_seq"],
                    gate["gate_kind"],
                    canonical_json_bytes(gate).decode("utf-8"),
                ),
            )
            for item in gate["items"]:
                connection.execute(
                    "INSERT INTO gate_items VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        gate["gate_id"],
                        item["stable_id"],
                        item["work_id"],
                        item["verdict"],
                        item["vet_identity"],
                        item["fidelity_identity"],
                    ),
                )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return summary


def load_current_records(database_path: Union[str, Path]) -> dict[str, JsonObject]:
    """Load materialized current model records from rebuilt SQLite state.

    Parameters
    ----------
    database_path:
        Rebuilt SQLite database.

    Returns
    -------
    dict[str, dict[str, Any]]
        Stable-ID keyed current records.
    """

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT stable_id, payload_json FROM current_models ORDER BY stable_id"
        ).fetchall()
    finally:
        connection.close()
    return {stable_id: json.loads(payload_json) for stable_id, payload_json in rows}
