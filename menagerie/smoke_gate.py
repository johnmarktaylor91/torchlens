"""Automated gate for menagerie smoke-test artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

from menagerie.cluster_runner import GIANT_REGISTRY
from menagerie.generate_menagerie import visual_mode_from_options
from menagerie.smoke_test import snapshot_verification_content
from menagerie.validate_menagerie import MANIFEST_STATUS_VALUES


def _fail(message: str) -> None:
    """Raise a gate failure.

    Parameters
    ----------
    message:
        Failure detail.
    """

    raise RuntimeError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows.

    Parameters
    ----------
    path:
        JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows.
    """

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file.

    Parameters
    ----------
    path:
        TSV path.

    Returns
    -------
    list[dict[str, str]]
        Parsed rows.
    """

    if not path.exists():
        _fail(f"missing TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _assert_validation_manifest(cases: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Assert validation manifest coverage and statuses.

    Parameters
    ----------
    cases:
        Smoke cases.
    out_dir:
        Smoke output directory.
    """

    rows = _read_tsv(out_dir / "validation" / "validation_manifest.tsv")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        stable_id = row.get("stable_id", "")
        if stable_id in by_id:
            _fail(f"{stable_id} | duplicate validation-manifest row")
        by_id[stable_id] = row
        status = row.get("status", "")
        if status not in MANIFEST_STATUS_VALUES:
            _fail(f"{stable_id} | invalid manifest status | {status}")
    expected_ids = {str(case["stable_id"]) for case in cases}
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        _fail(
            f"validation coverage mismatch | expected={sorted(expected_ids)} actual={sorted(actual_ids)}"
        )
    for case in cases:
        stable_id = str(case["stable_id"])
        expected = str(case["expected_status"])
        actual = by_id[stable_id]["status"]
        if actual != expected:
            _fail(f"{stable_id} | expected status={expected} | actual={actual}")


def _assert_smoke_ledger(cases: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Assert one current ledger row per smoke case.

    Parameters
    ----------
    cases:
        Smoke cases.
    out_dir:
        Smoke output directory.
    """

    ledger_db = out_dir / "ledger" / "verification_smoke.db"
    if not ledger_db.exists():
        _fail(f"missing smoke ledger | {ledger_db}")
    with sqlite3.connect(ledger_db) as connection:
        for case in cases:
            stable_id = str(case["stable_id"])
            count = connection.execute(
                "SELECT COUNT(*) FROM current_verification WHERE stable_id = ?",
                (stable_id,),
            ).fetchone()[0]
            if int(count) != 1:
                _fail(f"{stable_id} | smoke ledger current row count | actual={count}")


def _assert_production_unchanged(
    cases: Sequence[dict[str, Any]], out_dir: Path, production_verification_db: Path
) -> None:
    """Assert the smoke added or modified NO rows in production.

    The production-isolation invariant is logical, not physical: the smoke runs
    against an isolated ledger (``TORCHLENS_MENAGERIE_VERIFICATION_DB`` repointed
    to a per-run db), so it must touch ZERO rows in the production
    ``verification_runs`` table. The check compares a CONTENT fingerprint of that
    table -- ``COUNT(*)``, ``MAX(rowid)``, and a full-row hash -- captured before
    the smoke against the same fingerprint now: they must be byte-for-byte
    IDENTICAL.

    Why this is complete given the append-only ledger
    -------------------------------------------------
    ``verification_runs`` is append-only: ``menagerie.ledger`` installs
    ``verification_runs_no_update`` / ``verification_runs_no_delete`` triggers
    that ``RAISE(ABORT)`` on any UPDATE or DELETE, so the only mutation the
    schema permits is an INSERT (an append). An INSERT strictly raises both
    ``COUNT(*)`` and ``MAX(rowid)`` (the newest row gets the largest rowid). So
    "``runs_count`` unchanged AND ``max_rowid`` unchanged" PROVABLY means no row
    was appended, and -- because UPDATE/DELETE are impossible -- no row was
    modified or removed either. The full-row ``content_sha256`` is
    defense-in-depth: it changes on any field change, so the check no longer even
    relies on trusting the triggers. We additionally re-assert the original
    ZERO-fresh-smoke-rows query (any production row for a smoke stable_id stamped
    at/after the smoke start) as a direct, independent witness.

    Why this is immune to the WAL false positive
    --------------------------------------------
    The earlier check compared the file's ``sha256``/``mtime_ns``/``size``. The
    smoke only READS the production ledger, but a SQLite WAL-mode checkpoint
    rewrites the file bytes (and mtime/size) with NO logical content change, so
    the byte fingerprint drifted and tripped a false "changed". A content
    fingerprint is invariant under WAL churn (and any pure-bytes perturbation)
    while still changing on real row mutation -- a CORRECTNESS strengthening, not
    a loosening.

    Parameters
    ----------
    cases:
        Smoke cases.
    out_dir:
        Smoke output directory.
    production_verification_db:
        Production verification ledger.
    """

    snapshot_path = out_dir / "production_snapshot_before.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    before = payload["verification_db"]
    # Use the SAME canonical fingerprint function the producer used (never
    # forked), so before/after are computed identically.
    current = snapshot_verification_content(production_verification_db)
    for key in ("exists", "runs_count", "max_rowid", "content_sha256"):
        if before.get(key) != current.get(key):
            _fail(
                f"production verification.db changed | key={key} "
                f"before={before.get(key)!r} after={current.get(key)!r}"
            )
    if production_verification_db.exists():
        stable_ids = [str(case["stable_id"]) for case in cases]
        placeholders = ",".join("?" for _ in stable_ids)
        with sqlite3.connect(production_verification_db) as connection:
            count = connection.execute(
                f"SELECT COUNT(*) FROM verification_runs WHERE stable_id IN ({placeholders}) "
                "AND started_at >= ?",
                (*stable_ids, payload["smoke_start"]),
            ).fetchone()[0]
        if int(count) != 0:
            _fail(f"production ledger contains fresh smoke rows | count={count}")


def _assert_visuals(cases: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Assert expected SVGs are sane.

    Parameters
    ----------
    cases:
        Smoke cases.
    out_dir:
        Smoke output directory.
    """

    rows = _read_tsv(out_dir / "visuals" / "manifest.tsv")
    rendered = {row["stable_id"]: row for row in rows if row.get("status") == "rendered"}
    root = (out_dir / "visuals").resolve()
    for case in cases:
        stable_id = str(case["stable_id"])
        if case.get("expected_status") != "validated" or not bool(case.get("render", True)):
            if stable_id in rendered:
                _fail(f"{stable_id} | unexpected rendered visual")
            continue
        row = rendered.get(stable_id)
        if row is None:
            _fail(f"{stable_id} | missing rendered visual")
        render_path = Path(row["render_path"]).resolve()
        if not render_path.is_relative_to(root):
            _fail(f"{stable_id} | render path escaped | {render_path}")
        if render_path.suffix != ".svg" or render_path.stat().st_size <= 2048:
            _fail(f"{stable_id} | invalid SVG size/path | {render_path}")
        text = render_path.read_text(encoding="utf-8", errors="ignore")
        if "<svg" not in text:
            _fail(f"{stable_id} | SVG missing <svg | {render_path}")
        expected_visual_mode = str(
            case.get("expected_visual_mode")
            or visual_mode_from_options(
                case.get("vis_option") if isinstance(case.get("vis_option"), dict) else None
            )
        )
        actual_visual_mode = str(row.get("visual_mode") or "default")
        if actual_visual_mode != expected_visual_mode:
            _fail(
                f"{stable_id} | visual mode mismatch "
                f"expected={expected_visual_mode!r} actual={actual_visual_mode!r}"
            )


def _assert_metadata_and_csv(cases: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Assert trace-summary and CSV smoke-only alignment.

    Parameters
    ----------
    cases:
        Smoke cases.
    out_dir:
        Smoke output directory.
    """

    metadata_ids = {
        str(case["stable_id"])
        for case in cases
        if case.get("expected_status") == "validated" and bool(case.get("metadata", True))
    }
    trace_db = out_dir / "metadata" / "trace_summary.db"
    if trace_db.exists() and metadata_ids:
        with sqlite3.connect(trace_db) as connection:
            rows = connection.execute("SELECT stable_id FROM trace_summaries").fetchall()
        actual = {str(row[0]) for row in rows}
        if not actual.issubset(metadata_ids):
            _fail(f"trace_summary has non-smoke rows | actual={sorted(actual)}")
    csv_path = out_dir / "metadata" / "csv" / "menagerie.csv"
    rows = _read_tsv(csv_path) if csv_path.suffix == ".tsv" else _read_csv(csv_path)
    actual_csv_ids = {row["stable_id"] for row in rows}
    expected_csv_ids = {str(case["stable_id"]) for case in cases}
    if actual_csv_ids != expected_csv_ids:
        _fail(
            f"CSV stable-id mismatch | expected={sorted(expected_csv_ids)} actual={sorted(actual_csv_ids)}"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file.

    Parameters
    ----------
    path:
        CSV path.

    Returns
    -------
    list[dict[str, str]]
        Parsed rows.
    """

    if not path.exists():
        _fail(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_timings(out_dir: Path) -> None:
    """Assert run-all timings are positive.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    report = json.loads((out_dir / "RUN_ALL_REPORT.json").read_text(encoding="utf-8"))
    for timing in report.get("timings", []):
        if not timing.get("skipped") and float(timing.get("elapsed_sec", 0.0)) <= 0.0:
            _fail(f"nonpositive timing | {timing.get('name')}")


def _assert_cluster(cases: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Assert cluster proof rows when cluster cases are present.

    The cluster-routed expectation must follow the TRUE routing policy
    (cluster-only-when-required): a case is expected remote IFF the model
    genuinely force-routes to the shared cluster (``force_cluster=True`` in the
    static giant registry, the only preemptive remote route). A case tagged
    ``expected_runner="cluster"`` for a model that actually fits locally
    (``force_cluster=False``) is a STALE manifest expectation -- it demands a
    remote dispatch that the policy forbids -- so the gate fails loudly rather
    than silently. When no genuinely-forced giant is selected (the default smoke
    under cluster-only-when-required), ``cluster_cases`` is empty and zero
    cluster rows is the correct, passing outcome.

    Parameters
    ----------
    cases:
        Smoke cases.
    out_dir:
        Smoke output directory.
    """

    cluster_cases = [case for case in cases if case.get("expected_runner") == "cluster"]
    for case in cluster_cases:
        stable_id = str(case["stable_id"])
        entry = GIANT_REGISTRY.get(stable_id)
        if entry is None or not entry.force_cluster:
            _fail(
                f"{stable_id} | stale cluster expectation: model is not a "
                f"force_cluster giant and routes LOCAL under "
                f"cluster-only-when-required (force_cluster="
                f"{None if entry is None else entry.force_cluster}); "
                "expected_runner must follow the real routing decision"
            )
    if not cluster_cases:
        return
    ledger_db = out_dir / "ledger" / "verification_smoke.db"
    local_host = socket.gethostname()
    with sqlite3.connect(ledger_db) as connection:
        for case in cluster_cases:
            stable_id = str(case["stable_id"])
            row = connection.execute(
                "SELECT runner_host, finished_at FROM current_verification WHERE stable_id = ?",
                (stable_id,),
            ).fetchone()
            if row is None:
                _fail(f"{stable_id} | missing cluster ledger row")
            if str(row[0]) in {"", local_host}:
                _fail(f"{stable_id} | cluster runner_host not remote | actual={row[0]}")
        import_count = connection.execute("SELECT COUNT(*) FROM cluster_result_imports").fetchone()
        if import_count is None or int(import_count[0]) < len(cluster_cases):
            _fail("cluster_result_imports missing cluster proof rows")
    cluster_root = out_dir / "validation" / "cluster"
    if not cluster_root.exists():
        _fail(f"missing cluster artifact root | {cluster_root}")


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-gate parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--production-verification-db", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> None:
    """Run all smoke-gate assertions.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    """

    cases = _read_jsonl(args.smoke_manifest)
    out_dir = args.out_dir.resolve()
    _assert_validation_manifest(cases, out_dir)
    _assert_smoke_ledger(cases, out_dir)
    _assert_cluster(cases, out_dir)
    _assert_visuals(cases, out_dir)
    _assert_metadata_and_csv(cases, out_dir)
    _assert_timings(out_dir)
    _assert_production_unchanged(cases, out_dir, args.production_verification_db)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke-gate CLI."""

    args = build_parser().parse_args(argv)
    try:
        run(args)
    except RuntimeError as error:
        print(f"smoke gate failed: {error}", file=sys.stderr)
        return 1
    print("smoke gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
