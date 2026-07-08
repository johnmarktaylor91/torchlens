"""Automated gate for menagerie smoke-test artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sqlite3
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any, NoReturn, Sequence

from menagerie.cluster_runner import GIANT_REGISTRY
from menagerie.generate_menagerie import visual_mode_from_options
from menagerie.smoke_test import snapshot_verification_content
from menagerie.validate_menagerie import MANIFEST_STATUS_VALUES

KNOWN_JUSTIFIED_MANIFEST_STATUSES = (
    "validated",
    "failed:exception",
    "failed:killed",
    "failed:memory_cap",
    "failed:native_crash",
    "failed:oom",
    "failed:replay",
    "failed:timeout",
    "failed:trace_summary",
    "skipped:cluster_unavailable",
    "skipped:dependency_unavailable",
    "skipped:dry_run",
    "skipped:unsupported_input_recipe",
)
KNOWN_JUSTIFIED_SKIP_STATUSES = tuple(
    status for status in KNOWN_JUSTIFIED_MANIFEST_STATUSES if status.startswith("skipped:")
)
HIGH_SIGKILL_ESTIMATE_MB = 12 * 1024


def _fail(message: str) -> NoReturn:
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


def _assert_no_env_smoke_masking(out_dir: Path) -> None:
    """Assert NULL-forward rows cannot mask validated forward rows.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    ledger_db = out_dir / "ledger" / "verification_smoke.db"
    if not ledger_db.exists():
        _fail(f"missing smoke ledger | {ledger_db}")
    rows = _read_tsv(out_dir / "validation" / "validation_manifest.tsv")
    expected_validated = {row["stable_id"] for row in rows if row.get("status") == "validated"}
    with sqlite3.connect(ledger_db) as connection:
        current_rows = connection.execute(
            "SELECT stable_id, status, forward_pass FROM current_verification"
        ).fetchall()
        forward_pass_count = connection.execute(
            "SELECT COUNT(DISTINCT stable_id) FROM current_verification WHERE forward_pass = 1"
        ).fetchone()[0]
    masked = [
        stable_id
        for stable_id, status, forward_pass in current_rows
        if stable_id in expected_validated and (status != "passed" or forward_pass != 1)
    ]
    if masked:
        _fail(f"validated rows masked by non-forward current rows | stable_ids={masked}")
    if int(forward_pass_count) != len(expected_validated):
        _fail(
            "validated headline mismatch | "
            f"manifest_validated={len(expected_validated)} forward_pass_count={forward_pass_count}"
        )


def _assert_skip_taxonomy_closed(
    statuses: Collection[str] = MANIFEST_STATUS_VALUES,
    *,
    known_statuses: Collection[str] = KNOWN_JUSTIFIED_MANIFEST_STATUSES,
) -> None:
    """Assert every skip/fail manifest status is in the closed taxonomy.

    Parameters
    ----------
    statuses:
        Manifest statuses to inspect.
    known_statuses:
        Frozen status taxonomy.
    """

    relevant = {status for status in statuses if status.startswith(("failed:", "skipped:"))}
    known = {status for status in known_statuses if status.startswith(("failed:", "skipped:"))}
    if relevant != known:
        _fail(
            "manifest skip/fail taxonomy drift | "
            f"missing={sorted(known - relevant)} unexpected={sorted(relevant - known)}"
        )


def _assert_sigkill_surfaced(out_dir: Path) -> None:
    """Assert killed workers are surfaced as failures with a high memory floor.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    ledger_db = out_dir / "ledger" / "verification_smoke.db"
    if not ledger_db.exists():
        _fail(f"missing smoke ledger | {ledger_db}")
    with sqlite3.connect(ledger_db) as connection:
        killed_rows = connection.execute(
            "SELECT stable_id, status, peak_rss_mb FROM current_verification "
            "WHERE status IN ('killed', 'oom', 'native_crash') "
            "OR error_class IN ('failed:killed', 'failed:oom', 'failed:native_crash')"
        ).fetchall()
    for stable_id, _status, peak_rss_mb in killed_rows:
        estimate_mb = HIGH_SIGKILL_ESTIMATE_MB if peak_rss_mb is None else int(peak_rss_mb)
        if estimate_mb <= 4 * 1024:
            _fail(f"{stable_id} | SIGKILL scheduler estimate did not use high floor")


def _assert_plain_placeholder_tripwire(out_dir: Path) -> None:
    """Assert synthesized funcless placeholders fail during plain capture.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    rows = _read_tsv(out_dir / "validation" / "validation_manifest.tsv")
    for row in rows:
        stable_id = row.get("stable_id", "")
        if stable_id == "smoke_plain_placeholder_1":
            status = row.get("status", "")
            error = row.get("error", "")
            if not status.startswith("failed:") or "placeholder" not in error.casefold():
                _fail(f"{stable_id} | plain placeholder did not fail the tripwire")
            return


def _assert_render_writes_no_verification_rows(out_dir: Path) -> None:
    """Assert render artifacts did not append verification rows.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    report = json.loads((out_dir / "RUN_ALL_REPORT.json").read_text(encoding="utf-8"))
    validation_total = int(report.get("validation", {}).get("total", 0))
    ledger_db = out_dir / "ledger" / "verification_smoke.db"
    with sqlite3.connect(ledger_db) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0])
    if count != validation_total:
        _fail(
            "render/metadata wrote verification rows | "
            f"validation_total={validation_total} verification_rows={count}"
        )


def _assert_snapshots_unchanged(out_dir: Path) -> None:
    """Assert non-ledger production snapshots did not change.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    before = json.loads((out_dir / "production_snapshot_before.json").read_text(encoding="utf-8"))
    data_dir = Path(__file__).resolve().parent / "data"
    current = {
        "trace_summary_db": _path_snapshot(data_dir / "trace_summary.db"),
        "catalog_db": _path_snapshot(Path(before["catalog_db"]["path"])),
        "locks_dir": _path_snapshot(Path(__file__).resolve().parent / "locks"),
    }
    for key, snapshot in current.items():
        if before.get(key) != snapshot:
            _fail(f"production snapshot changed | key={key}")


def _path_snapshot(path: Path) -> dict[str, Any]:
    """Return the same path snapshot shape produced by smoke_test.

    Parameters
    ----------
    path:
        Path to snapshot.

    Returns
    -------
    dict[str, Any]
        Snapshot fields.
    """

    import hashlib

    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "mtime_ns": None, "size": None}
    digest: str | None = None
    if path.is_file():
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _assert_append_only_triggers_present(out_dir: Path) -> None:
    """Assert smoke ledger has append-only protection triggers.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    ledger_db = out_dir / "ledger" / "verification_smoke.db"
    with sqlite3.connect(ledger_db) as connection:
        triggers = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
    required = {"verification_runs_no_update", "verification_runs_no_delete"}
    if not required.issubset(triggers):
        _fail(f"append-only triggers missing | missing={sorted(required - triggers)}")


def _assert_no_stray_candidate(data_dir: Path | None = None) -> None:
    """Assert production data has no stray candidate migration files.

    Parameters
    ----------
    data_dir:
        Menagerie data directory. Defaults to production data.
    """

    root = data_dir or Path(__file__).resolve().parent / "data"
    candidates = sorted(path for path in root.rglob("*.candidate") if path.is_file())
    if candidates:
        _fail(f"stray candidate files found | paths={[str(path) for path in candidates]}")


def _assert_offline_isolation() -> None:
    """Assert smoke ran with model-download offline mode enabled."""

    missing = [
        key for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") if os.environ.get(key) != "1"
    ]
    if missing:
        _fail(f"offline isolation missing | env={missing}")


def _assert_det_algo_restored(out_dir: Path) -> None:
    """Assert deterministic-algorithm and thread globals were restored.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    before_path = out_dir / "runtime_snapshot_before.json"
    after_path = out_dir / "runtime_snapshot_after.json"
    if not before_path.exists() or not after_path.exists():
        _fail("missing runtime snapshots")
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    if before != after:
        _fail(f"runtime globals changed | before={before} after={after}")


def _assert_run_health_signal(out_dir: Path) -> None:
    """Assert run-all emitted a health block and surfaced degradations.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    """

    report = json.loads((out_dir / "RUN_ALL_REPORT.json").read_text(encoding="utf-8"))
    health = report.get("health")
    if not isinstance(health, dict):
        _fail("RUN_ALL_REPORT missing health block")
    if int(health.get("unexpected_failures", 0)) and health.get("ok") is True:
        _fail("RUN_ALL_REPORT health is ok despite unexpected validation failures")


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
    _assert_no_env_smoke_masking(out_dir)
    _assert_skip_taxonomy_closed()
    _assert_sigkill_surfaced(out_dir)
    _assert_plain_placeholder_tripwire(out_dir)
    _assert_cluster(cases, out_dir)
    _assert_visuals(cases, out_dir)
    _assert_metadata_and_csv(cases, out_dir)
    _assert_timings(out_dir)
    _assert_render_writes_no_verification_rows(out_dir)
    _assert_snapshots_unchanged(out_dir)
    _assert_append_only_triggers_present(out_dir)
    _assert_no_stray_candidate()
    _assert_offline_isolation()
    _assert_det_algo_restored(out_dir)
    _assert_run_health_signal(out_dir)
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
