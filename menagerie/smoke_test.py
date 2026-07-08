"""Smoke-test orchestration for the TorchLens menagerie pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from menagerie import envs, run_all
from menagerie.catalog import CATALOG_DB, load_rows
from menagerie.ledger import ENV_VERIFICATION_DB, VERIFICATION_DB


DEFAULT_SMOKE_ROOT = Path("/tmp/torchlens_menagerie_smoke")
DEFAULT_SMOKE_MANIFEST = Path(__file__).resolve().parent / "data" / "smoke_manifest.jsonl"
SYNTHETIC_BASE_MODEL_ID = 90_000_000


@dataclass(frozen=True)
class SmokeCase:
    """One smoke-test case.

    Parameters
    ----------
    payload:
        Raw JSON-compatible manifest row.
    """

    payload: dict[str, Any]

    @property
    def stable_id(self) -> str:
        """Return the stable ID for this case."""

        return str(self.payload["stable_id"])

    @property
    def include(self) -> str:
        """Return the inclusion group for this case."""

        return str(self.payload.get("include", "default"))

    @property
    def synthetic(self) -> bool:
        """Return whether this case is synthetic."""

        return bool(self.payload.get("synthetic", False))


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _sha256(path: Path) -> str | None:
    """Return a file SHA-256 digest, or ``None`` when absent.

    Parameters
    ----------
    path:
        File path.

    Returns
    -------
    str | None
        Hex digest, or ``None``.
    """

    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path(path: Path) -> dict[str, Any]:
    """Return a conservative production-path snapshot.

    Parameters
    ----------
    path:
        Path to snapshot.

    Returns
    -------
    dict[str, Any]
        Snapshot payload.
    """

    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "mtime_ns": None, "size": None}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256(path) if path.is_file() else None,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def snapshot_verification_content(path: Path) -> dict[str, Any]:
    """Return a CONTENT fingerprint of the production verification ledger.

    The smoke gate's production-isolation invariant is logical, not physical:
    *the smoke must add or modify NO rows in production* ``verification_runs``.
    The earlier file-byte fingerprint (``sha256``/``mtime_ns``/``size``) was a
    brittle proxy -- the smoke only READS the production ledger, but SQLite
    WAL-mode checkpoints rewrite the file bytes with NO logical content change,
    so the byte fingerprint drifted intermittently and tripped a false
    "production verification.db changed" (the magic-constant-proxy-fails-flaky
    pattern). This helper fingerprints the table CONTENT instead, so a WAL
    checkpoint (or any pure-bytes perturbation) leaves the fingerprint identical
    while any real row mutation changes it.

    The fingerprint is:

    * ``runs_count`` -- ``COUNT(*)`` of ``verification_runs``; any INSERT raises
      it, so an unchanged count proves no row was added.
    * ``max_rowid`` -- ``MAX(rowid)``; the ledger only ever appends, so the
      newest ``rowid`` strictly increases on any INSERT. Pinning it (together
      with ``runs_count``) catches an append even in the impossible-by-trigger
      case of a compensating delete.
    * ``content_sha256`` -- a SHA-256 over every row (``ORDER BY rowid``), which
      changes if ANY field of ANY row changes. This is defense-in-depth: the
      ledger's append-only UPDATE/DELETE triggers (see
      :mod:`menagerie.ledger`) already forbid in-place edits and deletes, so a
      stable ``runs_count`` + ``max_rowid`` already proves no pollution; the
      content hash makes the check independent of trusting the triggers.

    Cheap by construction: it is a single ordered scan of ``verification_runs``
    (sub-second for tens of thousands of rows) executed once per smoke run.

    Parameters
    ----------
    path:
        Production verification ledger path.

    Returns
    -------
    dict[str, Any]
        ``{"path", "exists", "runs_count", "max_rowid", "content_sha256"}``;
        the three content fields are ``None`` when the ledger is absent.
    """

    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": path.exists(),
            "runs_count": None,
            "max_rowid": None,
            "content_sha256": None,
        }
    # Read-only connection so the fingerprint scan never perturbs the production
    # ledger (and so it works even on a read-only mount).
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        runs_count = int(connection.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0])
        max_rowid_row = connection.execute("SELECT MAX(rowid) FROM verification_runs").fetchone()[0]
        max_rowid = None if max_rowid_row is None else int(max_rowid_row)
        digest = hashlib.sha256()
        for row in connection.execute("SELECT * FROM verification_runs ORDER BY rowid"):
            digest.update(repr(row).encode("utf-8"))
            digest.update(b"\0")
    finally:
        connection.close()
    return {
        "path": str(path),
        "exists": True,
        "runs_count": runs_count,
        "max_rowid": max_rowid,
        "content_sha256": digest.hexdigest(),
    }


def load_cases(path: Path) -> list[SmokeCase]:
    """Load smoke cases from JSONL.

    Parameters
    ----------
    path:
        Smoke manifest path.

    Returns
    -------
    list[SmokeCase]
        Loaded cases.
    """

    cases: list[SmokeCase] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                cases.append(SmokeCase(json.loads(stripped)))
    return cases


def select_cases(
    cases: Sequence[SmokeCase],
    *,
    all_islands: bool,
    with_heavy_giant: bool,
    no_cluster: bool,
) -> list[SmokeCase]:
    """Select cases for a smoke run.

    Parameters
    ----------
    cases:
        Available smoke cases.
    all_islands:
        Include extended island representatives.
    with_heavy_giant:
        Include the opt-in heavy large-graph case.
    no_cluster:
        Drop cluster-routed cases.

    Returns
    -------
    list[SmokeCase]
        Selected cases.
    """

    selected: list[SmokeCase] = []
    for case in cases:
        include = case.include
        if include == "all_islands" and not all_islands:
            continue
        if include == "heavy" and not with_heavy_giant:
            continue
        if include == "cluster" and no_cluster:
            continue
        # Under --no-cluster every case runs with --runner local, so any case
        # that expects a remote cluster runner (a genuine force_cluster giant)
        # would route local and contradict its expectation; drop it rather than
        # fabricate a forbidden remote dispatch.
        if no_cluster and case.payload.get("expected_runner") == "cluster":
            continue
        selected.append(case)
    return selected


def _write_resolved_manifest(cases: Sequence[SmokeCase], path: Path) -> None:
    """Write selected smoke cases as JSONL.

    Parameters
    ----------
    cases:
        Selected cases.
    path:
        Destination manifest path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.payload, sort_keys=True) + "\n")


def _copy_catalog(catalog_db: Path, destination: Path) -> None:
    """Copy the catalog DB into the smoke sandbox.

    Parameters
    ----------
    catalog_db:
        Source catalog DB.
    destination:
        Destination catalog DB.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(catalog_db, destination)


def _insert_synthetic_rows(catalog_db: Path, cases: Sequence[SmokeCase]) -> None:
    """Insert synthetic smoke rows into the copied catalog.

    Parameters
    ----------
    catalog_db:
        Smoke catalog DB.
    cases:
        Selected smoke cases.
    """

    synthetic_cases = [case for case in cases if case.synthetic]
    if not synthetic_cases:
        return
    with sqlite3.connect(catalog_db) as connection:
        max_model_id = connection.execute(
            "SELECT COALESCE(MAX(model_id), 0) FROM models"
        ).fetchone()[0]
        rows = []
        for index, case in enumerate(synthetic_cases, start=1):
            model_id = max(SYNTHETIC_BASE_MODEL_ID, int(max_model_id) + 1) + index
            rows.append(
                (
                    model_id,
                    model_id,
                    case.stable_id,
                    case.stable_id,
                    "",
                    "smoke_synthetic",
                    "smoke_synthetic",
                    "synthetic",
                    "smoke",
                    "smoke_synthetic",
                    "(1, 4)",
                    "float32",
                    "2026",
                    1,
                    f"synthetic smoke case {case.payload.get('synth_kind', '')}",
                    "smoke",
                    f"smoke-{case.stable_id}",
                    1,
                    "forward_required",
                    0,
                )
            )
        connection.executemany(
            "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _snapshot_production(out_dir: Path, smoke_start: str, stable_ids: Sequence[str]) -> None:
    """Write production-isolation snapshots for the gate.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    smoke_start:
        Smoke start ISO timestamp.
    stable_ids:
        Smoke stable IDs.
    """

    data_dir = Path(__file__).resolve().parent / "data"
    payload = {
        "smoke_start": smoke_start,
        "stable_ids": list(stable_ids),
        # CONTENT fingerprint, not file bytes: the smoke only READS this ledger,
        # and SQLite WAL checkpoints perturb the file bytes with no logical
        # change, so a byte fingerprint produced intermittent false "changed"
        # gate failures. The content fingerprint pins COUNT(*)/MAX(rowid)/a
        # full-row hash instead, which is invariant under WAL churn but changes
        # on any real row mutation.
        "verification_db": snapshot_verification_content(VERIFICATION_DB),
        "trace_summary_db": _snapshot_path(data_dir / "trace_summary.db"),
        "catalog_db": _snapshot_path(CATALOG_DB),
        "locks_dir": _snapshot_path(Path(__file__).resolve().parent / "locks"),
    }
    (out_dir / "production_snapshot_before.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _torch_runtime_snapshot() -> dict[str, Any]:
    """Return deterministic-algorithm and thread runtime state.

    Returns
    -------
    dict[str, Any]
        Runtime state used by the smoke gate to detect global leakage.
    """

    try:
        import torch
    except ModuleNotFoundError:
        return {"torch_available": False}
    return {
        "torch_available": True,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "num_threads": int(torch.get_num_threads()),
    }


def _write_runtime_snapshot(out_dir: Path, name: str) -> None:
    """Write one smoke runtime snapshot.

    Parameters
    ----------
    out_dir:
        Smoke output directory.
    name:
        Snapshot name suffix.
    """

    (out_dir / f"runtime_snapshot_{name}.json").write_text(
        json.dumps(_torch_runtime_snapshot(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preflight_assignments(cases: Sequence[SmokeCase], catalog_db: Path) -> None:
    """Assert live env assignment matches the smoke manifest.

    Parameters
    ----------
    cases:
        Selected cases.
    catalog_db:
        Smoke catalog DB.
    """

    rows = load_rows(db_path=catalog_db)
    by_id = {row.stable_id: row for row in rows}
    selected_rows = [by_id[case.stable_id] for case in cases]
    assignments = envs.assign(selected_rows)
    mismatches = [
        (case.stable_id, case.payload["expected_env"], assignments.get(case.stable_id))
        for case in cases
        if assignments.get(case.stable_id) != case.payload["expected_env"]
    ]
    if mismatches:
        stable_id, expected, actual = mismatches[0]
        raise RuntimeError(
            f"env preflight mismatch: {stable_id} expected={expected} actual={actual}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-test CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-manifest", type=Path, default=DEFAULT_SMOKE_MANIFEST)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--catalog-db", type=Path, default=CATALOG_DB)
    parser.add_argument("--all-islands", action="store_true")
    parser.add_argument("--with-heavy-giant", action="store_true")
    parser.add_argument("--no-cluster", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--file-format", default="svg")
    parser.add_argument("--skip-gate", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    """Run the smoke pipeline and gate.

    Parameters
    ----------
    args:
        Parsed smoke-test arguments.

    Returns
    -------
    pathlib.Path
        Smoke output directory.
    """

    out_dir = (args.out_dir or (args.out_root / _timestamp())).resolve()
    if out_dir.exists() and not args.force:
        raise RuntimeError(f"smoke out-dir already exists: {out_dir}")
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(
        load_cases(args.smoke_manifest),
        all_islands=args.all_islands,
        with_heavy_giant=args.with_heavy_giant,
        no_cluster=args.no_cluster,
    )
    stable_ids = [case.stable_id for case in cases]
    resolved_manifest = out_dir / "smoke_manifest.resolved.jsonl"
    smoke_catalog = out_dir / "smoke_catalog.db"
    smoke_ledger = out_dir / "ledger" / "verification_smoke.db"
    _write_resolved_manifest(cases, resolved_manifest)
    _copy_catalog(args.catalog_db, smoke_catalog)
    _insert_synthetic_rows(smoke_catalog, cases)
    _preflight_assignments(cases, smoke_catalog)
    smoke_start = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _snapshot_production(out_dir, smoke_start, stable_ids)
    env_keys = (
        ENV_VERIFICATION_DB,
        "TORCHLENS_MENAGERIE_SMOKE_LOCKS_READONLY",
        "TORCHLENS_MENAGERIE_ENV_CACHE_ROOT",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    previous_env = {key: os.environ.get(key) for key in env_keys}
    _write_runtime_snapshot(out_dir, "before")
    try:
        os.environ[ENV_VERIFICATION_DB] = str(smoke_ledger)
        os.environ["TORCHLENS_MENAGERIE_SMOKE_LOCKS_READONLY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Honor a caller-provided warm pixi-env cache (faster reruns); default to a
        # per-run cache under the smoke out-dir for full isolation.
        os.environ.setdefault("TORCHLENS_MENAGERIE_ENV_CACHE_ROOT", str(out_dir / "envs_cache"))
        run_args = [
            "--smoke",
            "--out-dir",
            str(out_dir),
            "--force",
            "--db",
            str(smoke_catalog),
            "--verification-db",
            str(smoke_ledger),
            "--smoke-manifest",
            str(resolved_manifest),
            "--stable-ids",
            *stable_ids,
            "--runner",
            "local" if args.no_cluster else "auto",
            "--jobs",
            str(args.jobs),
            "--file-format",
            args.file_format,
        ]
        exit_code = run_all.main(run_args)
        if exit_code != 0:
            raise RuntimeError(f"run-all smoke failed with exit code {exit_code}")
        if not args.skip_gate:
            from menagerie import smoke_gate

            gate_code = smoke_gate.main(
                [
                    "--out-dir",
                    str(out_dir),
                    "--smoke-manifest",
                    str(resolved_manifest),
                    "--production-verification-db",
                    str(VERIFICATION_DB),
                ]
            )
            if gate_code != 0:
                raise RuntimeError(f"smoke gate failed with exit code {gate_code}")
    finally:
        _write_runtime_snapshot(out_dir, "after")
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke-test CLI."""

    args = build_parser().parse_args(argv)
    try:
        out_dir = run(args)
    except RuntimeError as error:
        print(f"smoke-test failed: {error}", file=sys.stderr)
        return 1
    print(f"smoke-test out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
