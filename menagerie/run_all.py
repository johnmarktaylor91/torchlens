"""One-command orchestrator for honest menagerie validation artifacts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

from menagerie.ledger import ENV_VERIFICATION_DB


DEFAULT_OUT_ROOT = Path("/tmp/torchlens_menagerie_run_all")
REPORT_JSON = "RUN_ALL_REPORT.json"
REPORT_MD = "RUN_ALL_REPORT.md"
VALIDATION_DIR = "validation"
VISUALS_DIR = "visuals"
METADATA_DIR = "metadata"
CSV_DIR = "csv"


@dataclass(frozen=True)
class StepTiming:
    """Wall-clock timing for one orchestration step.

    Parameters
    ----------
    name:
        Stable step name.
    started_at:
        UTC ISO timestamp for step start.
    finished_at:
        UTC ISO timestamp for step end.
    elapsed_sec:
        Elapsed wall-clock seconds.
    skipped:
        Whether existing output caused the step to be skipped.
    """

    name: str
    started_at: str
    finished_at: str
    elapsed_sec: float
    skipped: bool


@dataclass(frozen=True)
class CombinedReport:
    """Combined run-all report.

    Parameters
    ----------
    out_dir:
        Run output directory.
    validation:
        Validation completeness numbers.
    render_count:
        Number of rendered visual artifacts recorded in the render manifest.
    metadata_row_count:
        Number of trace-summary rows exported.
    timings:
        Per-step timings.
    total_elapsed_sec:
        Total run-all elapsed wall-clock seconds.
    csv_export:
        CSV export status note.
    """

    out_dir: str
    validation: dict[str, int]
    render_count: int
    metadata_row_count: int
    timings: list[StepTiming]
    total_elapsed_sec: float
    csv_export: str


def _utc_now() -> datetime:
    """Return the current UTC time.

    Returns
    -------
    datetime
        Timezone-aware UTC timestamp.
    """

    return datetime.now(UTC)


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp.

    Returns
    -------
    str
        Timestamp formatted as ``YYYYmmddTHHMMSSZ``.
    """

    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _iso_timestamp(moment: datetime) -> str:
    """Return an ISO timestamp with UTC suffix.

    Parameters
    ----------
    moment:
        UTC datetime.

    Returns
    -------
    str
        ISO-8601 timestamp.
    """

    return moment.isoformat().replace("+00:00", "Z")


def _log(message: str) -> None:
    """Print one orchestration log line.

    Parameters
    ----------
    message:
        Human-readable log message.
    """

    print(f"[menagerie run-all] {message}", flush=True)


def _unique_timestamped_dir(out_root: Path) -> Path:
    """Return a timestamped run directory that does not already exist.

    Parameters
    ----------
    out_root:
        Parent output directory.

    Returns
    -------
    Path
        Fresh timestamped directory path.
    """

    base = out_root / _timestamp()
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = out_root / f"{base.name}-{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique run directory under {out_root}")


def _timed_step(
    name: str,
    output_exists: Callable[[], bool],
    run_step: Callable[[], None],
    *,
    force: bool,
) -> StepTiming:
    """Run or skip one timed orchestration step.

    Parameters
    ----------
    name:
        Step name for logs and report output.
    output_exists:
        Predicate returning whether the step output already exists.
    run_step:
        Step callable.
    force:
        Whether to run even if output exists.

    Returns
    -------
    StepTiming
        Captured timing.
    """

    started = _utc_now()
    monotonic_start = time.perf_counter()
    if output_exists() and not force:
        _log(f"{name} skip: output already exists")
        finished = _utc_now()
        return StepTiming(
            name=name,
            started_at=_iso_timestamp(started),
            finished_at=_iso_timestamp(finished),
            elapsed_sec=round(time.perf_counter() - monotonic_start, 6),
            skipped=True,
        )
    _log(f"{name} start: {_iso_timestamp(started)}")
    run_step()
    finished = _utc_now()
    elapsed = time.perf_counter() - monotonic_start
    _log(f"{name} done: {_iso_timestamp(finished)} elapsed={elapsed:.3f}s")
    return StepTiming(
        name=name,
        started_at=_iso_timestamp(started),
        finished_at=_iso_timestamp(finished),
        elapsed_sec=round(elapsed, 6),
        skipped=False,
    )


def _append_option(args: list[str], flag: str, value: Any | None) -> None:
    """Append one optional scalar CLI argument.

    Parameters
    ----------
    args:
        Mutable CLI argument list.
    flag:
        CLI flag name.
    value:
        Optional value to append.
    """

    if value is not None:
        args.extend((flag, str(value)))


def _append_repeated(args: list[str], flag: str, values: Sequence[Any] | None) -> None:
    """Append repeated CLI arguments.

    Parameters
    ----------
    args:
        Mutable CLI argument list.
    flag:
        CLI flag name.
    values:
        Optional values to append.
    """

    for value in values or ():
        args.extend((flag, str(value)))


def _common_selection_args(args: argparse.Namespace) -> list[str]:
    """Return catalog selection arguments common to validation.

    Parameters
    ----------
    args:
        Parsed run-all arguments.

    Returns
    -------
    list[str]
        CLI arguments.
    """

    forwarded: list[str] = []
    _append_option(forwarded, "--subset", args.subset)
    _append_option(forwarded, "--family", args.family)
    _append_option(forwarded, "--domain", args.domain)
    _append_option(forwarded, "--zoo", args.zoo)
    _append_repeated(forwarded, "--name", args.name)
    _append_repeated(forwarded, "--model-id", args.model_id)
    _append_option(forwarded, "--since", args.since)
    _append_option(forwarded, "--max-models", args.max_models)
    _append_option(forwarded, "--db", args.db)
    if args.verified_only:
        forwarded.append("--verified-only")
    if args.featured_only:
        forwarded.append("--featured-only")
    return forwarded


def _common_runtime_args(args: argparse.Namespace) -> list[str]:
    """Return runtime arguments shared by validation and rendering.

    Parameters
    ----------
    args:
        Parsed run-all arguments.

    Returns
    -------
    list[str]
        CLI arguments.
    """

    forwarded: list[str] = []
    _append_option(forwarded, "--jobs", args.jobs)
    _append_option(forwarded, "--gpu-jobs", args.gpu_jobs)
    _append_option(forwarded, "--device", args.device)
    _append_option(forwarded, "--timeout-sec", args.timeout_sec)
    _append_option(forwarded, "--timeout-base-sec", args.timeout_base_sec)
    _append_option(forwarded, "--timeout-scale", args.timeout_scale)
    _append_option(forwarded, "--timeout-ceiling-sec", args.timeout_ceiling_sec)
    _append_option(forwarded, "--install-timeout", args.install_timeout)
    _append_option(forwarded, "--min-free-gb", args.min_free_gb)
    _append_repeated(forwarded, "--pip-args", args.pip_args)
    if args.keep_cache:
        forwarded.append("--keep-cache")
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.install_deps is True:
        forwarded.append("--install-deps")
    if args.install_deps is False:
        forwarded.append("--no-install-deps")
    return forwarded


def _validation_args(args: argparse.Namespace, validation_dir: Path) -> list[str]:
    """Return arguments for ``menagerie.validate_menagerie.main``.

    Parameters
    ----------
    args:
        Parsed run-all arguments.
    validation_dir:
        Validation output directory.

    Returns
    -------
    list[str]
        CLI argument list.
    """

    forwarded = _common_selection_args(args) + _common_runtime_args(args)
    if args.stable_ids:
        forwarded.append("--stable-ids")
        forwarded.extend(str(value) for value in args.stable_ids)
    _append_option(forwarded, "--scope", args.scope)
    _append_option(forwarded, "--memory-budget-gb", args.memory_budget_gb)
    _append_option(forwarded, "--memory-floor-gb", args.memory_floor_gb)
    _append_option(forwarded, "--worker-memory-cap-gb", args.worker_memory_cap_gb)
    _append_option(forwarded, "--runner", args.runner)
    _append_option(forwarded, "--input-scale", args.input_scale)
    _append_option(forwarded, "--env-registry", args.env_registry)
    _append_option(forwarded, "--verification-db", args.verification_db)
    _append_option(forwarded, "--smoke-manifest", args.smoke_manifest)
    if args.revalidate_failed:
        forwarded.append("--revalidate-failed")
    forwarded.extend(
        (
            "--out-dir",
            str(validation_dir),
            "--manifest",
            str(validation_dir / "validation_manifest.tsv"),
        )
    )
    return forwarded


def _render_args(
    args: argparse.Namespace, visuals_dir: Path, model_ids: Sequence[int]
) -> list[str]:
    """Return arguments for ``menagerie.generate_menagerie.main``.

    Parameters
    ----------
    args:
        Parsed run-all arguments.
    visuals_dir:
        Visuals output directory.
    model_ids:
        Validated model IDs to render.

    Returns
    -------
    list[str]
        CLI argument list.
    """

    forwarded = _common_runtime_args(args)
    _append_option(forwarded, "--db", args.db)
    _append_repeated(forwarded, "--model-id", model_ids)
    _append_option(forwarded, "--file-format", args.file_format)
    _append_repeated(forwarded, "--vis-option", args.vis_option)
    _append_option(forwarded, "--smoke-manifest", args.smoke_manifest)
    if args.force:
        forwarded.append("--force")
    forwarded.extend(
        (
            "--out-dir",
            str(visuals_dir),
            "--manifest",
            str(visuals_dir / "manifest.tsv"),
        )
    )
    return forwarded


def _validation_outputs_exist(validation_dir: Path) -> bool:
    """Return whether validation output exists.

    Parameters
    ----------
    validation_dir:
        Validation output directory.

    Returns
    -------
    bool
        Whether summary and manifest files exist.
    """

    return (validation_dir / "validation_manifest.tsv").exists() and (
        validation_dir / "validation_summary.json"
    ).exists()


def _render_outputs_exist(visuals_dir: Path) -> bool:
    """Return whether visual render output exists.

    Parameters
    ----------
    visuals_dir:
        Visuals output directory.

    Returns
    -------
    bool
        Whether the render manifest exists.
    """

    return (visuals_dir / "manifest.tsv").exists()


def _metadata_outputs_exist(metadata_dir: Path) -> bool:
    """Return whether metadata output exists.

    Parameters
    ----------
    metadata_dir:
        Metadata output directory.

    Returns
    -------
    bool
        Whether the trace-summary database exists.
    """

    return (metadata_dir / "trace_summary.db").exists()


def _read_validation_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read validation manifest rows.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.

    Returns
    -------
    list[dict[str, str]]
        Manifest rows.
    """

    if not manifest_path.exists():
        raise RuntimeError(f"validation manifest is missing: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _validated_rows(validation_dir: Path) -> list[dict[str, str]]:
    """Return validation manifest rows with validated status.

    Parameters
    ----------
    validation_dir:
        Validation output directory.

    Returns
    -------
    list[dict[str, str]]
        Validated manifest rows.
    """

    rows = _read_validation_manifest(validation_dir / "validation_manifest.tsv")
    return [row for row in rows if row.get("status") == "validated"]


def _validated_stable_ids(validation_dir: Path) -> list[str]:
    """Return validated stable IDs from the validation manifest.

    Parameters
    ----------
    validation_dir:
        Validation output directory.

    Returns
    -------
    list[str]
        Stable IDs.
    """

    return [row["stable_id"] for row in _validated_rows(validation_dir) if row.get("stable_id")]


def _manifest_stable_ids(validation_dir: Path) -> list[str]:
    """Return all stable IDs from the validation manifest.

    Parameters
    ----------
    validation_dir:
        Validation output directory.

    Returns
    -------
    list[str]
        Stable IDs in manifest order.
    """

    return [
        row["stable_id"]
        for row in _read_validation_manifest(validation_dir / "validation_manifest.tsv")
        if row.get("stable_id")
    ]


def _read_smoke_manifest(smoke_manifest: Path | None) -> list[dict[str, Any]]:
    """Return smoke manifest rows, or an empty list when absent.

    Parameters
    ----------
    smoke_manifest:
        Optional smoke-manifest JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed smoke manifest rows.
    """

    if smoke_manifest is None or not Path(smoke_manifest).exists():
        return []
    rows: list[dict[str, Any]] = []
    with Path(smoke_manifest).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _smoke_synthetic_stable_ids(smoke_manifest: Path | None) -> set[str]:
    """Return synthetic stable IDs declared in the smoke manifest.

    Synthetic smoke rows exist only to exercise validation status paths; they
    have no real architecture to re-trace or render, so downstream metadata and
    render steps must skip them.

    Parameters
    ----------
    smoke_manifest:
        Optional smoke-manifest JSONL path.

    Returns
    -------
    set[str]
        Synthetic stable IDs.
    """

    return {
        str(row["stable_id"])
        for row in _read_smoke_manifest(smoke_manifest)
        if bool(row.get("synthetic", False)) and row.get("stable_id")
    }


def _smoke_excluded_stable_ids(smoke_manifest: Path | None, field: str) -> set[str]:
    """Return stable IDs the smoke manifest opts out of a downstream step.

    A smoke row is excluded from the render / metadata step when it is
    synthetic, when ``field`` (``"render"`` / ``"metadata"``) is explicitly
    false, or when it is routed to a non-base island. Render and metadata run in
    the base environment and re-instantiate the model, so an island model whose
    dependencies are absent from the base env cannot be re-traced there; its
    in-island validation is the coverage for that case.

    Parameters
    ----------
    smoke_manifest:
        Optional smoke-manifest JSONL path.
    field:
        Manifest field gating the step (``"render"`` or ``"metadata"``).

    Returns
    -------
    set[str]
        Excluded stable IDs.
    """

    excluded: set[str] = set()
    for row in _read_smoke_manifest(smoke_manifest):
        stable_id = row.get("stable_id")
        if not stable_id:
            continue
        non_base_env = str(row.get("expected_env", "base")) != "base"
        if bool(row.get("synthetic", False)) or not bool(row.get(field, True)) or non_base_env:
            excluded.add(str(stable_id))
    return excluded


def _validated_model_ids(validation_dir: Path) -> list[int]:
    """Return validated model IDs from the validation manifest.

    Parameters
    ----------
    validation_dir:
        Validation output directory.

    Returns
    -------
    list[int]
        Model IDs.
    """

    model_ids: list[int] = []
    for row in _validated_rows(validation_dir):
        raw_model_id = row.get("model_id")
        if raw_model_id:
            model_ids.append(int(raw_model_id))
    return model_ids


def _run_validation(args: argparse.Namespace, validation_dir: Path) -> None:
    """Run honest validation through the existing validator entry point.

    Parameters
    ----------
    args:
        Parsed run-all arguments.
    validation_dir:
        Validation output directory.
    """

    from menagerie import validate_menagerie

    exit_code = validate_menagerie.main(_validation_args(args, validation_dir))
    if exit_code != 0:
        raise RuntimeError(f"validation failed with exit code {exit_code}")


def _run_render(args: argparse.Namespace, validation_dir: Path, visuals_dir: Path) -> None:
    """Run visual rendering through the existing renderer entry point.

    Parameters
    ----------
    args:
        Parsed run-all arguments.
    validation_dir:
        Validation output directory.
    visuals_dir:
        Visuals output directory.
    """

    from menagerie import generate_menagerie

    excluded = _smoke_excluded_stable_ids(args.smoke_manifest, "render")
    model_ids = [
        int(row["model_id"])
        for row in _validated_rows(validation_dir)
        if row.get("model_id") and row.get("stable_id") not in excluded
    ]
    if not model_ids:
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (visuals_dir / "manifest.tsv").write_text(
            "\t".join(generate_menagerie.MANIFEST_COLUMNS) + "\n",
            encoding="utf-8",
        )
        _log("render skip: no validated models found")
        return
    exit_code = generate_menagerie.main(_render_args(args, visuals_dir, model_ids))
    if exit_code != 0:
        raise RuntimeError(f"render failed with exit code {exit_code}")


def _csv_export_note(metadata_dir: Path, args: argparse.Namespace) -> str:
    """Run optional CSV export if the module is importable.

    Parameters
    ----------
    metadata_dir:
        Metadata output directory.
    args:
        Parsed run-all arguments.

    Returns
    -------
    str
        CSV export status note.
    """

    try:
        csv_export = importlib.import_module("menagerie.csv_export")
    except ModuleNotFoundError as error:
        if error.name != "menagerie.csv_export":
            raise
        note = "skipped: menagerie.csv_export is not available"
        _log(f"csv_export {note}")
        return note
    csv_dir = metadata_dir / CSV_DIR
    csv_args = [
        "--out-dir",
        str(csv_dir),
        "--trace-summary-db",
        str(metadata_dir / "trace_summary.db"),
    ]
    if args.db is not None:
        csv_args.extend(("--catalog-db", str(args.db)))
    if args.verification_db is not None:
        csv_args.extend(("--verification-db", str(args.verification_db)))
    stable_ids = _manifest_stable_ids(metadata_dir.parent / VALIDATION_DIR)
    if stable_ids:
        csv_args.extend(("--stable-ids", *stable_ids))
    if args.csv_limit is not None:
        csv_args.extend(("--limit", str(args.csv_limit)))
    exit_code = csv_export.main(csv_args)
    if exit_code != 0:
        raise RuntimeError(f"csv_export failed with exit code {exit_code}")
    return "exported"


def _run_metadata(args: argparse.Namespace, validation_dir: Path, metadata_dir: Path) -> str:
    """Run trace-summary metadata export and optional CSV export.

    Parameters
    ----------
    args:
        Parsed run-all arguments.
    validation_dir:
        Validation output directory.
    metadata_dir:
        Metadata output directory.

    Returns
    -------
    str
        CSV export status note.
    """

    from menagerie import trace_summary

    excluded = _smoke_excluded_stable_ids(args.smoke_manifest, "metadata")
    stable_ids = [
        stable_id
        for stable_id in _validated_stable_ids(validation_dir)
        if stable_id not in excluded
    ]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    db_path = metadata_dir / "trace_summary.db"
    if stable_ids:
        trace_args = ["--stable-ids", *stable_ids, "--db", str(db_path)]
        if args.db is not None:
            trace_args.extend(("--catalog-db", str(args.db)))
        exit_code = trace_summary.main(trace_args)
        if exit_code != 0:
            raise RuntimeError(f"trace_summary failed with exit code {exit_code}")
    else:
        trace_summary.ensure_store(db_path)
        _log("trace_summary skip: no validated models found")
    return _csv_export_note(metadata_dir, args)


def _validation_numbers(validation_dir: Path) -> dict[str, int]:
    """Return validation completeness numbers.

    Parameters
    ----------
    validation_dir:
        Validation output directory.

    Returns
    -------
    dict[str, int]
        Totals keyed by validated, failed, skipped, and total.
    """

    summary_path = validation_dir / "validation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {key: int(value) for key, value in summary.get("totals", {}).items()}
    rows = _read_validation_manifest(validation_dir / "validation_manifest.tsv")
    totals = {"validated": 0, "failed": 0, "skipped": 0, "total": len(rows)}
    for row in rows:
        status = row.get("status", "")
        if status == "validated":
            totals["validated"] += 1
        elif status.startswith("skipped:"):
            totals["skipped"] += 1
        else:
            totals["failed"] += 1
    return totals


def _render_count(visuals_dir: Path) -> int:
    """Return rendered row count from the render manifest.

    Parameters
    ----------
    visuals_dir:
        Visuals output directory.

    Returns
    -------
    int
        Rendered row count.
    """

    manifest_path = visuals_dir / "manifest.tsv"
    if not manifest_path.exists():
        return 0
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return sum(
            1 for row in csv.DictReader(handle, delimiter="\t") if row.get("status") == "rendered"
        )


def _metadata_row_count(metadata_dir: Path) -> int:
    """Return trace-summary row count.

    Parameters
    ----------
    metadata_dir:
        Metadata output directory.

    Returns
    -------
    int
        Row count.
    """

    db_path = metadata_dir / "trace_summary.db"
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as connection:
        try:
            result = connection.execute("SELECT COUNT(*) FROM trace_summaries").fetchone()
        except sqlite3.DatabaseError:
            return 0
    return int(result[0]) if result is not None else 0


def _write_report(report: CombinedReport, out_dir: Path) -> None:
    """Write combined JSON and Markdown reports.

    Parameters
    ----------
    report:
        Combined report.
    out_dir:
        Run output directory.
    """

    report_json = {
        **asdict(report),
        "timings": [asdict(timing) for timing in report.timings],
    }
    (out_dir / REPORT_JSON).write_text(json.dumps(report_json, indent=2, sort_keys=True) + "\n")
    lines = [
        "# TorchLens Menagerie Run-All Report",
        "",
        f"- Output directory: {report.out_dir}",
        f"- Validation validated: {report.validation.get('validated', 0)}",
        f"- Validation failed: {report.validation.get('failed', 0)}",
        f"- Validation skipped: {report.validation.get('skipped', 0)}",
        f"- Validation total: {report.validation.get('total', 0)}",
        f"- Rendered visuals: {report.render_count}",
        f"- Metadata rows: {report.metadata_row_count}",
        f"- CSV export: {report.csv_export}",
        f"- Total elapsed seconds: {report.total_elapsed_sec:.3f}",
        "",
        "## Timings",
        "",
        "| Step | Skipped | Elapsed seconds | Started | Finished |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for timing in report.timings:
        lines.append(
            "| "
            f"{timing.name} | {timing.skipped} | {timing.elapsed_sec:.3f} | "
            f"{timing.started_at} | {timing.finished_at} |"
        )
    (out_dir / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_report(report: CombinedReport) -> None:
    """Print the combined report to stdout.

    Parameters
    ----------
    report:
        Combined report.
    """

    print("TorchLens menagerie run-all complete")
    print(f"out_dir: {report.out_dir}")
    print(
        "validation: "
        f"validated={report.validation.get('validated', 0)} "
        f"failed={report.validation.get('failed', 0)} "
        f"skipped={report.validation.get('skipped', 0)} "
        f"total={report.validation.get('total', 0)}"
    )
    print(f"render_count: {report.render_count}")
    print(f"metadata_row_count: {report.metadata_row_count}")
    print(f"csv_export: {report.csv_export}")
    for timing in report.timings:
        print(f"timing.{timing.name}: skipped={timing.skipped} elapsed={timing.elapsed_sec:.3f}s")
    print(f"total_elapsed: {report.total_elapsed_sec:.3f}s")


def build_parser() -> argparse.ArgumentParser:
    """Build the run-all CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="rerun steps even if output exists")
    parser.add_argument("--subset", type=int)
    parser.add_argument("--family")
    parser.add_argument("--domain")
    parser.add_argument("--zoo")
    parser.add_argument("--name", action="append")
    parser.add_argument("--model-id", action="append", type=int)
    parser.add_argument("--stable-ids", nargs="+")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--featured-only", action="store_true")
    parser.add_argument("--since", type=int)
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--gpu-jobs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--timeout-base-sec", type=float)
    parser.add_argument("--timeout-scale", type=float)
    parser.add_argument("--timeout-ceiling-sec", type=float)
    parser.add_argument("--install-timeout", type=float)
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--pip-args", action="append", default=[])
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--install-deps", dest="install_deps", action="store_true", default=None)
    parser.add_argument("--no-install-deps", dest="install_deps", action="store_false")
    parser.add_argument("--scope", choices=("forward", "forward+backward"))
    parser.add_argument("--revalidate-failed", action="store_true")
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--memory-floor-gb", type=float)
    parser.add_argument("--worker-memory-cap-gb", type=float)
    parser.add_argument(
        "--runner",
        choices=("auto", "local", "cluster"),
        default="auto",
        help=(
            "validation runner policy: auto routes RAM giants to the cluster, "
            "local forces legacy local validation, cluster sends non-native-crash rows "
            "to the cluster"
        ),
    )
    parser.add_argument("--input-scale", type=float)
    parser.add_argument("--env-registry", type=Path)
    parser.add_argument("--file-format", default="svg")
    parser.add_argument(
        "--vis-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Trace.draw() keyword argument forwarded to generate_menagerie",
    )
    parser.add_argument("--verification-db", type=Path)
    parser.add_argument("--smoke", action="store_true", help="run with smoke isolation env vars")
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--csv-limit", type=int)
    return parser


def run(args: argparse.Namespace) -> CombinedReport:
    """Run all menagerie artifact steps.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    CombinedReport
        Combined report.
    """

    total_start = time.perf_counter()
    env_keys = (
        ENV_VERIFICATION_DB,
        "TORCHLENS_MENAGERIE_SMOKE_LOCKS_READONLY",
        "TORCHLENS_MENAGERIE_ENV_CACHE_ROOT",
    )
    previous_env = {key: os.environ.get(key) for key in env_keys}
    if args.verification_db is not None:
        os.environ[ENV_VERIFICATION_DB] = str(args.verification_db)
    try:
        out_dir = (args.out_dir or _unique_timestamped_dir(args.out_root)).resolve()
        if args.smoke:
            os.environ.setdefault("TORCHLENS_MENAGERIE_SMOKE_LOCKS_READONLY", "1")
            os.environ.setdefault("TORCHLENS_MENAGERIE_ENV_CACHE_ROOT", str(out_dir / "envs_cache"))
        validation_dir = out_dir / VALIDATION_DIR
        visuals_dir = out_dir / VISUALS_DIR
        metadata_dir = out_dir / METADATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        timings = [
            _timed_step(
                "validation",
                lambda: _validation_outputs_exist(validation_dir),
                lambda: _run_validation(args, validation_dir),
                force=args.force,
            ),
            _timed_step(
                "render",
                lambda: _render_outputs_exist(visuals_dir),
                lambda: _run_render(args, validation_dir, visuals_dir),
                force=args.force,
            ),
        ]
        csv_note = "not run"

        def run_metadata() -> None:
            """Run metadata and record its CSV status."""

            nonlocal csv_note
            csv_note = _run_metadata(args, validation_dir, metadata_dir)

        timings.append(
            _timed_step(
                "metadata",
                lambda: _metadata_outputs_exist(metadata_dir),
                run_metadata,
                force=args.force,
            )
        )
        if timings[-1].skipped:
            csv_note = "skipped: metadata output already exists"
        report = CombinedReport(
            out_dir=str(out_dir),
            validation=_validation_numbers(validation_dir),
            render_count=_render_count(visuals_dir),
            metadata_row_count=_metadata_row_count(metadata_dir),
            timings=timings,
            total_elapsed_sec=round(time.perf_counter() - total_start, 6),
            csv_export=csv_note,
        )
        _write_report(report, out_dir)
        _print_report(report)
        return report
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the run-all CLI.

    Parameters
    ----------
    argv:
        Optional CLI argument vector.

    Returns
    -------
    int
        Process exit status.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except RuntimeError as error:
        print(f"run-all failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
