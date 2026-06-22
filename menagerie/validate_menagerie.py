"""Dependency-aware, disk-safe validator for the TorchLens model menagerie."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ContextManager, Mapping, Sequence, cast

from menagerie.catalog import CatalogRow, load_rows
from menagerie.recipe import (
    build_input_for_row,
    instantiate_model,
)
from menagerie.runtime import (
    CACHE_ROOTS,
    DependencyPlan,
    assert_min_free,
    combine_notes,
    cleanup_runtime,
    cuda_is_available,
    default_jobs,
    dependency_plan,
    device_note,
    disk_free_gb,
    group_by_dependency,
    install_dependency_plan,
    is_device_related_error,
    log_event,
    move_model_and_input_to_device,
    purge_new_cache_entries,
    safe_path_part,
    select_rows,
    snapshot_cache,
    unrenderable_reason,
)
from menagerie.ledger import (
    VerificationRun,
    Status,
    append_verification_run,
    base_env_hash,
    connect as connect_ledger,
    python_version,
    runner_host,
    utc_now,
)


DEFAULT_OUT_DIR = Path("/tmp/torchlens_menagerie_validation")
MANIFEST_COLUMNS = (
    "name",
    "model_id",
    "stable_id",
    "recipe_revision_sha256",
    "status",
    "n_ops",
    "validate_metadata_ok",
    "scope",
    "elapsed",
    "dependency_cluster",
    "error",
    "graph_shape_hash",
)
SUMMARY_JSON = "validation_summary.json"
REPORT_MD = "VALIDATION_REPORT.md"


@dataclass(frozen=True)
class ValidationResult:
    """One model validation result.

    Parameters
    ----------
    name:
        Catalog model name.
    model_id:
        Catalog model identifier.
    status:
        Validation status.
    n_ops:
        Number of traced forward ops, when available.
    validate_metadata_ok:
        Whether forward metadata validation completed cleanly.
    scope:
        Requested validation scope.
    elapsed:
        Elapsed seconds.
    dependency_cluster:
        Dependency cluster used for this row.
    error:
        Error text or skip note.
    graph_shape_hash:
        TorchLens architecture hash for deduplication.
    stable_id:
        Opaque durable model identity.
    recipe_revision_sha256:
        Frozen recipe fingerprint for the row's current construction recipe.
    """

    name: str
    model_id: int
    status: str
    n_ops: int | None
    validate_metadata_ok: bool
    scope: str
    elapsed: float
    dependency_cluster: str
    error: str
    graph_shape_hash: str = ""
    stable_id: str = ""
    recipe_revision_sha256: str = ""


def manifest_records(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Read latest validation manifest rows keyed by stable model identity.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.

    Returns
    -------
    dict[str, dict[str, str]]
        Latest manifest rows.
    """

    if not manifest_path.exists():
        return {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["stable_id"]: row for row in reader if row.get("stable_id")}


def completed_stable_ids(manifest_path: Path, revalidate_failed: bool) -> set[str]:
    """Return stable IDs that should be skipped for resumable validation.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.
    revalidate_failed:
        Whether non-validated rows should be retried.

    Returns
    -------
    set[str]
        Stable IDs to skip.
    """

    records = manifest_records(manifest_path)
    if not revalidate_failed:
        return set(records)
    return {stable_id for stable_id, row in records.items() if row.get("status") == "validated"}


def append_manifest(manifest_path: Path, result: ValidationResult) -> None:
    """Append one result row to the validation manifest.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.
    result:
        Validation result.
    """

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(
            (
                result.name,
                result.model_id,
                result.stable_id,
                result.recipe_revision_sha256,
                result.status,
                "" if result.n_ops is None else result.n_ops,
                str(result.validate_metadata_ok).lower(),
                result.scope,
                f"{result.elapsed:.3f}",
                result.dependency_cluster,
                result.error.replace("\n", " | "),
                result.graph_shape_hash,
            )
        )


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


def _torchlens_version() -> str:
    """Return the current TorchLens version.

    Returns
    -------
    str
        TorchLens package version.
    """

    import torchlens as tl

    return _package_version("torchlens", str(getattr(tl, "__version__", "unknown")))


def _torch_version() -> str:
    """Return the current PyTorch version.

    Returns
    -------
    str
        PyTorch package version.
    """

    import torch

    return str(torch.__version__)


def _ledger_status(status: str) -> str:
    """Map validation manifest status to ledger status.

    Parameters
    ----------
    status:
        Validation manifest status.

    Returns
    -------
    str
        Ledger status.
    """

    if status == "validated":
        return "passed"
    if status == "failed:timeout":
        return "timeout"
    if status.startswith("skipped:"):
        return "skipped"
    if status.startswith("failed:"):
        return "failed"
    return "error"


def _actual_device(result: ValidationResult, requested_device: str) -> str:
    """Infer the actual validation device from the result note.

    Parameters
    ----------
    result:
        Validation result.
    requested_device:
        Requested device mode.

    Returns
    -------
    str
        Actual device string.
    """

    match = re.search(r"(?:^|; )device=([^; |]+)", result.error)
    if match is not None:
        return match.group(1)
    return "cpu" if requested_device in {"cpu", "auto"} else requested_device


def append_validation_ledger(row: CatalogRow, result: ValidationResult, device: str) -> None:
    """Append one validation result to the verification ledger.

    Parameters
    ----------
    row:
        Catalog row.
    result:
        Validation result.
    device:
        Requested validation device.
    """

    ledger_status = _ledger_status(result.status)
    env_hash = os.environ.get("TORCHLENS_MENAGERIE_ENV_HASH") or base_env_hash()
    passed = ledger_status == "passed"
    started_at = utc_now()
    finished_at = utc_now()
    with connect_ledger() as conn:
        append_verification_run(
            conn,
            VerificationRun(
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                name=row.name,
                zoo=row.zoo,
                variant=row.variant,
                scope="forward",
                status=cast(Status, ledger_status),
                forward_pass=1 if passed else (None if ledger_status == "skipped" else 0),
                backward_pass=None,
                backward_na_reason=None,
                metadata_ok=(
                    int(result.validate_metadata_ok)
                    if not result.status.startswith("skipped:")
                    else None
                ),
                n_ops=result.n_ops if passed else None,
                graph_shape_hash=result.graph_shape_hash or None,
                svg_sha256=None,
                torchlens_version=_torchlens_version(),
                torch_version=_torch_version(),
                python_version=python_version(),
                device_requested=device,
                device_actual=_actual_device(result, device),
                env_hash=env_hash,
                runner_host=runner_host(),
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=result.elapsed,
                error_class=None if passed else result.status,
                error_message=None if passed else result.error,
            ),
        )


def result_from_payload(payload: Mapping[str, Any]) -> ValidationResult:
    """Build a validation result from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible result payload.

    Returns
    -------
    ValidationResult
        Parsed validation result.
    """

    raw_n_ops = payload.get("n_ops")
    n_ops = None if raw_n_ops in {None, ""} else int(raw_n_ops)
    return ValidationResult(
        name=str(payload["name"]),
        model_id=int(payload["model_id"]),
        status=str(payload["status"]),
        n_ops=n_ops,
        validate_metadata_ok=bool(payload["validate_metadata_ok"]),
        scope=str(payload["scope"]),
        elapsed=float(payload["elapsed"]),
        dependency_cluster=str(payload["dependency_cluster"]),
        error=str(payload["error"]),
        graph_shape_hash=str(payload.get("graph_shape_hash", "")),
        stable_id=str(payload.get("stable_id", "")),
        recipe_revision_sha256=str(payload.get("recipe_revision_sha256", "")),
    )


def catalog_row_from_payload(payload: Mapping[str, Any]) -> CatalogRow:
    """Build a catalog row from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible row payload.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    return CatalogRow(
        model_id=int(payload["model_id"]),
        display_index=int(payload.get("display_index", payload["model_id"])),
        stable_id=str(payload.get("stable_id", "")),
        name=str(payload["name"]),
        variant=str(payload.get("variant", "")),
        family=str(payload["family"]),
        family_normalized=str(payload["family_normalized"]),
        domain=str(payload["domain"]),
        zoo=str(payload["zoo"]),
        constructor_call=str(payload["constructor_call"]),
        input_shape=str(payload["input_shape"]),
        input_dtype=str(payload["input_dtype"]),
        era=str(payload["era"]),
        verified=bool(payload["verified"]),
        notes=str(payload["notes"]),
        source=str(payload.get("source", "catalog")),
        recipe_revision_sha256=str(payload.get("recipe_revision_sha256", "")),
    )


def _build_input(row: CatalogRow) -> Any:
    """Build the example input for a catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Example input.
    """

    return build_input_for_row(row)


def _sum_float_outputs(output: Any) -> Any:
    """Return a scalar sum over floating tensors in a nested output.

    Parameters
    ----------
    output:
        Model output object.

    Returns
    -------
    Any
        Scalar tensor loss.
    """

    import torch

    terms: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        """Collect floating tensor leaves from nested outputs."""

        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point or value.dtype.is_complex:
                terms.append(value.float().sum())
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(output)
    if not terms:
        raise ValueError("backward validation requires at least one floating tensor output")
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def _trace_n_ops_and_hash(model: Any, input_tensor: Any) -> tuple[int | None, str, str]:
    """Trace a model once to count forward ops and compute its architecture hash.

    Parameters
    ----------
    model:
        Instantiated model.
    input_tensor:
        Example input.

    Returns
    -------
    tuple[int | None, str, str]
        Number of traced ops, graph-shape hash, and error text. ``n_ops`` is
        ``None`` when trace summarization fails.
    """

    import torch
    import torchlens as tl

    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            trace = tl.trace(
                model,
                input_tensor,
                layers_to_save=None,
                save=None,
                save_rng_states=False,
                inference_only=True,
            )
        n_ops = int(getattr(trace, "num_ops", 0) or len(getattr(trace, "layer_logs", {}) or {}))
        graph_shape_hash = str(getattr(trace, "graph_shape_hash", "") or "")
    except Exception as error:
        return None, "", f"{error!r}\n{traceback.format_exc(limit=8)}"
    return n_ops, graph_shape_hash, ""


def validate_one(row: CatalogRow, dry_run: bool, scope: str, device: str) -> ValidationResult:
    """Instantiate and validate one menagerie model.

    Parameters
    ----------
    row:
        Catalog row.
    dry_run:
        Build recipe only when true.
    scope:
        Validation scope, ``"forward"`` or ``"forward+backward"``.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.

    Returns
    -------
    ValidationResult
        Validation result.
    """

    start = time.monotonic()
    plan = dependency_plan(row)
    skip_reason = unrenderable_reason(row)
    if skip_reason is not None:
        return ValidationResult(
            row.name,
            row.model_id,
            f"skipped:{skip_reason}",
            0,
            False,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            skip_reason,
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    try:
        input_tensor = _build_input(row)
    except Exception as error:
        return ValidationResult(
            row.name,
            row.model_id,
            "skipped:unsupported_input_recipe",
            0,
            False,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            str(error),
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    if dry_run:
        return ValidationResult(
            row.name,
            row.model_id,
            "skipped:dry_run",
            0,
            False,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            "validated recipe",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )

    model = instantiate_model(row)
    if hasattr(model, "eval"):
        model.eval()

    def attempt_validation(
        attempt_model: Any, attempt_input: Any, actual_device: str
    ) -> ValidationResult:
        """Validate the model on one resolved device.

        Parameters
        ----------
        attempt_model:
            Model prepared for the attempt device.
        attempt_input:
            Example input prepared for the attempt device.
        actual_device:
            Device used by this attempt.

        Returns
        -------
        ValidationResult
            Validation result for this attempt.
        """

        import torchlens as tl

        try:
            forward_result = tl.validate_forward_pass(
                attempt_model,
                attempt_input,
                validate_metadata=True,
            )
        except Exception as error:
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                False,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                combine_notes(
                    device_note(device, actual_device),
                    f"{error!r}\n{traceback.format_exc(limit=8)}",
                ),
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        if not bool(forward_result):
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:replay",
                0,
                False,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                combine_notes(device_note(device, actual_device), repr(forward_result)),
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )

        backward_error = ""
        if scope == "forward+backward":
            try:
                backward_result = tl.validate(
                    attempt_model,
                    attempt_input,
                    scope="backward",
                    loss_fn=_sum_float_outputs,
                    validate_metadata=True,
                )
            except Exception as error:
                return ValidationResult(
                    row.name,
                    row.model_id,
                    "failed:exception",
                    0,
                    True,
                    scope,
                    time.monotonic() - start,
                    plan.cluster_key,
                    combine_notes(
                        device_note(device, actual_device),
                        f"backward validation failed: {error!r}\n{traceback.format_exc(limit=8)}",
                    ),
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
            if not bool(backward_result):
                return ValidationResult(
                    row.name,
                    row.model_id,
                    "failed:replay",
                    0,
                    True,
                    scope,
                    time.monotonic() - start,
                    plan.cluster_key,
                    combine_notes(
                        device_note(device, actual_device),
                        f"backward validation returned {backward_result!r}",
                    ),
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
            backward_error = f"; backward={backward_result!r}"

        n_ops, graph_shape_hash, trace_error = _trace_n_ops_and_hash(attempt_model, attempt_input)
        if n_ops is None:
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:trace_summary",
                None,
                True,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                combine_notes(device_note(device, actual_device), trace_error),
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        return ValidationResult(
            row.name,
            row.model_id,
            "validated",
            n_ops,
            True,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            combine_notes(
                device_note(device, actual_device),
                f"forward={forward_result!r}{backward_error}",
            ),
            graph_shape_hash,
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )

    if device == "cuda":
        try:
            model, input_tensor = move_model_and_input_to_device(model, input_tensor, "cuda")
        except Exception as error:
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                False,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                f"device=cuda; {error!r}\n{traceback.format_exc(limit=8)}",
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        return attempt_validation(model, input_tensor, "cuda")

    if device == "auto":
        cpu_result = attempt_validation(model, input_tensor, "cpu")
        if (
            cpu_result.status == "failed:exception"
            and is_device_related_error(RuntimeError(cpu_result.error))
            and cuda_is_available()
        ):
            try:
                model, input_tensor = move_model_and_input_to_device(model, input_tensor, "cuda")
            except Exception as error:
                return ValidationResult(
                    row.name,
                    row.model_id,
                    "failed:exception",
                    0,
                    False,
                    scope,
                    time.monotonic() - start,
                    plan.cluster_key,
                    f"device=cuda; {error!r}\n{traceback.format_exc(limit=8)}",
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
            return attempt_validation(model, input_tensor, "cuda")
        if cpu_result.error:
            return ValidationResult(
                cpu_result.name,
                cpu_result.model_id,
                cpu_result.status,
                cpu_result.n_ops,
                cpu_result.validate_metadata_ok,
                cpu_result.scope,
                cpu_result.elapsed,
                cpu_result.dependency_cluster,
                combine_notes(device_note(device, "cpu"), cpu_result.error),
                cpu_result.graph_shape_hash,
                stable_id=cpu_result.stable_id,
                recipe_revision_sha256=cpu_result.recipe_revision_sha256,
            )
        return cpu_result

    return attempt_validation(model, input_tensor, "cpu")


def validate_with_timeout(
    row: CatalogRow,
    dry_run: bool,
    scope: str,
    device: str,
    timeout_sec: float,
    tmp_dir: Path | None = None,
) -> ValidationResult:
    """Run one validation in an isolated child process with a timeout.

    Parameters
    ----------
    row:
        Catalog row.
    dry_run:
        Build recipe only when true.
    scope:
        Validation scope.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.
    timeout_sec:
        Maximum wall time in seconds.
    tmp_dir:
        Optional per-model temporary directory routed to the worker via the
        ``TMPDIR``/``TEMP``/``TMP`` environment variables. Passed through the
        subprocess environment (not process globals) so concurrent workers each
        get an isolated scratch directory without mutating shared state.

    Returns
    -------
    ValidationResult
        Validation result.
    """

    plan = dependency_plan(row)
    command = [
        sys.executable,
        "-m",
        "menagerie.validate_menagerie",
        "--worker-row-json",
        json.dumps(row.__dict__),
        "--scope",
        scope,
        "--device",
        device,
    ]
    if dry_run:
        command.append("--dry-run")
    child_env = None
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        child_env = dict(os.environ)
        for key in ("TMPDIR", "TEMP", "TMP"):
            child_env[key] = str(tmp_dir)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            row.name,
            row.model_id,
            "failed:timeout",
            0,
            False,
            scope,
            timeout_sec,
            plan.cluster_key,
            f"timed out after {timeout_sec:.1f}s",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    if completed.returncode != 0:
        stderr_tail = " | ".join(completed.stderr.strip().splitlines()[-5:])
        return ValidationResult(
            row.name,
            row.model_id,
            "failed:exception",
            0,
            False,
            scope,
            0.0,
            plan.cluster_key,
            stderr_tail or f"worker exited with code {completed.returncode}",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "worker_result":
            return result_from_payload(payload["result"])
    return ValidationResult(
        row.name,
        row.model_id,
        "failed:exception",
        0,
        False,
        scope,
        0.0,
        plan.cluster_key,
        "worker did not emit a worker_result event",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
    )


def _status_bucket(status: str) -> str:
    """Return the high-level status bucket.

    Parameters
    ----------
    status:
        Manifest status.

    Returns
    -------
    str
        ``validated``, ``failed``, or ``skipped``.
    """

    if status == "validated":
        return "validated"
    if status.startswith("failed:"):
        return "failed"
    return "skipped"


def write_reports(out_dir: Path, manifest_path: Path, rows: Sequence[CatalogRow]) -> None:
    """Write validation summary JSON and Markdown reports.

    Parameters
    ----------
    out_dir:
        Output directory.
    manifest_path:
        Validation manifest path.
    rows:
        Selected catalog rows used for report context.
    """

    records = manifest_records(manifest_path)
    row_by_stable_id = {
        row.stable_id: row
        for row in load_rows(db_path=Path(__file__).parent / "data" / "catalog.db")
    }
    row_by_stable_id.update({row.stable_id: row for row in rows})
    statuses = Counter(_status_bucket(row.get("status", "")) for row in records.values())
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_zoo: dict[str, Counter[str]] = defaultdict(Counter)
    failures = []
    for stable_id, record in records.items():
        catalog_row = row_by_stable_id.get(stable_id)
        domain = catalog_row.domain if catalog_row else "unknown"
        zoo = catalog_row.zoo if catalog_row else "unknown"
        bucket = _status_bucket(record.get("status", ""))
        by_domain[domain][bucket] += 1
        by_zoo[zoo][bucket] += 1
        if bucket == "failed":
            failures.append(
                {
                    "name": record.get("name", ""),
                    "stable_id": stable_id,
                    "model_id": record.get("model_id", ""),
                    "status": record.get("status", ""),
                    "error": record.get("error", ""),
                }
            )
    headline = (
        "TorchLens forward validation has algorithmically verified saved activation "
        f"replay for {statuses['validated']} menagerie models."
    )
    summary = {
        "totals": {
            "validated": statuses["validated"],
            "failed": statuses["failed"],
            "skipped": statuses["skipped"],
            "total": sum(statuses.values()),
        },
        "by_domain": {key: dict(value) for key, value in sorted(by_domain.items())},
        "by_zoo": {key: dict(value) for key, value in sorted(by_zoo.items())},
        "headline": headline,
        "failures": failures,
        "manifest": str(manifest_path),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# TorchLens Menagerie Validation Report",
        "",
        headline,
        "",
        "## Totals",
        "",
        f"- Validated: {statuses['validated']}",
        f"- Failed: {statuses['failed']}",
        f"- Skipped: {statuses['skipped']}",
        f"- Total manifest rows: {sum(statuses.values())}",
        "",
        "## Counts by Domain",
        "",
        "| Domain | Validated | Failed | Skipped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain, counts in sorted(by_domain.items()):
        lines.append(
            f"| {domain} | {counts['validated']} | {counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(
        [
            "",
            "## Counts by Zoo",
            "",
            "| Zoo | Validated | Failed | Skipped |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for zoo, counts in sorted(by_zoo.items()):
        lines.append(
            f"| {zoo} | {counts['validated']} | {counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- {failure['name']} ({failure['model_id']}): "
                f"{failure['status']} - {failure['error']}"
            )
    else:
        lines.append("No failures recorded.")
    (out_dir / REPORT_MD).write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    """Run the dependency-aware disk-safe validator.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    out_dir = args.out_dir.resolve()
    manifest_path = (args.manifest or out_dir / "validation_manifest.tsv").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_rows(args)
    if args.report_only:
        write_reports(out_dir, manifest_path, selected)
        log_event("report_done", manifest=str(manifest_path), out_dir=str(out_dir))
        return 0

    run_cache_snapshots = [snapshot_cache(root) for root in CACHE_ROOTS]
    start_free_gb = disk_free_gb(out_dir)
    log_event("validation_run_start", out_dir=str(out_dir), free_gb=round(start_free_gb, 3))
    assert_min_free(out_dir, args.min_free_gb)

    done = completed_stable_ids(manifest_path, args.revalidate_failed)
    rows = [row for row in selected if row.stable_id not in done]
    log_event("selected", count=len(rows), skipped_existing=len(selected) - len(rows))

    # Phase 1: install dependencies per cluster (serial -- installs mutate the
    # shared interpreter/site-packages and must precede their rows). Clusters
    # whose dependencies are unavailable are recorded directly to the manifest.
    runnable: list[tuple[DependencyPlan, CatalogRow]] = []
    for plan, cluster_rows in group_by_dependency(rows):
        install_error = install_dependency_plan(plan, args)
        if install_error is not None:
            for row in cluster_rows:
                result = ValidationResult(
                    row.name,
                    row.model_id,
                    "skipped:dependency_unavailable",
                    0,
                    False,
                    args.scope,
                    0.0,
                    plan.cluster_key,
                    install_error,
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
                append_validation_ledger(row, result, args.device)
                append_manifest(
                    manifest_path,
                    result,
                )
            log_event(
                "cluster_skipped",
                cluster=plan.cluster_key,
                count=len(cluster_rows),
                error=install_error,
            )
            continue
        runnable.extend((plan, row) for row in cluster_rows)

    # Phase 2: validate runnable rows concurrently. Each model already runs in an
    # isolated child process (``validate_with_timeout``); threads here just
    # dispatch and await those subprocesses. The GPU semaphore caps in-flight
    # jobs when a device that may use CUDA is selected. The main thread does ALL
    # manifest appends and disk bookkeeping single-threaded as futures complete.
    jobs = max(1, args.jobs)
    use_gpu_cap = args.device in {"cuda", "auto"}
    gpu_jobs = max(1, args.gpu_jobs)
    effective_jobs = min(jobs, gpu_jobs) if use_gpu_cap else jobs
    gpu_semaphore = threading.Semaphore(gpu_jobs) if use_gpu_cap else None

    def process_one(plan: DependencyPlan, row: CatalogRow) -> tuple[ValidationResult, int]:
        """Validate one row in a worker thread and clean up its scratch state.

        Parameters
        ----------
        plan:
            Dependency plan for the row's cluster.
        row:
            Catalog row to validate.

        Returns
        -------
        tuple[ValidationResult, int]
            Validation result and the number of new cache entries removed.
        """

        cache_snapshots = [snapshot_cache(root) for root in CACHE_ROOTS]
        tmp_dir = out_dir / "_tmp" / f"{row.model_id:05d}_{safe_path_part(row.name)}"
        gate: ContextManager[Any] = gpu_semaphore if gpu_semaphore is not None else nullcontext()
        with gate:
            result = validate_with_timeout(
                row,
                args.dry_run,
                args.scope,
                args.device,
                args.timeout_sec,
                tmp_dir=tmp_dir,
            )
            removed = 0 if args.keep_cache else cleanup_runtime(cache_snapshots, tmp_dir)
        return result, removed

    processed = 0
    total = len(runnable)
    log_event(
        "parallel_start",
        jobs=jobs,
        effective_jobs=effective_jobs,
        gpu_jobs=gpu_jobs if use_gpu_cap else None,
        device=args.device,
        rows=total,
    )

    if total:
        try:
            assert_min_free(out_dir, args.min_free_gb)
        except RuntimeError:
            for snapshot in run_cache_snapshots:
                purge_new_cache_entries(snapshot)
            assert_min_free(out_dir, args.min_free_gb)

    with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
        futures: dict[Future[tuple[ValidationResult, int]], tuple[DependencyPlan, CatalogRow]] = {}
        for plan, row in runnable:
            before_free_gb = disk_free_gb(out_dir)
            log_event(
                "model_start",
                name=row.name,
                cluster=plan.cluster_key,
                free_gb=round(before_free_gb, 3),
            )
            futures[executor.submit(process_one, plan, row)] = (plan, row)

        for future in as_completed(futures):
            plan, row = futures[future]
            processed += 1
            result, removed = future.result()
            append_validation_ledger(row, result, args.device)
            append_manifest(manifest_path, result)
            after_free_gb = disk_free_gb(out_dir)
            log_event(
                "model_done",
                index=processed,
                total=total,
                name=row.name,
                status=result.status,
                n_ops=result.n_ops,
                cache_entries_removed=removed,
                after_free_gb=round(after_free_gb, 3),
                elapsed=round(result.elapsed, 3),
                error=result.error,
            )
            # Periodic disk-safety check: free space should not run dry as the
            # batch progresses.
            try:
                assert_min_free(out_dir, args.min_free_gb)
            except RuntimeError:
                for snapshot in run_cache_snapshots:
                    purge_new_cache_entries(snapshot)
                assert_min_free(out_dir, args.min_free_gb)

    write_reports(out_dir, manifest_path, selected)
    log_event(
        "validation_run_done",
        processed=processed,
        manifest=str(manifest_path),
        report=str(out_dir / REPORT_MD),
        summary=str(out_dir / SUMMARY_JSON),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the validator CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, help="process the first N rows after filters")
    parser.add_argument("--family")
    parser.add_argument("--domain")
    parser.add_argument("--zoo")
    parser.add_argument("--name", action="append", help="case-insensitive model-name substring")
    parser.add_argument("--model-id", action="append", type=int, help="exact catalog model id")
    parser.add_argument("--stable-ids", nargs="+", help="exact stable IDs to validate")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--featured-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--since", type=int, help="only process rows with model_id greater than this"
    )
    parser.add_argument(
        "--scope",
        choices=("forward", "forward+backward"),
        default="forward",
        help="validation scope",
    )
    parser.add_argument(
        "--revalidate-failed", action="store_true", help="retry non-validated manifest rows"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="number of models to validate concurrently (each in its own subprocess)",
    )
    parser.add_argument(
        "--gpu-jobs",
        type=int,
        default=4,
        help="max concurrent in-flight jobs when --device is cuda/auto (GPU OOM guard)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--db", type=Path, default=Path(__file__).resolve().parent / "data" / "catalog.db"
    )
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-models", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--install-timeout", type=float, default=600.0)
    parser.add_argument(
        "--pip-args", action="append", default=[], help="extra argument for pip install"
    )
    parser.add_argument("--install-deps", dest="install_deps", action="store_true", default=True)
    parser.add_argument("--no-install-deps", dest="install_deps", action="store_false")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker-row-json", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validator CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    # Pin BLAS/OMP threads to 1 so concurrent validate workers don't oversubscribe the CPU (see generate_menagerie).
    for _thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_var, "1")
    if args.worker_row_json:
        row = catalog_row_from_payload(json.loads(args.worker_row_json))
        try:
            result = validate_one(row, args.dry_run, args.scope, args.device)
        except Exception as error:
            plan = dependency_plan(row)
            result = ValidationResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                False,
                args.scope,
                0.0,
                plan.cluster_key,
                f"{error!r}\n{traceback.format_exc(limit=8)}",
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        print(json.dumps({"event": "worker_result", "result": result.__dict__}), flush=True)
        return 0
    try:
        return run(args)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
