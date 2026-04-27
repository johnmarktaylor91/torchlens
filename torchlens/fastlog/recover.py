"""Load and recover fastlog directory bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._io import TorchLensIOError
from .._io.manifest import Manifest, enforce_version_policy, sha256_of_file
from .._io.paths import resolve_bundle_blob_path
from .exceptions import BundleNotFinalizedError, RecoveryError
from .storage_disk import record_from_json
from .storage_ram import RamStorageBackend
from .types import ActivationRecord, Recording


def load(path: str | Path) -> Recording:
    """Load a finalized fastlog bundle.

    Parameters
    ----------
    path:
        Fastlog bundle directory.

    Returns
    -------
    Recording
        Loaded recording with ``recovered=False``.

    Raises
    ------
    BundleNotFinalizedError
        If the bundle has no valid manifest.
    TorchLensIOError
        If the finalized bundle is malformed.
    """

    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise BundleNotFinalizedError("fastlog bundle is partial; use tl.fastlog.recover()")
    try:
        manifest = Manifest.read(manifest_path)
    except TorchLensIOError as exc:
        raise BundleNotFinalizedError(
            "fastlog bundle manifest is invalid; use tl.fastlog.recover()"
        ) from exc
    _validate_fastlog_layout(bundle_path, manifest)
    return _load_from_index(bundle_path, recovered=False, recovery_warnings=[])


def recover(path: str | Path) -> Recording:
    """Recover a finalized or partial fastlog bundle.

    Parameters
    ----------
    path:
        Fastlog bundle directory or streaming temp directory.

    Returns
    -------
    Recording
        Loaded or recovered recording.

    Raises
    ------
    RecoveryError
        If no recoverable index exists.
    """

    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = Manifest.read(manifest_path)
            _validate_fastlog_layout(bundle_path, manifest)
        except TorchLensIOError:
            pass
        else:
            return _load_from_index(bundle_path, recovered=False, recovery_warnings=[])

    index_path = bundle_path / "fastlog_index.jsonl"
    if not index_path.exists():
        raise RecoveryError("no recoverable index")
    return _load_from_index(bundle_path, recovered=True, recovery_warnings=[])


def _load_from_index(
    bundle_path: Path,
    *,
    recovered: bool,
    recovery_warnings: list[str],
) -> Recording:
    """Load records by scanning ``fastlog_index.jsonl``."""

    metadata = _read_metadata(bundle_path / "metadata.json")
    records: list[ActivationRecord] = []
    warnings_out = list(recovery_warnings)
    lines = _read_index_lines(bundle_path / "fastlog_index.jsonl")
    for line_number, raw_line in enumerate(lines, start=1):
        if raw_line == "" and line_number == len(lines):
            warnings_out.append("truncated tail")
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                warnings_out.append("truncated tail")
            else:
                warnings_out.append(f"malformed line {line_number}")
            continue
        if not isinstance(data, dict):
            warnings_out.append(f"malformed line {line_number}")
            continue
        record = record_from_json(data)
        if not _blob_is_recoverable(bundle_path, record, warnings_out):
            continue
        records.append(record)
    recording = _recording_from_records(
        records,
        bundle_path=bundle_path,
        metadata=metadata,
        recovered=recovered,
        recovery_warnings=warnings_out,
    )
    RamStorageBackend(recording).finalize()
    return recording


def _read_index_lines(path: Path) -> list[str]:
    """Read index lines while preserving a missing trailing newline signal."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecoveryError("no recoverable index") from exc
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines[-1] = ""
    return lines


def _blob_is_recoverable(
    bundle_path: Path,
    record: ActivationRecord,
    recovery_warnings: list[str],
) -> bool:
    """Return whether a record's blob(s) are present and hash-valid.

    Both the raw activation blob and the transformed activation blob are
    validated when their metadata is present. A missing or hash-mismatched
    blob disqualifies the record.
    """

    raw_recoverable = _validate_blob_metadata(
        bundle_path,
        record.metadata.get("blob_id"),
        record.metadata.get("relative_path"),
        record.metadata.get("sha256"),
        recovery_warnings,
    )
    if not raw_recoverable:
        return False
    transformed_recoverable = _validate_blob_metadata(
        bundle_path,
        record.metadata.get("transformed_activation_blob_id"),
        record.metadata.get("transformed_activation_relative_path"),
        record.metadata.get("transformed_activation_sha256"),
        recovery_warnings,
    )
    return transformed_recoverable


def _validate_blob_metadata(
    bundle_path: Path,
    blob_id: Any,
    relative_path: Any,
    expected_sha256: Any,
    recovery_warnings: list[str],
) -> bool:
    """Validate a single blob entry from record metadata."""

    if blob_id is None or relative_path is None or expected_sha256 is None:
        return True
    try:
        blob_path = resolve_bundle_blob_path(bundle_path, str(relative_path))
    except TorchLensIOError:
        recovery_warnings.append(f"malformed blob path {blob_id}")
        return False
    if not blob_path.exists():
        recovery_warnings.append(f"missing blob {blob_id}")
        return False
    if sha256_of_file(blob_path) != expected_sha256:
        recovery_warnings.append(f"hash mismatch {blob_id}")
        return False
    return True


def _recording_from_records(
    records: list[ActivationRecord],
    *,
    bundle_path: Path,
    metadata: dict[str, Any],
    recovered: bool,
    recovery_warnings: list[str],
) -> Recording:
    """Build a Recording around loaded records."""

    return Recording(
        records=records,
        by_pass={},
        by_label={},
        by_module_address={},
        bundle_path=bundle_path,
        n_passes=int(metadata.get("n_passes", 1)),
        n_records=len(records),
        pass_start_times=list(metadata.get("pass_start_times", [])),
        pass_end_times=list(metadata.get("pass_end_times", [])),
        predicate_failures=[],
        predicate_failure_overflow_count=int(metadata.get("predicate_failure_overflow_count", 0)),
        keep_op_repr=metadata.get("keep_op_repr"),
        keep_module_repr=metadata.get("keep_module_repr"),
        history_size=int(metadata.get("history_size", 0)),
        activation_postfunc_repr=metadata.get("activation_postfunc_repr"),
        recovered=recovered,
        recovery_warnings=recovery_warnings,
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    """Read optional fastlog metadata JSON."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _validate_fastlog_layout(bundle_path: Path, manifest: Manifest) -> None:
    """Validate the finalized fastlog directory layout."""

    enforce_version_policy(manifest)
    if manifest.bundle_format != "fastlog-directory":
        raise TorchLensIOError("Expected fastlog-directory bundle format.")
    required = (
        "manifest.json",
        "fastlog_index.jsonl",
        "pass_index.json",
        "label_index.json",
        "metadata.json",
        "blobs",
    )
    for name in required:
        candidate = bundle_path / name
        if not candidate.exists():
            raise TorchLensIOError(f"Fastlog bundle is missing {name}.")
    if not (bundle_path / "blobs").is_dir():
        raise TorchLensIOError("Fastlog bundle blobs path is not a directory.")
