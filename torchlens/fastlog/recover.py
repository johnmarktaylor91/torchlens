"""Load and recover fastlog directory bundles."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

from safetensors import SafetensorError
from safetensors.torch import load_file

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
        rehydrated_record = _rehydrate_record_payloads(bundle_path, record, warnings_out)
        if rehydrated_record is None:
            continue
        records.append(rehydrated_record)
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

    Both the raw out blob and the transformed out blob are
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
        record.metadata.get("transformed_out_blob_id"),
        record.metadata.get("transformed_out_relative_path"),
        record.metadata.get("transformed_out_sha256"),
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


def _load_blob_tensor(blob_path: Path) -> Any:
    """Load the single tensor stored in one fastlog safetensors blob.

    Fastlog blobs are always written with exactly one tensor per file (see
    ``BundleStreamWriter._write_tensor_blob``), so the blob's sole value is the
    materialized payload; the storage key itself is not part of the public
    contract.

    Raises
    ------
    TorchLensIOError
        If the blob is missing, unreadable, or does not contain exactly one
        tensor.
    """

    try:
        tensor_map = load_file(str(blob_path))
    except ImportError as exc:
        raise TorchLensIOError(
            "Fastlog bundle payload materialization requires the safetensors "
            "backend. Install safetensors>=0.4."
        ) from exc
    except (OSError, SafetensorError, ValueError) as exc:
        raise TorchLensIOError(f"Failed to materialize fastlog blob at {blob_path}.") from exc
    if len(tensor_map) != 1:
        raise TorchLensIOError(f"Expected a single tensor in fastlog blob file {blob_path}.")
    return next(iter(tensor_map.values()))


def _rehydrate_record_payloads(
    bundle_path: Path,
    record: ActivationRecord,
    recovery_warnings: list[str],
) -> ActivationRecord | None:
    """Rehydrate a reloaded record's disk-persisted tensor payloads.

    ``_blob_is_recoverable`` already confirmed both the raw and transformed
    blobs (when present) exist and are hash-valid, so a subsequent failure to
    read them back here reflects a genuine I/O problem (e.g. a race with
    concurrent bundle mutation) rather than a corruption this function should
    silently mask. Such a record is skipped with a recovery warning, matching
    the existing missing-blob/hash-mismatch skip-and-warn behavior in this
    module, instead of raising out of ``load()``/``recover()``.

    Returns
    -------
    ActivationRecord | None
        A record with ``disk_payload``/``transformed_disk_payload`` populated
        from their persisted blobs, or ``None`` if materialization failed for
        a blob that ``_blob_is_recoverable`` had already validated.
    """

    disk_payload = None
    relative_path = record.metadata.get("relative_path")
    blob_id = record.metadata.get("blob_id")
    if relative_path is not None:
        try:
            disk_payload = _load_blob_tensor(
                resolve_bundle_blob_path(bundle_path, str(relative_path))
            )
        except TorchLensIOError:
            recovery_warnings.append(f"failed to materialize blob {blob_id}")
            return None

    transformed_disk_payload = None
    transformed_relative_path = record.metadata.get("transformed_out_relative_path")
    transformed_blob_id = record.metadata.get("transformed_out_blob_id")
    if transformed_relative_path is not None:
        try:
            transformed_disk_payload = _load_blob_tensor(
                resolve_bundle_blob_path(bundle_path, str(transformed_relative_path))
            )
        except TorchLensIOError:
            recovery_warnings.append(f"failed to materialize blob {transformed_blob_id}")
            return None

    if disk_payload is None and transformed_disk_payload is None:
        return record
    return dataclasses.replace(
        record,
        disk_payload=disk_payload,
        transformed_disk_payload=transformed_disk_payload,
    )


def _recording_from_records(
    records: list[ActivationRecord],
    *,
    bundle_path: Path,
    metadata: dict[str, Any],
    recovered: bool,
    recovery_warnings: list[str],
) -> Recording:
    """Build a Recording around loaded records."""

    halted = bool(metadata.get("halted", False))
    status: Literal["complete", "halted", "partial_error", "recovered"] = (
        "recovered" if recovered else "halted" if halted else "complete"
    )
    return Recording(
        records=records,
        by_pass={},
        by_label={},
        by_address={},
        orphan_records=list(metadata.get("orphan_records", [])),
        bundle_path=bundle_path,
        n_ops=int(metadata.get("n_passes", metadata.get("n_ops", 1))),
        start_times=list(metadata.get("start_times", [])),
        end_times=list(metadata.get("end_times", [])),
        predicate_failures=[],
        predicate_failure_overflow_count=int(metadata.get("predicate_failure_overflow_count", 0)),
        halted=halted,
        halt_reason=metadata.get("halt_reason"),
        halts_by_pass={
            int(pass_index): str(reason)
            for pass_index, reason in dict(metadata.get("halts_by_pass", {})).items()
        },
        keep_op_repr=metadata.get("keep_op_repr"),
        keep_module_repr=metadata.get("keep_module_repr"),
        history_size=int(metadata.get("history_size", 0)),
        _activation_transform_repr=metadata.get("_activation_transform_repr"),
        recovered=recovered,
        status=status,
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
