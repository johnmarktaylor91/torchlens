"""Load and recover fastlog directory bundles."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Literal

from safetensors import SafetensorError
from safetensors.torch import load_file as load_safetensors_file

from .._io import (
    ArtifactRuntimeIncompatibleError,
    ArtifactVersionAboveRuntimeError,
    ArtifactVersionBelowFloorError,
    TorchLensIOError,
)
from .._io._json import _MAX_JSON_BYTES, loads_bounded, read_bounded
from .._io.manifest import Manifest, enforce_version_policy
from .._io.paths import resolve_bundle_blob_path
from .._io.streaming import PARTIAL_SENTINEL, REASON_SENTINEL
from .exceptions import BundleNotFinalizedError, RecoveryError
from .storage_disk import record_from_json
from .storage_ram import RamStorageBackend
from .types import ActivationRecord, Recording

# Recovery scans an UNTRUSTED, possibly crash-torn index. Both ceilings keep a
# hostile or corrupted bundle from turning ``recover()`` into an allocation or
# warning-ledger DoS while leaving every realistic bundle untouched.
_INDEX_MAX_BYTES = _MAX_JSON_BYTES
_MAX_RECOVERY_WARNINGS = 200


#: Governed compatibility refusals recover() must RE-RAISE, never salvage
#: (D-ECO-9): salvaging a below-floor / above-ceiling / runtime-incompatible
#: bundle as ``recovered=True`` defeats the version window. recover() keeps
#: salvaging CORRUPTION -- that is its job; only governed refusals pass through.
_GOVERNED_COMPATIBILITY_REFUSALS = (
    ArtifactVersionBelowFloorError,
    ArtifactVersionAboveRuntimeError,
    ArtifactRuntimeIncompatibleError,
)


class _RecoveryWarningSink(list):
    """Recovery-warning ledger with a hard cap and an explicit overflow summary.

    A malformed-line flood (a multi-million-line corrupt index) used to append
    one warning per line without bound. The sink keeps the first
    ``_MAX_RECOVERY_WARNINGS`` diagnostics and counts the rest, so the ledger
    stays honest about suppression instead of silently growing or truncating.
    """

    def __init__(self, initial: list[str] | None = None) -> None:
        super().__init__()
        self.suppressed = 0
        for item in initial or []:
            self.append(item)

    def append(self, item: str) -> None:
        """Append ``item`` unless the cap is reached; count it as suppressed then."""

        if len(self) >= _MAX_RECOVERY_WARNINGS:
            self.suppressed += 1
            return
        super().append(item)

    def finalized(self) -> list[str]:
        """Return the plain capped list plus a suppression summary line."""

        out = list(self)
        if self.suppressed:
            out.append(f"{self.suppressed} additional recovery warnings suppressed")
        return out


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
    return _load_from_index(
        bundle_path,
        recovered=False,
        recovery_warnings=[],
        strict_integrity=True,
    )


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
        except _GOVERNED_COMPATIBILITY_REFUSALS:
            # Drop-not-resurrect (R10-4, widened by G3/D-ECO-9): a bundle the
            # version window or runtime policy REFUSES is refused here too;
            # letting the bare-pass salvage below resurrect it as
            # recovered=True defeated the governed refusal entirely.
            raise
        except TorchLensIOError:
            pass
        else:
            return _load_from_index(
                bundle_path,
                recovered=False,
                recovery_warnings=[],
                strict_integrity=False,
            )

    index_path = bundle_path / "fastlog_index.jsonl"
    if not index_path.exists():
        raise RecoveryError("no recoverable index")
    return _load_from_index(
        bundle_path,
        recovered=True,
        recovery_warnings=[],
        strict_integrity=False,
    )


def _load_from_index(
    bundle_path: Path,
    *,
    recovered: bool,
    recovery_warnings: list[str],
    strict_integrity: bool,
) -> Recording:
    """Load records by scanning ``fastlog_index.jsonl``."""

    metadata = _read_metadata(bundle_path / "metadata.json")
    records: list[ActivationRecord] = []
    warnings_out = _RecoveryWarningSink(recovery_warnings)
    # WT1 A-IV item 21 (lane A08): an aborted bundle's PARTIAL/REASON.txt
    # debris IS the on-disk failure record. Recovery used to discard it and
    # rebuild the Recording with ``failed=False`` and no error evidence --
    # laundering a failed capture into a benign "recovered". Carry it.
    abort_reason = _abort_debris_reason(bundle_path) if recovered else None
    if abort_reason is not None:
        warnings_out.append(f"bundle was aborted mid-write: {abort_reason}")
    lines = _read_index_lines(bundle_path / "fastlog_index.jsonl")
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            # Bounded: a depth-bomb line used to escape the JSONDecodeError
            # handler as a raw RecursionError out of public recover().
            data = loads_bounded(raw_line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                warnings_out.append("truncated tail")
            else:
                warnings_out.append(f"malformed line {line_number}")
            continue
        if not isinstance(data, dict):
            warnings_out.append(f"malformed line {line_number}")
            continue
        try:
            record = record_from_json(data)
        except (KeyError, TypeError, ValueError, AttributeError, TorchLensIOError):
            # Structurally valid JSON that is not a well-formed record (e.g. a
            # missing "ctx" key) is recovery debris, not a crash: skip and warn,
            # matching the malformed-line contract. The strict finalized-load
            # path still refuses on any warning below.
            warnings_out.append(f"malformed record line {line_number}")
            continue
        blob_recoverable, validated_payloads = _blob_is_recoverable(
            bundle_path,
            record,
            warnings_out,
        )
        if not blob_recoverable:
            continue
        rehydrated_record = _rehydrate_record_payloads(
            record,
            validated_payloads,
        )
        if rehydrated_record is None:
            continue
        records.append(rehydrated_record)
    final_warnings = warnings_out.finalized()
    if strict_integrity and final_warnings:
        raise TorchLensIOError(_format_strict_integrity_error(final_warnings))
    recording = _recording_from_records(
        records,
        bundle_path=bundle_path,
        metadata=metadata,
        recovered=recovered or bool(final_warnings),
        recovery_warnings=final_warnings,
        abort_reason=abort_reason,
    )
    RamStorageBackend(recording).finalize()
    return recording


def _abort_debris_reason(bundle_path: Path) -> str | None:
    """Return the persisted abort reason when the bundle was aborted mid-write.

    ``BundleStreamWriter.abort`` (and the fastlog storage backend routed
    through it) marks the temp bundle with a ``PARTIAL`` sentinel and writes
    the scrubbed failure reason to ``REASON.txt``. Absence of the sentinel
    means no abort evidence, never proof of success.
    """

    if not (bundle_path / PARTIAL_SENTINEL).exists():
        return None
    try:
        text = (bundle_path / REASON_SENTINEL).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    text = " ".join(text.split())
    if len(text) > 500:
        text = text[:500] + "..."
    return text or "no abort reason recorded"


def _format_strict_integrity_error(recovery_warnings: list[str]) -> str:
    """Return a finalized-load integrity error message.

    Parameters
    ----------
    recovery_warnings:
        Recovery diagnostics accumulated while scanning the JSONL index.

    Returns
    -------
    str
        Human-readable error explaining why a finalized bundle load refused to
        continue.
    """

    if not recovery_warnings:
        return "Finalized fastlog bundle failed integrity validation."
    primary = recovery_warnings[0]
    if len(recovery_warnings) == 1:
        return (
            "Finalized fastlog bundle failed integrity validation: "
            f"{primary}. Use tl.fastlog.recover() to inspect salvageable records."
        )
    return (
        "Finalized fastlog bundle failed integrity validation: "
        f"{primary} (and {len(recovery_warnings) - 1} more issue(s)). "
        "Use tl.fastlog.recover() to inspect salvageable records."
    )


def _read_index_lines(path: Path) -> list[str]:
    """Read index lines from a recoverable fastlog index file, size-bounded.

    ``read_text`` materialized the whole attacker-sized file before any
    ceiling could apply. At most ``_INDEX_MAX_BYTES + 1`` bytes are read; an
    over-ceiling index refuses typed rather than allocating without bound.
    Undecodable bytes (a crash can tear a multibyte sequence) degrade to
    replacement characters so the surrounding intact lines stay salvageable;
    the torn line itself fails JSON parsing and is warned about normally.
    """

    try:
        with path.open("rb") as handle:
            # R10-3 fstat-first: refuse on the stat size BEFORE reading so an
            # over-ceiling index never materializes a transient ~512 MiB
            # buffer. The post-read length check stays as the truth for a file
            # that grew after the stat.
            stat_size = os.fstat(handle.fileno()).st_size
            if stat_size > _INDEX_MAX_BYTES:
                raise TorchLensIOError(
                    f"Fastlog index at {path} exceeds the {_INDEX_MAX_BYTES}-byte ceiling.",
                    code="fastlog_index_too_large",
                )
            # Stat-sized bounded read (R33-1): reading CEILING+1 pre-allocated
            # a transient ~512 MiB buffer for every index regardless of its
            # actual size. One sentinel byte over the stat size detects growth
            # between fstat and read; the ceiling refusal above stays the
            # authority for over-ceiling files.
            data = handle.read(stat_size + 1)
    except OSError as exc:
        raise RecoveryError("no recoverable index") from exc
    if len(data) > stat_size:
        raise TorchLensIOError(
            f"Fastlog index at {path} grew between its size check and read.",
            code="fastlog_index_too_large",
        )
    return data.decode("utf-8", errors="replace").splitlines()


def _blob_is_recoverable(
    bundle_path: Path,
    record: ActivationRecord,
    recovery_warnings: list[str],
) -> tuple[bool, dict[str, Any]]:
    """Return whether a record's blob(s) are present and hash-valid.

    Both the raw out blob and the transformed out blob are
    validated when their metadata is present. A missing or hash-mismatched
    blob disqualifies the record. Successful validation also returns the
    materialized payloads so the caller does not have to read the file again.
    """

    raw_recoverable, raw_payload = _validate_blob_metadata(
        bundle_path,
        record.metadata.get("blob_id"),
        record.metadata.get("relative_path"),
        record.metadata.get("sha256"),
        recovery_warnings,
    )
    if not raw_recoverable:
        return False, {}
    transformed_recoverable, transformed_payload = _validate_blob_metadata(
        bundle_path,
        record.metadata.get("transformed_out_blob_id"),
        record.metadata.get("transformed_out_relative_path"),
        record.metadata.get("transformed_out_sha256"),
        recovery_warnings,
    )
    if not transformed_recoverable:
        return False, {}
    return True, {
        "disk_payload": raw_payload,
        "transformed_disk_payload": transformed_payload,
    }


def _validate_blob_metadata(
    bundle_path: Path,
    blob_id: Any,
    relative_path: Any,
    expected_sha256: Any,
    recovery_warnings: list[str],
) -> tuple[bool, Any | None]:
    """Validate a single blob entry from record metadata."""

    if blob_id is None and relative_path is None and expected_sha256 is None:
        return True, None
    if blob_id is None or relative_path is None or expected_sha256 is None:
        warning_id = "unknown"
        if blob_id is not None:
            warning_id = str(blob_id)
        elif relative_path is not None:
            warning_id = str(relative_path)
        recovery_warnings.append(f"incomplete blob metadata {warning_id}")
        return False, None
    try:
        blob_path = resolve_bundle_blob_path(bundle_path, str(relative_path))
    except TorchLensIOError:
        recovery_warnings.append(f"malformed blob path {blob_id}")
        return False, None
    if not blob_path.exists():
        recovery_warnings.append(f"missing blob {blob_id}")
        return False, None
    try:
        payload = _load_verified_blob_tensor(blob_path, str(expected_sha256))
    except TorchLensIOError:
        recovery_warnings.append(f"hash mismatch {blob_id}")
        return False, None
    return True, payload


def _load_verified_blob_tensor(blob_path: Path, expected_sha256: str) -> Any:
    """Verify one fastlog blob's digest, then materialize it.

    The digest is computed by a CHUNKED streaming hash over the on-disk bytes
    (``sha256_of_file``), never ``read_bytes()``: the prior code read the ENTIRE
    attacker-controlled blob into memory BEFORE the digest check, so a hostile
    bundle whose blob does not even match its claimed hash still forced a full
    allocation (KNOWN HIGH, never closed -- ee29700f bounded only the JSON reads).
    A hash mismatch is now rejected in constant memory, and the tensor is
    materialized through the mmap-backed safetensors file loader (matching the
    main bundle path's ``sha256_of_file`` + ``load_file`` discipline).
    """

    from .._io.lazy import _file_identity
    from .._io.manifest import sha256_of_file

    # R59 TOCTOU (lazy.py discipline): the digest and the mmap-backed load are
    # two opens of the same path, so bracket the hash with the file identity and
    # re-check before loading -- a rename-replace in the window is refused
    # rather than admitting bytes that were never hashed.
    try:
        pre_hash_identity = _file_identity(blob_path.stat())
        observed_sha256 = sha256_of_file(blob_path)
    except OSError as exc:
        raise TorchLensIOError(f"Failed to read fastlog blob at {blob_path}.") from exc
    if observed_sha256 != expected_sha256:
        raise TorchLensIOError(f"Checksum mismatch for fastlog blob at {blob_path}.")
    try:
        pre_load_identity = _file_identity(blob_path.stat())
    except OSError as exc:
        raise TorchLensIOError(f"Failed to read fastlog blob at {blob_path}.") from exc
    if pre_load_identity != pre_hash_identity:
        raise TorchLensIOError(
            f"Fastlog blob at {blob_path} changed between integrity check and load."
        )
    return _load_blob_tensor_from_file(blob_path)


def _load_blob_tensor_from_file(blob_path: Path) -> Any:
    """Load the single tensor stored in one fastlog safetensors blob.

    Fastlog blobs are always written with exactly one tensor per file (see
    ``BundleStreamWriter._write_tensor_blob``), so the blob's sole value is the
    materialized payload; the storage key itself is not part of the public
    contract. The mmap-backed file loader never fully pre-copies the blob into a
    bytes object.

    Raises
    ------
    TorchLensIOError
        If the blob is missing, unreadable, or does not contain exactly one
        tensor.
    """

    try:
        tensor_map = load_safetensors_file(str(blob_path))
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
    record: ActivationRecord,
    validated_payloads: dict[str, Any],
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

    disk_payload = validated_payloads.get("disk_payload")
    transformed_disk_payload = validated_payloads.get("transformed_disk_payload")

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
    abort_reason: str | None = None,
) -> Recording:
    """Build a Recording around loaded records.

    ``abort_reason`` is the PARTIAL/REASON.txt debris text when the bundle was
    aborted mid-write: the rebuilt Recording then carries ``failed=True`` plus
    the string evidence, and its derived outcome is FAILED -- recovery
    salvages records, never the failure verdict (WT1 A-IV item 21).
    """

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
        history_size=int(metadata.get("history_size", 0)),
        _activation_transform_repr=metadata.get("_activation_transform_repr"),
        recovered=recovered,
        status=status,
        recovery_warnings=recovery_warnings,
        failed=abort_reason is not None,
        error_repr=(None if abort_reason is None else f"recovered aborted bundle: {abort_reason}"),
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    """Read optional fastlog metadata JSON through the bounded reader.

    Best-effort by contract (missing or malformed metadata degrades to ``{}``),
    so the bounded reader's size/depth refusals -- surfaced as
    ``JSONDecodeError`` -- degrade the same way instead of escaping as a raw
    ``RecursionError`` or unbounded allocation.
    """

    try:
        data = read_bounded(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _validate_fastlog_layout(bundle_path: Path, manifest: Manifest) -> None:
    """Validate the finalized fastlog directory layout.

    Notes
    -----
    This finalized-layout tripwire currently validates the required fastlog
    sidecar files by presence only. ``load()`` reconstructs the effective
    lookup indexes from ``fastlog_index.jsonl`` rather than trusting the
    sidecar contents directly; the sidecar authority contract remains an owner
    decision.
    """

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
