"""Synchronous disk storage backend for fastlog recordings."""

from __future__ import annotations

import json
import platform
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .. import __version__ as TORCHLENS_VERSION
from .._io import IO_FORMAT_VERSION, TorchLensIOError
from .._io.manifest import Manifest, TensorEntry
from .._io.streaming import BundleStreamWriter
from ._indexes import write_label_index, write_pass_index
from ._storage_resolver import _resolve_storage
from .exceptions import PredicateError, RecordingConfigError
from .options import RecordingOptions
from .storage_ram import RamStorageBackend
from .types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    RecordContext,
    Recording,
    StorageIntent,
)

_GRAD_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
}


class DiskStorageBackend:
    """Persist fastlog records to a sync directory bundle."""

    def __init__(self, options: RecordingOptions, recording: Recording) -> None:
        """Initialize a disk-backed fastlog storage backend.

        Parameters
        ----------
        options:
            Recording options for the active session.
        recording:
            Recording receiving retained records and indexes.

        Raises
        ------
        RecordingConfigError
            If static options request ``keep_grad=True`` in disk-only mode.
        """

        if options.streaming is None or options.streaming.bundle_path is None:
            raise RecordingConfigError("DiskStorageBackend requires streaming.bundle_path")
        self.options = options
        self.recording = recording
        self.disk_only = not options.streaming.retain_in_memory
        self._validate_static_keep_grad()
        self.writer = BundleStreamWriter(options.streaming.bundle_path)
        self.index_path = self.writer.tmp_path / "fastlog_index.jsonl"
        self._ram_backend = RamStorageBackend(recording)
        self._tensor_entries: list[TensorEntry] = []
        self._warned_device_move = False
        self._finalized = False

    def append(self, record: ActivationRecord) -> None:
        """Append one record, writing its disk blob first when present.

        Parameters
        ----------
        record:
            Fastlog record selected by a predicate.

        Raises
        ------
        PredicateError
            If ``keep_grad=True`` is invalid for this storage mode or tensor.
        """

        self._validate_record_keep_grad(record)
        stored_record = record
        wrote_blob = False
        if record.disk_payload is not None:
            blob_id = self.writer.next_blob_id()
            entry = self.writer.write_blob(
                blob_id,
                record.disk_payload,
                kind="activation",
                label=record.ctx.label,
            )
            self._tensor_entries.append(entry)
            stored_record.metadata.update(_entry_to_record_metadata(record, entry))
            wrote_blob = True
        if record.transformed_disk_payload is not None:
            blob_id = self.writer.next_blob_id()
            transformed_label = f"{record.ctx.label}::transformed_activation"
            entry = self.writer.write_blob(
                blob_id,
                record.transformed_disk_payload,
                kind="transformed_activation",
                label=transformed_label,
            )
            self._tensor_entries.append(entry)
            stored_record.metadata.update(_transformed_entry_to_record_metadata(record, entry))
            wrote_blob = True
        if wrote_blob or record.spec.save_metadata:
            self._append_index_line(stored_record)
        self._ram_backend.append(stored_record)

    def resolve_payloads(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        intent: StorageIntent,
        *,
        options: RecordingOptions,
        ctx: "RecordContext | None" = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve tensor payloads for disk-backed capture.

        Parameters
        ----------
        tensor:
            Tensor selected for capture.
        spec:
            Capture policy for the selected tensor.
        intent:
            Storage intent resolved from streaming options.
        options:
            Recording options carrying ``activation_postfunc`` /
            ``save_raw_activation``.
        ctx:
            Record context used to enrich postfunc error messages.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
            ``(ram_payload, disk_payload, transformed_ram_payload,
            transformed_disk_payload)``. Any element may be ``None``.
        """

        return _resolve_storage(
            tensor,
            spec,
            intent,
            activation_postfunc=options.activation_postfunc,
            save_raw_activation=options.save_raw_activation,
            ctx=ctx,
        )

    def finalize(self) -> None:
        """Finalize the fastlog directory bundle."""

        if self._finalized:
            return
        self._ram_backend.finalize()
        try:
            write_pass_index(self.writer.tmp_path, self.recording)
            write_label_index(self.writer.tmp_path, self.recording)
            _write_metadata(self.writer.tmp_path / "metadata.json", self.recording, self.options)
            manifest = _build_fastlog_manifest(self._tensor_entries)
            manifest.write(self.writer.tmp_path / "manifest.json")
            self.writer.tmp_path.rename(self.writer.final_path)
        except (OSError, TorchLensIOError, ValueError) as exc:
            self.abort(f"Failed to finalize fastlog bundle: {exc}")
            raise
        self.writer._closed = True
        self.writer._finalized = True
        self._finalized = True
        object.__setattr__(self.recording, "bundle_path", self.writer.final_path)

    def abort(self, reason: str) -> None:
        """Abort the underlying streaming writer.

        Parameters
        ----------
        reason:
            Human-readable failure reason.
        """

        if not self._finalized:
            self.writer.abort(reason)

    def _validate_static_keep_grad(self) -> None:
        """Reject static keep_grad defaults in disk-only mode."""

        if not self.disk_only:
            return
        for name, default in (
            ("default_op", self.options.default_op),
            ("default_module", self.options.default_module),
        ):
            if isinstance(default, CaptureSpec) and default.keep_grad:
                raise RecordingConfigError(
                    f"{name} cannot use keep_grad=True with disk-only fastlog storage"
                )

    def _validate_record_keep_grad(self, record: ActivationRecord) -> None:
        """Validate keep_grad constraints before persisting one record."""

        if not record.spec.keep_grad:
            return
        if self.disk_only:
            raise PredicateError("keep_grad=True is not valid for disk-only fastlog storage")
        dtype = record.ctx.tensor_dtype
        if dtype is not None and dtype not in _GRAD_DTYPES:
            raise PredicateError("keep_grad=True is not valid for integer or bool tensors")
        if (
            record.spec.device is not None
            and record.ctx.tensor_device is not None
            and not self._warned_device_move
        ):
            warnings.warn(
                "fastlog keep_grad=True with an explicit device target may make autograd "
                "expensive.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_device_move = True

    def _append_index_line(self, record: ActivationRecord) -> None:
        """Append one JSONL metadata line for a retained record."""

        line = json.dumps(record_to_json(record), sort_keys=True)
        try:
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        except OSError as exc:
            self.abort(f"Failed to append fastlog index: {exc}")
            raise TorchLensIOError(f"Failed to append fastlog index at {self.index_path}.") from exc


def record_to_json(record: ActivationRecord) -> dict[str, Any]:
    """Convert one activation record into JSON-serializable metadata."""

    return {
        "ctx": _ctx_to_json(record.ctx),
        "spec": _spec_to_json(record.spec),
        "metadata": dict(record.metadata),
        "recorded_at": record.recorded_at,
    }


def record_from_json(data: dict[str, Any]) -> ActivationRecord:
    """Build an activation record from JSON-decoded fastlog metadata."""

    ctx = _ctx_from_json(data["ctx"])
    spec = _spec_from_json(data["spec"])
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return ActivationRecord(
        ctx=ctx,
        spec=spec,
        metadata=metadata,
        recorded_at=float(data.get("recorded_at", 0.0)),
    )


def _entry_to_record_metadata(record: ActivationRecord, entry: TensorEntry) -> dict[str, Any]:
    """Return persisted raw activation blob metadata for one record."""

    metadata = entry.to_dict()
    metadata["shape"] = (
        entry.shape if record.ctx.tensor_shape is None else list(record.ctx.tensor_shape)
    )
    return metadata


def _transformed_entry_to_record_metadata(
    record: ActivationRecord,
    entry: TensorEntry,
) -> dict[str, Any]:
    """Return persisted transformed activation blob metadata for one record.

    Stored under a ``transformed_activation_*`` key namespace so it can be
    rehydrated independently from the raw activation blob.
    """

    return {
        "transformed_activation_blob_id": entry.blob_id,
        "transformed_activation_kind": entry.kind,
        "transformed_activation_relative_path": entry.relative_path,
        "transformed_activation_shape": list(entry.shape),
        "transformed_activation_dtype": entry.dtype,
        "transformed_activation_backend": entry.backend,
        "transformed_activation_bytes": entry.bytes,
        "transformed_activation_sha256": entry.sha256,
        "transformed_activation_layout": entry.layout,
        "transformed_activation_device_at_save": entry.device_at_save,
    }


def _ctx_to_json(ctx: Any) -> dict[str, Any]:
    """Convert a RecordContext into JSON data without recursive history."""

    data = asdict(ctx)
    data["module_stack"] = [asdict(frame) for frame in ctx.module_stack]
    data["recent_events"] = []
    data["recent_ops"] = []
    data["tensor_dtype"] = _dtype_to_name(ctx.tensor_dtype)
    data["tensor_device"] = None if ctx.tensor_device is None else str(ctx.tensor_device)
    return data


def _ctx_from_json(data: dict[str, Any]) -> Any:
    """Convert JSON data into a RecordContext."""

    from .types import RecordContext

    values = dict(data)
    values["module_stack"] = tuple(ModuleStackFrame(**frame) for frame in values["module_stack"])
    values["recent_events"] = ()
    values["recent_ops"] = ()
    values["parent_labels"] = tuple(values.get("parent_labels", ()))
    values["tensor_shape"] = (
        None if values.get("tensor_shape") is None else tuple(values["tensor_shape"])
    )
    values["tensor_dtype"] = _dtype_from_name(values.get("tensor_dtype"))
    values["tensor_device"] = (
        None if values.get("tensor_device") is None else torch.device(values["tensor_device"])
    )
    return RecordContext(**values)


def _spec_to_json(spec: CaptureSpec) -> dict[str, Any]:
    """Convert a CaptureSpec into JSON data."""

    return {
        "save_activation": spec.save_activation,
        "save_metadata": spec.save_metadata,
        "keep_grad": spec.keep_grad,
        "device": None if spec.device is None else str(spec.device),
        "dtype": _dtype_to_name(spec.dtype),
    }


def _spec_from_json(data: dict[str, Any]) -> CaptureSpec:
    """Build a CaptureSpec from JSON data."""

    return CaptureSpec(
        save_activation=bool(data.get("save_activation", True)),
        save_metadata=bool(data.get("save_metadata", True)),
        keep_grad=bool(data.get("keep_grad", False)),
        device=data.get("device"),
        dtype=_dtype_from_name(data.get("dtype")),
    )


def _dtype_to_name(dtype: torch.dtype | None) -> str | None:
    """Return a portable dtype name."""

    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    """Resolve a portable dtype name to a torch dtype."""

    if name is None:
        return None
    return getattr(torch, name)


def _write_metadata(path: Path, recording: Recording, options: RecordingOptions) -> None:
    """Write fastlog JSON metadata."""

    metadata = {
        "n_passes": recording.n_passes,
        "n_records": len(recording.records),
        "pass_start_times": recording.pass_start_times,
        "pass_end_times": recording.pass_end_times,
        "predicate_failure_overflow_count": recording.predicate_failure_overflow_count,
        "keep_op_repr": recording.keep_op_repr,
        "keep_module_repr": recording.keep_module_repr,
        "activation_postfunc_repr": recording.activation_postfunc_repr,
        "save_raw_activation": options.save_raw_activation,
        "history_size": options.history_size,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_fastlog_manifest(tensor_entries: list[TensorEntry]) -> Manifest:
    """Build a manifest for a finalized fastlog directory bundle."""

    return Manifest(
        io_format_version=IO_FORMAT_VERSION,
        torchlens_version=TORCHLENS_VERSION,
        torch_version=torch.__version__,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=f"{platform.system().lower()}-{platform.machine().lower()}",
        created_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        bundle_format="fastlog-directory",
        n_layers=0,
        n_activation_blobs=sum(1 for entry in tensor_entries if entry.kind == "activation"),
        n_gradient_blobs=sum(1 for entry in tensor_entries if entry.kind == "gradient"),
        n_auxiliary_blobs=sum(
            1 for entry in tensor_entries if entry.kind not in {"activation", "gradient"}
        ),
        tensors=tensor_entries,
        unsupported_tensors=[],
    )
