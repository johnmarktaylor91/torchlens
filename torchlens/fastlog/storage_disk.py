"""Synchronous disk storage backend for fastlog recordings."""

from __future__ import annotations

import json
import platform
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from .. import __version__ as TORCHLENS_VERSION
from .._io import TLSPEC_VERSION, TorchLensIOError
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
    GradRecordContext,
    ModuleStackFrame,
    RecordContext,
    Recording,
    StorageIntent,
)
from ..ir.refs import DeviceRef, DtypeRef

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
                kind="out",
                label=record.ctx.label,
            )
            self._tensor_entries.append(entry)
            stored_record.metadata.update(_entry_to_record_metadata(record, entry))
            wrote_blob = True
        if record.transformed_disk_payload is not None:
            blob_id = self.writer.next_blob_id()
            transformed_label = f"{record.ctx.label}::transformed_out"
            entry = self.writer.write_blob(
                blob_id,
                record.transformed_disk_payload,
                kind="transformed_out",
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
        ctx: "RecordContext | GradRecordContext | None" = None,
        kind: str = "activation",
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
            Recording options carrying ``activation_transform`` /
            ``save_raw_activations``.
        ctx:
            Record context used to enrich transform error messages.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
            ``(ram_payload, disk_payload, transformed_ram_payload,
            transformed_disk_payload)``. Any element may be ``None``.
        """

        transform = options.grad_transform if kind == "grad" else options.activation_transform
        save_raw = options.save_raw_gradients if kind == "grad" else options.save_raw_activations
        return _resolve_storage(
            tensor,
            spec,
            intent,
            activation_transform=transform,
            save_raw_activations=save_raw,
            ctx=ctx,
            kind="grad" if kind == "grad" else "activation",
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
        dtype = _torch_dtype_from_ref(record.ctx.dtype)
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
    """Convert one out record into JSON-serializable metadata."""

    return {
        "ctx": _ctx_to_json(record.ctx),
        "spec": _spec_to_json(record.spec),
        "metadata": dict(record.metadata),
        "recorded_at": record.recorded_at,
    }


def record_from_json(data: dict[str, Any]) -> ActivationRecord:
    """Build an out record from JSON-decoded fastlog metadata."""

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
    """Return persisted raw out blob metadata for one record."""

    metadata = entry.to_dict()
    metadata["shape"] = entry.shape if record.ctx.shape is None else list(record.ctx.shape)
    return metadata


def _transformed_entry_to_record_metadata(
    record: ActivationRecord,
    entry: TensorEntry,
) -> dict[str, Any]:
    """Return persisted transformed out blob metadata for one record.

    Stored under a ``transformed_out_*`` key namespace so it can be
    rehydrated independently from the raw out blob.
    """

    return {
        "transformed_out_blob_id": entry.blob_id,
        "transformed_out_kind": entry.kind,
        "transformed_out_relative_path": entry.relative_path,
        "transformed_out_shape": list(entry.shape),
        "transformed_out_dtype": entry.dtype,
        "transformed_out_backend": entry.backend,
        "transformed_out_bytes": entry.bytes,
        "transformed_out_sha256": entry.sha256,
        "transformed_out_layout": entry.layout,
        "transformed_out_device_at_save": entry.device_at_save,
    }


def _ctx_to_json(ctx: Any) -> dict[str, Any]:
    """Convert a RecordContext into JSON data without recursive history."""

    data = asdict(ctx)
    data["module_stack"] = [asdict(frame) for frame in ctx.module_stack]
    data["recent_events"] = []
    data["recent_ops"] = []
    data["dtype"] = None if ctx.dtype is None else str(ctx.dtype)
    data["tensor_device"] = None if ctx.tensor_device is None else str(ctx.tensor_device)
    return data


def _torch_dtype_from_ref(dtype: Any) -> torch.dtype | None:
    """Resolve a neutral dtype reference to a torch dtype when possible.

    Parameters
    ----------
    dtype
        ``DtypeRef``, torch dtype, string, or ``None``.

    Returns
    -------
    torch.dtype | None
        Torch dtype for validation checks, or ``None`` when unavailable.
    """

    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).replace("torch.", "")
    return cast(torch.dtype | None, getattr(torch, name, None))


def _ctx_from_json(data: dict[str, Any]) -> Any:
    """Convert JSON data into a RecordContext."""

    from .types import RecordContext

    values = dict(data)
    values["module_stack"] = tuple(ModuleStackFrame(**frame) for frame in values["module_stack"])
    values["recent_events"] = ()
    values["recent_ops"] = ()
    values["parent_labels"] = tuple(values.get("parent_labels", ()))
    values["shape"] = None if values.get("shape") is None else tuple(values["shape"])
    values["dtype"] = DtypeRef.from_value(values.get("dtype"))
    values["tensor_device"] = DeviceRef.from_value(values.get("tensor_device"))
    return RecordContext(**values)


def _spec_to_json(spec: CaptureSpec) -> dict[str, Any]:
    """Convert a CaptureSpec into JSON data."""

    return {
        "save_out": spec.save_out,
        "save_metadata": spec.save_metadata,
        "keep_grad": spec.keep_grad,
        "device": None if spec.device is None else str(spec.device),
        "dtype": _dtype_to_name(spec.dtype),
    }


def _spec_from_json(data: dict[str, Any]) -> CaptureSpec:
    """Build a CaptureSpec from JSON data."""

    return CaptureSpec(
        save_out=bool(data.get("save_out", True)),
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
    return cast(torch.dtype, getattr(torch, name))


def _write_metadata(path: Path, recording: Recording, options: RecordingOptions) -> None:
    """Write fastlog JSON metadata."""

    metadata = {
        "n_ops": recording.n_ops,
        "n_records": len(recording.records),
        "start_times": recording.start_times,
        "end_times": recording.end_times,
        "predicate_failure_overflow_count": recording.predicate_failure_overflow_count,
        "halted": recording.halted,
        "halt_reason": recording.halt_reason,
        "halts_by_pass": recording.halts_by_pass,
        "keep_op_repr": recording.keep_op_repr,
        "keep_module_repr": recording.keep_module_repr,
        "_activation_transform_repr": recording._activation_transform_repr,
        "save_raw_activations": options.save_raw_activations,
        "history_size": options.history_size,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_fastlog_manifest(tensor_entries: list[TensorEntry]) -> Manifest:
    """Build a manifest for a finalized fastlog directory bundle."""

    return Manifest(
        tlspec_version=TLSPEC_VERSION,
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
        n_out_blobs=sum(1 for entry in tensor_entries if entry.kind == "out"),
        n_grad_blobs=sum(1 for entry in tensor_entries if entry.kind == "grad"),
        n_auxiliary_blobs=sum(1 for entry in tensor_entries if entry.kind not in {"out", "grad"}),
        tensors=tensor_entries,
        unsupported_tensors=[],
    )
