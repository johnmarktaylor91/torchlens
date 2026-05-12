"""In-memory storage backend for fastlog recordings."""

from __future__ import annotations

import torch

from ._storage_resolver import _resolve_storage
from .options import RecordingOptions
from .types import (
    ActivationRecord,
    CaptureSpec,
    GradRecordContext,
    RecordContext,
    Recording,
    StorageIntent,
)


class RamStorageBackend:
    """Append fastlog records to an in-memory recording."""

    def __init__(self, recording: Recording) -> None:
        """Initialize the RAM backend.

        Parameters
        ----------
        recording:
            Recording instance that receives retained records.
        """

        self.recording = recording

    def resolve_payloads(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        intent: StorageIntent,
        *,
        options: RecordingOptions,
        ctx: RecordContext | GradRecordContext | None,
        kind: str = "activation",
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve tensor payloads for in-memory capture.

        Parameters
        ----------
        tensor:
            Tensor selected for capture.
        spec:
            Capture policy for the selected tensor.
        intent:
            Storage intent resolved from streaming options.
        options:
            Active recording options carrying the out postfunc and
            ``save_raw_outs`` flag.
        ctx:
            Record context used to enrich postfunc error messages.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
            Raw RAM payload, raw disk payload (always ``None`` for the RAM
            backend), transformed RAM payload, and transformed disk payload.
        """

        transform = options.grad_transform if kind == "grad" else options.out_transform
        save_raw = options.save_raw_grads if kind == "grad" else options.save_raw_outs
        return _resolve_storage(
            tensor,
            spec,
            intent,
            out_postfunc=transform,
            save_raw_outs=save_raw,
            ctx=ctx,
            kind="grad" if kind == "grad" else "activation",
        )

    def append(self, record: ActivationRecord) -> None:
        """Append one retained record and update recording indexes.

        Parameters
        ----------
        record:
            Fastlog out record to retain.
        """

        index = len(self.recording.records)
        self.recording.records.append(record)
        self.recording.by_pass.setdefault(record.ctx.pass_index, []).append(index)
        self.recording.by_label.setdefault(record.ctx.label, []).append(
            (record.ctx.pass_index, index)
        )
        if record.ctx.raw_label is not None:
            self.recording.by_label.setdefault(record.ctx.raw_label, []).append(
                (record.ctx.pass_index, index)
            )
        if record.ctx.address is not None:
            self.recording.by_address.setdefault(record.ctx.address, []).append(index)

    def finalize(self) -> None:
        """Finalize RAM indexes after a pass."""

        self.recording.by_pass.clear()
        self.recording.by_label.clear()
        self.recording.by_address.clear()
        for index, record in enumerate(self.recording.records):
            self.recording.by_pass.setdefault(record.ctx.pass_index, []).append(index)
            self.recording.by_label.setdefault(record.ctx.label, []).append(
                (record.ctx.pass_index, index)
            )
            if record.ctx.raw_label is not None:
                self.recording.by_label.setdefault(record.ctx.raw_label, []).append(
                    (record.ctx.pass_index, index)
                )
            if record.ctx.address is not None:
                self.recording.by_address.setdefault(record.ctx.address, []).append(index)

    def abort(self, reason: str) -> None:
        """Ignore abort requests for RAM-only recordings.

        Parameters
        ----------
        reason:
            Human-readable abort reason.
        """

        _ = reason
