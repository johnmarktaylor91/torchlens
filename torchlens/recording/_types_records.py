"""Frozen capture decisions, storage intents, and retained event records."""

from __future__ import annotations

__tl_layer__ = "L2"

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .._errors import InvalidArgumentError
from .._ir.predicate import (
    EventKind,
    ModuleStackFrame as ModuleStackFrame,
    RecordContext,
    RetroactiveCaptureDecision,
)
from .._utils.tensor_utils import SaveMode


def _distinct_label_index_keys(label: str, raw_label: str | None) -> tuple[str, ...]:
    """Return the distinct label keys (primary, then raw alias) that index one record."""
    if raw_label is None or raw_label == label:
        return (label,)
    return (label, raw_label)


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """Capture policy returned by predicate callbacks.

    Parameters
    ----------
    save_out:
        Whether tensor payloads should be retained for this event.
    save_metadata:
        Whether non-payload metadata should be retained for this event.
    save_grad:
        Whether the in-RAM tensor clone should stay attached to autograd.
    device:
        Optional target device for retained payloads.
    dtype:
        Optional target dtype for retained payloads.
    save_mode:
        Tensor retention mode for saved payloads.
    """

    save_out: bool = True
    save_metadata: bool = True
    save_grad: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    save_mode: SaveMode = "copy"

    def __post_init__(self) -> None:
        """Normalize and validate capture save-mode settings."""

        from .._utils.tensor_utils import SAVE_MODES

        if self.save_mode not in SAVE_MODES:
            raise InvalidArgumentError(
                "save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'; "
                f"received {self.save_mode!r}",
                code="save_mode_invalid",
                remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
                argument="save_mode",
            )
        if self.save_mode == "view" and not self.save_grad:
            object.__setattr__(self, "save_grad", True)


CaptureDecision = bool | CaptureSpec | None
PredicateDecision = CaptureDecision | RetroactiveCaptureDecision


@dataclass(frozen=True, slots=True)
class StorageIntent:
    """Resolved storage destinations for a capture decision."""

    in_ram: bool
    on_disk: bool


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """One retained fastlog event.

    Parameters
    ----------
    ctx:
        Frozen record context produced for the underlying event.
    spec:
        Resolved capture policy for this record.
    ram_payload:
        Raw out copy retained in memory, or ``None`` when not stored
        either because the record is metadata-only or the caller opted out
        via ``save_raw_activations=False``.
    disk_payload:
        Raw out copy persisted to disk, or ``None`` when no disk
        target is active or the caller opted out via
        ``save_raw_activations=False``.
    transformed_ram_payload:
        Output of ``activation_transform`` retained in memory. ``None`` when
        no transform is configured for the recording.
    transformed_disk_payload:
        Output of ``activation_transform`` persisted to disk. ``None`` when
        no transform is configured for the recording.
    metadata:
        Auxiliary record metadata, including disk blob entries when present.
    recorded_at:
        Wall-clock time the record was created.
    """

    ctx: RecordContext
    spec: CaptureSpec
    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    transformed_ram_payload: torch.Tensor | None = None
    transformed_disk_payload: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_time: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        """Return a bounded one-line record summary.

        Payloads render as metadata descriptors (dtype/shape/storage), never
        contents: the dataclass auto-repr printed full tensor payloads AND
        the recursive ``ctx`` lookback (lovely bug 1, CRITICAL).
        """

        stores = [
            name
            for name, payload in (
                ("ram", self.ram_payload),
                ("disk", self.disk_payload),
                ("transformed_ram", self.transformed_ram_payload),
                ("transformed_disk", self.transformed_disk_payload),
            )
            if payload is not None
        ]
        payload = self.ram_payload if self.ram_payload is not None else self.disk_payload
        if payload is None:
            descriptor = "metadata_only"
        else:
            dtype = str(payload.dtype).replace("torch.", "")
            descriptor = f"{dtype}[{','.join(str(d) for d in payload.shape)}]"
        return (
            f"ActivationRecord(label={self.ctx.label!r}, pass={self.ctx.pass_number}, "
            f"payload={descriptor}, stored={'+'.join(stores) if stores else 'none'})"
        )


@dataclass(frozen=True, slots=True)
class GradRecordContext:
    """Predicate input schema for one fastlog backward gradient event.

    Parameters
    ----------
    label:
        Label assigned to the autograd node during the backward walk.
    layer_label:
        Forward fastlog label joined by ``grad_fn_handle`` identity, when available.
    op_label:
        Alias of the joined forward operation label for selector parity.
    module_stack:
        Forward module stack captured for the joined operation.
    has_forward_op:
        Whether this backward node corresponds to a predicate-mode forward op.
    has_op:
        Whether this backward node has a joined forward op.
    """

    label: str
    grad_fn_class_name: str
    kind: str
    backward_call_index: int
    grad_kind: Literal["grad_input", "grad_output"]
    grad_input_index: int | None = None
    grad_output_index: int | None = None
    layer_label: str | None = None
    op_label: str | None = None
    module_stack: tuple[Any, ...] = ()
    has_forward_op: bool = False
    has_op: bool = False
    pass_number: int | None = None
    order: int | None = None
    event_index: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    tensor_device: torch.device | None = None

    @property
    def effective_label(self) -> str:
        """Return the forward label when joined, otherwise the grad-fn label."""

        return self.layer_label or self.label


@dataclass(frozen=True, slots=True)
class GradRecord:
    """One retained fastlog gradient event."""

    ctx: GradRecordContext
    spec: CaptureSpec
    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    transformed_ram_payload: torch.Tensor | None = None
    transformed_disk_payload: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PredicateFailure:
    """One captured predicate exception."""

    event_index: int
    kind: EventKind
    label: str
    traceback: str


# Preserve public type identity and pickle paths through the host's re-exports.
CaptureSpec.__module__ = "torchlens.recording.types"
StorageIntent.__module__ = "torchlens.recording.types"
ActivationRecord.__module__ = "torchlens.recording.types"
GradRecordContext.__module__ = "torchlens.recording.types"
GradRecord.__module__ = "torchlens.recording.types"
PredicateFailure.__module__ = "torchlens.recording.types"
