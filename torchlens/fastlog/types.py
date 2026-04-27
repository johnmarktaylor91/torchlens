"""Core dataclasses for fastlog predicate recording."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

EventKind = Literal["op", "module_enter", "module_exit", "input", "buffer"]


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """Capture policy returned by predicate callbacks.

    Parameters
    ----------
    save_activation:
        Whether tensor payloads should be retained for this event.
    save_metadata:
        Whether non-payload metadata should be retained for this event.
    keep_grad:
        Whether the in-RAM tensor clone should stay attached to autograd.
    device:
        Optional target device for retained payloads.
    dtype:
        Optional target dtype for retained payloads.
    """

    save_activation: bool = True
    save_metadata: bool = True
    keep_grad: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None


@dataclass(frozen=True, slots=True)
class StorageIntent:
    """Resolved storage destinations for a capture decision."""

    in_ram: bool
    on_disk: bool


@dataclass(frozen=True, slots=True)
class ModuleStackFrame:
    """One frame in the active module stack."""

    module_address: str
    module_type: str
    module_id: int
    pass_index: int


@dataclass(frozen=True, slots=True)
class RecordContext:
    """Predicate input schema for one chronological fastlog event."""

    kind: EventKind
    label: str
    raw_label: str | None
    pass_index: int
    event_index: int
    op_index: int | None
    layer_type: str | None
    layer_type_num: int | None
    creation_order: int | None
    func_name: str | None
    module_address: str | None
    module_type: str | None
    module_pass_index: int | None
    module_stack: tuple[ModuleStackFrame, ...]
    recent_events: tuple["RecordContext", ...]
    recent_ops: tuple["RecordContext", ...]
    parent_labels: tuple[str, ...]
    input_output_address: str | None
    tensor_shape: tuple[int, ...] | None
    tensor_dtype: torch.dtype | None
    tensor_device: torch.device | None
    tensor_requires_grad: bool | None
    output_index: int | None
    is_bottom_level_func: bool | None
    time_since_pass_start: float
    sample_id: str | int | None = None


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """One retained fastlog event."""

    ctx: RecordContext
    spec: CaptureSpec
    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PredicateFailure:
    """One captured predicate exception."""

    event_index: int
    kind: EventKind
    label: str
    traceback: str


@dataclass(frozen=True, slots=True)
class RecordingTrace:
    """Predicate dry-run trace without retained tensor payloads."""

    contexts: tuple[RecordContext, ...]
    predicate_failures: tuple[PredicateFailure, ...] = ()


@dataclass(slots=True)
class Recording:
    """Result of a fastlog recording session."""

    records: list[ActivationRecord]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_module_address: dict[str, list[int]]
    bundle_path: Path | None
    n_passes: int
    n_records: int
    pass_start_times: list[float]
    pass_end_times: list[float]
    predicate_failures: list[PredicateFailure]
    predicate_failure_overflow_count: int
    keep_op_repr: str | None
    keep_module_repr: str | None
    history_size: int
    recovered: bool = False
    recovery_warnings: list[str] = field(default_factory=list)

    def __getitem__(self, key: int | str) -> ActivationRecord | list[ActivationRecord]:
        """Return records by integer index or raw/final label."""

        if isinstance(key, int):
            return self.records[key]
        indexes = self.by_label[key]
        return [self.records[index] for _, index in indexes]

    def __iter__(self) -> Iterator[ActivationRecord]:
        """Iterate over retained activation records."""

        return iter(self.records)

    def __len__(self) -> int:
        """Return the number of retained records."""

        return len(self.records)

    def iter_pass(self, pass_num: int) -> Iterator[ActivationRecord]:
        """Iterate over records retained for one pass."""

        for index in self.by_pass.get(pass_num, []):
            yield self.records[index]

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame representation of retained records."""

        import pandas as pd

        rows = [
            {
                "kind": record.ctx.kind,
                "label": record.ctx.label,
                "pass_index": record.ctx.pass_index,
                "event_index": record.ctx.event_index,
                "save_activation": record.spec.save_activation,
                "save_metadata": record.spec.save_metadata,
            }
            for record in self.records
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a concise human-readable recording summary."""

        return f"Recording(n_passes={self.n_passes}, n_records={len(self.records)})"

    def enrich(self, steps: list[str]) -> "Recording":
        """Return this recording after validating enrichment step names."""

        if steps:
            raise NotImplementedError("fastlog enrichment is implemented in a later step")
        return self
