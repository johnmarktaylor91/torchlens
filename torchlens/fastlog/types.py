"""Core dataclasses for fastlog predicate recording."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

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


CaptureDecision = bool | CaptureSpec | None


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

    def __getattr__(self, name: str) -> Any:
        """Raise a schema-specific error for unknown predicate fields.

        Parameters
        ----------
        name:
            Missing attribute name.

        Raises
        ------
        RecordContextFieldError
            Always raised for missing fields.
        """

        from .exceptions import RecordContextFieldError

        raise RecordContextFieldError(name)


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
        Raw activation copy retained in memory, or ``None`` when not stored
        either because the record is metadata-only or the caller opted out
        via ``save_raw_activation=False``.
    disk_payload:
        Raw activation copy persisted to disk, or ``None`` when no disk
        target is active or the caller opted out via
        ``save_raw_activation=False``.
    transformed_ram_payload:
        Output of ``activation_postfunc`` retained in memory. ``None`` when
        no postfunc is configured for the recording.
    transformed_disk_payload:
        Output of ``activation_postfunc`` persisted to disk. ``None`` when
        no postfunc is configured for the recording.
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
    decisions: tuple[bool, ...] = ()
    predicate_failures: tuple[PredicateFailure, ...] = ()

    @property
    def events(self) -> tuple[RecordContext, ...]:
        """Return chronological dry-run events."""

        return self.contexts

    def print_tree(self) -> str:
        """Return a unicode-indented event tree for this trace."""

        from ..visualization.fastlog_live import print_tree

        return print_tree(self)

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame representation of trace events."""

        from ..visualization.fastlog_live import to_pandas

        return to_pandas(self)

    def show_graph(self, **kwargs: Any) -> str:
        """Render a flat Graphviz graph of trace operation events."""

        from ..visualization.fastlog_live import show_graph

        return show_graph(self, **kwargs)

    def summary(self) -> str:
        """Return a concise human-readable dry-run summary."""

        from ..visualization.fastlog_live import summary

        return summary(self)

    def timeline_html(self) -> Any:
        """Return an IPython HTML timeline for this trace."""

        from ..visualization.fastlog_live import timeline_html

        return timeline_html(self)

    def repredicate(
        self,
        other_keep_op: Callable[[RecordContext], CaptureDecision] | None = None,
        other_keep_module: Callable[[RecordContext], CaptureDecision] | None = None,
    ) -> "RecordingTrace":
        """Return a new trace with decisions from new predicates.

        Parameters
        ----------
        other_keep_op:
            Predicate for op, input, and buffer events.
        other_keep_module:
            Predicate for module entry and exit events.

        Returns
        -------
        RecordingTrace
            New trace sharing the same event tuple and predicate failures.
        """

        from ._predicate import _normalize_capture_decision

        decisions: list[bool] = []
        for ctx in self.contexts:
            predicate = (
                other_keep_module if ctx.kind in {"module_enter", "module_exit"} else other_keep_op
            )
            result = predicate(ctx) if predicate is not None else False
            spec = _normalize_capture_decision(result, ctx, False)
            decisions.append(spec.save_activation or spec.save_metadata)
        return RecordingTrace(
            contexts=self.contexts,
            decisions=tuple(decisions),
            predicate_failures=self.predicate_failures,
        )


@dataclass(frozen=True, slots=True)
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
    activation_postfunc_repr: str | None = None
    recovered: bool = False
    recovery_warnings: list[str] = field(default_factory=list)

    @property
    def activation_transform_repr(self) -> str | None:
        """Canonical repr for the activation transform callable.

        Returns
        -------
        str | None
            Callable repr captured at recording time, if any.
        """

        return self.activation_postfunc_repr

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

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

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

    def enrich(self, steps: list[str] | str) -> "Recording":
        """Return a new recording with requested incremental enrichments.

        Parameters
        ----------
        steps:
            Enrichment names, or ``"all-feasible"`` for all currently computable
            enrichments.

        Returns
        -------
        Recording
            New immutable recording value with enriched records.
        """

        from ..postprocess.incremental import enrich_recording

        return enrich_recording(self, steps)
