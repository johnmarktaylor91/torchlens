"""Core dataclasses for fastlog predicate recording."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

from ..captured_run import CapturedRun
from ..ir.predicate import EventKind, ModuleStackFrame, RecordContext
from ..utils.tensor_utils import SaveMode

if TYPE_CHECKING:
    from ..capture.projections import RecordingState
    from ..data_classes.trace import Trace


def _public_fastlog_layer_label(ctx: RecordContext) -> str:
    """Return a compact public label for a predicate-mode operation context."""

    if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
        return f"{ctx.layer_type}_{ctx.type_index}"
    return ctx.label


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """Capture policy returned by predicate callbacks.

    Parameters
    ----------
    save_out:
        Whether tensor payloads should be retained for this event.
    save_metadata:
        Whether non-payload metadata should be retained for this event.
    keep_grad:
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
    keep_grad: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    save_mode: SaveMode = "copy"

    def __post_init__(self) -> None:
        """Normalize and validate capture save-mode settings."""

        if self.save_mode not in {"copy", "reference", "view", "cpu_async"}:
            raise ValueError("save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'")
        if self.save_mode == "view" and not self.keep_grad:
            object.__setattr__(self, "keep_grad", True)


CaptureDecision = bool | CaptureSpec | None


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
    recorded_at: float = field(default_factory=time.time)


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
    type: str
    backward_call_index: int
    grad_kind: Literal["grad_input", "grad_output"]
    grad_input_index: int | None = None
    grad_output_index: int | None = None
    layer_label: str | None = None
    op_label: str | None = None
    module_stack: tuple[Any, ...] = ()
    has_forward_op: bool = False
    has_op: bool = False
    pass_index: int | None = None
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
class GradientRecord:
    """One retained fastlog gradient event."""

    ctx: GradRecordContext
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

    def draw(self, **kwargs: Any) -> str:
        """Render a flat Graphviz graph of trace operation events."""

        from ..visualization.fastlog_live import draw

        return draw(self, **kwargs)

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

        from ..capture.predicates import _normalize_capture_decision

        decisions: list[bool] = []
        for ctx in self.contexts:
            predicate = (
                other_keep_module if ctx.kind in {"module_enter", "module_exit"} else other_keep_op
            )
            result = predicate(ctx) if predicate is not None else False
            spec = _normalize_capture_decision(result, ctx, False)
            if not isinstance(spec, CaptureSpec):
                decisions.append(True)
                continue
            decisions.append(spec.save_out or spec.save_metadata)
        return RecordingTrace(
            contexts=self.contexts,
            decisions=tuple(decisions),
            predicate_failures=self.predicate_failures,
        )


@dataclass(frozen=True, slots=True)
class Recording(CapturedRun):
    """Result of a fastlog recording session."""

    records: list[ActivationRecord]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]
    bundle_path: Path | None
    n_ops: int
    n_records: int
    start_times: list[float]
    end_times: list[float]
    predicate_failures: list[PredicateFailure]
    predicate_failure_overflow_count: int
    keep_op_repr: str | None
    keep_module_repr: str | None
    history_size: int
    orphan_records: list[dict[str, Any]] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None
    halts_by_pass: dict[int, str] = field(default_factory=dict)
    grad_records: list[GradientRecord] = field(default_factory=list)
    grad_by_pass: dict[int, list[int]] = field(default_factory=dict)
    grad_by_label: dict[str, list[int]] = field(default_factory=dict)
    grad_by_grad_fn_label: dict[str, list[int]] = field(default_factory=dict)
    save_grads_repr: str | None = None
    _grad_transform_repr: str | None = None
    _activation_transform_repr: str | None = None
    recovered: bool = False
    recovery_warnings: list[str] = field(default_factory=list)
    _capture_events: Any | None = field(default=None, repr=False, compare=False)
    _output_tensors: list[torch.Tensor] = field(default_factory=list, repr=False, compare=False)
    _output_tensor_addresses: list[str] = field(default_factory=list, repr=False, compare=False)
    _records_built: bool = field(default=True, repr=False, compare=False)
    _recording_trace: RecordingTrace | None = field(default=None, repr=False, compare=False)
    _recording_state: Any | None = field(default=None, repr=False, compare=False)

    def __getattribute__(self, name: str) -> Any:
        """Populate lazy record projections when ``records`` is read."""

        if name == "records":
            ensure = object.__getattribute__(self, "_ensure_records")
            ensure()
        return object.__getattribute__(self, name)

    @classmethod
    def from_capture_events(cls, session: Any) -> "Recording":
        """Build a lazy Recording projection from a predicate capture session.

        Parameters
        ----------
        session:
            Trace-like session exposing ``capture_events`` and
            ``_fastlog_recording`` metadata.

        Returns
        -------
        Recording
            Recording whose retained records are built lazily from events.
        """

        base = session._fastlog_recording
        object.__setattr__(base, "_capture_events", session.capture_events)
        object.__setattr__(
            base,
            "_output_tensors",
            list(getattr(session, "output_tensors", [])),
        )
        object.__setattr__(
            base,
            "_output_tensor_addresses",
            list(getattr(session, "output_tensor_addresses", [])),
        )
        object.__setattr__(base, "_recording_state", getattr(session, "recording_state", None))
        object.__setattr__(base, "_records_built", bool(base.records))
        object.__setattr__(base, "_recording_trace", None)
        return base

    def _ensure_records(self) -> None:
        """Populate retained records from CaptureEvents on first record access."""

        if self._records_built:
            return
        from ..capture.projections import activation_record_from_event

        records = object.__getattribute__(self, "records")
        records.clear()
        self.by_pass.clear()
        self.by_label.clear()
        self.by_address.clear()
        if self._capture_events is not None:
            for event in self._capture_events.op_events:
                record = activation_record_from_event(event)
                if record is None:
                    continue
                index = len(records)
                records.append(record)
                self.by_pass.setdefault(record.ctx.pass_index, []).append(index)
                self.by_label.setdefault(record.ctx.label, []).append(
                    (record.ctx.pass_index, index)
                )
                if record.ctx.raw_label is not None:
                    self.by_label.setdefault(record.ctx.raw_label, []).append(
                        (record.ctx.pass_index, index)
                    )
                if record.ctx.address is not None:
                    self.by_address.setdefault(record.ctx.address, []).append(index)
        object.__setattr__(self, "n_records", len(records))
        object.__setattr__(self, "_records_built", True)

    @property
    def recording_trace(self) -> RecordingTrace:
        """Return a lazy trace projection over all capture events."""

        if self._recording_trace is None:
            from ..capture.projections import recording_trace_from_events

            contexts = (
                ()
                if self._capture_events is None
                else recording_trace_from_events(self._capture_events)
            )
            object.__setattr__(
                self,
                "_recording_trace",
                RecordingTrace(
                    contexts=contexts,
                    decisions=tuple(
                        bool(getattr(event, "predicate_matched", False))
                        for event in getattr(self._capture_events, "op_events", ())
                    ),
                    predicate_failures=tuple(self.predicate_failures),
                ),
            )
        trace = self._recording_trace
        if trace is None:
            raise RuntimeError("recording_trace projection was not initialized")
        return trace

    @property
    def activation_transform_repr(self) -> str | None:
        """Canonical repr for the out transform callable.

        Returns
        -------
        str | None
            Callable repr captured at recording time, if any.
        """

        return self._activation_transform_repr

    @property
    def grad_transform_repr(self) -> str | None:
        """Canonical repr for the gradient transform callable."""

        return self._grad_transform_repr

    def add_grad_record(self, record: GradientRecord) -> None:
        """Append one retained gradient record and update indexes."""

        index = len(self.grad_records)
        self.grad_records.append(record)
        if record.ctx.pass_index is not None:
            self.grad_by_pass.setdefault(record.ctx.pass_index, []).append(index)
        if record.ctx.layer_label is not None:
            self.grad_by_label.setdefault(record.ctx.layer_label, []).append(index)
        self.grad_by_label.setdefault(record.ctx.label, []).append(index)
        self.grad_by_grad_fn_label.setdefault(record.ctx.label, []).append(index)

    def log_backward(
        self,
        loss: torch.Tensor,
        *,
        save_grads: Callable[[GradRecordContext], CaptureDecision]
        | bool
        | CaptureSpec
        | None = None,
        default_grad: bool | CaptureSpec | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> "Recording":
        """Run ``loss.backward`` while capturing selected fastlog gradients.

        Parameters
        ----------
        loss:
            Loss tensor whose autograd graph should be walked.
        save_grads:
            Optional per-gradient predicate overriding the recording default.
        default_grad:
            Default capture decision when no predicate is configured.
        retain_graph:
            Forwarded to ``Tensor.backward``.
        create_graph:
            Forwarded to ``Tensor.backward``.

        Returns
        -------
        Recording
            This recording, mutated with gradient records.
        """

        if self.halted:
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                f"Cannot call log_backward on halted Recording (halt_reason={self.halt_reason!r})."
            )

        from ..backends.torch.backward import log_recording_backward

        return log_recording_backward(
            self,
            loss,
            save_grads=save_grads,
            default_grad=default_grad,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

    def __getitem__(self, key: int | str) -> ActivationRecord | list[ActivationRecord]:
        """Return records by integer index or raw/final label."""

        self._ensure_records()
        if isinstance(key, int):
            return self.records[key]
        indexes = self.by_label[key]
        return [self.records[index] for _, index in indexes]

    def __iter__(self) -> Iterator[ActivationRecord]:
        """Iterate over retained out records."""

        self._ensure_records()
        return iter(self.records)

    def __len__(self) -> int:
        """Return the number of retained records."""

        self._ensure_records()
        return len(self.records)

    def iter_pass(self, call_index: int) -> Iterator[ActivationRecord]:
        """Iterate over records retained for one pass."""

        self._ensure_records()
        for index in self.by_pass.get(call_index, []):
            yield self.records[index]

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame representation of retained records."""

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        self._ensure_records()
        rows = [
            {
                "kind": record.ctx.kind,
                "label": record.ctx.label,
                "pass_index": record.ctx.pass_index,
                "event_index": record.ctx.event_index,
                "save_out": record.spec.save_out,
                "save_metadata": record.spec.save_metadata,
            }
            for record in self.records
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a concise human-readable recording summary."""

        return (
            f"Recording(n_ops={self.n_ops}, n_records={len(self)}, "
            f"n_grad_records={len(self.grad_records)})"
        )

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

    def to_trace(self) -> "Trace":
        """Cook this recording's event stream into a full ``Trace``.

        Returns
        -------
        Trace
            Trace built by the normal Step-0 materializer and postprocess pipeline.

        Raises
        ------
        RuntimeError
            If the recording does not retain the topology-complete event stream.
        """

        if self._capture_events is None:
            raise RuntimeError(
                "Recording.to_trace() requires retained capture events; disk-recovered "
                "recordings do not contain enough topology metadata."
            )
        from ..data_classes.trace import Trace

        trace = Trace(model_class_name="RecordedModel")
        trace.capture_mode = "exhaustive"
        trace._predicate_save_options = object()
        trace._replay_arg_version_data_complete = False
        trace.capture_events = self._capture_events
        trace.output_layers = [
            event.label_raw
            for event in self._capture_events.op_events
            if getattr(event, "is_output_parent", False)
        ]
        trace._postprocess(
            list(self._output_tensors),
            list(self._output_tensor_addresses),
        )
        return trace


def _mark_recording_halted(recording: Recording, pass_index: int, reason: str) -> None:
    """Set halt state on a frozen ``Recording``.

    Parameters
    ----------
    recording:
        Recording to mutate via ``object.__setattr__``.
    pass_index:
        Recorder pass index that observed the halt.
    reason:
        User-supplied halt reason. Empty string means no reason was provided.
    """

    recording.halts_by_pass.setdefault(pass_index, reason)
    if recording.halted:
        return
    object.__setattr__(recording, "halted", True)
    object.__setattr__(recording, "halt_reason", reason)


def build_grad_record_context(
    recording_state: "RecordingState",
    grad_fn_handle: Any,
    grad: torch.Tensor | None,
    *,
    label: str,
    grad_kind: Literal["grad_input", "grad_output"],
    backward_call_index: int,
    grad_input_index: int | None = None,
    grad_output_index: int | None = None,
) -> GradRecordContext:
    """Build a fastlog gradient context from a backward node and optional join."""

    forward_ctx = recording_state.grad_fn_to_context.get(grad_fn_handle)
    shape = tuple(grad.shape) if grad is not None else None
    dtype = grad.dtype if grad is not None else None
    tensor_device = grad.device if grad is not None else None
    grad_fn_type = type(grad_fn_handle).__name__.removesuffix("Backward0").lower()
    if forward_ctx is None:
        return GradRecordContext(
            label=label,
            grad_fn_class_name=type(grad_fn_handle).__name__,
            type=grad_fn_type,
            backward_call_index=backward_call_index,
            grad_kind=grad_kind,
            grad_input_index=grad_input_index,
            grad_output_index=grad_output_index,
            shape=shape,
            dtype=dtype,
            tensor_device=tensor_device,
        )
    return GradRecordContext(
        label=label,
        grad_fn_class_name=type(grad_fn_handle).__name__,
        type=grad_fn_type,
        backward_call_index=backward_call_index,
        grad_kind=grad_kind,
        grad_input_index=grad_input_index,
        grad_output_index=grad_output_index,
        layer_label=_public_fastlog_layer_label(forward_ctx),
        op_label=_public_fastlog_layer_label(forward_ctx),
        module_stack=forward_ctx.module_stack,
        has_forward_op=True,
        has_op=True,
        pass_index=forward_ctx.pass_index,
        event_index=forward_ctx.event_index,
        shape=shape,
        dtype=dtype,
        tensor_device=tensor_device,
    )
