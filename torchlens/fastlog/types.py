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

__all__ = [
    "ActivationRecord",
    "CaptureSpec",
    "GradRecordContext",
    "ModuleStackFrame",
    "PredicateFailure",
    "RecordContext",
    "Recording",
    "StorageIntent",
]

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
    """Result of a fastlog recording session.

    Notes
    -----
    Failed partial recordings contain everything captured up to but excluding
    the failing op. user-op failures exclude the failing call; TL-side capture
    failures may include a skipped/partial current-call event. ``last_event_*``
    fields are best-effort details about the last captured event, not an
    authoritative description of the failing op. For reused multi-pass
    recorders, ``n_ops_completed`` is the total count of op-kind events captured
    across all completed passes in the recorder before the failure, not just
    the count from the failing pass.
    """

    records: list[ActivationRecord]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]
    bundle_path: Path | None
    n_ops: int
    start_times: list[float]
    end_times: list[float]
    predicate_failures: list[PredicateFailure]
    predicate_failure_overflow_count: int
    keep_op_repr: str | None
    keep_module_repr: str | None
    history_size: int
    orphan_records: list[dict[str, Any]] = field(default_factory=list)
    halted: bool = False
    status: Literal["complete", "halted", "partial_error", "recovered"] = "complete"
    failed: bool = False
    error_repr: str | None = None
    error_traceback: str | None = None
    n_ops_completed: int = 0
    last_successful_op_label: str | None = None
    last_event_label: str | None = None
    last_event_func: str | None = None
    last_event_source_line: str | None = None
    last_event_input_meta: str | None = None
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

    @property
    def n_passes(self) -> int:
        """Return the number of model-call passes captured by this recording.

        Returns
        -------
        int
            Number of explicit ``record()`` or ``Recorder.log()`` forward passes.
        """

        return self.n_ops

    def __getattribute__(self, name: str) -> Any:
        """Populate lazy record projections when ``records`` is read."""

        if name == "records":
            ensure = object.__getattribute__(self, "_ensure_records")
            ensure()
        return object.__getattribute__(self, name)

    @classmethod
    def from_capture_events(cls: type["Recording"], session: Any) -> "Recording":
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
        object.__setattr__(base, "_records_built", bool(object.__getattribute__(base, "records")))
        object.__setattr__(base, "_recording_trace", None)
        return base

    @property
    def n_records(self) -> int:
        """Return the current number of retained activation records."""

        return len(self.records)

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

        if self.failed:
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                "Cannot call log_backward on failed partial Recording; "
                "user-op failures exclude the failing call; TL-side capture failures may "
                "include a skipped/partial current-call event."
            )
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

        if self.failed:
            return (
                f"Recording(status={self.status!r}, n_passes={self.n_passes}, "
                f"n_records={len(self)}, n_ops_completed={self.n_ops_completed}, "
                "caveat='n_ops_completed counts op-kind events across completed "
                "passes in this recorder; user-op failures exclude the failing "
                "call; TL-side capture failures may include a skipped/partial "
                "current-call event.')"
            )
        return (
            f"Recording(n_passes={self.n_passes}, n_records={len(self)}, "
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
            Trace built by the normal Step-0 materializer and postprocess
            pipeline. A halted recording produces a valid halted Trace
            (``.halted`` True, frontier bound as the output node), mirroring the
            exhaustive ``tl.trace(model, x, halt=...)`` finalization path.

        Notes
        -----
        Predicate-mode buffer-write limitation. The fastlog/predicate capture
        that backs ``record(...)`` does NOT track registered-buffer *writes*
        (``install_buffer_write_tracker`` is gated to exhaustive capture). In a
        Trace cooked from a Recording this has two visible consequences that a
        Trace from exhaustive ``tl.trace()`` does not share:

        * ``Op.buffer_write_kind`` / ``Op.buffer_value_changed`` stay ``None`` on
          every buffer Op, so a buffer that was provably MUTATED during the
          forward pass (e.g. a training-mode ``BatchNorm``'s running stats)
          looks identical to a genuinely read-only buffer. A ``None``
          ``buffer_write_kind`` on a cooked Trace therefore means "buffer writes
          were not tracked in predicate mode", NOT "this buffer was
          provably not written". Buffer *reads* are still captured. Use
          ``tl.trace(model, x)`` if you need faithful buffer-write provenance.
        * Because the write-back / ``num_batches_tracked`` bump ops are not
          captured, a training-mode buffer-writing model yields FEWER graph
          nodes than exhaustive capture, which shifts the global-index component
          of every final label downstream of the first buffer write. For such
          models ``halt_reason`` / ``halt_frontier`` (and other final labels)
          can differ from ``tl.trace(model, x, halt=...)`` even though both
          describe the same semantic halt point. Eval-mode and
          buffer-write-free models are unaffected: halt provenance matches
          exhaustive capture exactly (both are remapped to final labels).

        Raises
        ------
        RuntimeError
            If the recording is a failed partial capture, if it does not retain
            the topology-complete event stream (e.g. disk-recovered), or if it
            is halted but retained no raw activation payload to bind as the
            output frontier.
        """

        if self.failed:
            raise RuntimeError(
                "Recording.to_trace() cannot materialize a failed partial Recording because "
                "the topology is incomplete; user-op failures exclude the failing call; "
                "TL-side capture failures may include a skipped/partial current-call event."
            )
        if self._capture_events is None:
            raise RuntimeError(
                "Recording.to_trace() requires retained capture events; disk-recovered "
                "recordings do not contain enough topology metadata."
            )
        from ..data_classes.trace import Trace
        from .options import RecordingOptions

        trace = Trace(model_class_name="RecordedModel")
        trace.capture_mode = "exhaustive"
        trace._predicate_save_options = RecordingOptions()
        trace._replay_arg_version_data_complete = False
        # Hand postprocess a STRUCTURAL COPY, never this frozen Recording's own
        # `_capture_events`. Later graph traversal replaces operation events in
        # place, so aliasing the Recording's buffer would mutate its read-only
        # event stream. The copy shares frozen OpEvents and tensor payloads by
        # reference (cheap; no activation cloning) while giving postprocess an
        # independent working projection. NOTE: `output_layers`,
        # `input_layers`, `buffer_layers`, `internal_source_ops`, the
        # `_layer_counter` seed, and `_recover_halt_frontier()` all read from
        # `self._capture_events` (the original, intact) below -- only the
        # materialized `trace.capture_events` is the copy.
        events_for_replay = self._capture_events.copy_for_replay()
        trace.capture_events = events_for_replay
        trace.output_layers = [
            event.label_raw
            for event in self._capture_events.op_events
            if getattr(event, "is_output_parent", False)
        ]
        # Halt-finalization parity. A halted recording never reached the
        # model's real return, so no captured event carries is_output_parent
        # (the output-marking step that stamps it only runs on a completed
        # pass). The exhaustive path handles this in _finalize_halted_trace
        # (capture/trace.py): it recovers a frontier-output tensor, sets
        # halted/halt_reason/halt_frontier, and marks that frontier as the
        # output parent so postprocess Step 1 synthesizes a dedicated output_N
        # node. Mirror that here so a halted Recording.to_trace() yields a
        # VALID halted Trace whose .halted is True (not silently False) and
        # whose output_layers/trace_self_consistency invariants hold -- instead
        # of the previous silent-wrong-.halted + "No output layers found"
        # crash. Self._output_tensors is empty for a halted pass, so seed the
        # frontier's saved payload as the sole output tensor (no fabrication --
        # the tensor is the recording's own retained raw activation).
        halt_output_tensors = list(self._output_tensors)
        halt_output_addresses = list(self._output_tensor_addresses)
        if self.halted:
            frontier_label, frontier_tensor = self._recover_halt_frontier()
            trace.halted = True
            trace.halt_reason = self.halt_reason
            trace.halt_frontier = self.halt_reason
            trace.raw_output = None
            trace.output_layers = [frontier_label]
            halt_output_tensors = [frontier_tensor]
            halt_output_addresses = [""]
        # Symmetric with the output_layers back-fill above: the fastlog
        # predicate-capture path stamps OpEvent.kind from the raw
        # RecordContext.kind ("input"/"buffer"/"op"/...), not the "source"
        # convention `_materialize.py` checks (`event.kind == "source"`,
        # which torch/backends/ops.py's canonical event builder only ever
        # produces from an exhaustive-mode capture). The per-layer is_input
        # flag that actually lands on each Op is authoritatively normalized
        # from layer_type alone (`_normalize_io_role_flags`,
        # torchlens/postprocess/labeling.py: `is_input = layer_type ==
        # "input"`), independent of event.kind. Match that exact criterion
        # here so the replayed Trace satisfies check_metadata_invariants'
        # bidirectional input_layers <-> is_input consistency check, instead
        # of leaving input_layers empty while individual Ops still carry
        # is_input=True.
        trace.input_layers = [
            event.label_raw
            for event in self._capture_events.op_events
            if event.layer_type == "input"
        ]
        # Complete the special-layer-list backfill for the remaining two of the
        # five (list <-> per-Op-flag) pairs the `special_layer_lists` invariant
        # checks (torchlens/validation/invariants.py `_SPECIAL_LIST_FLAG_PAIRS`:
        # input_layers, output_layers, buffer_layers, internal_source_ops,
        # internal_sink_ops). `input_layers`/`output_layers` are seeded above;
        # `internal_sink_ops` is NOT seeded here on purpose -- it is computed
        # fresh inside `_postprocess()` (graph_traversal.py
        # `_log_internally_terminated_tensor`; buffer-write sinks are stamped by
        # Step 6 `_fix_buffer_layers`), so it self-heals from the replayed event
        # stream. `buffer_layers` and `internal_source_ops`, by contrast, are
        # populated by `.append()` calls made directly on the LIVE per-pass Trace
        # DURING capture (backends/torch/sources.py, ops.py, model_prep.py) --
        # exactly the same mechanism as `input_layers` -- and are therefore
        # dropped when replaying events into a brand-new Trace unless seeded here.
        #
        # `buffer_layers` is load-bearing, not just invariant bookkeeping:
        # postprocess Step 6 `_fix_buffer_layers` (control_flow.py) iterates it to
        # link `buffer_source`, deduplicate same-value buffers, assign
        # `buffer_pass`, and mark buffer-write sinks; leaving it empty silently
        # disables all of that (buffer_pass stays None, Module.buffer_layers comes
        # back empty for every buffer-bearing module -- BatchNorm running stats,
        # registered constants). Seed both from raw labels; Step 10
        # `_rename_model_history_layer_names` (labeling.py) remaps these lists
        # from raw to final labels exactly as it already does for input/output.
        #
        # Criteria mirror the authoritative per-Op flag normalization so the
        # seeded lists stay bidirectionally consistent with the flags the
        # invariant cross-checks: is_buffer <- layer_type == "buffer"
        # (postprocess/labeling.py `_normalize_io_role_flags`); is_internal_source
        # <- layer_type != "input" and no parents (postprocess/_materialize.py,
        # capture/projections.py -- note buffers are ALSO internal-source ops, so
        # they legitimately appear in both lists, matching exhaustive tl.trace()).
        trace.buffer_layers = [
            event.label_raw
            for event in self._capture_events.op_events
            if event.layer_type == "buffer"
        ]
        trace.internal_source_ops = [
            event.label_raw
            for event in self._capture_events.op_events
            if event.layer_type != "input" and not event.parents
        ]

        # The live capture path (torchlens/capture/trace.py) sets
        # capture_start_time/setup_duration/forward_duration/forward_peak_memory/
        # forward_memory_backend on the Trace it builds *during* the forward
        # pass, before _postprocess() runs. Replaying captured events into a
        # brand-new Trace here skips that live-dispatch setup entirely, so
        # without this back-fill capture_start_time stays at the dataclass
        # default of 0 and _log_time_elapsed (postprocess Step 14) computes
        # cleanup_duration as `time.time() - 0 - 0 - 0`, a garbage
        # multi-decade Duration. Seed capture_start_time from the Recording's
        # own directly-measured first pass-start timestamp (always populated
        # for a non-failed recording with retained capture events), and copy
        # the remaining timing/memory fields from the fastlog Recorder's
        # internal runtime_trace when reachable -- that trace genuinely
        # measured them via the same _forward_peak_memory_bracket helper the
        # normal capture path uses (predicate-mode primary pass, see
        # torchlens/capture/trace.py's _run_predicate_forward_with_root_frame),
        # they are just otherwise discarded once the pass completes. Fields
        # that were not actually measured are left at their honest Trace
        # defaults -- no fabrication.
        recording_state = getattr(self, "_recording_state", None)
        runtime_trace = getattr(recording_state, "runtime_trace", None)
        if self.start_times:
            trace.capture_start_time = self.start_times[0]
        elif runtime_trace is not None:
            trace.capture_start_time = runtime_trace.capture_start_time
        if runtime_trace is not None:
            trace.setup_duration = runtime_trace.setup_duration
            trace.forward_duration = runtime_trace.forward_duration
            trace.forward_peak_memory = runtime_trace.forward_peak_memory
            trace.forward_memory_backend = runtime_trace.forward_memory_backend
            # Backfill the weak source-model reference the live predicate-mode
            # primary pass set (capture/trace.py run_and_log_inputs_through_model).
            # _postprocess() below (graph_traversal.py _resolve_output_parent_labels)
            # needs it on THIS trace to late-log a registered buffer that was
            # returned directly from forward() without ever being touched by a
            # traced op -- otherwise a buffer-only-output model crashes here with
            # "could not attribute a model output tensor to any traced op" even
            # though the primary capture pass already identified it correctly.
            if getattr(trace, "_source_model_ref", None) is None:
                trace._source_model_ref = getattr(runtime_trace, "_source_model_ref", None)
            # The predicate-mode primary pass (capture/trace.py) genuinely seeds
            # and records the RNG seed on its runtime_trace (self.random_seed,
            # set even when the caller passed random_seed=None). random_seed is a
            # FieldPolicy.KEEP (serialized) field, so backfill it from the trace
            # that actually used it rather than leaving Trace.random_seed at its
            # None dataclass default -- no fabrication, it is the real seed.
            runtime_seed = getattr(runtime_trace, "random_seed", None)
            if runtime_seed is not None:
                trace.random_seed = runtime_seed

        # Seed the raw-index high-water mark before postprocess. The live torch
        # capture path advances trace._layer_counter once per real op *during*
        # the forward pass (backends/torch/ops.py, sources.py), so by the time
        # postprocess Step 1 (_add_output_layers, graph_traversal.py) synthesizes
        # the dedicated output node(s) via `self._layer_counter += 1`, the counter
        # already sits at the last captured op's raw_index and each new output
        # node gets a fresh, strictly-larger raw_index. Replaying events straight
        # into a brand-new Trace here skips that live-dispatch bookkeeping, so
        # without this seed _layer_counter stays at 0 and the first output node is
        # stamped raw_index=1 -- colliding with input_1 (which carries its own
        # captured raw_index=1) and violating the graph_ordering invariant's
        # raw_index-uniqueness/monotonicity contract. Seed from the true
        # event-stream high-water mark so output-node numbering continues from
        # there, matching the live-capture path exactly.
        if self._capture_events.op_events:
            trace._layer_counter = max(event.raw_index for event in self._capture_events.op_events)

        # Snapshot each retained output tensor's raw label BEFORE handing it to
        # _postprocess(). Step 12 (_undecorate_all_saved_tensors,
        # postprocess/finalization.py) strips the private `._tl` metadata --
        # including `label_raw` -- off every saved tensor, and `halt_output_tensors`
        # here are the SAME tensor objects retained across every to_trace() call on
        # this frozen Recording (never copied; see the copy_for_replay() note above
        # -- tensors are intentionally shared by reference for cost reasons, so that
        # fix structurally cannot protect this side-channel). Without restoring the
        # label afterward, a second to_trace() call falls into
        # graph_traversal.py's _resolve_output_parent_labels() slow path (reached
        # whenever the output-tensor count doesn't equal the output-label count --
        # e.g. a model returning the same tensor twice, `return (y, y)`), which
        # reads get_tensor_label() and finds it already cleared by the FIRST call,
        # raising "could not attribute a model output tensor to any traced op".
        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        _retained_output_labels = [get_tensor_label(t) for t in halt_output_tensors]

        trace._postprocess(
            halt_output_tensors,
            halt_output_addresses,
        )

        # Restore any label Step 12 just cleared so the NEXT to_trace() call on
        # this same Recording can still resolve it. Only restores what was
        # actually there before -- no fabrication -- and only touches tensors
        # that lost their label (leaves anything Step 12 didn't clear alone).
        for _t, _label in zip(halt_output_tensors, _retained_output_labels):
            if _label is not None and get_tensor_label(_t) is None:
                set_tensor_label(_t, _label)

        return trace

    def _recover_halt_frontier(self) -> "tuple[str, torch.Tensor]":
        """Recover the frontier (output-parent label, tensor) for a halted recording.

        Mirrors ``_finalize_halted_trace``'s frontier recovery
        (``torchlens/capture/trace.py``): prefer the halting op itself
        (``self.halt_reason`` is the frontier op's raw label), otherwise fall
        back to the last captured op that retained a raw activation payload.
        The exhaustive path raises when no tensor frontier can be identified;
        match that contract here rather than fabricating a placeholder tensor.

        Returns
        -------
        tuple[str, torch.Tensor]
            The frontier op's raw label and its retained raw activation tensor.

        Raises
        ------
        RuntimeError
            When no captured op retained a raw activation payload, so no honest
            output frontier exists for the halted partial graph.
        """

        assert self._capture_events is not None  # guarded by to_trace() caller
        payload_by_label_raw: dict[str, torch.Tensor] = {}
        for record in self.records:
            payload = record.ram_payload
            if payload is None:
                continue
            label_raw = getattr(record.ctx, "label_raw", None) or getattr(record.ctx, "label", None)
            if label_raw is not None and label_raw not in payload_by_label_raw:
                payload_by_label_raw[label_raw] = payload

        # Primary: the halt frontier op, when its activation was retained.
        halt_label = self.halt_reason
        if halt_label and halt_label in payload_by_label_raw:
            return halt_label, payload_by_label_raw[halt_label]

        # Fallback: last captured op with a retained raw activation.
        for event in reversed(self._capture_events.op_events):
            payload = payload_by_label_raw.get(event.label_raw)
            if payload is not None:
                return event.label_raw, payload

        raise RuntimeError(
            "Recording.to_trace() cannot materialize a halted Recording that retained no "
            "raw activation payload: there is no tensor frontier to bind the halted graph's "
            "output node to. Re-run record(...) with a save= predicate that captures at least "
            "the halt frontier layer, or use tl.trace(model, x, halt=...) for the exhaustive "
            "halted-capture path."
        )


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
    object.__setattr__(recording, "status", "halted")
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
