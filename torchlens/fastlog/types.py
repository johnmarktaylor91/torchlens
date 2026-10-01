"""Core dataclasses for fastlog predicate recording."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch

from .._errors import CaptureContextError, InvalidArgumentError
from ..captured_run import CapturedRun
from ..ir.predicate import EventKind, ModuleStackFrame, RecordContext, RetroactiveCaptureDecision
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
    from ..capture.session import CapturedRunCore
    from ..data_classes.trace import Trace


def _backfill_cooked_ancestry(events: Any) -> None:
    """Derive the per-op ancestry closures the sparse recorder never tracked.

    Predicate-mode capture appends journal records without ancestry facts
    (the live exhaustive path computes them incrementally per op at capture
    time). A cooked Trace materializes its Op rows straight from these
    records, so without a backfill every cooked row carries empty
    ``root_ancestors`` / ``internal_source_ancestors`` and the
    ``ancestry_closure`` metadata invariant correctly fails on the first
    input layer. Recompute the exact closure the invariant checks, in journal
    order (parents precede children within one sealed pass), over the AMENDED
    view (graph-edge-insertion amendments contribute parents):

    * ``input_ancestors``           = own label for input rows, else parent union;
    * ``internal_source_ancestors`` = ``{self}`` for parentless non-input rows
      (matching the live source-minting convention), else parent union;
    * ``root_ancestors``            = ``input_ancestors | internal_source_ancestors``.

    Mutates ``events`` in place: this runs only on the ``copy_for_replay``
    projection a cook owns (the sanctioned mutation surface -- postprocess
    graph traversal replaces events on the same projection), never on the
    sealed Recording stream. The fold cache keys on lane lengths, so it is
    explicitly invalidated after the in-place replacement.

    Parameters
    ----------
    events
        Replay-projection ``CaptureEvents`` whose op lane should be
        ancestry-backfilled before Trace postprocessing.
    """

    from dataclasses import replace as _dc_replace

    from ..ir.op_record import AncestryFacet, OpRecord

    folded = events.amended_op_records()
    closures: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    by_raw_label = events.live_index.by_raw_label
    mutated = False
    for index, record in enumerate(folded):
        layer_type = getattr(record, "layer_type", None)
        if layer_type in (None, "module_enter", "module_exit"):
            continue
        label_raw = record.label_raw
        parent_labels = [edge.parent_label_raw for edge in record.parents]
        if layer_type == "input":
            input_ancestors = frozenset((label_raw,))
            internal_source_ancestors: frozenset[str] = frozenset()
        elif not parent_labels:
            input_ancestors = frozenset()
            internal_source_ancestors = frozenset((label_raw,))
        else:
            input_ancestors = frozenset().union(
                *(closures[parent][0] for parent in parent_labels if parent in closures)
            )
            internal_source_ancestors = frozenset().union(
                *(closures[parent][1] for parent in parent_labels if parent in closures)
            )
        closures[label_raw] = (input_ancestors, internal_source_ancestors)
        ancestry = AncestryFacet(
            input_ancestors=input_ancestors,
            internal_source_ancestors=internal_source_ancestors,
            root_ancestors=input_ancestors | internal_source_ancestors,
            has_internal_source_ancestor=bool(internal_source_ancestors),
        )
        raw_record = events.op_events[index]
        if isinstance(raw_record, OpRecord):
            updated = _dc_replace(raw_record, ancestry=ancestry)
        else:
            updated = _dc_replace(
                raw_record,
                input_ancestors=input_ancestors,
                internal_source_ancestors=internal_source_ancestors,
                root_ancestors=ancestry.root_ancestors,
                has_internal_source_ancestor=ancestry.has_internal_source_ancestor,
            )
        events.op_events[index] = updated
        mutated = True
        if by_raw_label.get(label_raw) is raw_record:
            by_raw_label[label_raw] = updated
    if mutated:
        events._amended_fold_cache = None


def _distinct_label_index_keys(label: str, raw_label: str | None) -> tuple[str, ...]:
    """Return the distinct label keys that should index one activation record.

    Parameters
    ----------
    label
        Primary public label for the retained record.
    raw_label
        Optional raw label alias for the same retained record.

    Returns
    -------
    tuple[str, ...]
        Unique label keys that should reference the record exactly once.
    """

    if raw_label is None or raw_label == label:
        return (label,)
    return (label, raw_label)


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

        from ..utils.tensor_utils import SAVE_MODES

        if self.save_mode not in SAVE_MODES:
            raise InvalidArgumentError(
                "save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'; "
                f"received {self.save_mode!r}",
                code="save_mode_invalid",
                remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
                argument="save_mode",
            )
        if self.save_mode == "view" and not self.keep_grad:
            object.__setattr__(self, "keep_grad", True)


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
        other_keep_op: Callable[[RecordContext], PredicateDecision] | None = None,
    ) -> RecordingTrace:
        """Return a new trace with decisions from a new op predicate.

        Parameters
        ----------
        other_keep_op:
            Predicate for op, input, and buffer events. Module boundary
            events have no predicate slot and are never re-selected.

        Returns
        -------
        RecordingTrace
            New trace sharing the same event tuple and predicate failures.
        """

        from ..capture.predicates import _normalize_capture_decision

        decisions: list[bool] = []
        for ctx in self.contexts:
            predicate = None if ctx.kind in {"module_enter", "module_exit"} else other_keep_op
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


#: Closed RecordContext field set exported by :meth:`Recording.raw_metadata`.
#: Metadata-only by construction: payload-bearing and lookback-view fields
#: (``recent_events``/``recent_ops``, deferred-value booleans) stay out so a
#: row can never force a payload read.
_RAW_METADATA_FIELDS = (
    "kind",
    "label",
    "raw_label",
    "pass_index",
    "event_index",
    "step_index",
    "layer_type",
    "type_index",
    "raw_index",
    "func_name",
    "address",
    "module_type",
    "module_pass_index",
    "module_stack",
    "parent_labels",
    "input_output_address",
    "shape",
    "dtype",
    "tensor_device",
    "output_index",
    "is_bottom_level_func",
    "func_call_id",
    "is_output_parent",
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

    # Sanctioned outcome-gate marker (torchlens/capture/outcome.py): this
    # slots-backed product's validating ``outcome`` property IS the settlement
    # truth (stamped-record re-parse + conservative derivation, never blessing
    # COMPLETE). Without it ``outcome_for`` read the absent ``__dict__`` as
    # UNKNOWN and every capability gate refused settled Recordings with a
    # false hand-built-object warning (WT1 A-IV item 21, lane A08).
    _OUTCOME_SELF_AUTHORITY: ClassVar[bool] = True

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
    _captured_run_cores: tuple[CapturedRunCore, ...] = field(default=(), repr=False, compare=False)
    # Settled capture outcome stamped by the recorder settlement adapter
    # (torchlens/capture/outcome.py); ``outcome`` below derives conservatively
    # for unstamped legacy/recovered recordings.
    _outcome: Any | None = field(default=None, repr=False, compare=False)

    @property
    def outcome(self) -> Any:
        """Return the settled (or conservatively derived) capture outcome.

        Stamped recordings return the settlement authority's record. Legacy
        pickles (whose ``_outcome`` slot may be unset) and recovered/unstamped
        recordings derive from the construction status: ``halted`` halt
        markers are construction-time proofs, ``partial_error`` is FAILED,
        ``recovered`` with carried abort evidence (``failed=True`` from the
        PARTIAL/REASON.txt debris) is FAILED, ``recovered`` otherwise is
        UNKNOWN (or reconstructed HALTED where the halt markers survived)
        with ``recovered=True``, and ``complete`` derives UNATTESTED -- the
        status string on a deserialized object is a plain spoofable field,
        and a derivation never blesses COMPLETE (R06; same doctrine as the
        trace-side structural lattice). All ``derived=True``, never a
        settle-stamp upgrade.
        """

        from ..capture.outcome import CaptureOutcome, CaptureStatus, FailureOrigin

        stamped = getattr(self, "_outcome", None)
        if isinstance(stamped, CaptureOutcome):
            # R10-5: a plain-pickled Recording's stamped outcome is spoofable
            # bytes like any other unpickled field. Re-validate it through the
            # same string-only parse + coherence matrix the trace-side load
            # uses, so an incoherent or forged record degrades (never upgrades)
            # instead of being adopted verbatim.
            from ..capture.outcome import parse_outcome_payload

            try:
                return parse_outcome_payload(stamped.to_payload())
            except Exception:  # noqa: BLE001 - fail closed on hostile payloads
                return CaptureOutcome(status=CaptureStatus.UNKNOWN, derived=True)
        status = self.status
        if status == "partial_error":
            return CaptureOutcome(
                status=CaptureStatus.FAILED,
                origin=FailureOrigin.UNKNOWN,
                reason=self.error_repr,
                n_ops_committed=self.n_ops_completed,
                derived=True,
            )
        if status == "recovered" and self.failed:
            # WT1 A-IV item 21 (lane A08): recover() used to rebuild aborted
            # bundles (PARTIAL sentinel + REASON.txt debris) with
            # ``failed=False`` and no error evidence -- laundering the failure
            # record into a benign UNKNOWN. A recovered bundle carrying abort
            # evidence derives FAILED, never a blank slate.
            return CaptureOutcome(
                status=CaptureStatus.FAILED,
                origin=FailureOrigin.UNKNOWN,
                reason=self.error_repr,
                n_ops_committed=self.n_ops_completed,
                recovered=True,
                derived=True,
            )
        if status == "halted" or (status == "recovered" and self.halted):
            return CaptureOutcome(
                status=CaptureStatus.HALTED,
                reason=self.halt_reason,
                boundary_label=self.halt_reason,
                recovered=status == "recovered",
                derived=True,
            )
        if status == "recovered":
            return CaptureOutcome(
                status=CaptureStatus.UNKNOWN,
                recovered=True,
                derived=True,
            )
        if status == "complete":
            return CaptureOutcome(status=CaptureStatus.UNATTESTED, derived=True)
        return CaptureOutcome(status=CaptureStatus.UNKNOWN, derived=True)

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

        if name == "records" and not object.__getattribute__(self, "_records_built"):
            ensure = object.__getattribute__(self, "_ensure_records")
            ensure()
        return object.__getattribute__(self, name)

    @classmethod
    def from_capture_events(cls: type[Recording], session: Any) -> Recording:
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
        object.__setattr__(
            base,
            "_captured_run_cores",
            tuple(getattr(session, "captured_run_cores", ())),
        )
        has_core_projection = bool(getattr(session, "captured_run_cores", ()))
        object.__setattr__(
            base,
            "_records_built",
            bool(object.__getattribute__(base, "records")) and not has_core_projection,
        )
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
        records = object.__getattribute__(self, "records")
        records.clear()
        self.by_pass.clear()
        self.by_label.clear()
        self.by_address.clear()
        if self._captured_run_cores:
            from ..capture.projectors import RecordingProjector

            projection = RecordingProjector().project(self._captured_run_cores)
            records.extend(projection.records)
            self.by_pass.update(projection.by_pass)
            self.by_label.update(projection.by_label)
            self.by_address.update(projection.by_address)
        elif self._capture_events is not None:
            from ..capture.projections import activation_record_from_event

            for event in self._capture_events.amended_op_records():
                record = activation_record_from_event(event)
                if record is None:
                    continue
                index = len(records)
                records.append(record)
                self.by_pass.setdefault(record.ctx.pass_index, []).append(index)
                for label_key in _distinct_label_index_keys(record.ctx.label, record.ctx.raw_label):
                    self.by_label.setdefault(label_key, []).append((record.ctx.pass_index, index))
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
                        for event in (
                            self._capture_events.amended_op_records()
                            if self._capture_events is not None
                            else ()
                        )
                    ),
                    predicate_failures=tuple(self.predicate_failures),
                ),
            )
        trace = self._recording_trace
        if trace is None:
            raise RuntimeError("recording_trace projection was not initialized")
        return trace

    def raw_metadata(self) -> tuple[dict[str, Any], ...]:
        """Return payload-free raw metadata for every captured event.

        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification).
        One plain dict per chronological capture event — retained or not —
        read straight from the recorder's raw event stream with no cooking
        and no payload access, so it works on failed partial recordings too.
        Each row carries the closed RecordContext metadata field set plus
        ``retained`` (whether the predicate kept the event's record).

        Returns
        -------
        tuple[dict[str, Any], ...]
            Chronological per-event metadata rows.

        Raises
        ------
        RecorderStateError
            ``recording_event_stream_unavailable`` when this recording no
            longer holds its raw event stream (explicitly cleaned, or
            restored from a payload-only projection) — never a silently
            empty result.
        """

        from .exceptions import RecorderStateError

        if self.event_stream is None:
            raise RecorderStateError(
                "this Recording no longer holds its raw capture event stream "
                "(explicitly cleaned, or restored from a payload-only "
                "projection), so raw per-event metadata is unavailable",
                code="recording_event_stream_unavailable",
                remedy=(
                    "read retained-record metadata via recording.records / "
                    "to_pandas(), or keep the event stream alive"
                ),
            )
        self._ensure_records()
        retained_keys = {(record.ctx.pass_index, record.ctx.event_index) for record in self.records}
        rows: list[dict[str, Any]] = []
        for ctx in self.recording_trace.contexts:
            row = {name: getattr(ctx, name) for name in _RAW_METADATA_FIELDS}
            row["retained"] = (ctx.pass_index, ctx.event_index) in retained_keys
            rows.append(row)
        return tuple(rows)

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
    ) -> Recording:
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

        # Refusals carry stable machine-branchable codes (branch on
        # exc.fields["code"], never message text). The failed arm mirrors the
        # capability table's backward/FAILED cell (N3). The halted arm is a
        # DELIBERATE Recording-scoped strictness beyond the table's
        # HALTED-allow cell, which describes Trace-side backward (a halted
        # Trace holds the prefix autograd graph); a halted Recording is a
        # sparse event product whose frontier pass retained no complete
        # output to root the backward walk -- the same capability boundary
        # that refuses halted-no-payload ``to_trace()``. Documented in
        # docs/reference/capture_outcomes.md.
        if self.failed:
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                "Cannot call log_backward on failed partial Recording; "
                "user-op failures exclude the failing call; TL-side capture failures may "
                "include a skipped/partial current-call event.",
                code="N3",
                capability="backward",
                status="failed",
            )
        if self.halted:
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                f"Cannot call log_backward on halted Recording (halt_reason={self.halt_reason!r}).",
                code="recording_backward_halted",
                capability="backward",
                status="halted",
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
        from .._capture_honesty import attach_dataframe_honesty

        return attach_dataframe_honesty(pd.DataFrame(rows), self)

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

    def enrich(self, steps: list[str] | str) -> Recording:
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

    def to_trace(self) -> Trace:
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
            the topology-complete event stream (e.g. disk-recovered), if it
            spans multiple recorded passes, or if it is halted but retained no
            raw activation payload to bind as the output frontier.
        """

        if self.failed:
            raise CaptureContextError(
                "Recording.to_trace() cannot materialize a failed partial Recording because "
                "the topology is incomplete; user-op failures exclude the failing call; "
                "TL-side capture failures may include a skipped/partial current-call event",
                code="recording_failed_not_convertible",
                remedy="fix the failing forward and re-record before converting",
            )
        if not self._captured_run_cores:
            raise CaptureContextError(
                "Recording.to_trace() requires retained capture events; disk-recovered "
                "recordings do not contain enough topology metadata",
                code="recording_events_not_retained",
                remedy="convert the in-session Recording rather than a disk-recovered one",
            )
        if self.n_passes > 1:
            raise CaptureContextError(
                "Recording.to_trace() does not support multi-pass Recordings because "
                "replaying multiple Recorder.log() passes into one Trace is not yet "
                "structurally defined",
                code="recording_multipass_not_convertible",
                remedy="record one pass per Recording before converting",
            )
        from ..capture.projectors import RecordingProjector
        from ..data_classes.trace import Trace
        from .options import RecordingOptions

        projection = RecordingProjector().project(self._captured_run_cores)
        if projection.capture_events is None:
            raise RuntimeError("Recording.to_trace() core has no replay event facts.")

        trace = Trace(model_class_name="RecordedModel")
        # capture_mode drives exhaustive-style postprocess behavior for the
        # cooked projection; the honest provenance fact is the marker below,
        # which records that this Trace was cooked from a Recording rather
        # than captured live.
        trace.capture_mode = "exhaustive"
        trace._cooked_from = "recording"
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
        events_for_replay = cast(Any, projection.capture_events).copy_for_replay()
        # The sparse recorder never tracks ancestry at capture time; derive the
        # closures on the cook's own projection before postprocess materializes
        # Op rows from it (the ancestry_closure invariant checks exactly this).
        _backfill_cooked_ancestry(events_for_replay)
        trace.capture_events = events_for_replay
        projection.prepare_trace(trace)
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
        halt_output_tensors = list(projection.output_tensors)
        halt_output_addresses = list(projection.output_tensor_addresses)
        if self.halted:
            frontier_label, frontier_tensor = self._recover_halt_frontier()
            projection.bind_halt_frontier(frontier_tensor, frontier_label)
            trace.halted = True
            trace.halt_reason = self.halt_reason
            trace.halt_frontier = self.halt_reason
            trace.raw_output = None
            trace.output_layers = [frontier_label]
            halt_output_tensors = [frontier_tensor]
            halt_output_addresses = [""]
        trace._postprocess(
            halt_output_tensors,
            halt_output_addresses,
        )

        # Settle at the cook seam (settlement authority path 9): the cooked
        # Trace is a real product and must carry an attested outcome; a halted
        # cooked trace is HALTED (so the runnable/live-replay gates see it),
        # never a silently-blessed complete. The frontier label is read back
        # post-postprocess so it is the FINAL remapped label.
        from ..capture.outcome import stamp_cooked

        cooked_frontier = None
        cooked_reason = None
        cooked_boundary = None
        cooked_boundary_kind = None
        if self.halted:
            output_labels = list(getattr(trace, "output_layers", ()))
            cooked_frontier = str(output_labels[0]) if output_labels else None
            # R06: the halted postprocess remapped the persisted halt fields to
            # FINAL labels; the settled record mirrors them (settle_halted
            # parity) instead of stamping the Recording-space raw label into an
            # outcome whose frontier is final. boundary_kind rides the
            # Recording's own settled outcome; the raw halt_reason stays the
            # fallback when the remap did not run.
            remapped_reason = getattr(trace, "halt_reason", None)
            cooked_reason = (
                remapped_reason if isinstance(remapped_reason, str) else self.halt_reason
            )
            remapped_frontier = getattr(trace, "halt_frontier", None)
            cooked_boundary = (
                remapped_frontier if isinstance(remapped_frontier, str) else cooked_reason
            )
            cooked_boundary_kind = getattr(self.outcome, "boundary_kind", None)
        stamp_cooked(
            trace,
            halted=self.halted,
            reason=cooked_reason,
            boundary_kind=cooked_boundary_kind,
            boundary_label=cooked_boundary,
            frontier_label=cooked_frontier,
        )
        return trace

    def _recover_halt_frontier(self) -> tuple[str, torch.Tensor]:
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

        assert self._captured_run_cores  # guarded by to_trace() caller
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
        core_events = tuple(event for core in self._captured_run_cores for event in core.events)
        for event in reversed(core_events):
            payload = payload_by_label_raw.get(event.label_raw)
            if payload is not None:
                return event.label_raw, payload

        raise CaptureContextError(
            "Recording.to_trace() cannot materialize a halted Recording that retained no "
            "raw activation payload: there is no tensor frontier to bind the halted graph's "
            "output node to",
            code="recording_halt_frontier_missing",
            remedy=(
                "re-run record(...) with a save= predicate that captures at least the halt "
                "frontier layer, or use tl.trace(model, x, halt=...) for the exhaustive "
                "halted-capture path"
            ),
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
