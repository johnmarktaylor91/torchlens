"""Core dataclasses for fastlog predicate recording."""

from __future__ import annotations

__tl_layer__ = "L1"
__tl_vocabulary__ = True

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast

import torch

from .._captured_run import TraceLike
from .._errors import CaptureNotActiveError
from .._ir import predicate as _predicate
from .._utils.tensor_utils import SaveMode as SaveMode

__all__ = [
    "ActivationRecord",
    "CaptureSpec",
    "GradRecordContext",
    "ModuleStackFrame",
    "PredicateFailure",
    "RecordContext",
    "Recording",
    "RecordingSettings",
    "StorageIntent",
]

ActivationRecord = _predicate.ActivationRecord
CaptureDecision = _predicate.CaptureDecision
CaptureSpec = _predicate.CaptureSpec
EventKind = _predicate.EventKind
GradRecord = _predicate.GradRecord
GradRecordContext = _predicate.GradRecordContext
ModuleStackFrame = _predicate.ModuleStackFrame
PredicateDecision = _predicate.PredicateDecision
PredicateFailure = _predicate.PredicateFailure
RecordContext = _predicate.RecordContext
RetroactiveCaptureDecision = _predicate.RetroactiveCaptureDecision
StorageIntent = _predicate.StorageIntent
_distinct_label_index_keys = _predicate._distinct_label_index_keys

if TYPE_CHECKING:
    from .._capture.session import CapturedRunCore
    from .._data_classes.trace import Trace
    from ..intervention.predicates import InterventionPredicate
    from ..options import StorageOptions
    from ..types import _ActivationPostfunc, _GradientPostfunc
    from ..viz._drawing import Drawing
    from .options import (
        ForwardErrorMode,
        GradPredicateFn,
        HaltPredicateFn,
        LookbackPayloadPolicy,
        PredicateErrorMode,
        PredicateFn,
        _RecordingConfig,
    )


@dataclass(frozen=True, slots=True)
class DryRun:
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

        from .._visualization.fastlog_live import print_tree

        return print_tree(self)

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame representation of trace events."""

        from .._visualization.fastlog_live import to_pandas

        return to_pandas(self)

    def draw(self, **kwargs: Any) -> Drawing:
        """Render a flat Graphviz graph of trace operation events.

        Returns
        -------
        Drawing
            The picture; ``str()`` of it is the DOT source.
        """

        from .._visualization.fastlog_live import draw

        return draw(self, **kwargs)

    def summary(self) -> str:
        """Return a concise human-readable dry-run summary."""

        from .._visualization.fastlog_live import summary

        return summary(self)

    def timeline_html(self) -> Any:
        """Return an IPython HTML timeline for this trace."""

        from .._visualization.fastlog_live import timeline_html

        return timeline_html(self)

    def filter(
        self,
        other_keep_op: Callable[[RecordContext], PredicateDecision] | None = None,
    ) -> DryRun:
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

        from .._capture.predicates import _normalize_capture_decision

        decisions: list[bool] = []
        for ctx in self.contexts:
            predicate = None if ctx.kind in {"module_enter", "module_exit"} else other_keep_op
            result = predicate(ctx) if predicate is not None else False
            spec = _normalize_capture_decision(result, ctx, False)
            if not isinstance(spec, CaptureSpec):
                decisions.append(True)
                continue
            decisions.append(spec.save_out or spec.save_metadata)
        return DryRun(
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
    "pass_number",
    "event_index",
    "step_index",
    "layer_type",
    "type_index",
    "raw_index",
    "func_name",
    "address",
    "module_type",
    "module_pass_number",
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


# --- Model-output readback (Recording.output / Trace.output) ---------------
# The output slot and its constructors live in ``torchlens._output_slot`` (vocabulary).

#: ``RecordingSettings`` field -> the ``_RecordingConfig`` slot it reads. The
#: record speaks the public spellings: the capture knobs keep their
#: ``CaptureOptions`` names, the recorder knobs their flat ``tl.record``
#: keyword names, and the three door-carried slots their door keywords.
_SETTINGS_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("keep_op", "keep_op"),
    ("default_op", "default_op"),
    ("default_module", "default_module"),
    ("history_size", "history_size"),
    ("include_source_events", "include_source_events"),
    ("intervene", "intervene"),
    ("max_predicate_failures", "max_predicate_failures"),
    ("on_predicate_error", "on_predicate_error"),
    ("on_forward_error", "on_forward_error"),
    ("default_grad", "default_grad"),
    ("postprocess", "postprocess"),
    ("lookback", "lookback"),
    ("lookback_payload_policy", "lookback_payload_policy"),
    ("seed", "seed"),
    ("activation_transform", "activation_transform"),
    ("save_raw_activations", "save_raw_activations"),
    ("save_grads", "save_grads"),
    ("grad_transform", "grad_transform"),
    ("save_raw_grads", "save_raw_grads"),
    ("detach_saved_activations", "detach_saved_activations"),
    ("stop_after", "halt"),
    ("storage", "streaming"),
    ("narrate", "echo"),
)

#: Value types a saved recording stores verbatim (bundle ``metadata.json``).
_PERSISTED_SETTING_TYPES: Final[tuple[type, ...]] = (bool, int, float, str)


@dataclass(frozen=True, slots=True, init=False)
class RecordingSettings:
    """The settings one recording was made with; read-only.

    Returned by ``Recording.options``. One record answers "how was this
    recording made?" across every door that configured it: the capture knobs
    (``capture=CaptureOptions(...)``: ``save_grads``, ``activation_transform``,
    ``grad_transform``, ``lookback``, ...), the recorder knobs
    (the flat ``tl.record`` keywords: ``keep_op``, ``default_op``,
    ``history_size``, ...) and the door keywords ``stop_after=``, ``storage=``
    and ``narrate=``. Values are the live objects the recorder used (the
    predicate or transform callable itself, the ``CaptureSpec``), never repr
    strings. Built by the recorder; there is no public constructor.

    A recording recovered from disk carries the plain values (bools, numbers,
    strings) that saved recordings store; a setting whose value is a callable
    or an object (predicates, transforms, ``CaptureSpec`` values, storage and
    narration options) is not stored with saved recordings and reads ``None``
    there.

    Attributes
    ----------
    keep_op:
        The save predicate deciding which op events keep a record (``save=``).
    default_op:
        Capture decision for op events the save predicate leaves undecided.
    default_module:
        Capture decision for module events the save predicate leaves undecided.
    history_size:
        How many recent event contexts a predicate could look back over.
    include_source_events:
        Whether input and buffer source events were offered to the predicates.
    intervene:
        The live intervention predicate applied during the forward.
    max_predicate_failures:
        How many predicate failures were kept before counting the rest.
    on_predicate_error:
        What a raising predicate did: ``auto``, ``accumulate`` or ``fail-fast``.
    on_forward_error:
        What a raising forward did: ``raise``, ``attach_partial`` or ``return_partial``.
    default_grad:
        Capture decision for gradient events no gradient predicate decided.
    postprocess:
        The post-capture finalization pass (``none`` by default).
    lookback:
        How many earlier events each record context carried.
    lookback_payload_policy:
        Which payload the lookback contexts carried.
    seed:
        The random seed set before the forward, if any.
    activation_transform:
        The function applied to each saved activation.
    save_raw_activations:
        Whether the untransformed activation was kept beside the transformed one.
    save_grads:
        Which gradients were saved: a predicate, a bool or a ``CaptureSpec``.
    grad_transform:
        The function applied to each saved gradient.
    save_raw_grads:
        Whether the untransformed gradient was kept beside the transformed one.
    detach_saved_activations:
        Whether saved activations were detached from the autograd graph.
    stop_after:
        The stop predicate from ``stop_after=``, if any.
    storage:
        The disk sink from ``storage=``, if any.
    narrate:
        The live narration settings (a ``NarrationOptions``) that printed each captured
        event as the forward ran, from the door's ``narrate=``; ``None`` when off.
    """

    keep_op: PredicateFn | None
    default_op: bool | CaptureSpec | None
    default_module: bool | CaptureSpec | None
    history_size: int | None
    include_source_events: bool | None
    intervene: InterventionPredicate | None
    max_predicate_failures: int | None
    on_predicate_error: PredicateErrorMode | None
    on_forward_error: ForwardErrorMode | None
    default_grad: bool | CaptureSpec | None
    postprocess: str | None
    lookback: int | None
    lookback_payload_policy: LookbackPayloadPolicy | None
    seed: int | None
    activation_transform: _ActivationPostfunc | None
    save_raw_activations: bool | None
    save_grads: GradPredicateFn | bool | CaptureSpec | None
    grad_transform: _GradientPostfunc | None
    save_raw_grads: bool | None
    detach_saved_activations: bool | None
    stop_after: HaltPredicateFn | None
    storage: StorageOptions | None
    narrate: Any | None

    @classmethod
    def _from_values(cls: type[RecordingSettings], values: Mapping[str, Any]) -> RecordingSettings:
        """Build the record; a field absent from ``values`` reads ``None``."""

        record = object.__new__(cls)
        for name, _slot in _SETTINGS_SLOTS:
            object.__setattr__(record, name, values.get(name))
        return record

    @classmethod
    def _from_config(cls: type[RecordingSettings], config: _RecordingConfig) -> RecordingSettings:
        """Project the recorder's resolved config onto the public spellings."""

        return cls._from_values({name: getattr(config, slot) for name, slot in _SETTINGS_SLOTS})

    def _persisted_values(self) -> dict[str, Any]:
        """Return the plain values a saved recording stores (``metadata.json``)."""

        return {
            name: value
            for name, _slot in _SETTINGS_SLOTS
            if isinstance(value := getattr(self, name), _PERSISTED_SETTING_TYPES)
        }


@dataclass(frozen=True, slots=True, init=False)
class Recording(TraceLike):
    """Result of a fastlog recording session.

    Built by ``tl.record`` and ``Recorder``; there is no public constructor
    (the recorder and the bundle recovery build it through the private
    ``_build``). ``options`` returns the settings the recording was made with.

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

    # Sanctioned outcome-gate marker (torchlens/_capture/outcome.py): this slots-backed product's
    # validating ``outcome`` property IS the settlement truth (stamped-record re-parse +
    # conservative derivation, never blessing COMPLETE). Without it ``outcome_for`` read the absent
    # ``__dict__`` as UNKNOWN and every capability gate refused settled Recordings with a false
    # hand-built-object warning (WT1 A-IV item 21, lane A08).
    _OUTCOME_SELF_AUTHORITY: ClassVar[bool] = True

    records: list[ActivationRecord]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]
    path: Path | None
    num_ops: int
    start_times: list[float]
    end_times: list[float]
    predicate_failures: list[PredicateFailure]
    num_predicate_failure_overflows: int
    history_size: int
    orphan_records: list[dict[str, Any]] = field(default_factory=list)
    is_stopped: bool = False
    status: Literal["complete", "halted", "partial_error", "recovered"] = "complete"
    is_failed: bool = False
    error_repr: str | None = None
    error_traceback: str | None = None
    num_committed_ops: int = 0
    last_successful_op_label: str | None = None
    last_event_label: str | None = None
    last_event_func: str | None = None
    last_event_source_line: str | None = None
    last_event_input_meta: str | None = None
    stop_reason: str | None = None
    stops_by_pass: dict[int, str] = field(default_factory=dict)
    grad_records: list[GradRecord] = field(default_factory=list)
    grad_by_pass: dict[int, list[int]] = field(default_factory=dict)
    grad_by_label: dict[str, list[int]] = field(default_factory=dict)
    grad_by_grad_fn_label: dict[str, list[int]] = field(default_factory=dict)
    is_recovered: bool = False
    recovery_warnings: list[str] = field(default_factory=list)
    _capture_events: Any | None = field(default=None, repr=False, compare=False)
    _output_tensors: list[torch.Tensor] = field(default_factory=list, repr=False, compare=False)
    _output_tensor_addresses: list[str] = field(default_factory=list, repr=False, compare=False)
    _records_built: bool = field(default=True, repr=False, compare=False)
    _recording_trace: DryRun | None = field(default=None, repr=False, compare=False)
    _recording_state: Any | None = field(default=None, repr=False, compare=False)
    _captured_run_cores: tuple[CapturedRunCore, ...] = field(default=(), repr=False, compare=False)
    # Settled capture outcome stamped by the recorder settlement adapter
    # (torchlens/_capture/outcome.py); ``outcome`` below derives conservatively
    # for unstamped legacy/recovered recordings.
    _outcome: Any | None = field(default=None, repr=False, compare=False)
    # The resolved settings record behind ``options`` (a ``RecordingSettings``).
    _options: Any | None = field(default=None, repr=False, compare=False)
    # The live model output behind ``output`` (an ``OutputSlot``); memory only,
    # never persisted or pickled.
    _output: Any | None = field(default=None, repr=False, compare=False)

    @classmethod
    def _build(cls: type[Recording], **values: Any) -> Recording:
        """Build a Recording from field values; omitted fields take their defaults.

        The one construction path (the class has no generated ``__init__``):
        the recorder's empty session record, bundle recovery and tests. A
        required field left out, or an unknown name, fails on assignment.
        """

        recording = object.__new__(cls)
        for item in dataclasses.fields(cls):
            if item.name in values:
                value = values.pop(item.name)
            elif item.default_factory is not dataclasses.MISSING:
                value = item.default_factory()
            elif item.default is not dataclasses.MISSING:
                value = item.default
            else:
                value = values[item.name]  # a required field left out: KeyError names it
            object.__setattr__(recording, item.name, value)
        for name, value in values.items():
            object.__setattr__(recording, name, value)
        return recording

    def _replace(self, **changes: Any) -> Recording:
        """Return a copy with ``changes`` applied (``dataclasses.replace`` needs ``__init__``)."""

        values = {
            item.name: object.__getattribute__(self, item.name) for item in dataclasses.fields(self)
        }
        values.update(changes)
        return type(self)._build(**values)

    @property
    def output(self) -> Any:
        """Return the model's return value, exactly as the model returned it.

        The tensor, tuple, dict, Hugging Face output object or distribution the
        forward returned (for a multi-pass ``Recorder``, the last pass's). Held in
        memory only: the recording keeps it alive for as long as the recording
        exists (for a Hugging Face model this includes ``past_key_values``); it
        is released when the recording is dropped.

        Returns
        -------
        Any
            The model output.

        Raises
        ------
        PayloadUnavailableError
            ``output_not_stored`` on a recording loaded from disk or unpickled,
            ``output_forward_failed`` on a failed partial recording
            (``on_forward_error="return_partial"``), and
            ``output_capture_stopped`` when ``stop_after=`` ended the capture
            before the model returned. Never returns ``None`` for a missing
            output.
        """

        from .._output_slot import read_output

        return read_output(self._output, owner="Recording")

    @property
    def options(self) -> RecordingSettings:
        """Return the settings this recording was made with (read-only).

        Returns
        -------
        RecordingSettings
            The resolved capture and recorder settings, as live values. On a
            recording recovered from disk, settings that saved recordings do
            not store (callables and objects) read ``None``.
        """

        stored = self._options
        if isinstance(stored, RecordingSettings):
            return stored
        return RecordingSettings._from_values({})

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

        from .._capture.outcome import CaptureOutcome, CaptureStatus, FailureOrigin

        stamped = getattr(self, "_outcome", None)
        if isinstance(stamped, CaptureOutcome):
            # R10-5: a plain-pickled Recording's stamped outcome is spoofable
            # bytes like any other unpickled field. Re-validate it through the
            # same string-only parse + coherence matrix the trace-side load
            # uses, so an incoherent or forged record degrades (never upgrades)
            # instead of being adopted verbatim.
            from .._capture.outcome import parse_outcome_payload

            try:
                return parse_outcome_payload(stamped.to_payload())
            except Exception:  # noqa: BLE001 - fail closed on hostile payloads
                return CaptureOutcome(status=CaptureStatus.UNKNOWN, is_derived=True)
        status = self.status
        if status == "partial_error":
            return CaptureOutcome(
                status=CaptureStatus.FAILED,
                origin=FailureOrigin.UNKNOWN,
                reason=self.error_repr,
                num_committed_ops=self.num_committed_ops,
                is_derived=True,
            )
        if status == "recovered" and self.is_failed:
            # WT1 A-IV item 21 (lane A08): recover() used to rebuild aborted
            # bundles (PARTIAL sentinel + REASON.txt debris) with
            # ``failed=False`` and no error evidence -- laundering the failure
            # record into a benign UNKNOWN. A recovered bundle carrying abort
            # evidence derives FAILED, never a blank slate.
            return CaptureOutcome(
                status=CaptureStatus.FAILED,
                origin=FailureOrigin.UNKNOWN,
                reason=self.error_repr,
                num_committed_ops=self.num_committed_ops,
                is_recovered=True,
                is_derived=True,
            )
        if status == "halted" or (status == "recovered" and self.is_stopped):
            return CaptureOutcome(
                status=CaptureStatus.STOPPED,
                reason=self.stop_reason,
                boundary_label=self.stop_reason,
                is_recovered=status == "recovered",
                is_derived=True,
            )
        if status == "recovered":
            return CaptureOutcome(
                status=CaptureStatus.UNKNOWN,
                is_recovered=True,
                is_derived=True,
            )
        if status == "complete":
            return CaptureOutcome(status=CaptureStatus.UNATTESTED, is_derived=True)
        return CaptureOutcome(status=CaptureStatus.UNKNOWN, is_derived=True)

    @property
    def num_passes(self) -> int:
        """Return the number of model-call passes captured by this recording.

        Returns
        -------
        int
            Number of explicit ``record()`` or ``Recorder(...)`` forward passes.
        """

        return self.num_ops

    def __repr__(self) -> str:
        """Return an O(1) identity/status card.

        Never materializes lazy records and never renders payloads: the
        dataclass auto-repr grew ~34x per saved site (recursive
        ``RecordContext`` lookback times full tensor payloads) and OOMed an
        8 GiB box at four saved sites (lovely bug 1, CRITICAL).
        """

        built = object.__getattribute__(self, "_records_built")
        records = "lazy" if not built else str(len(object.__getattribute__(self, "records")))
        grads = len(self.grad_records)
        return (
            f"Recording(status={self.status}, records={records}, "
            f"passes={self.num_passes}, grad_records={grads}, "
            f"bundle={str(self.path) if self.path else 'none'})"
        )

    def __str__(self) -> str:
        """Return a bounded per-pass status table (<= 24 lines).

        Status precedes detail; failure evidence renders above the fold and
        lazy records stay lazy (a repr is never a compute trigger).
        """

        lines = [self.__repr__()]
        built = object.__getattribute__(self, "_records_built")
        if built:
            pass_rows = sorted(self.by_pass.items())
            for pass_index, indexes in pass_rows[:16]:
                lines.append(f"  pass {pass_index}: {len(indexes)} records")
            if len(pass_rows) > 16:
                lines.append(f"  (+{len(pass_rows) - 16} more passes)")
        else:
            lines.append("  records: lazy (materialized on first read)")
        if self.is_stopped:
            lines.append(f"  halted: {self.stop_reason or 'halt predicate fired'}")
        if self.is_failed:
            lines.append(f"  FAILED after {self.num_committed_ops} ops: {self.error_repr}")
            if self.last_event_label is not None:
                lines.append(
                    f"  last event: {self.last_event_label} ({self.last_event_func or '?'})"
                )
        if self.is_recovered:
            lines.append("  recovered: from a persisted partial bundle")
        return "\n".join(lines[:24])

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
            from .._capture.projectors import RecordingProjector

            projection = RecordingProjector().project(self._captured_run_cores)
            records.extend(projection.records)
            self.by_pass.update(projection.by_pass)
            self.by_label.update(projection.by_label)
            self.by_address.update(projection.by_address)
        elif self._capture_events is not None:
            from .._capture.projections import activation_record_from_event

            for event in self._capture_events.amended_op_records():
                record = activation_record_from_event(event)
                if record is None:
                    continue
                index = len(records)
                records.append(record)
                self.by_pass.setdefault(record.ctx.pass_number, []).append(index)
                for label_key in _distinct_label_index_keys(record.ctx.label, record.ctx.raw_label):
                    self.by_label.setdefault(label_key, []).append((record.ctx.pass_number, index))
                if record.ctx.address is not None:
                    self.by_address.setdefault(record.ctx.address, []).append(index)
        object.__setattr__(self, "_records_built", True)

    @property
    def lazy_trace(self) -> DryRun:
        """Return a lazy trace projection over all capture events."""

        if self._recording_trace is None:
            from .._capture.projections import recording_trace_from_events

            contexts = (
                ()
                if self._capture_events is None
                else recording_trace_from_events(self._capture_events)
            )
            object.__setattr__(
                self,
                "_recording_trace",
                DryRun(
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
        retained_keys = {
            (record.ctx.pass_number, record.ctx.event_index) for record in self.records
        }
        rows: list[dict[str, Any]] = []
        for ctx in self.lazy_trace.contexts:
            row = {name: getattr(ctx, name) for name in _RAW_METADATA_FIELDS}
            row["retained"] = (ctx.pass_number, ctx.event_index) in retained_keys
            rows.append(row)
        return tuple(rows)

    def add_grad_record(self, record: GradRecord) -> None:
        """Append one retained gradient record and update indexes."""

        index = len(self.grad_records)
        self.grad_records.append(record)
        if record.ctx.pass_number is not None:
            self.grad_by_pass.setdefault(record.ctx.pass_number, []).append(index)
        if record.ctx.layer_label is not None:
            self.grad_by_label.setdefault(record.ctx.layer_label, []).append(index)
        self.grad_by_label.setdefault(record.ctx.label, []).append(index)
        self.grad_by_grad_fn_label.setdefault(record.ctx.label, []).append(index)

    def backward(
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
        if self.is_failed:
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                "Cannot call backward on failed partial Recording; "
                "user-op failures exclude the failing call; TL-side capture failures may "
                "include a skipped/partial current-call event.",
                code="N3",
                capability="backward",
                status="failed",
            )
        if self.is_stopped:
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                f"Cannot call backward on halted Recording (stop_reason={self.stop_reason!r}).",
                code="recording_backward_halted",
                capability="backward",
                status="halted",
            )

        # A record without a backward opt-in installed no gradient hooks, and a
        # forward under no_grad / inference_mode implied inference_only and built
        # no autograd graph: either backward would capture nothing.
        state = getattr(self, "_recording_state", None)
        runtime_trace = getattr(state, "runtime_trace", None)
        runtime_options = getattr(runtime_trace, "options", None)
        if runtime_options is not None and getattr(runtime_options, "inference_only", False):
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                "Cannot call backward on this Recording: its forward ran under "
                "torch.no_grad() or torch.inference_mode(), so tl.record implied "
                "inference_only=True and no autograd graph was built. Remedy: record again "
                "with gradients enabled and pass backward_ready=True or save_grads= to tl.record.",
                code="recording_backward_inference_only",
                capability="backward",
            )
        if not getattr(runtime_trace, "capture_tensor_grad_hooks", True):
            from .exceptions import RecorderStateError

            raise RecorderStateError(
                "Cannot call backward on this Recording: tl.record installs gradient hooks "
                "only when a backward is requested up front, so this backward would record "
                "nothing. Remedy: pass backward_ready=True or save_grads= to tl.record (or "
                "Recorder) and record again.",
                code="recording_backward_not_requested",
                capability="backward",
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

    def records_in_pass(self, call_index: int) -> Iterator[ActivationRecord]:
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
                "pass_number": record.ctx.pass_number,
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

        if self.is_failed:
            return (
                f"Recording(status={self.status!r}, n_passes={self.num_passes}, "
                f"n_records={len(self)}, n_ops_completed={self.num_committed_ops}, "
                "caveat='n_ops_completed counts op-kind events across completed "
                "passes in this recorder; user-op failures exclude the failing "
                "call; TL-side capture failures may include a skipped/partial "
                "current-call event.')"
            )
        return (
            f"Recording(n_passes={self.num_passes}, n_records={len(self)}, "
            f"n_grad_records={len(self.grad_records)})"
        )

    def narrate(self, last: int | None = None, *, select: Any = None) -> str:
        """Render this recording's events in the live-narration grammar.

        One renderer, three carriers (snoop D5): failed partial recordings
        (``on_forward_error="attach_partial"``/``"return_partial"``) render
        their captured frontier under the failure header; the printed labels
        are this recording's own lookup keys.

        Parameters
        ----------
        last:
            Keep only the last ``last`` rows.
        select:
            Optional filter: a substring, or a callable over
            :class:`torchlens.types.NarrationEvent` rows.

        Returns
        -------
        str
            Rendered narration block (no trailing newline).
        """

        from torchlens._snoop._narrate import narrate_recording

        return narrate_recording(self, last=last, select=select)

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

        from .._postprocess.incremental import enrich_recording

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
          models ``stop_reason`` / ``halt_frontier`` (and other final labels)
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

        if self.is_failed:
            raise CaptureNotActiveError(
                "Recording.to_trace() cannot materialize a failed partial Recording because "
                "the topology is incomplete; user-op failures exclude the failing call; "
                "TL-side capture failures may include a skipped/partial current-call event",
                code="recording_failed_not_convertible",
                remedy="fix the failing forward and re-record before converting",
            )
        if not self._captured_run_cores:
            raise CaptureNotActiveError(
                "Recording.to_trace() requires retained capture events; disk-recovered "
                "recordings do not contain enough topology metadata",
                code="recording_events_not_retained",
                remedy="convert the in-session Recording rather than a disk-recovered one",
            )
        if self.num_passes > 1:
            raise CaptureNotActiveError(
                "Recording.to_trace() does not support multi-pass Recordings because "
                "replaying multiple Recorder(...) passes into one Trace is not yet "
                "structurally defined",
                code="recording_multipass_not_convertible",
                remedy="record one pass per Recording before converting",
            )
        from .._capture.projectors import RecordingProjector
        from .._data_classes.trace import Trace
        from ._types_ancestry import _backfill_cooked_ancestry
        from .options import _RecordingConfig

        projection = RecordingProjector().project(self._captured_run_cores)
        if projection.capture_events is None:
            raise RuntimeError("Recording.to_trace() core has no replay event facts.")

        trace = Trace(model_class_name="RecordedModel")
        # capture_mode drives exhaustive-style postprocess behavior for the
        # cooked projection; the honest provenance fact is the marker below,
        # which records that this Trace was cooked from a Recording rather
        # than captured live.
        trace._options_mode = "exhaustive"
        trace._cooked_from = "recording"
        trace._predicate_save_options = _RecordingConfig()
        trace._replay_arg_version_data_complete = False
        # Hand postprocess a STRUCTURAL COPY, never this frozen Recording's own `_capture_events`.
        # Later graph traversal replaces operation events in place, so aliasing the Recording's
        # buffer would mutate its read-only event stream. The copy shares frozen OpEvents and tensor
        # payloads by reference (cheap; no activation cloning) while giving postprocess an
        # independent working projection. NOTE: `output_layers`, `input_layers`, `buffer_layers`,
        # `internal_source_ops`, the `_layer_counter` seed, and `_recover_halt_frontier()` all read
        # from `self._capture_events` (the original, intact) below -- only the materialized
        # `trace.capture_events` is the copy.
        events_for_replay = cast(Any, projection.capture_events)._copy_for_replay()
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
        # (_capture/trace.py): it recovers a frontier-output tensor, sets
        # halted/stop_reason/halt_frontier, and marks that frontier as the
        # output parent so postprocess Step 1 synthesizes a dedicated output_N
        # node. Mirror that here so a halted Recording.to_trace() yields a
        # VALID halted Trace whose .halted is True (not silently False) and
        # whose output_layers/trace_self_consistency invariants hold -- instead
        # of the previous silent-wrong-.halted + "No output layers found"
        # crash. Self._output_tensors is empty for a halted pass, so seed the
        # frontier's saved payload as the sole output tensor (no fabrication --
        # the tensor is the recording's own retained raw activation).
        halt_output_tensors = cast("list[torch.Tensor]", list(projection.output_tensors))
        halt_output_addresses = list(projection.output_tensor_addresses)
        if self.is_stopped:
            frontier_label, frontier_tensor = self._recover_halt_frontier()
            projection.bind_halt_frontier(frontier_tensor, frontier_label)
            trace.is_stopped = True
            trace.stop_reason = self.stop_reason
            trace._graph_stop_frontier = self.stop_reason
            trace.transformed_output = None
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
        from .._capture.outcome import stamp_cooked

        cooked_frontier = None
        cooked_reason = None
        cooked_boundary = None
        cooked_boundary_kind = None
        if self.is_stopped:
            output_labels = list(getattr(trace, "output_layers", ()))
            cooked_frontier = str(output_labels[0]) if output_labels else None
            # R06: the halted postprocess remapped the persisted halt fields to
            # FINAL labels; the settled record mirrors them (settle_halted
            # parity) instead of stamping the Recording-space raw label into an
            # outcome whose frontier is final. boundary_kind rides the
            # Recording's own settled outcome; the raw stop_reason stays the
            # fallback when the remap did not run.
            remapped_reason = getattr(trace, "stop_reason", None)
            cooked_reason = (
                remapped_reason if isinstance(remapped_reason, str) else self.stop_reason
            )
            remapped_frontier = trace._graph_stop_frontier
            cooked_boundary = (
                remapped_frontier if isinstance(remapped_frontier, str) else cooked_reason
            )
            cooked_boundary_kind = getattr(self.outcome, "boundary_kind", None)
        stamp_cooked(
            trace,
            halted=self.is_stopped,
            reason=cooked_reason,
            boundary_kind=cooked_boundary_kind,
            boundary_label=cooked_boundary,
            frontier_label=cooked_frontier,
        )
        return trace

    def _recover_halt_frontier(self) -> tuple[str, torch.Tensor]:
        """Recover the frontier (output-parent label, tensor) for a halted recording.

        Mirrors ``_finalize_halted_trace``'s frontier recovery
        (``torchlens/_capture/trace.py``): prefer the halting op itself
        (``self.stop_reason`` is the frontier op's raw label), otherwise fall
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
        halt_label = self.stop_reason
        if halt_label and halt_label in payload_by_label_raw:
            return halt_label, payload_by_label_raw[halt_label]

        # Fallback: last captured op with a retained raw activation.
        core_events = tuple(event for core in self._captured_run_cores for event in core.events)
        for event in reversed(core_events):
            payload = payload_by_label_raw.get(event.label_raw)
            if payload is not None:
                return event.label_raw, payload

        raise CaptureNotActiveError(
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


def _recording_getstate(self: Recording) -> list[Any]:
    """Pickle every field except the memory-only model output."""

    return [
        None if item.name == "_output" else object.__getattribute__(self, item.name)
        for item in dataclasses.fields(self)
    ]


def _recording_setstate(self: Recording, state: list[Any]) -> None:
    """Restore a pickled Recording field by field."""

    for item, value in zip(dataclasses.fields(self), state, strict=True):
        object.__setattr__(self, item.name, value)


# Assigned after the class body: ``dataclass(slots=True, frozen=True)`` installs
# its own pickling pair on Python 3.10 even when the class defines one, and that
# pair would pickle the live model output.
Recording.__getstate__ = _recording_getstate  # type: ignore[method-assign,assignment]
Recording.__setstate__ = _recording_setstate  # type: ignore[attr-defined]


def _mark_recording_halted(recording: Recording, pass_index: int, reason: str) -> None:
    """Set halt state on a frozen ``Recording`` (mutated via ``object.__setattr__``); an empty
    ``reason`` means none was given.
    """

    recording.stops_by_pass.setdefault(pass_index, reason)
    if recording.is_stopped:
        return
    object.__setattr__(recording, "is_stopped", True)
    object.__setattr__(recording, "status", "halted")
    object.__setattr__(recording, "stop_reason", reason)
