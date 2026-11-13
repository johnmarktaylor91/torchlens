"""The lean event interface a non-torch backend's ``tl.record`` producer emits.

A backend whose ``capability_implementations["fastlog"]`` factory resolves to a
:class:`BackendRecordProducer` records through this module instead of the torch
``Recorder``. The backend recorder drives the producer one forward pass at a time
against a :class:`RecordSink`:

* the producer emits one frozen :class:`BackendOpEvent` (metadata only, never a
  value) for every op it runs, in execution order;
* the sink builds that event's ``RecordContext``, runs the save and halt predicate
  helpers the torch recorder runs, and answers with an :class:`EventDecision`;
* only when ``decision.request`` is set does the producer hand the op's value to
  :meth:`RecordSink.deliver` (at once, or later in the pass for a lazy framework
  that computes values only when asked), so unselected values are never copied;
* when ``decision.stop`` is set the producer delivers its outstanding requests and
  returns without running the rest of the forward. An event emitted after the stop
  refuses: the interpreted suffix never runs.

Payloads route through the producer's :class:`PayloadCodec` into the recording's
RAM store and, with ``storage=``, the disk bundle. Records land in event order even
when a lazy producer delivers out of order. The sink keeps one context per op only
in the lookback window; it never builds the full journal ``tl.trace`` builds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, TypeGuard, runtime_checkable

import torch

from .._capture.predicates import _evaluate_keep_op
from .._capture.projections import _build_record_context
from .._capture.stop import StopDirective
from ._backend_options import refuse_record_option
from ._exceptions import RecorderStateError
from ._halt import StopSignal
from ._types import ActivationRecord, CaptureSpec, ModuleStackFrame, RecordContext

if TYPE_CHECKING:
    from .._capture.projections import RecordingState

#: Event kinds a producer may emit. Module boundaries arrive as the
#: ``module_stack`` of each event, not as events of their own.
BACKEND_EVENT_KINDS: frozenset[str] = frozenset({"op", "input", "buffer"})


@dataclass(frozen=True, slots=True)
class BackendOpEvent:
    """Metadata for one executed op, as a backend producer reports it.

    ``raw_label`` is the backend's raw capture label (the same spelling its
    ``tl.trace`` gives the op), ``layer_type`` and ``type_index`` its normalized
    type and per-type counter, ``raw_index`` its one-based op index. ``dtype`` and
    ``device`` take the backend's own objects or strings; the sink normalizes them.
    ``module_stack`` lists the active module calls, outermost first.
    """

    raw_label: str
    layer_type: str
    type_index: int
    raw_index: int
    parent_labels: tuple[str, ...] = ()
    func_name: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: Any = None
    device: Any = None
    requires_grad: bool | None = None
    output_index: int | None = None
    is_bottom_level_func: bool | None = True
    module_stack: tuple[ModuleStackFrame, ...] = ()
    func_call_id: int | None = None
    kind: str = "op"

    def __post_init__(self) -> None:
        """Refuse an event the sink cannot turn into a predicate context."""

        if self.kind not in BACKEND_EVENT_KINDS:
            refuse_backend_record_state(
                f"backend record event kind must be one of {sorted(BACKEND_EVENT_KINDS)}; "
                f"received {self.kind!r} (modules travel as module_stack)"
            )
        if not isinstance(self.raw_label, str) or not self.raw_label:
            refuse_backend_record_state(
                "backend record event needs the non-empty raw label the backend's "
                "tl.trace gives this op"
            )


@dataclass(frozen=True, slots=True)
class PayloadRequest:
    """The sink's request for one selected op's value (pass it to ``deliver``)."""

    token: int
    raw_label: str


@dataclass(frozen=True, slots=True)
class EventDecision:
    """The sink's answer to one event: a payload request and whether to stop."""

    request: PayloadRequest | None
    stop: bool


@runtime_checkable
class PayloadCodec(Protocol):
    """Backend-owned payload conversions used when a selected value is stored."""

    def ram_copy(self, value: Any) -> Any:
        """Return the value the recording keeps in memory (an independent copy)."""

    def to_disk_tensor(self, value: Any) -> torch.Tensor:
        """Return a CPU ``torch.Tensor`` the disk bundle writer can persist."""


@runtime_checkable
class BackendRecordProducer(Protocol):
    """What a backend's ``capability_implementations["fastlog"]`` factory returns.

    ``prepare`` runs once per recorder (model discovery is cached across passes);
    ``run_pass`` runs one forward against the sink and returns the model output.
    ``supported_options`` names the record options the producer honours beyond
    the neutral set (for example ``"seed"``); every other one refuses typed.
    """

    backend: str
    codec: PayloadCodec
    supported_options: frozenset[str]

    def prepare(self, model: Any) -> Any:
        """Discover the model once; the result is reused for every pass."""

    def run_pass(
        self,
        prepared: Any,
        input_args: Any,
        input_kwargs: Mapping[str, Any],
        sink: RecordSink,
    ) -> Any:
        """Run one forward, emitting events to ``sink``; return the model output."""


def is_backend_record_producer(candidate: object) -> TypeGuard[BackendRecordProducer]:
    """Return whether ``candidate`` satisfies :class:`BackendRecordProducer`."""

    return isinstance(candidate, BackendRecordProducer) and isinstance(
        getattr(candidate, "codec", None), PayloadCodec
    )


@dataclass(slots=True)
class _Pending:
    """One selected event waiting for its payload, in event order."""

    ctx: RecordContext
    spec: CaptureSpec
    record: ActivationRecord | None = None


@dataclass(slots=True)
class _PassFacts:
    """What the sink saw in one pass, for halted and failed recordings."""

    num_op_events: int = 0
    last_op_label: str | None = None
    last_event: BackendOpEvent | None = None
    last_ctx: RecordContext | None = None


@dataclass(slots=True)
class RecordSink:
    """Turns one pass of producer events into retained ``ActivationRecord`` values."""

    state: RecordingState
    codec: PayloadCodec
    backend: str
    pause_capture: Callable[[], AbstractContextManager[Any]] = nullcontext
    stop: StopSignal | None = None
    facts: _PassFacts = field(default_factory=_PassFacts)
    _pending: dict[int, _Pending] = field(default_factory=dict)
    _next_token: int = 0
    _flushed_token: int = 0
    _start_time: float = field(default_factory=time.time)

    def emit(self, event: BackendOpEvent) -> EventDecision:
        """Decide one event: build its context, run the predicates, maybe request its value."""

        if self.stop is not None:
            refuse_backend_record_state(
                f"backend {self.backend!r} emitted {event.raw_label!r} after the recording "
                "stopped; a producer must return once a decision says stop"
            )
        ctx = self._context_for(event)
        spec = self._save_decision(ctx)
        selected = spec.save_out or spec.save_metadata
        self.state.append_context(ctx, selected=selected)
        self._note(event, ctx)
        request = self._queue(ctx, spec) if selected else None
        self._evaluate_halt(ctx)
        return EventDecision(request=request, stop=self.stop is not None)

    def deliver(self, request: PayloadRequest, value: Any) -> None:
        """Store the value for an earlier request through the codec."""

        pending = self._pending.get(request.token)
        if pending is None or pending.record is not None:
            refuse_backend_record_state(
                f"backend {self.backend!r} delivered {request.raw_label!r} for a request "
                "that is unknown or already delivered"
            )
        pending.record = self._payload_record(pending, value)
        self._flush()

    def finish_pass(self) -> None:
        """Refuse a pass that returned with requested values never delivered."""

        undelivered = [p.ctx.label for p in self._pending.values() if p.record is None]
        if undelivered:
            refuse_backend_record_state(
                f"backend {self.backend!r} ended the pass without delivering the values of "
                f"{len(undelivered)} selected op(s), first {undelivered[0]!r}"
            )

    def iter_pending_labels(self) -> Iterator[str]:
        """Yield the labels of requests still waiting for a value."""

        return (p.ctx.label for p in self._pending.values() if p.record is None)

    def _context_for(self, event: BackendOpEvent) -> RecordContext:
        """Build the frozen predicate context, advancing the pass counters."""

        state = self.state
        event_index = state.event_index
        state.event_index += 1
        step_index: int | None = None
        if event.kind == "op":
            step_index = state.step_index
            state.step_index += 1
            state.op_counts[event.layer_type] = state.op_counts.get(event.layer_type, 0) + 1
        inner = event.module_stack[-1] if event.module_stack else None
        return _build_record_context(
            kind=event.kind,
            op_log_or_op_data=_event_fields(event, inner),
            module_stack=event.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=event_index,
            step_index=step_index,
            time_since_pass_start=time.time() - self._start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
        )

    def _save_decision(self, ctx: RecordContext) -> CaptureSpec:
        """Run the save predicate as the torch recorder does; a halt() inside it stops."""

        try:
            with self.pause_capture():
                decision = _evaluate_keep_op(ctx, self.state.options)
        except StopSignal as stop:
            self.stop = stop
            return CaptureSpec(save_out=False, save_metadata=False)
        except Exception as exc:
            self.state.handle_predicate_exception(ctx, exc)
            return CaptureSpec(save_out=False, save_metadata=False)
        if not isinstance(decision, CaptureSpec):
            refuse_record_option(self.backend, "a retroactive save= decision (followed_by)")
        _refuse_spec_fields(self.backend, decision)
        return decision

    def _evaluate_halt(self, ctx: RecordContext) -> None:
        """Run ``stop_after`` after the save decision, as the torch recorder does."""

        if self.stop is not None:
            return
        try:
            with self.pause_capture():
                StopDirective(halt_options=self.state.options).evaluate_halt(ctx)
        except StopSignal as stop:
            self.stop = stop

    def _note(self, event: BackendOpEvent, ctx: RecordContext) -> None:
        """Remember the last event (and op) for halted and failed recordings."""

        self.facts.last_event = event
        self.facts.last_ctx = ctx
        if event.kind == "op":
            self.facts.num_op_events += 1
            self.facts.last_op_label = event.raw_label

    def _queue(self, ctx: RecordContext, spec: CaptureSpec) -> PayloadRequest | None:
        """Queue a selected event; a metadata-only one is complete at once."""

        token = self._next_token
        self._next_token += 1
        pending = _Pending(ctx=ctx, spec=spec)
        self._pending[token] = pending
        if not spec.save_out:
            pending.record = ActivationRecord(ctx=ctx, spec=spec)
            self._flush()
            return None
        return PayloadRequest(token=token, raw_label=ctx.raw_label or ctx.label)

    def _payload_record(self, pending: _Pending, value: Any) -> ActivationRecord:
        """Route one value to RAM and disk through the codec."""

        intent = self.state.storage_intent
        keep_raw = self.state.options.save_raw_activations
        ram_payload = self.codec.ram_copy(value) if intent.in_ram and keep_raw else None
        disk_payload = self.codec.to_disk_tensor(value) if intent.on_disk and keep_raw else None
        return ActivationRecord(
            ctx=pending.ctx,
            spec=pending.spec,
            ram_payload=ram_payload,
            disk_payload=disk_payload,
        )

    def _flush(self) -> None:
        """Append completed records to the store, in event order."""

        while self._flushed_token in self._pending:
            pending = self._pending[self._flushed_token]
            if pending.record is None:
                return
            self.state.add_record(pending.record)
            del self._pending[self._flushed_token]
            self._flushed_token += 1


def _event_fields(event: BackendOpEvent, inner: ModuleStackFrame | None) -> dict[str, Any]:
    """Return the field mapping ``_build_record_context`` reads for one event."""

    return {
        "label": event.raw_label,
        "raw_label": event.raw_label,
        "_label_raw": event.raw_label,
        "raw_index": event.raw_index,
        "type": event.layer_type,
        "type_index": event.type_index,
        "func_name": event.func_name,
        "parent_labels": event.parent_labels,
        "shape": event.shape,
        "dtype": event.dtype,
        "tensor_device": event.device,
        "tensor_requires_grad": event.requires_grad,
        "output_index": event.output_index,
        "is_bottom_level_func": event.is_bottom_level_func,
        "address": None if inner is None else inner.address,
        "module_type": None if inner is None else inner.module_type,
        "module_pass_number": None if inner is None else inner.pass_number,
    }


def _refuse_spec_fields(backend: str, spec: CaptureSpec) -> None:
    """Refuse capture-spec fields the neutral path has no codec step for."""

    if spec.save_grad:
        refuse_record_option(backend, "CaptureSpec(save_grad=True)")
    if spec.device is not None or spec.dtype is not None:
        refuse_record_option(backend, "CaptureSpec(device=..., dtype=...)")
    if spec.save_mode != "copy":
        refuse_record_option(backend, f"CaptureSpec(save_mode={spec.save_mode!r})")


def refuse_backend_record_state(problem: str) -> NoReturn:
    """Raise the uncoded ``RecorderStateError`` the torch ``Recorder`` raises for misuse.

    The backend recorder's lifecycle misuse and producer-contract breaches all
    raise here, uncoded, as the torch recorder's misuse refusals are.
    """

    raise RecorderStateError(problem)
