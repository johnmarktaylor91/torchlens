"""Fastlog projections over unified capture events."""

from __future__ import annotations

import time
import traceback
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, cast

import torch

from ..fastlog.exceptions import PredicateError
from ..fastlog.types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    PredicateFailure,
    RecordContext,
    Recording,
    StorageIntent,
)
from ..ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ..ir.refs import TensorRef
from ..ir.predicate import EventKind
from ..ir.semantics import BackendSemantics, CapturePolicy

if TYPE_CHECKING:
    from ..fastlog.options import RecordingOptions

_active_recording_state: "RecordingState | None" = None


class _StorageBackend(Protocol):
    """Protocol implemented by fastlog storage backends."""

    def append(self, record: ActivationRecord) -> None:
        """Append one retained record."""

    def resolve_payloads(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        intent: StorageIntent,
        *,
        options: "RecordingOptions",
        ctx: RecordContext | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads for one selected tensor."""

    def finalize(self) -> None:
        """Finalize storage."""

    def abort(self, reason: str) -> None:
        """Abort storage."""


def _resolve_storage_intent(options: "RecordingOptions") -> StorageIntent:
    """Resolve storage destinations from StreamingOptions."""

    if options.streaming is None or options.streaming.bundle_path is None:
        return StorageIntent(in_ram=True, on_disk=False)
    return StorageIntent(
        in_ram=options.streaming.retain_in_memory,
        on_disk=True,
    )


def _empty_recording(options: "RecordingOptions") -> Recording:
    """Create an empty lazy Recording for a predicate capture session."""

    return Recording(
        records=[],
        by_pass={},
        by_label={},
        by_address={},
        bundle_path=(
            None
            if options.streaming is None or options.streaming.bundle_path is None
            else Path(options.streaming.bundle_path)
        ),
        n_ops=0,
        n_records=0,
        start_times=[],
        end_times=[],
        predicate_failures=[],
        predicate_failure_overflow_count=0,
        keep_op_repr=repr(options.keep_op) if options.keep_op is not None else None,
        keep_module_repr=repr(options.keep_module) if options.keep_module is not None else None,
        history_size=options.history_size,
        _out_transform_repr=(
            repr(options.out_transform) if options.out_transform is not None else None
        ),
    )


@dataclass(slots=True)
class RecordingState:
    """Mutable state for one active predicate recording pass."""

    options: "RecordingOptions"
    recording: Recording
    history: deque[RecordContext] = field(default_factory=deque)
    op_counts: dict[str, int] = field(default_factory=dict)
    module_stack: list[ModuleStackFrame] = field(default_factory=list)
    predicate_failures: list[PredicateFailure] = field(default_factory=list)
    predicate_failure_overflow_count: int = 0
    error_slot: BaseException | None = None
    sample_id: str | int | None = None
    pass_index: int = 0
    event_index: int = 0
    compute_index: int = 0
    no_tensor_capture: bool = False
    all_contexts: list[RecordContext] = field(default_factory=list)
    storage_intent: StorageIntent = field(init=False)
    storage_backend: _StorageBackend = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived storage policy."""

        self.storage_intent = _resolve_storage_intent(self.options)
        if self.storage_intent.on_disk:
            from ..fastlog.storage_disk import DiskStorageBackend

            self.storage_backend = DiskStorageBackend(self.options, self.recording)
        else:
            from ..fastlog.storage_ram import RamStorageBackend

            self.storage_backend = RamStorageBackend(self.recording)

    def append_context(self, ctx: RecordContext) -> None:
        """Append an event context to the bounded sliding window."""

        self.all_contexts.append(ctx)
        if self.options.history_size == 0:
            return
        self.history.append(ctx)
        while len(self.history) > self.options.history_size:
            self.history.popleft()

    def add_record(self, record: ActivationRecord) -> None:
        """Append a retained out record and update indexes."""

        self.storage_backend.append(record)
        object.__setattr__(self.recording, "n_records", len(self.recording.records))

    def resolve_storage(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        *,
        ctx: RecordContext | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads through the active storage backend."""

        if self.no_tensor_capture:
            return None, None, None, None
        return self.storage_backend.resolve_payloads(
            tensor,
            spec,
            self.storage_intent,
            options=self.options,
            ctx=ctx,
        )

    def finalize_storage(self) -> None:
        """Finalize the active storage backend."""

        self.storage_backend.finalize()

    def abort_storage(self, reason: str) -> None:
        """Abort the active storage backend after a failed pass."""

        self.storage_backend.abort(reason)

    def effective_predicate_error_mode(self) -> str:
        """Return the concrete predicate exception policy for this session."""

        if self.options.on_predicate_error != "auto":
            return self.options.on_predicate_error
        return "fail-fast" if self.storage_intent.on_disk else "accumulate"

    def add_predicate_failure(self, ctx: RecordContext, exc: BaseException) -> None:
        """Record a predicate failure subject to the configured cap."""

        failure = PredicateFailure(
            event_index=ctx.event_index,
            kind=cast(EventKind, ctx.kind),
            label=ctx.label,
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        if len(self.predicate_failures) < self.options.max_predicate_failures:
            self.predicate_failures.append(failure)
        else:
            self.predicate_failure_overflow_count += 1
        object.__setattr__(self.recording, "predicate_failures", list(self.predicate_failures))
        object.__setattr__(
            self.recording,
            "predicate_failure_overflow_count",
            self.predicate_failure_overflow_count,
        )

    def handle_predicate_exception(self, ctx: RecordContext, exc: BaseException) -> None:
        """Apply the configured predicate exception policy."""

        self.add_predicate_failure(ctx, exc)
        if self.effective_predicate_error_mode() == "fail-fast":
            raise exc

    def raise_accumulated_predicate_error(self) -> None:
        """Raise a final PredicateError when accumulated failures exist."""

        if self.effective_predicate_error_mode() != "accumulate":
            return
        if not self.predicate_failures and self.predicate_failure_overflow_count == 0:
            return
        details = ""
        if self.predicate_failures:
            details = f": {self.predicate_failures[0].traceback.strip().splitlines()[-1]}"
        raise PredicateError(
            f"fastlog predicate failed during recording{details}",
            failures=list(self.predicate_failures),
            total_count=len(self.predicate_failures) + self.predicate_failure_overflow_count,
            overflow=self.predicate_failure_overflow_count,
        )


def get_active_recording_state() -> RecordingState:
    """Return the active recording state or fail clearly."""

    if _active_recording_state is None:
        raise RuntimeError("fastlog predicate state is not active")
    return _active_recording_state


@contextmanager
def active_recording_state(state: RecordingState) -> Iterator[RecordingState]:
    """Install fastlog recording state for one active logging scope."""

    global _active_recording_state
    previous = _active_recording_state
    _active_recording_state = state
    try:
        yield state
    finally:
        _active_recording_state = previous


def _read_field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping or object with a default."""

    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalize_module_stack(
    module_stack: Iterable[ModuleStackFrame | Mapping[str, Any]] | None,
) -> tuple[ModuleStackFrame, ...]:
    """Normalize module stack input to ModuleStackFrame instances."""

    if module_stack is None:
        return ()
    frames: list[ModuleStackFrame] = []
    for frame in module_stack:
        if isinstance(frame, ModuleStackFrame):
            frames.append(frame)
            continue
        frames.append(
            ModuleStackFrame(
                address=str(frame.get("address", "")),
                module_type=str(frame.get("module_type", "")),
                module_id=int(frame.get("module_id", 0)),
                pass_index=int(frame.get("pass_index", 0)),
            )
        )
    return tuple(frames)


def _recent_ops_for_event(
    recent_events: Sequence[RecordContext],
    include_source_events: bool,
) -> tuple[RecordContext, ...]:
    """Return the operation-visible subset of recent events."""

    visible_kinds = {"op", "input", "buffer"} if include_source_events else {"op"}
    return tuple(event for event in recent_events if event.kind in visible_kinds)


def _build_record_context(
    *,
    kind: str,
    op_log_or_op_data: Any = None,
    module_stack: Iterable[ModuleStackFrame | Mapping[str, Any]] | None = None,
    history: Sequence[RecordContext] = (),
    op_counts: Mapping[str, int] | None = None,
    pass_index: int = 0,
    event_index: int = 0,
    compute_index: int | None = None,
    time_since_pass_start: float = 0.0,
    include_source_events: bool = False,
    sample_id: str | int | None = None,
) -> RecordContext:
    """Build the single source-of-truth RecordContext schema."""

    data = op_log_or_op_data
    stack = _normalize_module_stack(module_stack)
    recent_events = tuple(history)
    layer_type = _read_field(data, "layer_type")
    if layer_type is None:
        layer_type = _read_field(data, "type")
    if layer_type is None:
        layer_type = _read_field(data, "func_name")
    if isinstance(layer_type, str):
        layer_type = layer_type.lower().replace("_", "")
    type_index = _read_field(data, "type_index")
    if type_index is None and layer_type is not None and op_counts is not None:
        type_index = op_counts.get(cast(str, layer_type))
    tensor = _read_field(data, "tensor")
    shape = _read_field(data, "shape")
    dtype = _read_field(data, "dtype")
    tensor_device = _read_field(data, "tensor_device")
    tensor_requires_grad = _read_field(data, "tensor_requires_grad")
    if isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        tensor_device = tensor.device
        tensor_requires_grad = tensor.requires_grad
    raw_label = _read_field(data, "_label_raw", _read_field(data, "raw_label"))
    label = _read_field(data, "label", raw_label)
    if label is None:
        label = f"{kind}_{event_index}"
    return RecordContext(
        kind=kind,
        label=str(label),
        raw_label=raw_label,
        pass_index=pass_index,
        event_index=event_index,
        compute_index=compute_index,
        layer_type=layer_type,
        type_index=type_index,
        capture_index=_read_field(data, "capture_index"),
        func_name=_read_field(data, "func_name"),
        address=_read_field(data, "address"),
        module_type=_read_field(data, "module_type"),
        module_pass_index=_read_field(data, "module_pass_index"),
        module_stack=stack,
        recent_events=recent_events,
        recent_ops=_recent_ops_for_event(recent_events, include_source_events),
        parent_labels=tuple(_read_field(data, "parent_labels", ())),
        input_output_address=_read_field(data, "input_output_address"),
        shape=shape,
        dtype=dtype,
        tensor_device=tensor_device,
        tensor_requires_grad=tensor_requires_grad,
        output_index=_read_field(data, "output_index"),
        is_bottom_level_func=_read_field(data, "is_bottom_level_func"),
        time_since_pass_start=time_since_pass_start,
        sample_id=sample_id,
    )


def _module_frames_from_record_context(ctx: RecordContext) -> tuple[ModuleFrame, ...]:
    """Convert fastlog module-stack frames to IR module frames."""

    return tuple(
        ModuleFrame(
            address=frame.address,
            address_normalized=None,
            module_type=frame.module_type,
            call_index=frame.pass_index,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for frame in ctx.module_stack
    )


def _record_context_from_event(event: OpEvent) -> RecordContext:
    """Project one ``OpEvent`` back into the fastlog predicate schema."""

    ctx = getattr(event, "record_context", None)
    if isinstance(ctx, RecordContext):
        return ctx
    tensor = event.output.tensor
    return _build_record_context(
        kind=event.kind,
        op_log_or_op_data={
            "label": event.label_raw,
            "raw_label": event.label_raw,
            "_label_raw": event.label_raw,
            "capture_index": event.capture_index,
            "type": event.layer_type,
            "type_index": event.type_index,
            "func_name": event.function.func_name,
            "parent_labels": tuple(parent.parent_label_raw for parent in event.parents),
            "shape": tensor.shape,
            "dtype": torch.dtype(tensor.dtype) if isinstance(tensor.dtype, torch.dtype) else None,
            "tensor_device": torch.device(tensor.device) if tensor.device is not None else None,
            "tensor_requires_grad": tensor.requires_grad,
            "output_index": event.output.multi_output_index,
            "is_bottom_level_func": event.is_bottom_level,
        },
        event_index=event.capture_index,
        compute_index=event.compute_index,
    )


def _event_from_record(
    ctx: RecordContext,
    spec: CaptureSpec,
    *,
    tensor: torch.Tensor | None = None,
    ram_payload: torch.Tensor | None = None,
    transformed_ram_payload: torch.Tensor | None = None,
    predicate_matched: bool,
) -> OpEvent:
    """Build a lightweight fastlog ``OpEvent`` without materializing an OpLog."""

    label_raw = ctx.raw_label or ctx.label
    tensor_ref = TensorRef(
        label_raw=label_raw,
        shape=ctx.shape,
        dtype=str(ctx.dtype) if ctx.dtype is not None else None,
        device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
        requires_grad=ctx.tensor_requires_grad,
        memory=None,
        payload=ram_payload,
        blob_ref=None,
        backend_handle_id=str(id(tensor)) if tensor is not None else None,
    )
    transformed_ref = None
    if transformed_ram_payload is not None:
        transformed_ref = TensorRef(
            label_raw=label_raw,
            shape=tuple(transformed_ram_payload.shape),
            dtype=str(transformed_ram_payload.dtype),
            device=str(transformed_ram_payload.device),
            requires_grad=transformed_ram_payload.requires_grad,
            memory=None,
            payload=transformed_ram_payload,
            blob_ref=None,
            backend_handle_id=str(id(transformed_ram_payload)),
        )
    event = OpEvent(
        kind=ctx.kind,
        label_raw=label_raw,
        layer_label_raw=label_raw,
        layer_type=ctx.layer_type or ctx.kind,
        capture_index=ctx.capture_index or ctx.event_index,
        type_index=ctx.type_index or 0,
        compute_index=ctx.compute_index or 0,
        source_trace_id=None,
        tracing_finished=False,
        construction_done=True,
        function=FunctionCallRef(
            func=None,
            func_name=ctx.func_name,
            func_qualname=None,
            func_call_id=None,
            code_context=(),
            func_duration=None,
            flops_forward=None,
            flops_backward=None,
            func_rng_states=None,
            func_autocast_state=None,
            arg_names=(),
            num_args_total=0,
            num_pos_args=0,
            num_kwargs=0,
            non_tensor_pos_args=(),
            non_tensor_kwargs=(),
            func_non_tensor_args=(),
            is_inplace=False,
            func_config=(),
        ),
        output=OutputRef(
            tensor=tensor_ref,
            transformed_tensor=transformed_ref,
            has_saved_outs=bool(spec.save_out and (ram_payload is not None)),
            output_device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
            out_postfunc=None,
            detach_saved_activations=not spec.keep_grad,
            visualizer_path=None,
            multi_output_index=ctx.output_index,
            is_part_of_iterable_output=False,
            container_path=(),
            container_spec=None,
            child_versions=(),
        ),
        templates=ArgTemplateRef(
            saved_args=None,
            saved_kwargs=None,
            args_template=None,
            kwargs_template=None,
            has_saved_args=False,
        ),
        parents=tuple(
            ParentEdge(parent_label_raw=parent, arg_position=None, edge_use="arg")
            for parent in ctx.parent_labels
        ),
        params=(),
        module_stack=_module_frames_from_record_context(ctx),
        backend_semantics=BackendSemantics(
            grad_fn_id=None,
            grad_fn_name=None,
            autograd_saved_memory=0,
            num_autograd_saved_tensors=0,
            mutates_inputs=(),
            bytes_delta_at_call=None,
            bytes_peak_at_call=None,
        ),
        policy=CapturePolicy(
            must_keep_topology=False,
            save_payload=spec.save_out,
            requires_isolation=False,
            save_args=False,
            save_code=False,
            save_rng=False,
            save_grad=spec.keep_grad,
            stream=False,
        ),
        predicate_matched=predicate_matched,
        is_bottom_level=bool(ctx.is_bottom_level_func),
        is_scalar_bool=None,
        bool_value=None,
        intervention_fired=False,
        intervention_replaced=False,
        fire_results=(),
        intervention_template_ref=None,
        record_context=ctx,
        capture_spec=spec,
    )
    return event


def append_projected_event(
    trace: Any,
    ctx: RecordContext,
    spec: CaptureSpec,
    *,
    tensor: torch.Tensor | None = None,
    ram_payload: torch.Tensor | None = None,
    transformed_ram_payload: torch.Tensor | None = None,
    predicate_matched: bool,
) -> None:
    """Append one lightweight predicate event to ``trace.capture_events``."""

    if not hasattr(trace, "capture_events"):
        from ..ir import CaptureEvents

        trace.capture_events = CaptureEvents()
    trace.capture_events.append(
        _event_from_record(
            ctx,
            spec,
            tensor=tensor,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram_payload,
            predicate_matched=predicate_matched,
        )
    )


def activation_record_from_event(event: OpEvent) -> ActivationRecord | None:
    """Project a retained ``OpEvent`` into an ``ActivationRecord``."""

    if not event.predicate_matched:
        return None
    spec = getattr(event, "capture_spec", CaptureSpec(save_out=False, save_metadata=True))
    ctx = _record_context_from_event(event)
    ram_payload = event.output.tensor.payload if spec.save_out else None
    transformed_ram_payload = (
        event.output.transformed_tensor.payload
        if event.output.transformed_tensor is not None and spec.save_out
        else None
    )
    return ActivationRecord(
        ctx=ctx,
        spec=spec,
        ram_payload=ram_payload if isinstance(ram_payload, torch.Tensor) else None,
        transformed_ram_payload=(
            transformed_ram_payload if isinstance(transformed_ram_payload, torch.Tensor) else None
        ),
    )


def recording_trace_from_events(events: Any) -> tuple[RecordContext, ...]:
    """Project capture events into fastlog ``RecordContext`` objects."""

    return tuple(_record_context_from_event(event) for event in events.op_events)
