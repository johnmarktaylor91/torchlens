"""Fastlog projections over unified capture events."""

from __future__ import annotations

import time
import traceback
import weakref
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, cast

import torch

from .._state import pause_logging
from ..fastlog.exceptions import PredicateError
from ..fastlog.types import (
    ActivationRecord,
    CaptureSpec,
    GradRecordContext,
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
from ..ir.refs import DeviceRef, DtypeRef, TensorRef
from ..ir.predicate import EventKind, coerce_deferred_value
from ..ir.semantics import BackendSemantics, CapturePolicy
from ..utils.tensor_utils import get_memory_amount_from_metadata

if TYPE_CHECKING:
    from ..fastlog.options import RecordingOptions
    from ..data_classes.trace import Trace
    from ..ir import LiveOpRecord

_active_recording_state: "RecordingState | None" = None


class _GradFnContextMap:
    """Map autograd nodes to record contexts with weak keys when supported."""

    def __init__(self) -> None:
        """Initialize weak-key and object-key fallback storage."""

        self._weak: weakref.WeakKeyDictionary[Any, RecordContext] = weakref.WeakKeyDictionary()
        self._strong: dict[Any, RecordContext] = {}

    def __setitem__(self, key: Any, value: RecordContext) -> None:
        """Store a context by grad_fn_handle object."""

        try:
            self._weak[key] = value
        except TypeError:
            self._strong[key] = value

    def get(self, key: Any, default: RecordContext | None = None) -> RecordContext | None:
        """Return the context for a grad_fn_handle object, if present."""

        try:
            value = self._weak.get(key)
        except TypeError:
            value = None
        if value is not None:
            return value
        return self._strong.get(key, default)


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
        ctx: RecordContext | GradRecordContext | None,
        kind: str = "activation",
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
        orphan_records=[],
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
        halted=False,
        halt_reason=None,
        halts_by_pass={},
        keep_op_repr=repr(options.keep_op) if options.keep_op is not None else None,
        keep_module_repr=repr(options.keep_module) if options.keep_module is not None else None,
        history_size=options.history_size,
        save_grads_repr=repr(options.save_grads) if options.save_grads is not None else None,
        _activation_transform_repr=(
            repr(options.activation_transform) if options.activation_transform is not None else None
        ),
        _grad_transform_repr=(
            repr(options.grad_transform) if options.grad_transform is not None else None
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
    step_index: int = 0
    no_tensor_capture: bool = False
    all_contexts: list[RecordContext] = field(default_factory=list)
    storage_intent: StorageIntent = field(init=False)
    storage_backend: _StorageBackend = field(init=False)
    grad_fn_to_context: _GradFnContextMap = field(default_factory=_GradFnContextMap)
    runtime_trace: "Trace | None" = None
    active_save_grads_record_policy: Any | None = None

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
        window_size = self.options.lookback or self.options.history_size
        if window_size == 0:
            return
        self.history.append(ctx)
        while len(self.history) > window_size:
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
        ctx: RecordContext | GradRecordContext | None = None,
        kind: str = "activation",
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
            kind=kind,
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
    step_index: int | None = None,
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
        dtype = DtypeRef.from_value(tensor.dtype)
        tensor_device = DeviceRef.from_value(tensor.device)
        tensor_requires_grad = tensor.requires_grad
    else:
        dtype = DtypeRef.from_value(dtype)
        tensor_device = DeviceRef.from_value(tensor_device)
    raw_label = _read_field(data, "_label_raw", _read_field(data, "raw_label"))
    label = _read_field(data, "label", raw_label)
    if label is None:
        label = f"{kind}_{event_index}"
    parent_labels = tuple(_read_field(data, "parent_labels", ()))
    return RecordContext(
        kind=kind,
        label=str(label),
        raw_label=raw_label,
        pass_index=pass_index,
        event_index=event_index,
        step_index=step_index,
        layer_type=layer_type,
        type_index=type_index,
        raw_index=_read_field(data, "raw_index"),
        func_name=_read_field(data, "func_name"),
        address=_read_field(data, "address"),
        module_type=_read_field(data, "module_type"),
        module_pass_index=_read_field(data, "module_pass_index"),
        module_stack=stack,
        recent_events=recent_events,
        recent_ops=_recent_ops_for_event(recent_events, include_source_events),
        parent_labels=parent_labels,
        input_output_address=_read_field(data, "input_output_address"),
        shape=shape,
        dtype=dtype,
        tensor_device=tensor_device,
        tensor_requires_grad=tensor_requires_grad,
        output_index=_read_field(data, "output_index"),
        is_bottom_level_func=_read_field(data, "is_bottom_level_func"),
        time_since_pass_start=time_since_pass_start,
        sample_id=sample_id,
        label_raw=str(raw_label) if raw_label is not None else "",
        label_prefix=str(label).rsplit("_", 2)[0] if isinstance(label, str) else "",
        parent_labels_raw=parent_labels,
        is_transform=bool(_read_field(data, "is_transform", False)),
        transform_kind=_read_field(data, "transform_kind"),
    )


def _module_frames_from_record_context(ctx: RecordContext) -> tuple[ModuleFrame, ...]:
    """Convert fastlog module-stack frames to IR module frames."""

    return tuple(
        ModuleFrame(
            address=_module_frame_address(frame),
            address_normalized=None,
            module_type=frame.module_type,
            call_index=frame.pass_index,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for frame in ctx.module_stack
    )


def _module_frame_address(frame: ModuleStackFrame) -> str:
    """Return the Trace-facing address for a predicate module-stack frame.

    Parameters
    ----------
    frame
        Predicate module frame from a capture-time ``RecordContext``.

    Returns
    -------
    str
        Public module address used by postprocess.
    """

    return frame.address or "self"


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
            "raw_index": event.raw_index,
            "type": event.layer_type,
            "type_index": event.type_index,
            "func_name": event.function.func_name,
            "parent_labels": tuple(parent.parent_label_raw for parent in event.parents),
            "shape": tensor.shape,
            "dtype": DtypeRef.from_value(tensor.dtype),
            "tensor_device": DeviceRef.from_value(tensor.device),
            "tensor_requires_grad": tensor.requires_grad,
            "output_index": event.output.multi_output_index,
            "is_bottom_level_func": event.is_bottom_level,
        },
        event_index=event.raw_index,
        step_index=event.step_index,
    )


def _event_from_record(
    ctx: RecordContext,
    spec: CaptureSpec,
    *,
    tensor: torch.Tensor | None = None,
    ram_payload: torch.Tensor | None = None,
    transformed_ram_payload: torch.Tensor | None = None,
    predicate_matched: bool,
    backend_semantics: BackendSemantics | None = None,
    function: FunctionCallRef | None = None,
    container_path: tuple[Any, ...] = (),
) -> OpEvent:
    """Build a lightweight fastlog ``OpEvent`` without materializing an Op."""

    label_raw = ctx.raw_label or ctx.label
    memory = (
        get_memory_amount_from_metadata(tensor, ctx.shape or tuple(tensor.shape), tensor.dtype)
        if tensor is not None
        else 0
    )
    tensor_requires_grad = cast(bool | None, coerce_deferred_value(ctx.tensor_requires_grad))
    is_scalar_bool = cast(bool | None, coerce_deferred_value(ctx.is_scalar_bool))
    bool_value = cast(bool | None, coerce_deferred_value(ctx.bool_value))
    tensor_ref = TensorRef(
        label_raw=label_raw,
        shape=ctx.shape,
        dtype=str(ctx.dtype) if ctx.dtype is not None else None,
        device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
        requires_grad=tensor_requires_grad,
        memory=memory,
        payload=ram_payload,
        blob_ref=None,
        backend_handle_id=str(id(tensor)) if tensor is not None else None,
    )
    transformed_ref = None
    if transformed_ram_payload is not None:
        transformed_shape = tuple(transformed_ram_payload.shape)
        transformed_dtype = transformed_ram_payload.dtype
        transformed_memory = get_memory_amount_from_metadata(
            transformed_ram_payload,
            transformed_shape,
            transformed_dtype,
        )
        transformed_ref = TensorRef(
            label_raw=label_raw,
            shape=transformed_shape,
            dtype=str(transformed_dtype),
            device=str(transformed_ram_payload.device),
            requires_grad=transformed_ram_payload.requires_grad,
            memory=transformed_memory,
            payload=transformed_ram_payload,
            blob_ref=None,
            backend_handle_id=str(id(transformed_ram_payload)),
        )
    event = OpEvent(
        kind=ctx.kind,
        label_raw=label_raw,
        layer_label_raw=label_raw,
        layer_type=ctx.layer_type or ctx.kind,
        raw_index=ctx.raw_index or ctx.event_index,
        type_index=ctx.type_index or 0,
        step_index=ctx.step_index or 0,
        source_trace=None,
        source_trace_id=None,
        tracing_finished=False,
        construction_done=True,
        function=function
        or FunctionCallRef(
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
            has_saved_activation=bool(spec.save_out and (ram_payload is not None)),
            output_device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
            activation_transform=None,
            detach_saved_activations=not spec.keep_grad,
            visualizer_path=None,
            multi_output_index=ctx.output_index,
            in_multi_output=bool(container_path),
            container_path=container_path,
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
            ParentEdge(parent_label_raw=parent, arg_position=None, edge_use="unknown")
            for parent in ctx.parent_labels
        ),
        parent_arg_positions={"args": {}, "kwargs": {}},
        _edge_uses=(),
        params=(),
        parent_params=(),
        module_stack=_module_frames_from_record_context(ctx),
        modules=tuple(
            (_module_frame_address(frame), frame.pass_index) for frame in ctx.module_stack
        ),
        backend_semantics=backend_semantics
        if backend_semantics is not None
        else BackendSemantics(
            backend_grad_handle=None,
            grad_fn_class_name=None,
            autograd_memory=0,
            num_autograd_tensors=0,
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
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
            save_mode=spec.save_mode,
        ),
        predicate_matched=predicate_matched,
        pass_index=ctx.pass_index,
        grad_fn_class_qualname=None,
        grad_fn_handle=None,
        equivalence_class=None,
        is_transform=False,
        transform_kind=None,
        transform_chain=(),
        transform_config={"_tl_annotations": _reference_annotations(spec.save_mode, ram_payload)},
        transform_fn_name=None,
        transform_fn_qualname=None,
        transform_fn_source=None,
        unattributed_tensor_args=(),
        is_output_parent=ctx.is_output_parent,
        has_internal_source_ancestor=False,
        internal_source_ancestors=frozenset(),
        input_ancestors=frozenset(),
        root_ancestors=frozenset(),
        func_call_id=ctx.func_call_id,
        is_bottom_level=bool(ctx.is_bottom_level_func),
        is_scalar_bool=is_scalar_bool,
        bool_value=bool_value,
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
    backend_semantics: BackendSemantics | None = None,
    function: FunctionCallRef | None = None,
    container_path: tuple[Any, ...] = (),
) -> None:
    """Append one lightweight predicate event to ``trace.capture_events``."""

    if not hasattr(trace, "capture_events"):
        from ..ir import CaptureEvents

        trace.capture_events = CaptureEvents()
    label_raw = ctx.raw_label or ctx.label
    if tensor is not None and ctx.kind == "op":
        from ..backends.torch.tensor_tracking import _add_tensor_backward_hook

        public_label = _public_record_context_label(ctx)
        trace.__dict__.setdefault("_raw_to_final_layer_labels", {})[label_raw] = public_label
        trace.__dict__.setdefault("_fastlog_grad_contexts", {})[public_label] = ctx
        _add_tensor_backward_hook(trace, tensor, label_raw)
    trace.capture_events.append(
        _event_from_record(
            ctx,
            spec,
            tensor=tensor,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram_payload,
            predicate_matched=predicate_matched,
            backend_semantics=backend_semantics,
            function=function,
            container_path=container_path,
        )
    )


class LiveOpViewFieldNotYetWritten(AttributeError):
    """Raised when a live op view field is populated only by postprocess."""


_OPLOG_FIELDS_KNOWN_LATE = frozenset(
    {
        "final_out",
        "layer_label",
        "layer_label_short",
        "label",
        "label_short",
        "layer_label",
        "layer_label_short",
    }
)


def _event_live_field(trace: "Trace", event: OpEvent, name: str) -> Any:
    """Return a forward-time field projected from an operation event.

    Parameters
    ----------
    trace
        Active trace owning the live index.
    event
        Operation event to project.
    name
        Op-style field name requested by a capture-time consumer.

    Returns
    -------
    Any
        Event-backed field value.
    """

    output = event.output
    function = event.function
    semantics = event.backend_semantics
    templates = event.templates
    simple: dict[str, Any] = {
        "_label_raw": event.label_raw,
        "_layer_label_raw": event.layer_label_raw,
        "raw_index": event.raw_index,
        "step_index": event.step_index,
        "source_trace": event.source_trace or trace,
        "_tracing_finished": event.tracing_finished,
        "_construction_done": event.construction_done,
        "type": event.layer_type,
        "type_index": event.type_index,
        "pass_index": event.pass_index,
        "num_passes": 1,
        "lookup_keys": [],
        "out": output.tensor.payload,
        "transformed_out": None
        if output.transformed_tensor is None
        else output.transformed_tensor.payload,
        "has_saved_activation": output.has_saved_activation,
        "activation_transform": output.activation_transform,
        "annotations": _event_annotations(event, output.tensor.payload),
        "output_device": output.output_device,
        "detach_saved_activations": output.detach_saved_activations,
        "has_saved_args": False if templates is None else templates.has_saved_args,
        "saved_args": None if templates is None else templates.saved_args,
        "saved_kwargs": None if templates is None else templates.saved_kwargs,
        "args_template": None if templates is None else templates.args_template,
        "kwargs_template": None if templates is None else templates.kwargs_template,
        "shape": output.tensor.shape,
        "transformed_out_shape": None
        if output.transformed_tensor is None
        else output.transformed_tensor.shape,
        "dtype": output.tensor.dtype,
        "transformed_out_dtype": None
        if output.transformed_tensor is None
        else output.transformed_tensor.dtype,
        "activation_memory": output.tensor.memory,
        "transformed_activation_memory": None
        if output.transformed_tensor is None
        else output.transformed_tensor.memory,
        "visualizer_path": output.visualizer_path,
        "bytes_delta_at_call": semantics.bytes_delta_at_call,
        "bytes_peak_at_call": semantics.bytes_peak_at_call,
        "autograd_memory": semantics.autograd_memory,
        "num_autograd_tensors": semantics.num_autograd_tensors,
        "has_out_variations": bool(output.child_versions),
        "out_versions_by_child": dict(output.child_versions),
        "func": function.func,
        "func_call_id": function.func_call_id,
        "func_name": function.func_name,
        "func_qualname": function.func_qualname,
        "code_context": list(function.code_context),
        "func_duration": function.func_duration or 0,
        "flops_forward": function.flops_forward or 0,
        "flops_backward": function.flops_backward or 0,
        "func_rng_states": function.func_rng_states,
        "func_autocast_state": function.func_autocast_state,
        "arg_names": tuple(function.arg_names),
        "num_args_total": function.num_args_total,
        "num_pos_args": function.num_pos_args,
        "num_kwargs": function.num_kwargs,
        "non_tensor_pos_args": list(function.non_tensor_pos_args),
        "non_tensor_kwargs": dict(function.non_tensor_kwargs),
        "func_non_tensor_args": list(function.func_non_tensor_args),
        "is_inplace": function.is_inplace,
        "grad_fn_class_name": semantics.grad_fn_class_name,
        "grad_fn_class_qualname": event.grad_fn_class_qualname,
        "grad_fn_object_id": None if event.grad_fn_handle is None else id(event.grad_fn_handle),
        "grad_fn_handle": event.grad_fn_handle,
        "grad_fn": None,
        "in_multi_output": output.in_multi_output,
        "multi_output_index": output.multi_output_index,
        "multi_output_name": None,
        "container_path": output.container_path,
        "container_spec": output.container_spec,
        "parent_params": list(event.parent_params),
        "_param_barcodes": [param.barcode for param in event.params],
        "parent_param_ops": {param.barcode: event.pass_index for param in event.params},
        "param_shapes": [param.shape for param in event.params],
        "num_params": sum(
            0 if param.shape is None else prod(param.shape) for param in event.params
        ),
        "equivalence_class": event.equivalence_class,
        "equivalent_ops": {event.label_raw},
        "recurrent_ops": [],
        "parents": [edge.parent_label_raw for edge in event.parents],
        "parent_arg_positions": event.parent_arg_positions,
        "_edge_uses": list(event._edge_uses),
        "root_ancestors": set(event.root_ancestors),
        "children": list(trace.capture_events.live_index.children(event.label_raw)),
        "has_children": bool(trace.capture_events.live_index.children(event.label_raw)),
        "is_input": event.kind == "source" and event.layer_type == "input",
        "has_input_ancestor": bool(event.input_ancestors),
        "input_ancestors": set(event.input_ancestors),
        "is_output": False,
        "is_output_parent": event.is_output_parent,
        "is_final_output": False,
        "has_output_descendant": False,
        "output_descendants": set(),
        "is_orphan": False,
        "io_role": None,
        "is_buffer": event.kind == "source" and event.layer_type == "buffer",
        "is_internal_source": event.layer_type != "input" and not event.parents,
        "has_internal_source_ancestor": event.has_internal_source_ancestor,
        "internal_source_parents": [],
        "internal_source_ancestors": set(event.internal_source_ancestors),
        "is_internal_sink": False,
        "is_scalar_bool": event.is_scalar_bool,
        "bool_value": event.bool_value,
        "module": event.modules[-1] if event.modules else None,
        "modules": list(event.modules),
        "module_call_stack": list(
            trace.capture_events.live_index.module_stack_membership(event.label_raw)
        ),
        "input_to_module_calls": [],
        "module_entry_arg_keys": defaultdict(list),
        "output_of_modules": [],
        "output_of_module_calls": [],
        "is_module_output": False,
        "is_atomic_module": False,
        "atomic_module_call": None,
        "interventions": [
            result.fire_record for result in event.fire_results if result.fire_record is not None
        ],
        "intervention_replaced": event.intervention_replaced,
        "func_config": dict(function.func_config),
    }
    if name in simple:
        return simple[name]
    if name in _OPLOG_FIELDS_KNOWN_LATE:
        raise LiveOpViewFieldNotYetWritten(
            f"LiveOpView.{name!r} is populated by postprocess Step 0; "
            "it is not available inside a forward-time callback."
        )
    raise AttributeError(f"LiveOpView has no attribute {name!r}.")


def _event_annotations(event: OpEvent, payload: Any) -> dict[str, Any]:
    """Return Op annotations projected from an operation event."""

    raw_annotations = event.transform_config.get("_tl_annotations")
    annotations = dict(raw_annotations) if isinstance(raw_annotations, Mapping) else {}
    annotations.update(_reference_annotations(event.policy.save_mode, payload))
    return annotations


def _reference_annotations(save_mode: str, payload: Any) -> dict[str, Any]:
    """Return saved-payload annotations needed by reference-mode tripwires."""

    if save_mode != "reference" or not isinstance(payload, torch.Tensor):
        return {}
    return {
        "save_mode": "reference",
        "saved_out_version": getattr(payload, "_version", None),
    }


class LiveOpView:
    """Read-only Op-shaped adapter over a capture-time operation event."""

    __slots__ = ("_trace_ref", "_record")

    def __init__(self, trace: "Trace", record: "LiveOpRecord | OpEvent") -> None:
        """Initialize the live view.

        Parameters
        ----------
        trace
            Active trace that owns the live record.
        record
            Operation event, or a legacy live operation record.
        """

        object.__setattr__(self, "_trace_ref", weakref.ref(trace))
        object.__setattr__(self, "_record", record)

    @property
    def _trace(self) -> "Trace":
        """Return the owning trace while it is still alive.

        Returns
        -------
        Trace
            Active trace.
        """

        trace = object.__getattribute__(self, "_trace_ref")()
        if trace is None:
            raise RuntimeError(
                "LiveOpView outlived its Trace (capture context ended); this view is invalid."
            )
        return trace

    def __getattr__(self, name: str) -> Any:
        """Return a live field value.

        Parameters
        ----------
        name
            Field name to read.

        Returns
        -------
        Any
            Current live field value.
        """

        record = object.__getattribute__(self, "_record")
        if isinstance(record, OpEvent):
            return _event_live_field(self._trace, record, name)
        if name in record.fields:
            return record.fields[name]
        if hasattr(record, name):
            return getattr(record, name)
        if name in _OPLOG_FIELDS_KNOWN_LATE:
            raise LiveOpViewFieldNotYetWritten(
                f"LiveOpView.{name!r} is populated by postprocess Step 0; "
                "it is not available inside a forward-time callback."
            )
        raise AttributeError(f"LiveOpView has no attribute {name!r}.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject direct mutation of live views.

        Parameters
        ----------
        name
            Field name.
        value
            Ignored attempted value.

        Returns
        -------
        None
            This method always raises.
        """

        raise AttributeError(f"LiveOpView is read-only mid-forward; cannot set {name}")


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


def sync_recording_grad_records_from_sidecar(state: RecordingState) -> None:
    """Rebuild fastlog gradient records from the unified backward sidecar.

    Parameters
    ----------
    state:
        Active recording state whose runtime trace owns backward events.

    Returns
    -------
    None
        ``state.recording.grad_records`` and its lookup indexes are replaced.
    """

    from ..fastlog.types import GradientRecord
    from ..ir.events import GradFnFired, OpGradObserved

    trace = state.runtime_trace
    if trace is None:
        return
    state.recording.grad_records.clear()
    state.recording.grad_by_pass.clear()
    state.recording.grad_by_label.clear()
    state.recording.grad_by_grad_fn_label.clear()
    backward_passes = getattr(trace, "backward_pass_logs", {})
    for event in getattr(trace, "backward_events", ()):
        if not isinstance(event, OpGradObserved):
            if isinstance(event, GradFnFired):
                _maybe_add_grad_fn_metadata_record(state, trace, event)
            continue
        if event.payload_ref is None and event.transformed_payload_ref is None:
            continue
        ctx = _grad_record_context_from_op_grad_event(trace, event, backward_passes)
        spec = CaptureSpec(
            save_out=event.payload_ref is not None or event.transformed_payload_ref is not None,
            save_metadata=True,
            keep_grad=False,
        )
        state.recording.add_grad_record(
            GradientRecord(
                ctx=ctx,
                spec=spec,
                ram_payload=event.payload_ref
                if isinstance(event.payload_ref, torch.Tensor)
                else None,
                transformed_ram_payload=(
                    event.transformed_payload_ref
                    if isinstance(event.transformed_payload_ref, torch.Tensor)
                    else None
                ),
                metadata={"timestamp": event.timestamp, "seq": event.seq},
                recorded_at=event.timestamp,
            )
        )


def _maybe_add_grad_fn_metadata_record(state: RecordingState, trace: "Trace", event: Any) -> None:
    """Append a metadata-only grad-fn record when the active policy selects it."""

    from ..fastlog.types import GradientRecord

    grad_fn = getattr(trace, "grad_fn_logs", {}).get(event.object_id)
    if grad_fn is None or getattr(grad_fn, "has_op", False):
        return
    pass_record = getattr(trace, "backward_pass_logs", {}).get(event.pass_index)
    ctx = GradRecordContext(
        label=grad_fn.label,
        grad_fn_class_name=grad_fn.class_name,
        type=grad_fn.type,
        backward_call_index=event.pass_index,
        grad_kind="grad_output",
        has_forward_op=False,
        has_op=False,
        pass_index=event.pass_index,
        order=getattr(pass_record, "order", None),
        event_index=event.seq,
    )
    policy = state.active_save_grads_record_policy
    decision = policy(ctx) if callable(policy) else policy
    if not isinstance(decision, CaptureSpec) or not decision.save_metadata:
        return
    state.recording.add_grad_record(
        GradientRecord(
            ctx=ctx,
            spec=CaptureSpec(save_out=False, save_metadata=True, keep_grad=False),
            metadata={"timestamp": event.timestamp, "seq": event.seq},
            recorded_at=event.timestamp,
        )
    )


def _grad_record_context_from_op_grad_event(
    trace: "Trace",
    event: Any,
    backward_passes: Mapping[int, Any],
) -> GradRecordContext:
    """Build a ``GradRecordContext`` from one op-gradient sidecar event."""

    pass_record = backward_passes.get(event.pass_index)
    if event.op_label not in getattr(trace, "layer_dict_all_keys", {}):
        fastlog_ctx = getattr(trace, "_fastlog_grad_contexts", {}).get(event.op_label)
        if fastlog_ctx is not None:
            return GradRecordContext(
                label=event.op_label,
                grad_fn_class_name="",
                type=fastlog_ctx.layer_type or "op",
                backward_call_index=event.pass_index,
                grad_kind="grad_output",
                grad_output_index=0,
                layer_label=event.op_label,
                op_label=event.op_label,
                module_stack=tuple(getattr(fastlog_ctx, "module_stack", ()) or ()),
                has_forward_op=True,
                has_op=True,
                pass_index=event.pass_index,
                order=getattr(pass_record, "order", None),
                event_index=event.seq,
                shape=event.shape,
                dtype=_torch_dtype_from_string(event.dtype),
                tensor_device=_torch_device_from_string(
                    getattr(fastlog_ctx, "tensor_device", None)
                ),
            )
        return GradRecordContext(
            label=event.op_label,
            grad_fn_class_name="",
            type="op",
            backward_call_index=event.pass_index,
            grad_kind="grad_output",
            has_forward_op=False,
            has_op=False,
            pass_index=event.pass_index,
            order=getattr(pass_record, "order", None),
            event_index=event.seq,
            shape=event.shape,
            dtype=_torch_dtype_from_string(event.dtype),
            tensor_device=None,
        )
    op = trace[event.op_label]
    return GradRecordContext(
        label=event.op_label,
        grad_fn_class_name=getattr(op, "grad_fn_class_name", None) or "",
        type=getattr(op, "layer_type", None) or "op",
        backward_call_index=event.pass_index,
        grad_kind="grad_output",
        grad_output_index=0,
        layer_label=event.op_label,
        op_label=event.op_label,
        module_stack=tuple(getattr(op, "module_stack", ()) or ()),
        has_forward_op=True,
        has_op=True,
        pass_index=event.pass_index,
        order=getattr(pass_record, "order", None),
        event_index=event.seq,
        shape=event.shape,
        dtype=_torch_dtype_from_string(event.dtype),
        tensor_device=_torch_device_from_string(getattr(op, "output_device", None)),
    )


def _public_record_context_label(ctx: RecordContext) -> str:
    """Return the compact public label for a predicate-mode op context."""

    if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
        return f"{ctx.layer_type}_{ctx.type_index}"
    return ctx.label


def _torch_dtype_from_string(dtype_name: str | None) -> torch.dtype | None:
    """Return a ``torch.dtype`` for canonical dtype strings when possible."""

    if dtype_name is None:
        return None
    if dtype_name.startswith("torch."):
        dtype_attr = dtype_name.removeprefix("torch.")
        dtype = getattr(torch, dtype_attr, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    return None


def _torch_device_from_string(device_name: Any) -> torch.device | None:
    """Return a ``torch.device`` from a string-like field when possible."""

    if device_name is None:
        return None
    try:
        return torch.device(str(device_name))
    except (TypeError, RuntimeError):
        return None


def recording_trace_from_events(events: Any) -> tuple[RecordContext, ...]:
    """Project capture events into fastlog ``RecordContext`` objects."""

    return tuple(_record_context_from_event(event) for event in events.op_events)
