"""Canonical RecordContext construction for fastlog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import torch

from .types import EventKind, ModuleStackFrame, RecordContext


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
                module_address=str(frame.get("module_address", "")),
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
    kind: EventKind,
    layer_pass_log_or_op_data: Any = None,
    module_stack: Iterable[ModuleStackFrame | Mapping[str, Any]] | None = None,
    history: Sequence[RecordContext] = (),
    op_counts: Mapping[str, int] | None = None,
    pass_index: int = 0,
    event_index: int = 0,
    op_index: int | None = None,
    time_since_pass_start: float = 0.0,
    include_source_events: bool = False,
    sample_id: str | int | None = None,
) -> RecordContext:
    """Build the single source-of-truth RecordContext schema.

    Parameters
    ----------
    kind:
        Chronological event kind.
    layer_pass_log_or_op_data:
        Mapping or object containing operation/source/module fields.
    module_stack:
        Active module stack at event time.
    history:
        Sliding-window event history before this event.
    op_counts:
        Per-layer-type counts available to predicates.
    pass_index:
        Forward pass index for this recording session.
    event_index:
        Chronological event index within the pass.
    op_index:
        Operation index for torch-function events.
    time_since_pass_start:
        Elapsed seconds since the pass began.
    include_source_events:
        Whether source events should appear in ``recent_ops``.
    sample_id:
        Optional user-supplied rollout/sample id.

    Returns
    -------
    RecordContext
        Frozen predicate context with the canonical schema.
    """

    data = layer_pass_log_or_op_data
    stack = _normalize_module_stack(module_stack)
    recent_events = tuple(history)
    layer_type = _read_field(data, "layer_type")
    if layer_type is None:
        layer_type = _read_field(data, "func_name")
    if isinstance(layer_type, str):
        layer_type = layer_type.lower().replace("_", "")
    layer_type_num = _read_field(data, "layer_type_num")
    if layer_type_num is None and layer_type is not None and op_counts is not None:
        layer_type_num = op_counts.get(cast(str, layer_type))
    tensor = _read_field(data, "tensor")
    tensor_shape = _read_field(data, "tensor_shape")
    tensor_dtype = _read_field(data, "tensor_dtype")
    tensor_device = _read_field(data, "tensor_device")
    tensor_requires_grad = _read_field(data, "tensor_requires_grad")
    if isinstance(tensor, torch.Tensor):
        tensor_shape = tuple(tensor.shape)
        tensor_dtype = tensor.dtype
        tensor_device = tensor.device
        tensor_requires_grad = tensor.requires_grad
    raw_label = _read_field(data, "tensor_label_raw", _read_field(data, "raw_label"))
    label = _read_field(data, "label", raw_label)
    if label is None:
        label = f"{kind}_{event_index}"
    return RecordContext(
        kind=kind,
        label=str(label),
        raw_label=raw_label,
        pass_index=pass_index,
        event_index=event_index,
        op_index=op_index,
        layer_type=layer_type,
        layer_type_num=layer_type_num,
        creation_order=_read_field(data, "creation_order"),
        func_name=_read_field(data, "func_name"),
        module_address=_read_field(data, "module_address"),
        module_type=_read_field(data, "module_type"),
        module_pass_index=_read_field(data, "module_pass_index"),
        module_stack=stack,
        recent_events=recent_events,
        recent_ops=_recent_ops_for_event(recent_events, include_source_events),
        parent_labels=tuple(_read_field(data, "parent_labels", ())),
        input_output_address=_read_field(data, "input_output_address"),
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        tensor_device=tensor_device,
        tensor_requires_grad=tensor_requires_grad,
        output_index=_read_field(data, "output_index"),
        is_bottom_level_func=_read_field(data, "is_bottom_level_func"),
        time_since_pass_start=time_since_pass_start,
        sample_id=sample_id,
    )
