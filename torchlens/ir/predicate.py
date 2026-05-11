"""Unified predicate context for capture filtering and selector adapters."""

from __future__ import annotations

from dataclasses import dataclass

from .events import ModuleFrame


@dataclass(frozen=True, slots=True)
class RecordContext:
    """Predicate input schema for one chronological capture event."""

    kind: str
    label: str
    raw_label: str | None
    pass_index: int
    event_index: int
    compute_index: int | None
    layer_type: str | None
    type_index: int | None
    capture_index: int | None
    func_name: str | None
    address: str | None
    module_type: str | None
    module_pass_index: int | None
    module_stack: tuple[ModuleFrame, ...]
    recent_events: tuple["RecordContext", ...]
    recent_ops: tuple["RecordContext", ...]
    parent_labels: tuple[str, ...]
    input_output_address: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    tensor_device: str | None
    tensor_requires_grad: bool | None
    output_index: int | None
    is_bottom_level_func: bool | None
    time_since_pass_start: float
    sample_id: str | int | None
    label_raw: str
    label_prefix: str
    func_call_id: int | None
    parent_labels_raw: tuple[str, ...]
    is_output_parent: bool
    backend_requires_isolation: bool
    is_scalar_bool: bool | None
    bool_value: bool | None
