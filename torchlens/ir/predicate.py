"""Unified predicate context for capture filtering and selector adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .refs import DeviceRef, DtypeRef

EventKind = Literal["op", "module_enter", "module_exit", "input", "buffer"]


@dataclass(frozen=True, slots=True)
class ModuleStackFrame:
    """One frame in the active module stack."""

    address: str
    module_type: str
    module_id: int
    pass_index: int


@dataclass(frozen=True, slots=True)
class RecordContext:
    """Predicate input schema for one chronological capture event."""

    kind: EventKind | str
    label: str
    raw_label: str | None
    pass_index: int
    event_index: int
    step_index: int | None
    layer_type: str | None
    type_index: int | None
    raw_index: int | None
    func_name: str | None
    address: str | None
    module_type: str | None
    module_pass_index: int | None
    module_stack: tuple[Any, ...]
    recent_events: tuple["RecordContext", ...]
    recent_ops: tuple["RecordContext", ...]
    parent_labels: tuple[str, ...]
    input_output_address: str | None
    shape: tuple[int, ...] | None
    dtype: DtypeRef | None
    tensor_device: DeviceRef | None
    tensor_requires_grad: bool | None
    output_index: int | None
    is_bottom_level_func: bool | None
    time_since_pass_start: float
    sample_id: str | int | None = None
    label_raw: str = ""
    label_prefix: str = ""
    func_call_id: int | None = None
    parent_labels_raw: tuple[str, ...] = ()
    is_output_parent: bool = False
    backend_requires_isolation: bool = False
    is_scalar_bool: bool | None = None
    bool_value: bool | None = None

    def __getattr__(self, name: str) -> Any:
        """Raise a schema-specific error for unknown predicate fields."""

        from ..fastlog.exceptions import RecordContextFieldError

        raise RecordContextFieldError(name)
