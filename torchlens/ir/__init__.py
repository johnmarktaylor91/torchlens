"""Internal backend-neutral IR for TorchLens capture unification."""

from __future__ import annotations

from .backward import BackwardSidecar
from .buffer import (
    CaptureEvents,
    LiveOpRecord,
    live_record_for_label,
    register_live_event,
    replace_op_event,
)
from .events import (
    ArgTemplateRef,
    BlobRef,
    BufferEvent,
    BufferWriteEvent,
    ConditionalEvent,
    ContainerSpec,
    EdgeUseKind,
    FunctionCallRef,
    InterventionState,
    ModuleEnterEvent,
    ModuleEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
    OpEvent,
    OpEventKind,
    OutputRef,
    OutputVersionEvent,
    ParentEdge,
    TraceBuildState,
)
from .intervention import FireResult, FunctionEventInput, InterventionTemplateRef
from .live_index import LiveIndex, LiveIndexWindowError
from .predicate import (
    MLXValueUnavailableError,
    RecordContext,
    _DEFERRED_VALUE,
    coerce_deferred_value,
    is_deferred_value,
)
from .refs import DeferredRef, DeviceRef, DtypeRef, ParamRef, ReservedLabel, TensorRef
from .semantics import BackendSemantics, CapturePolicy

__all__ = [
    "ArgTemplateRef",
    "BackendSemantics",
    "BackwardSidecar",
    "BlobRef",
    "BufferEvent",
    "BufferWriteEvent",
    "CaptureEvents",
    "CapturePolicy",
    "ConditionalEvent",
    "ContainerSpec",
    "DeferredRef",
    "DeviceRef",
    "DtypeRef",
    "EdgeUseKind",
    "FireResult",
    "FunctionCallRef",
    "FunctionEventInput",
    "InterventionState",
    "InterventionTemplateRef",
    "LiveOpRecord",
    "LiveIndex",
    "LiveIndexWindowError",
    "MLXValueUnavailableError",
    "ModuleEnterEvent",
    "ModuleEvent",
    "ModuleExitEvent",
    "ModuleFrame",
    "ModulePrepEvent",
    "OpEvent",
    "OpEventKind",
    "OutputRef",
    "OutputVersionEvent",
    "ParamRef",
    "ParentEdge",
    "RecordContext",
    "ReservedLabel",
    "TensorRef",
    "TraceBuildState",
    "_DEFERRED_VALUE",
    "coerce_deferred_value",
    "is_deferred_value",
    "live_record_for_label",
    "register_live_event",
    "replace_op_event",
]
