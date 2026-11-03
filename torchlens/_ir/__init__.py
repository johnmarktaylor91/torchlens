"""Internal backend-neutral IR for TorchLens capture unification."""

from __future__ import annotations

from .capture_events import (
    register_live_event as register_live_event,
)
from .container import ContainerSpec
from .container_registry import (
    ContainerLeafOccurrence,
    ContainerRecord,
    ContainerRegistry,
    ContainerSnapshot as ContainerSnapshot,
    FuncSite,
    IdentityEntry,
    ModelSite,
    ModuleSite as ModuleSite,
    Phase,
    Role,
    WalkResult as WalkResult,
    walk_container,
)
from .events import (
    ArgTemplateRef,
    BackwardCoverageGap,
    BackwardPassEnd,
    BackwardPassStart,
    BlobRef,
    BufferWriteEvent,
    EdgeUseKind as EdgeUseKind,
    FunctionCallRef,
    GradFnDiscovered,
    GradFnFired,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
    OpEvent,
    OpEventKind as OpEventKind,
    OpGradObserved,
    OutputRef,
    OutputVersionEvent as OutputVersionEvent,
    ParamGradObserved,
    ParentEdge,
    edge_use_kind,
    is_control_edge_use,
    is_value_edge_use,
)
from .live_index import LiveIndex
from .predicate import (
    _DEFERRED_VALUE,
    coerce_deferred_value as coerce_deferred_value,
    is_deferred_value as is_deferred_value,
)
from .refs import DeferredRef, DeviceRef, DtypeRef, ParamRef, TensorRef
from .workspaces import (
    LEGACY_TRACE_BUILD_STATE_KEYS,
    RawGraphWorkspace,
    WrapperRuntimeWorkspace,
)

__all__ = [
    "ArgTemplateRef",
    "BackwardPassEnd",
    "BackwardPassStart",
    "BlobRef",
    "BufferWriteEvent",
    "ContainerSpec",
    "ContainerLeafOccurrence",
    "ContainerRecord",
    "ContainerRegistry",
    "DeferredRef",
    "DeviceRef",
    "DtypeRef",
    "edge_use_kind",
    "FunctionCallRef",
    "FuncSite",
    "IdentityEntry",
    "is_control_edge_use",
    "is_value_edge_use",
    "GradFnDiscovered",
    "GradFnFired",
    "LiveIndex",
    "ModuleEnterEvent",
    "ModuleExitEvent",
    "ModuleFrame",
    "ModulePrepEvent",
    "ModelSite",
    "OpEvent",
    "OpGradObserved",
    "OutputRef",
    "ParamGradObserved",
    "BackwardCoverageGap",
    "ParamRef",
    "ParentEdge",
    "Phase",
    "Role",
    "walk_container",
    "TensorRef",
    "LEGACY_TRACE_BUILD_STATE_KEYS",
    "RawGraphWorkspace",
    "WrapperRuntimeWorkspace",
    "_DEFERRED_VALUE",
]
