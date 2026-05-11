"""Internal backend-neutral IR for TorchLens capture unification."""

from __future__ import annotations

from .backward import BackwardSidecar
from .buffer import CaptureEvents
from .events import (
    ArgTemplateRef,
    BlobRef,
    BufferEvent,
    ConditionalEvent,
    ContainerSpec,
    EdgeUseKind,
    FunctionCallRef,
    InterventionState,
    ModuleEvent,
    ModuleFrame,
    OpEvent,
    OpEventKind,
    OutputRef,
    ParentEdge,
    TraceBuildState,
)
from .intervention import FireResult, FunctionEventInput, InterventionTemplateRef
from .predicate import RecordContext
from .refs import DeferredRef, ParamRef, ReservedLabel, TensorRef
from .semantics import BackendSemantics, CapturePolicy

__all__ = [
    "ArgTemplateRef",
    "BackendSemantics",
    "BackwardSidecar",
    "BlobRef",
    "BufferEvent",
    "CaptureEvents",
    "CapturePolicy",
    "ConditionalEvent",
    "ContainerSpec",
    "DeferredRef",
    "EdgeUseKind",
    "FireResult",
    "FunctionCallRef",
    "FunctionEventInput",
    "InterventionState",
    "InterventionTemplateRef",
    "ModuleEvent",
    "ModuleFrame",
    "OpEvent",
    "OpEventKind",
    "OutputRef",
    "ParamRef",
    "ParentEdge",
    "RecordContext",
    "ReservedLabel",
    "TensorRef",
    "TraceBuildState",
]
