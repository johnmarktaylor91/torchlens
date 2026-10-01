"""Core data structures for representing a logged forward pass."""

from .backward_pass import BackwardPass, BackwardPassAccessor
from .buffer import Buffer, BufferAccessor
from .func_call_location import FuncCallLocation
from .grad_fn import GradFn, GradFnAccessor
from .grad_fn_call import GradFnCall
from .internal_types import (
    FuncExecutionContext as FuncExecutionContext,
    VisualizationOverrides as VisualizationOverrides,
)
from .module import Module, ModuleAccessor, ModuleCall
from .param import Param, ParamAccessor
from .prehook import ModuleInputSnapshot, PreHookEffect, TensorInputObservation

__all__ = [
    "BackwardPass",
    "BackwardPassAccessor",
    "Buffer",
    "BufferAccessor",
    "FuncCallLocation",
    "GradFn",
    "GradFnAccessor",
    "GradFnCall",
    "Module",
    "ModuleAccessor",
    "ModuleCall",
    "ModuleInputSnapshot",
    "Param",
    "ParamAccessor",
    "PreHookEffect",
    "TensorInputObservation",
]

# Trace, Layer, Op, and TensorLog are intentionally NOT
# re-exported here to avoid circular imports. Import them directly:
#   from .data_classes.trace import Trace
#   from .data_classes.layer import Layer
#   from .data_classes.op import Op, TensorLog
