"""Public TorchLens type aliases and rarely used data classes."""

from collections.abc import Callable

#: Role declaration (architecture memo build item 2, seeding the layer map
#: the C01 lint consumes): this module is a FACADE-role cross-layer
#: re-export aggregator -- thin and logic-free; its upward import edges are
#: licensed by the facade role (Rule F), not by a layer claim of its own.
__tl_role__ = "FACADE"

import torch

from ._data_classes.aten_op import AtenOp, OpRef
from ._data_classes.attribute_write import AttributeWrite
from ._data_classes.backward_pass import BackwardPass
from ._data_classes.buffer import Buffer
from ._data_classes.func_call_location import FuncCallLocation
from ._data_classes.grad_fn import GradFn
from ._data_classes.grad_fn_call import GradFnCall
from ._data_classes.module import Module, ModuleCall
from ._data_classes.op import TensorLog
from ._data_classes.param import Param
from ._data_classes.prehook import ModuleInputSnapshot, PreHookEffect, TensorInputObservation
from ._quantities import Bytes, Duration, Flops, Macs, Quantity
from .capture.outcome import CaptureOutcome, CapturePhase, CaptureStatus, FailureOrigin
from .intervention import SaveLevel, SiteTable, SpecCompat, TargetManifestDiff, TensorSliceSpec

ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "ActivationPostfunc",
    "AtenOp",
    "AttributeWrite",
    "BackwardPass",
    "Buffer",
    "Bytes",
    "CaptureOutcome",
    "CapturePhase",
    "CaptureStatus",
    "Duration",
    "FailureOrigin",
    "Flops",
    "FuncCallLocation",
    "GradientPostfunc",
    "GradFn",
    "GradFnCall",
    "Macs",
    "Module",
    "ModuleCall",
    "ModuleInputSnapshot",
    "OpRef",
    "Param",
    "PreHookEffect",
    "Quantity",
    "SaveLevel",
    "SiteTable",
    "SpecCompat",
    "TargetManifestDiff",
    "TensorLog",
    "TensorInputObservation",
    "TensorSliceSpec",
]
