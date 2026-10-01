"""Public TorchLens type aliases and rarely used data classes."""

from collections.abc import Callable

#: Role declaration (architecture memo build item 2, seeding the layer map
#: the C01 lint consumes): this module is a FACADE-role cross-layer
#: re-export aggregator -- thin and logic-free; its upward import edges are
#: licensed by the facade role (Rule F), not by a layer claim of its own.
__tl_role__ = "FACADE"

import torch

from .capture.outcome import CaptureOutcome, CapturePhase, CaptureStatus, FailureOrigin
from .data_classes.aten_op import AtenOp, OpRef
from .data_classes.backward_pass import BackwardPass
from .data_classes.buffer import Buffer
from .data_classes.func_call_location import FuncCallLocation
from .data_classes.grad_fn import GradFn
from .data_classes.grad_fn_call import GradFnCall
from .data_classes.module import Module, ModuleCall
from .data_classes.op import TensorLog
from .data_classes.param import Param
from .data_classes.prehook import ModuleInputSnapshot, PreHookEffect, TensorInputObservation
from .intervention import SaveLevel, SiteTable, SpecCompat, TargetManifestDiff, TensorSliceSpec
from .quantities import Bytes, Duration, Flops, Macs, Quantity

ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "ActivationPostfunc",
    "AtenOp",
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
