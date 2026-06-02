"""Public TorchLens type aliases and rarely used data classes."""

from collections.abc import Callable

import torch

from .data_classes.buffer import Buffer
from .data_classes.func_call_location import FuncCallLocation
from .data_classes.grad_fn import GradFn
from .data_classes.grad_fn_call import GradFnCall
from .data_classes.op import TensorLog
from .data_classes.module import Module, ModuleCall
from .data_classes.param import Param
from .intervention import SaveLevel, SiteTable, SpecCompat, TargetManifestDiff, TensorSliceSpec
from .quantities import Bytes, Duration, Flops, Macs, Quantity

ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "ActivationPostfunc",
    "Buffer",
    "Bytes",
    "Duration",
    "Flops",
    "FuncCallLocation",
    "GradientPostfunc",
    "GradFn",
    "GradFnCall",
    "Macs",
    "Module",
    "ModuleCall",
    "Param",
    "Quantity",
    "SaveLevel",
    "SiteTable",
    "SpecCompat",
    "TargetManifestDiff",
    "TensorLog",
    "TensorSliceSpec",
]
