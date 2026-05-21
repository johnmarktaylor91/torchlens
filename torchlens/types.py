"""Public TorchLens type aliases and rarely used data classes."""

from collections.abc import Callable

import torch

from .data_classes.buffer_log import Buffer
from .data_classes.func_call_location import FuncCallLocation
from .data_classes.grad_fn_log import GradFn
from .data_classes.grad_fn_call_log import GradFnCall
from .data_classes.op_log import TensorLog
from .data_classes.module_log import Module, ModuleCall
from .data_classes.param_log import Param
from .intervention import SaveLevel, SiteTable, SpecCompat, TargetManifestDiff, TensorSliceSpec

ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "ActivationPostfunc",
    "Buffer",
    "FuncCallLocation",
    "GradientPostfunc",
    "GradFn",
    "GradFnCall",
    "Module",
    "ModuleCall",
    "Param",
    "SaveLevel",
    "SiteTable",
    "SpecCompat",
    "TargetManifestDiff",
    "TensorLog",
    "TensorSliceSpec",
]
