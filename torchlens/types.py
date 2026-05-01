"""Public TorchLens type aliases and rarely used data classes."""

from collections.abc import Callable

import torch

from .data_classes.buffer_log import BufferLog
from .data_classes.func_call_location import FuncCallLocation
from .data_classes.grad_fn_log import GradFnLog
from .data_classes.grad_fn_pass_log import GradFnPassLog
from .data_classes.layer_pass_log import TensorLog
from .data_classes.module_log import ModuleLog, ModulePassLog
from .data_classes.param_log import ParamLog
from .intervention import SaveLevel, SiteTable, SpecCompat, TargetManifestDiff, TensorSliceSpec

ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "ActivationPostfunc",
    "BufferLog",
    "FuncCallLocation",
    "GradientPostfunc",
    "GradFnLog",
    "GradFnPassLog",
    "ModuleLog",
    "ModulePassLog",
    "ParamLog",
    "SaveLevel",
    "SiteTable",
    "SpecCompat",
    "TargetManifestDiff",
    "TensorLog",
    "TensorSliceSpec",
]
