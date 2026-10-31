"""Public TorchLens type aliases and rarely used data classes."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

#: Role declaration (architecture memo build item 2, seeding the layer map
#: the C01 lint consumes): this module is a FACADE-role cross-layer
#: re-export aggregator -- thin and logic-free; its upward import edges are
#: licensed by the facade role (Rule F), not by a layer claim of its own.
__tl_role__ = "FACADE"

import torch

from .capture.outcome import CaptureOutcome, CapturePhase, CaptureStatus, FailureOrigin
from .data_classes.aten_op import AtenOp, OpRef
from .data_classes.attribute_write import AttributeWrite
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
    "NarrationEvent",
    "RunnableDiagnostic",
    "RunnableCompatibility",
    "RunReport",
    "ReadinessReport",
    "ContractCheck",
    "MergedFinding",
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


if TYPE_CHECKING:
    from torchlens.merged._errors import MergedFinding
    from torchlens.runnable import (
        ContractCheck,
        ReadinessReport,
        RunnableCompatibility,
        RunnableDiagnostic,
        RunReport,
    )
    from torchlens.snoop._event import NarrationEvent

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "NarrationEvent": ("torchlens.snoop._event", "NarrationEvent"),
    "RunnableDiagnostic": ("torchlens.runnable", "RunnableDiagnostic"),
    "RunnableCompatibility": ("torchlens.runnable", "RunnableCompatibility"),
    "RunReport": ("torchlens.runnable", "RunReport"),
    "ReadinessReport": ("torchlens.runnable", "ReadinessReport"),
    "ContractCheck": ("torchlens.runnable", "ContractCheck"),
    "MergedFinding": ("torchlens.merged._errors", "MergedFinding"),
}


def __getattr__(name: str) -> Any:
    """Resolve exports whose defining modules would import-cycle if loaded eagerly.

    Parameters
    ----------
    name:
        Attribute requested from this namespace.

    Returns
    -------
    Any
        The exported object.

    Raises
    ------
    AttributeError
        If ``name`` is not an export of this namespace.
    """

    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
