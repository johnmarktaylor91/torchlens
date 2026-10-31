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

from ._data_classes._accessor_base import Accessor
from ._data_classes.aten_op import AtenOp, OpRef
from ._data_classes.attribute_write import AttributeWrite
from ._data_classes.backward_pass import BackwardPass
from ._data_classes.buffer import Buffer
from ._data_classes.func_call_location import FuncCallLocation
from ._data_classes.grad_fn import GradFn
from ._data_classes.grad_fn_call import GradFnCall
from ._data_classes.module import Module, ModuleCall
from ._data_classes.param import Param
from ._data_classes.prehook import ModuleInputSnapshot, PreHookEffect, TensorInputObservation
from ._quantities import Bytes, Duration, Flops, Macs, Quantity
from ._runnable import NumericAttestationStatus, PathFaithfulness, ReadinessStatus, RunProvider
from ._selection import Site
from ._selection_subspace import BasisProvenance
from .capture.outcome import CaptureOutcome, CapturePhase, CaptureStatus, FailureOrigin
from .captured_run import TraceLike
from .intervention import SpecCompat, TargetManifestDiff
from .intervention.types import Relationship

if TYPE_CHECKING:
    from ._data_classes.container import Container
    from ._ir.container import ContainerSpec, register_container
    from ._ir.container_registry import ContainerLeafOccurrence
    from ._merged._enums import MergeAlignment, MergeValueStatus
    from ._merged._presenter import MergedTrace
    from ._partial import PartialTrace
    from ._quickstart._grammar import InputSpec
    from ._quickstart._provenance import InputProvenance
    from ._trace_slice import TraceSlice

_ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
_GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = [
    "Accessor",
    "TraceLike",
    "NarrationEvent",
    "RunnableDiagnostic",
    "RunnableCompatibility",
    "RunReport",
    "ReadinessReport",
    "ContractCheck",
    "MergedFinding",
    "AtenOp",
    "AttributeWrite",
    "BackwardPass",
    "BasisProvenance",
    "Buffer",
    "Bytes",
    "CaptureOutcome",
    "CapturePhase",
    "CaptureStatus",
    "Container",
    "ContainerLeafOccurrence",
    "ContainerSpec",
    "Duration",
    "FailureOrigin",
    "Flops",
    "FuncCallLocation",
    "GradFn",
    "GradFnCall",
    "InputProvenance",
    "InputSpec",
    "Macs",
    "MergeAlignment",
    "MergeValueStatus",
    "MergedTrace",
    "Module",
    "ModuleCall",
    "ModuleInputSnapshot",
    "NumericAttestationStatus",
    "OpRef",
    "Param",
    "PathFaithfulness",
    "PartialTrace",
    "PreHookEffect",
    "Quantity",
    "ReadinessStatus",
    "Relationship",
    "RunProvider",
    "Site",
    "SpecCompat",
    "TargetManifestDiff",
    "TensorInputObservation",
    "TraceSlice",
    "register_container",
]


if TYPE_CHECKING:
    from torchlens._merged._errors import MergedFinding
    from torchlens._runnable import (
        ContractCheck,
        ReadinessReport,
        RunnableCompatibility,
        RunnableDiagnostic,
        RunReport,
    )
    from torchlens._snoop._event import NarrationEvent

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "NarrationEvent": ("torchlens._snoop._event", "NarrationEvent"),
    "RunnableDiagnostic": ("torchlens._runnable", "RunnableDiagnostic"),
    "RunnableCompatibility": ("torchlens._runnable", "RunnableCompatibility"),
    "RunReport": ("torchlens._runnable", "RunReport"),
    "ReadinessReport": ("torchlens._runnable", "ReadinessReport"),
    "ContractCheck": ("torchlens._runnable", "ContractCheck"),
    "MergedFinding": ("torchlens._merged._errors", "MergedFinding"),
    "Container": ("torchlens._data_classes.container", "Container"),
    "ContainerLeafOccurrence": ("torchlens._ir.container_registry", "ContainerLeafOccurrence"),
    "ContainerSpec": ("torchlens._ir.container", "ContainerSpec"),
    "InputProvenance": ("torchlens._quickstart._provenance", "InputProvenance"),
    "InputSpec": ("torchlens._quickstart._grammar", "InputSpec"),
    "MergeAlignment": ("torchlens._merged._enums", "MergeAlignment"),
    "MergeValueStatus": ("torchlens._merged._enums", "MergeValueStatus"),
    "MergedTrace": ("torchlens._merged._presenter", "MergedTrace"),
    "PartialTrace": ("torchlens._partial", "PartialTrace"),
    "TraceSlice": ("torchlens._trace_slice", "TraceSlice"),
    "register_container": ("torchlens._ir.container", "register_container"),
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
