"""Public TorchLens type aliases and rarely used data classes."""

from collections.abc import Callable
from typing import TYPE_CHECKING

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
from ._data_classes.param import Param
from ._data_classes.prehook import ModuleInputSnapshot, PreHookEffect, TensorInputObservation
from ._quantities import Bytes, Duration, Flops, Macs, Quantity
from ._runnable import NumericAttestationStatus, PathFaithfulness, ReadinessStatus, RunProvider
from .capture.outcome import CaptureOutcome, CapturePhase, CaptureStatus, FailureOrigin
from .captured_run import TraceLike
from .intervention import SiteTable, SpecCompat, TargetManifestDiff

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
    "TraceLike",
    "AtenOp",
    "AttributeWrite",
    "BackwardPass",
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
    "RunProvider",
    "SiteTable",
    "SpecCompat",
    "TargetManifestDiff",
    "TensorInputObservation",
    "TraceSlice",
    "register_container",
]

#: Public records whose implementation lives in a private package; loaded
#: on first access so importing tl.types stays cheap.
_LAZY_TYPES: dict[str, str] = {
    "Container": "torchlens._data_classes.container",
    "ContainerLeafOccurrence": "torchlens._ir.container_registry",
    "ContainerSpec": "torchlens._ir.container",
    "InputProvenance": "torchlens._quickstart._provenance",
    "InputSpec": "torchlens._quickstart._grammar",
    "MergeAlignment": "torchlens._merged._enums",
    "MergeValueStatus": "torchlens._merged._enums",
    "MergedTrace": "torchlens._merged._presenter",
    "PartialTrace": "torchlens._partial",
    "TraceSlice": "torchlens._trace_slice",
    "register_container": "torchlens._ir.container",
}


def __getattr__(name: str) -> object:
    """Resolve a lazily exported record type."""

    module = _LAZY_TYPES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
