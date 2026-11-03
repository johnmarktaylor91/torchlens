"""Leaf builders of a sparse (predicate-mode) op record: refs, facets, dtype parsing.

These construct IR objects from already-resolved values and import nothing
from the capture layer, so ``projections`` can call them at module top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .._ir.events import ModuleFrame, OutputRef
from .._ir.op_record import GraphFacet, ParamsFacet
from .._ir.predicate import ModuleStackFrame, RecordContext
from .._ir.refs import ParamRef, TensorRef
from .._utils._torch_symbols import torch_dtype_named

__all__ = (
    "_SparseFreezeValues",
    "_module_frames_from_record_context",
    "_sparse_graph_facet",
    "_sparse_output_ref",
    "_sparse_params_facet",
    "_torch_device_from_string",
    "_torch_dtype_from_string",
)


@dataclass(slots=True)
class _SparseFreezeValues:
    """Shared value bundle both sparse freeze shapes construct from."""

    label_raw: str
    tensor_ref: TensorRef
    transformed_ref: TensorRef | None
    module_stack: tuple[ModuleFrame, ...]
    modules: tuple[tuple[str, int], ...]
    is_scalar_bool: bool | None
    bool_value: bool | None


def _sparse_output_ref(
    ctx: RecordContext,
    values: _SparseFreezeValues,
    *,
    has_saved_activation: bool,
    detach_saved_activations: bool,
    container_path: tuple[Any, ...],
) -> OutputRef:
    """Return the sparse commit's output ref (shared by both freeze shapes).

    The caller resolves both flags from the capture spec: the recorded detach
    flag is ``False`` whenever the payload was kept graph-connected, either by
    ``save_grad`` or by an attached session.
    """

    return OutputRef(
        tensor=values.tensor_ref,
        transformed_tensor=values.transformed_ref,
        has_saved_activation=has_saved_activation,
        output_device=str(ctx.tensor_device) if ctx.tensor_device is not None else None,
        activation_transform=None,
        detach_saved_activations=detach_saved_activations,
        visualizer_path=None,
        # Exhaustive parity (``_ops_activations``): a slot only inside a container output.
        multi_output_index=ctx.output_index if container_path else None,
        in_multi_output=bool(container_path),
        container_path=container_path,
        container_spec=None,
        child_versions=(),
    )


def _sparse_graph_facet(ctx: RecordContext, equivalence_class: str | None) -> Any:
    """The graph facet of a sparse record: output-parent flag plus its loop-grouping class."""

    return GraphFacet(
        parent_arg_positions={"args": {}, "kwargs": {}},
        is_output_parent=ctx.is_output_parent,
        equivalence_class=equivalence_class,
    )


def _sparse_params_facet(params: tuple[ParamRef, ...]) -> Any:
    """The params facet of a sparse record, absent when the op read no prepared parameter."""

    return ParamsFacet(params=tuple(params)) if params else None


def _torch_dtype_from_string(dtype_name: str | None) -> torch.dtype | None:
    """Return a ``torch.dtype`` for canonical dtype strings when possible."""

    if dtype_name is None:
        return None
    if dtype_name.startswith("torch."):
        return torch_dtype_named(dtype_name.removeprefix("torch."))
    return None


def _torch_device_from_string(device_name: Any) -> torch.device | None:
    """Return a ``torch.device`` from a string-like field when possible."""

    if device_name is None:
        return None
    try:
        return torch.device(str(device_name))
    except (TypeError, RuntimeError):
        return None


def _module_frames_from_record_context(ctx: RecordContext) -> tuple[ModuleFrame, ...]:
    """Convert fastlog module-stack frames to IR module frames."""

    return tuple(
        ModuleFrame(
            address=_module_frame_address(frame),
            address_normalized=None,
            module_type=frame.module_type,
            call_index=frame.pass_number,
            fx_qualpath=None,
            entry_argnames=(),
        )
        for frame in ctx.module_stack
        # The root frame (address "") is the Recording's own bookkeeping; exhaustive
        # capture never lists the root as an op's containing module, so neither does
        # the cooked Trace (module, modules, the loop-grouping suffix).
        if frame.address
    )


def _module_frame_address(frame: ModuleStackFrame) -> str:
    """Return the Trace-facing address for a predicate module-stack frame."""

    return frame.address or "self"
