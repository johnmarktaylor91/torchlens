"""Leaf builders of a sparse (predicate-mode) op record: refs, facets, dtype parsing.

These construct IR objects from already-resolved values and import nothing
from the capture layer, so ``projections`` can call them at module top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._ir.events import FunctionCallRef, ModuleFrame, OutputRef
from .._ir.op_record import GraphFacet, ParamsFacet
from .._ir.predicate import ModuleStackFrame, RecordContext
from .._ir.refs import ParamRef, TensorRef

__all__ = (
    "_SparseFreezeValues",
    "_module_frames_from_record_context",
    "_sparse_function_ref",
    "_sparse_graph_facet",
    "_sparse_output_ref",
    "_sparse_params_facet",
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


#: Name-only shell function facets, one per function name. ``FunctionCallRef``
#: is frozen and tuple-valued, so every shell event of one function shares it.
_SHELL_FUNCTION_REFS: dict[str | None, FunctionCallRef] = {}


def _sparse_function_ref(ctx: RecordContext, function: FunctionCallRef | None) -> FunctionCallRef:
    """Return the sparse commit's function facet (name-only SHELL fallback)."""

    if function is not None:
        return function
    shell = _SHELL_FUNCTION_REFS.get(ctx.func_name)
    if shell is not None:
        return shell
    shell = _SHELL_FUNCTION_REFS[ctx.func_name] = FunctionCallRef(
        func=None,
        func_name=ctx.func_name,
        func_qualname=None,
        func_call_id=None,
        code_context=(),
        func_duration=None,
        flops_forward=None,
        flops_backward=None,
        func_rng_states=None,
        func_autocast_state=None,
        arg_names=(),
        num_args_total=0,
        num_pos_args=0,
        num_kwargs=0,
        non_tensor_pos_args=(),
        non_tensor_kwargs=(),
        func_non_tensor_args=(),
        is_inplace=False,
        func_config=(),
    )
    return shell


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
