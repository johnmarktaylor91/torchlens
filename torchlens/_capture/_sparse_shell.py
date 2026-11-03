"""Shell facets shared by sparse (predicate) journal records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._ir.events import FunctionCallRef

if TYPE_CHECKING:
    from ..recording.types import RecordContext

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
