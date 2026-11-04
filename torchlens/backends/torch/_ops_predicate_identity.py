"""Grouping identity of predicate-mode op outputs: parameter refs, call-site chain, equivalence class.

The builders exhaustive capture uses (``_ops_activations``, ``_ops_shared_fields``), asked by
``PredicateOpIdentity`` in ``_ops_predicate_events`` so ``Recording.to_trace()`` groups as ``tl.trace``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..._ir.predicate import RecordContext
from ..._ir.refs import ParamRef
from ..._utils.introspection import _get_code_context
from .tensor_tracking import (
    _append_module_suffix_to_equivalence_class,
    _get_equivalence_class,
    _make_raw_param_group_barcode,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = (
    "_predicate_call_site_chain",
    "_predicate_equivalence_class",
    "_prepared_param_refs",
)


def _predicate_call_site_chain(trace: "Trace") -> tuple[Any, ...]:
    """Return the call-site chain of the torch call being recorded.

    The same frame walk exhaustive capture runs for its code context
    (``_ops_shared_fields``), asked only for the chain: no source lines, no
    column offsets, and the per-capture anchor in ``trace._code_context_cache``
    stops the walk at the capture call. Empty when no Python frame lies
    inside the innermost module call. The key and class string it feeds are
    the ones exhaustive capture mints (``_ops_activations``), so the cooked
    Trace groups by call site (``callsite_v1``) with the same result.
    """

    context_cache = trace._code_context_cache
    call_site_out: list[Any] = []
    _get_code_context(
        0,
        source_loading_enabled=False,
        disable_col_offset=True,
        context_cache=context_cache,
        call_site_out=call_site_out,
    )
    return call_site_out[0] if call_site_out else ()


def _prepared_param_refs(
    state: Any,
    arg_parameters: list[torch.nn.Parameter],
) -> tuple[ParamRef, ...]:
    """Resolve the Parameters an op read to their prep-time identities.

    Parameters
    ----------
    state
        Active recording state holding ``prepared_param_refs``.
    arg_parameters
        Parameters found among the call's arguments.

    Returns
    -------
    tuple[ParamRef, ...]
        One ref per prepared parameter occurrence, in argument order. A
        Parameter the prep never saw (created inside the forward) has no
        identity and is skipped, as exhaustive capture skips unstamped ones.
    """

    if not arg_parameters:
        return ()
    registry = state.prepared_param_refs
    if not registry:
        return ()
    return tuple(ref for param in arg_parameters if (ref := registry.get(id(param))) is not None)


def _predicate_equivalence_class(
    state: Any,
    ctx: RecordContext,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    param_refs: tuple[ParamRef, ...],
    *,
    func_name: str,
    layer_type: str,
    output_index: int,
    in_multi_output: bool,
    is_inplace_dunder: Callable[[str], bool],
) -> str:
    """Build the loop-grouping fingerprint of one predicate-mode op output.

    The same two fingerprints exhaustive capture mints (``_ops_activations``):
    parameterized ops key on ``layer_type`` plus the sorted parameter barcodes,
    parameter-free ops on ``layer_type`` plus a hash of the non-tensor
    arguments, both suffixed with the output slot of a multi-output call and
    the module stack.
    """

    _frames, modules = state.module_fields_for(ctx)
    if param_refs:
        barcodes = [ref.barcode for ref in param_refs]
        base = _make_raw_param_group_barcode(
            barcodes, layer_type, output_index=output_index if in_multi_output else None
        )
        return _append_module_suffix_to_equivalence_class(base, modules)
    is_inplace_output = is_inplace_dunder(func_name) or (
        func_name.endswith("_") and not func_name.startswith("__")
    )
    equivalence_layer_type = f"{layer_type}_inplace" if is_inplace_output else layer_type
    base = _get_equivalence_class(
        args,
        kwargs,
        output_index,
        equivalence_layer_type,
        {"in_multi_output": in_multi_output, "module": modules[-1] if modules else None},
    )
    return _append_module_suffix_to_equivalence_class(base, modules)
