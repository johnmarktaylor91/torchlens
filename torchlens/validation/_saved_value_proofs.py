"""Posthoc value proofs for parents that provably cannot reach an op's output.

Each decision here answers one question after a perturbed replay left an op's
output unchanged: do the SAVED call arguments prove that the perturbed parent's
value cannot reach the output at this input? Every proof is keyed on one op
family, one argument position of the perturbed parent, and a value test on the
saved co-arguments, and returns non-exempt for anything else:

- division: the saved dividend is all zero, so the divisor (``args[1]``) only
  scales a zero;
- bitwise/logical and: the other saved operand is all zero/False;
- bool or: the other saved operand is all True (the out-of-place spellings are
  proved first by ``_operand_value_proofs``; this covers the in-place ones);
- pow: the saved exponent is all zero, so the base (``args[0]``) is raised to 0,
  or the saved base is all one, so the exponent (``args[1]``) is irrelevant;
- sdpa: there is one key position and every softmax row is well defined, so
  the lone softmax weight is exactly one whatever the additive float mask
  (``args[3]``) holds (the query and key edges are the ``_sdpa_value_proofs``);
- ``__setitem__``: the perturbed parent is an integer index component that
  addresses a destination dim of size 1, where every valid index is position 0.

Zero einsum operands, eval-mode batch norm at its running mean, unread
``scatter_reduce`` destinations and empty split chunks or matmul products are
proved by ``_operand_value_proofs`` and the overwrite decision before this
module is consulted.

Split from ``validation/exemptions.py`` (R43 file-size ratchet), which dispatches
here from its posthoc value-proof decision.
"""

from typing import Any

import numpy as np
import torch

from .._data_classes.op import Op
from ._sdpa_value_proofs import _output_equals_value_rows, sdpa_softmax_rows_well_defined
from ._value_predicates import _is_all_zero_value, _saved_output_all_finite
from .exemptions import PosthocPerturbDecision

_NOT_PROVED = PosthocPerturbDecision(False, "not_value_proved")

#: Keyword spellings read as their positional slot, per op family.
_KWARG_SLOTS = {
    "scaled_dot_product_attention": {"query": 0, "key": 1, "value": 2, "attn_mask": 3},
}

DIVISION_FUNC_NAMES = frozenset(
    {
        "__truediv__",
        "__itruediv__",
        "truediv",
        "div",
        "div_",
        "divide",
        "divide_",
        "true_divide",
        "true_divide_",
    }
)
CONJUNCTION_FUNC_NAMES = frozenset(
    {
        "__and__",
        "__rand__",
        "__iand__",
        "bitwise_and",
        "bitwise_and_",
        "logical_and",
        "logical_and_",
    }
)
DISJUNCTION_FUNC_NAMES = frozenset(
    {
        "__or__",
        "__ror__",
        "__ior__",
        "bitwise_or",
        "bitwise_or_",
        "logical_or",
        "logical_or_",
    }
)
POW_FUNC_NAMES = frozenset({"__pow__", "__ipow__", "pow", "pow_"})
VALUE_PROOF_FUNC_NAMES = (
    DIVISION_FUNC_NAMES
    | CONJUNCTION_FUNC_NAMES
    | DISJUNCTION_FUNC_NAMES
    | POW_FUNC_NAMES
    | frozenset({"__setitem__", "scaled_dot_product_attention"})
)


def _perturbed_arg_keys(layer: Op, layers_to_perturb: list[str]) -> set[Any] | None:
    """Return every positional key the single perturbed parent occupies.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is inspected.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    set or None
        Positional keys (ints, or tuples for nested arguments) holding the
        perturbed parent; ``None`` when more than one parent is perturbed or the
        parent also sits in a keyword argument, so callers fail closed.
    """

    if len(layers_to_perturb) != 1:
        return None
    positions = getattr(layer, "parent_arg_positions", None) or {}
    label = layers_to_perturb[0]
    kwarg_slots = _KWARG_SLOTS.get(getattr(layer, "func_name", None) or "", {})
    keys: set[Any] = set()
    for name, parent in (positions.get("kwargs", {}) or {}).items():
        if parent == label:
            if name not in kwarg_slots:
                return None
            keys.add(kwarg_slots[name])
    arg_positions = positions.get("args", {}) or {}
    return keys | {key for key, parent in arg_positions.items() if parent == label}


def _output_all_equal(layer: Op, value: float) -> bool:
    """Return whether the saved output is a non-empty tensor equal to ``value`` everywhere."""

    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor) or out.numel() == 0:
        return False
    return bool(torch.all(torch.eq(out, value)).item())


def _division_decision(layer: Op, keys: set[Any], args: tuple[Any, ...]) -> PosthocPerturbDecision:
    """Exempt the divisor of an all-zero dividend whose quotient is finite zero."""

    if keys != {1} or len(args) < 2 or not _is_all_zero_value(args[0]):
        return _NOT_PROVED
    if not (_saved_output_all_finite(layer) and _output_all_equal(layer, 0.0)):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "zero_dividend_annihilator",
        "the saved dividend is all zero and the quotient is finite, so any non-zero "
        "divisor yields the same zero output",
    )


def _conjunction_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt one operand of a bitwise/logical and whose other operand is all zero."""

    other = _sibling_operand_slot(keys)
    if other is None or len(args) <= other or not _is_all_zero_value(args[other]):
        return _NOT_PROVED
    if not _output_all_equal(layer, 0):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "conjunction_zero_annihilator",
        f"the non-perturbed and-operand at args[{other}] is all zero/False, so the "
        "conjunction is zero whatever the perturbed operand holds",
    )


def _sibling_operand_slot(keys: set[Any]) -> int | None:
    """Return the other operand slot of a binary op whose perturbed parent fills one slot."""

    if keys == {0}:
        return 1
    if keys == {1}:
        return 0
    return None


def _is_all_true_bool(value: Any) -> bool:
    """Return whether ``value`` is ``True`` or a non-empty all-True bool tensor."""

    if value is True:
        return True
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.numel() == 0:
        return False
    return bool(value.all().item())


def _disjunction_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt one operand of a bool or whose other operand is all True."""

    other = _sibling_operand_slot(keys)
    if other is None or len(args) <= other or not _is_all_true_bool(args[other]):
        return _NOT_PROVED
    if not _is_all_true_bool(getattr(layer, "out", None)):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "disjunction_saturated",
        f"the non-perturbed bool or-operand at args[{other}] is all True, so the "
        "disjunction is True whatever the perturbed operand holds",
    )


def _is_all_one_value(value: Any) -> bool:
    """Return whether ``value`` is a real scalar or non-empty tensor equal to one everywhere."""

    if isinstance(value, torch.Tensor):
        return value.numel() > 0 and bool(torch.all(torch.eq(value, 1)).item())
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 1


def _pow_decision(layer: Op, keys: set[Any], args: tuple[Any, ...]) -> PosthocPerturbDecision:
    """Exempt a base under a zero exponent, or an exponent over a unit base."""

    if keys == {0}:
        return _pow_zero_exponent_decision(layer, args)
    if keys != {1} or not args or not _is_all_one_value(args[0]):
        return _NOT_PROVED
    if not _output_all_equal(layer, 1.0):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "pow_unit_base",
        "the saved base is all one, so every exponent value yields one",
    )


def _pow_zero_exponent_decision(layer: Op, args: tuple[Any, ...]) -> PosthocPerturbDecision:
    """Exempt the base of a power whose saved exponent is all zero."""

    kwargs = getattr(layer, "saved_kwargs", None) or {}
    exponent = args[1] if len(args) > 1 else kwargs.get("exponent")
    if exponent is None or not _is_all_zero_value(exponent):
        return _NOT_PROVED
    if not _output_all_equal(layer, 1.0):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "pow_zero_exponent",
        "the saved exponent is all zero, so every base value yields one",
    )


def _is_bool_mask(component: Any) -> bool:
    """Return whether an index component is a torch or numpy boolean mask."""

    if isinstance(component, torch.Tensor):
        return component.dtype == torch.bool
    return isinstance(component, np.ndarray) and component.dtype == np.bool_


def _sdpa_singleton_key_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt sdpa's additive float mask when there is one key position.

    Softmax over a single key position is exactly one for any finite score, so
    with every softmax row well defined (finite query and key, no dropout, a
    finite scale and a mask that keeps the lone key in every row) and the saved
    output equal to the value rows, an additive float mask cannot reach the
    output. A bool mask can mask the lone key out and is never exempted. The
    query and key edges of the same call are the ``sdpa_single_key_identity``
    and ``sdpa_single_key_key_edge`` proofs, which carry the same precondition.
    """

    if keys != {3} or len(args) < 3 or not isinstance(args[1], torch.Tensor):
        return _NOT_PROVED
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    if not (isinstance(mask, torch.Tensor) and mask.is_floating_point()):
        return _NOT_PROVED
    if args[1].ndim < 2 or args[1].shape[-2] != 1:
        return _NOT_PROVED
    if not sdpa_softmax_rows_well_defined(layer, args):
        return _NOT_PROVED
    if not _output_equals_value_rows(getattr(layer, "out", None), args[2]):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "sdpa_singleton_key",
        "there is one key position and every softmax row is well defined, so its "
        "softmax weight is exactly one and the additive mask values cannot reach the output",
    )


def _index_component_dims(ndim: int, components: tuple[Any, ...]) -> list[int | None] | None:
    """Map each ``__setitem__`` index component to the destination dim it addresses.

    Parameters
    ----------
    ndim:
        Destination rank.
    components:
        Index components in order.

    Returns
    -------
    list or None
        The first addressed destination dim per component (``None`` for a
        component that addresses no dim), or ``None`` for any component kind
        whose dim footprint is not modelled, so callers fail closed.
    """

    widths: list[int | None] = []
    for component in components:
        if component is None:
            widths.append(0)
        elif component is Ellipsis:
            widths.append(None)
        elif _is_bool_mask(component):
            widths.append(component.ndim)
        elif isinstance(component, (torch.Tensor, slice)) or type(component) is int:
            widths.append(1)
        else:
            return None
    if widths.count(None) > 1:
        return None
    explicit = sum(width for width in widths if width is not None)
    dims: list[int | None] = []
    cursor = 0
    for width in widths:
        width = ndim - explicit if width is None else width
        dims.append(cursor if width else None)
        cursor += width
    return dims if cursor <= ndim else None


def _setitem_singleton_index_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt an integer index component that addresses a destination dim of size 1."""

    if len(args) < 2 or not isinstance(args[0], torch.Tensor) or not keys:
        return _NOT_PROVED
    destination, index = args[0], args[1]
    components = index if isinstance(index, tuple) else (index,)
    dims = _index_component_dims(destination.ndim, components)
    if dims is None:
        return _NOT_PROVED
    if not all(
        _key_is_integer_index_into_singleton_dim(key, index, components, dims, destination)
        for key in keys
    ):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "setitem_singleton_index_dim",
        "the perturbed integer index addresses a destination dim of size 1, where "
        "every valid index selects the same position",
    )


def _key_is_integer_index_into_singleton_dim(
    key: Any,
    index: Any,
    components: tuple[Any, ...],
    dims: list[int | None],
    destination: torch.Tensor,
) -> bool:
    """Return whether a perturbed ``__setitem__`` index key addresses a size-1 dim.

    ``key`` is ``1`` for a bare (non-tuple) index or ``(1, position)`` for one
    component of a tuple index; the component must be an integer tensor whose
    destination dim has size 1.
    """

    if key == 1 and not isinstance(index, tuple):
        component_position = 0
    elif isinstance(key, tuple) and len(key) == 2 and key[0] == 1:
        component_position = key[1]
    else:
        return False
    component = components[component_position]
    dim = dims[component_position]
    if not isinstance(component, torch.Tensor) or component.dtype == torch.bool:
        return False
    if component.is_floating_point() or component.is_complex() or dim is None:
        return False
    return bool(destination.shape[dim] == 1)


_DECISIONS_BY_FAMILY = (
    (DIVISION_FUNC_NAMES, _division_decision),
    (CONJUNCTION_FUNC_NAMES, _conjunction_decision),
    (DISJUNCTION_FUNC_NAMES, _disjunction_decision),
    (POW_FUNC_NAMES, _pow_decision),
    (frozenset({"__setitem__"}), _setitem_singleton_index_decision),
    (frozenset({"scaled_dot_product_attention"}), _sdpa_singleton_key_decision),
)


def saved_value_proof_decision(
    layer: Op, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Dispatch to the value proof for ``layer``'s op family.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        The family's decision, or non-exempt when no family matches or the
        perturbed parent's slots cannot be read.
    """

    keys = _perturbed_arg_keys(layer, layers_to_perturb)
    if keys is None:
        return _NOT_PROVED
    for func_names, decide in _DECISIONS_BY_FAMILY:
        if layer.func_name in func_names:
            return decide(layer, keys, args)
    return _NOT_PROVED
