"""Per-call argument layout plans: what a call's argument walks decide from types alone.

T1 part 5. The shared-fields builder walks one call's arguments several times: the
non-tensor split, the slot snapshot's tensor walk and the completeness witness's
tree walk. On a *flat* call (every positional and keyword value is a tensor or a
value none of those walks looks inside) each walk's answer depends only on which
positions hold tensors, and that is a function of the argument types. The plan for
one layout is built from the first call with that layout and reused by later calls
whose guard matches:

- the guard key is ``(func_name, argument types, keyword names, keyword types)``,
  plus ``plan.func is func``, checked on every call;
- a plan records which positions hold tensors and which hold other values;
- a layout with any non-flat type (a container, a mapping, or a class that
  overrides ``__class__``, which ``isinstance`` would consult) gets a plan that
  says so, and every walk then runs its general path.

Tensor values themselves are never cached: every consumer reads the live tensor at
the planned position. Plans live in the capture's raw-graph workspace
(``RawGraphWorkspace._call_layouts``) and go away with it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..._capture._arg_snapshot import split_non_tensor_args
from ._ops_arg_slots import _peek_raw_graph_workspace, _SlotCall

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = ("CallLayout", "planned_slot_call", "planned_tensor_items", "split_call_arguments")

#: Value types the argument walks look inside: ``_tensor_items`` (mappings, tuples,
#: lists), ``_walk_arg_tree`` (lists, tuples, dicts) and ``split_non_tensor_args``
#: (lists, dicts, sets, bytearrays, tuples).
_OPEN_TYPES = (list, tuple, dict, set, frozenset, bytearray, Mapping)


@dataclass(slots=True)
class CallLayout:
    """One argument layout's plan: which positions hold tensors.

    ``flat`` is False when any argument type is one the walks look inside; the other
    fields are then empty and every consumer runs its general path. ``mutates_receiver``
    is the witness's receiver-mutation verdict for ``func_name`` (part of the key),
    filled by the witness on first use.
    """

    func: Callable[..., Any]
    flat: bool
    tensor_slots: tuple[tuple[str, Any], ...] = ()
    other_args: tuple[int, ...] = ()
    other_kwargs: tuple[str, ...] = ()
    has_out: bool = False
    mutates_receiver: bool | None = None
    #: The guard key's type part, ``(argument types, keyword names, keyword types)``.
    types: tuple[Any, ...] = ((), (), ())
    #: The replay template plan of this layout (``_args_template_cache``), built on
    #: first use; the layout's guard is the template plan's key.
    template_plan: tuple[Any, ...] | None = None


def _is_flat_type(value_type: type) -> bool:
    """Return whether no argument walk looks inside a value of exact type ``value_type``.

    A class that defines ``__class__`` anywhere below ``object`` is not flat:
    ``isinstance`` consults ``__class__`` as well as the type.
    """

    if issubclass(value_type, _OPEN_TYPES):
        return False
    return not any("__class__" in vars(klass) for klass in value_type.__mro__[:-1])


def _new_layout(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    types: tuple[Any, ...],
) -> CallLayout:
    """Build the plan of the layout ``args`` and ``kwargs`` have."""

    values = (*args, *kwargs.values())
    if not all(_is_flat_type(type(value)) for value in values):
        return CallLayout(func, False)
    tensor = torch.Tensor
    slots: list[tuple[str, Any]] = [
        ("args", i) for i, value in enumerate(args) if isinstance(value, tensor)
    ]
    slots += [("kwargs", key) for key, value in kwargs.items() if isinstance(value, tensor)]
    return CallLayout(
        func,
        True,
        tuple(slots),
        tuple([i for i, value in enumerate(args) if not isinstance(value, tensor)]),
        tuple([key for key, value in kwargs.items() if not isinstance(value, tensor)]),
        "out" in kwargs,
        None,
        types,
    )


def planned_slot_call(
    trace: Trace,
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _SlotCall:
    """Return the call with its layout plan, or with none when the capture keeps no plans.

    The guard: the layout key (name and exact types) selects the plan, and the plan
    must belong to this very callable. A miss builds the plan from this call.
    """

    workspace = _peek_raw_graph_workspace(trace)
    if workspace is None:
        return _SlotCall(func, func_name, args, kwargs)
    key: tuple[Any, ...] = (
        (func_name, tuple(map(type, args)), tuple(kwargs), tuple(map(type, kwargs.values())))
        if kwargs
        else (func_name, tuple(map(type, args)))
    )
    layouts = workspace._call_layouts
    layout = layouts.get(key)
    if layout is None or layout.func is not func:
        types = (key[1], key[2], key[3]) if kwargs else (key[1], (), ())
        layout = layouts[key] = _new_layout(func, args, kwargs, types)
    return _SlotCall(func, func_name, args, kwargs, layout if layout.flat else None)


def split_call_arguments(
    call: _SlotCall, is_tensor: Callable[[Any], bool]
) -> tuple[list[Any], dict[str, Any]]:
    """Return ``split_non_tensor_args(call.args, call.kwargs, is_tensor)``.

    On a flat layout the non-tensor values are the planned non-tensor positions,
    each kept by reference, which is what the general split decides for a value that
    is neither a tensor nor a container.
    """

    layout = call.layout
    if layout is None:
        return split_non_tensor_args(call.args, call.kwargs, is_tensor)
    args, kwargs = call.args, call.kwargs
    return [args[i] for i in layout.other_args], {key: kwargs[key] for key in layout.other_kwargs}


def planned_tensor_items(call: _SlotCall, layout: CallLayout) -> list[tuple[str, Any, Any, bool]]:
    """Return ``list(_tensor_items(call))`` for a call with a flat layout.

    Every tensor of a flat call is a top-level value, which the template walk reaches.
    """

    args, kwargs = call.args, call.kwargs
    return [
        (kind, key, args[key] if kind == "args" else kwargs[key], True)
        for kind, key in layout.tensor_slots
    ]
