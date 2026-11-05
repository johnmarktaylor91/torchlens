"""Credit the alias autograd takes of a custom ``Function``'s output after its forward.

A custom ``torch.autograd.Function`` whose forward saves its own output
(``ctx.save_for_backward(y)`` with ``y`` returned) makes ``Function.apply`` dispatch
one ``aten.detach`` of ``y`` in C++ after the Python forward returns and before
``apply`` returns: the saved-variable alias autograd keeps for backward. No torch
wrapper runs there, so the completeness census saw the dispatch with only the
enclosing module forward as owner and counted it as a dropped op.

The credit is exactly as narrow as that event:

* at dispatch time, a pure alias (``aten.detach`` / ``aten.alias``) is marked only
  when the Python frame that entered the dispatcher is ``Function.apply`` itself
  (:data:`_APPLY_CODES`), so the dispatch ran in the C++ part of ``apply`` and not
  inside the user's forward or ``setup_context`` (each of those is its own Python
  frame);
* after that ``apply`` returns, :func:`credit_saved_output_aliases` re-owns a marked
  event of the call's own census window only when its operand IS (object identity)
  one of the tensors the call returned, its result is not one of them, it did not
  mutate, and only the enclosing module forward owned it.

Every other dispatch in the epilogue, an alias of any other tensor, and every
dispatch inside the user's forward keep their owner and still count.
"""

from __future__ import annotations

import dataclasses
import sys
import types
import weakref
from typing import Any

import torch

SAVED_ALIAS_OWNER = "autograd_function:saved_output_alias"
"""Owner token name of a credited epilogue alias."""

_PURE_ALIAS_OPERATORS = frozenset({"aten.detach", "aten.alias"})
_APPLY_CODES: set[types.CodeType] = set()
"""Code objects of the ``Function.apply`` frame that enters the C++ ``apply``."""


def register_apply_code(apply_function: Any) -> None:
    """Record the code object of the unwrapped ``Function.apply`` callable.

    Parameters
    ----------
    apply_function:
        The plain callable :func:`_function_boundary.apply_slot_function` returned
        (the Python classmethod body, or the closure that calls the C descriptor).
    """

    code = getattr(apply_function, "__code__", None)
    if isinstance(code, types.CodeType):
        _APPLY_CODES.add(code)


def epilogue_credit_refs(
    event: Any, func: Any, args: tuple[Any, ...], result: Any
) -> tuple[weakref.ref[torch.Tensor], ...]:
    """Return the event's credit references, marking a pure alias made by C++ ``apply``.

    Called from the witness dispatch callback only for output-scoped (module forward)
    owners, in place of ``_completeness_dispatch._dispatch_credit_refs``, whose rows it
    keeps: the tensor results (one level deep), then a pure alias's operand. Kept here
    so this module imports nothing from the witness import cycle. Frame 0 is this
    function, frame 1 the dispatch callback, frame 2 the Python frame that entered the
    dispatcher.

    Parameters
    ----------
    event:
        The dispatch event being finalized (``event.operator`` is overload-qualified).
    func:
        The dispatched operator (unused beyond the event's name; kept for the call shape).
    args:
        Its positional arguments.
    result:
        What it returned.

    Returns
    -------
    tuple[weakref.ref[torch.Tensor], ...]
        Weak references to the results and, for a pure alias, its operand.
    """

    del func
    refs: list[weakref.ref[torch.Tensor]] = []
    if isinstance(result, torch.Tensor):
        refs.append(weakref.ref(result))
    elif isinstance(result, (list, tuple)):
        refs.extend(weakref.ref(item) for item in result if isinstance(item, torch.Tensor))
    operator = event.operator
    base = operator.rsplit(".", 1)[0] if operator.count(".") >= 2 else operator
    if base in _PURE_ALIAS_OPERATORS and args and isinstance(args[0], torch.Tensor):
        refs.append(weakref.ref(args[0]))
        if _APPLY_CODES:
            event.apply_epilogue_alias = sys._getframe(2).f_code in _APPLY_CODES
    return tuple(refs)


def _is_saved_output_alias(event: Any, outputs: list[torch.Tensor]) -> bool:
    """Return whether one marked event is a pure alias OF a returned tensor."""

    if not event.apply_epilogue_alias or event.mutates or event.outcome != "returned_tensor":
        return False
    if not getattr(event.owner, "boundary_credit_is_output_scoped", False):
        return False
    if len(event.credit_refs) != 2:
        return False
    result, operand = (ref() for ref in event.credit_refs)
    if operand is None or not any(operand is output for output in outputs):
        return False
    return not any(result is output for output in outputs)


def credit_saved_output_aliases(witness_state: Any, start: int, out: Any, token: Any) -> None:
    """Re-own the epilogue aliases of one ``apply`` call's own returned tensors.

    Parameters
    ----------
    witness_state:
        The armed completeness-witness state, or ``None`` when no census runs.
    start:
        Index of the first witness event of the ``apply`` call.
    out:
        What ``apply`` returned (a tensor or a flat tuple).
    token:
        A capture-accounted owner token with no call id and no boundary outputs, so a
        credited alias leaves both censuses (it is no captured op and no dropped one).
    """

    if witness_state is None:
        return
    items = out if isinstance(out, (tuple, list)) else (out,)
    outputs = [item for item in items if isinstance(item, torch.Tensor)]
    if not outputs:
        return
    owned = dataclasses.replace(token, census_scope="owned")
    for event in witness_state.events[start:]:
        if _is_saved_output_alias(event, outputs):
            event.owner = owned
