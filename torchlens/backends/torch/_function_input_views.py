"""Record the view torch makes when an ``autograd.Function`` forward returns an input.

A custom ``Function`` whose forward hands back one of its input tensors unchanged
(``BackwardHookFunction``, which ``register_full_backward_hook`` applies to a
module's inputs and outputs, does exactly that) gets back ``input.view_as(input)``
from torch, so the output can carry the Function's ``grad_fn``. Newer torch makes
that view through the Python ``view_as``, which TorchLens wraps and records. Torch
2.1 makes it in C++, so no wrapper sees it and the returned tensor reaches its
consumer (or the model output) with no producing op.

:func:`log_native_input_views` closes that gap: after ``apply`` returns, each
output with no TorchLens label that is a same-geometry view of a labeled input
argument is recorded as the ``view_as`` call torch made, with the input as its
parent. Where torch already routed the view through Python the output carries a
label and nothing is recorded, so no capability flag is needed. The op itself is logged
by the caller's ``log_view`` (``_function_boundary`` owns the op logger and the census
token), so this module stays outside the capture import cycle.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from ... import _state
from ..._utils import _torch_compat
from ..._utils._tensor_label import get_tensor_label


def _same_view(output: torch.Tensor, source: torch.Tensor) -> bool:
    """Return whether ``output`` is a view of ``source`` with ``source``'s geometry.

    Parameters
    ----------
    output:
        A tensor ``Function.apply`` returned.
    source:
        One tensor argument of that call.

    Returns
    -------
    bool
        ``True`` when both share one base, data pointer, shape, stride and dtype.
    """

    root = source._base if source._is_view() else source
    return (
        output._base is root
        and output.data_ptr() == source.data_ptr()
        and output.shape == source.shape
        and output.stride() == source.stride()
        and output.dtype == source.dtype
    )


def own_view_dispatch(witness_state: Any, start: int, token: Any) -> None:
    """Hand the last unowned aten view of the ``apply`` call to the recorded ``view_as`` op.

    The C++ ``view_as`` ran outside every torch wrapper, so the completeness census saw
    its ``aten.view`` with no owner (or only the enclosing module forward's); the op
    recorded for it now owns that one dispatch.

    Parameters
    ----------
    witness_state:
        The armed completeness-witness state, or ``None`` when no census runs.
    start:
        Index of the first witness event of the ``apply`` call.
    token:
        The recorded op's capture-accounted owner token; it is re-scoped as an
        ordinary owned wrapper edge, which is what a Python ``view_as`` would get.
    """

    if witness_state is None:
        return
    owned = dataclasses.replace(token, census_scope="owned")
    for event in reversed(witness_state.events[start:]):
        # No owner, or only the enclosing module's forward (no torch wrapper between).
        module_only = getattr(event.owner, "boundary_credit_is_output_scoped", False)
        unowned = event.owner is None or module_only
        if unowned and event.operator.split(".")[:2] in (["aten", "view"], ["aten", "alias"]):
            event.owner = owned
            return


def _function_outputs(out: Any) -> list[torch.Tensor]:
    """Return the tensors ``Function.apply`` returned (it returns a tensor or a flat tuple)."""

    outputs = out if isinstance(out, (tuple, list)) else (out,)
    return [output for output in outputs if isinstance(output, torch.Tensor)]


def log_native_input_views(
    trace: Any, args: tuple[Any, ...], out: Any, marks: Any, log_view: Callable[..., None]
) -> None:
    """Record ``view_as`` for each ``apply`` output torch aliased to an input in C++.

    Parameters
    ----------
    trace:
        Active capture trace.
    args:
        Positional ``apply`` arguments (a Function's tensors are passed positionally).
    out:
        What ``apply`` returned.
    marks:
        The call's ``BoundaryMarks`` (its completeness-witness state and start index).
    log_view:
        ``log_view(trace, view_as, source, output, marks)`` logs one such view as an op.
    """

    sources = [arg for arg in args if isinstance(arg, torch.Tensor) and get_tensor_label(arg)]
    if not sources:
        return
    base_class = _torch_compat.get_tensorbase_class()
    view_as = getattr(base_class, "view_as", None) if base_class is not None else None
    if view_as is None:
        return
    with _state.pause_logging():
        views = []
        for output in _function_outputs(out):
            if get_tensor_label(output) is None and output._is_view():
                source = next((arg for arg in sources if _same_view(output, arg)), None)
                if source is not None:
                    views.append((output, source))
    for output, source in views:
        log_view(trace, view_as, source, output, marks)
