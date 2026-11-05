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
label and nothing is recorded, so no capability flag is needed.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from ... import _state
from ..._data_classes.internal_types import FuncExecutionContext
from ..._utils import _torch_compat
from ..._utils.rng import log_current_autocast_state


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


def _own_view_dispatch(witness_state: Any, start: int, view_as: Any, func_call_id: int) -> None:
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
    view_as:
        The ``view_as`` callable the recorded op names.
    func_call_id:
        The recorded op's call id.
    """

    from .escape_detection import _OUTPUT_SCOPED_CREDIT_PREFIX, ExpectedOriginalToken

    if witness_state is None:
        return
    for event in reversed(witness_state.events[start:]):
        # No owner, or only the enclosing module's forward (no torch wrapper between).
        owner_wrapper = getattr(event.owner, "wrapper_name", None)
        unowned = event.owner is None or str(owner_wrapper).startswith(_OUTPUT_SCOPED_CREDIT_PREFIX)
        if unowned and event.operator.split(".")[:2] in (["aten", "view"], ["aten", "alias"]):
            event.owner = ExpectedOriginalToken(
                raw_id=id(view_as),
                original=view_as,
                wrapper_name="view_as",
                wrapper_frame_id=0,
                owner_thread_id=threading.get_ident(),
                pending=False,
                func_name="view_as",
                func_call_id=func_call_id,
                capture_accounted=True,
            )
            return


def log_native_input_views(trace: Any, args: tuple[Any, ...], out: Any, marks: Any) -> None:
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
    """

    from ._ops_emission import log_function_output_tensors
    from ._tl import get_tensor_label
    from .wrappers import _collect_output_tensors

    sources = [arg for arg in args if isinstance(arg, torch.Tensor) and get_tensor_label(arg)]
    if not sources:
        return
    base_class = _torch_compat.get_tensorbase_class()
    view_as = getattr(base_class, "view_as", None) if base_class is not None else None
    if view_as is None:
        return
    with _state.pause_logging():
        views = []
        for output in _collect_output_tensors(out):
            if get_tensor_label(output) is None and output._is_view():
                source = next((arg for arg in sources if _same_view(output, arg)), None)
                if source is not None:
                    views.append((output, source))
    for output, source in views:
        exec_ctx = FuncExecutionContext(
            time_elapsed=0.0, rng_states={}, autocast_state=log_current_autocast_state()
        )
        func_call_id = _state.next_func_call_id()
        call: tuple[Any, ...] = (
            trace,
            view_as,
            "view_as",
            (source, source),
            {},
            (source, source),
            {},
            output,
            exec_ctx,
            True,
            func_call_id,
        )
        # Hide this bookkeeping from the opt-in user-op census, as the boundary op does.
        if _state.completeness_witness_is_shadow():
            with _state.pause_logging():
                log_function_output_tensors(*call)
        else:
            log_function_output_tensors(*call)
        _own_view_dispatch(marks.witness_state, marks.witness_mark, view_as, func_call_id)
