"""Opt-in label rows for backward nodes (U08.5; ``draw_backward`` keywords).

``draw_backward(show_module_paths=True)`` adds the owning module (or, on
AccumulateGrad nodes, the parameter) to each backward node;
``show_grad_values=True`` adds a compact norm and largest-magnitude summary of
the node's captured output gradient. Neither row appears unless requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ._render_leaf import _first_tensor_in_obj, _module_key_for_forward_op

if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn

__all__ = ["BackwardRowOptions", "backward_opt_in_rows"]


@dataclass(frozen=True)
class BackwardRowOptions:
    """Opt-in label rows for backward nodes (U08.5; ``draw_backward`` keywords).

    Parameters
    ----------
    trace:
        Trace owning the grad_fns (parameter attribution for AccumulateGrad).
    module_paths:
        Add a ``module <address>`` row (``param <address>`` on AccumulateGrad).
    grad_values:
        Add a ``grad norm ..., absmax ...`` row from the captured gradient.
    """

    trace: Any
    module_paths: bool = False
    grad_values: bool = False


def backward_opt_in_rows(
    grad_fn_handle: GradFn,
    call: Any | None,
    rows: BackwardRowOptions,
) -> list[str]:
    """Return the requested opt-in rows (U08.5) for one backward node.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    call:
        GradFnCall in unrolled mode; ``None`` reads the latest captured call.
    rows:
        Which rows were requested.

    Returns
    -------
    list[str]
        Zero, one or two rows. A module row is omitted when no module owns
        the node (an intervening grad_fn); the value row says when no
        gradient tensor was captured rather than inventing a number.
    """

    extra: list[str] = []
    if rows.module_paths:
        module_row = _backward_module_path_row(grad_fn_handle, rows.trace)
        if module_row is not None:
            extra.append(module_row)
    if rows.grad_values:
        extra.append(_backward_grad_value_row(grad_fn_handle, call))
    return extra


def _backward_module_path_row(grad_fn_handle: GradFn, trace: Any) -> str | None:
    """Return ``module <address>`` (or ``param <address>``) for a backward node.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to label.
    trace:
        Owning trace, used for AccumulateGrad parameter attribution.

    Returns
    -------
    str | None
        The row, or ``None`` when the node has no owning module.
    """

    op = grad_fn_handle.op
    if op is not None:
        module_key = _module_key_for_forward_op(op)
        if module_key is None:
            return None
        return f"module {module_key.split(':', 1)[0]}"
    if grad_fn_handle.type == "accumulategrad" and trace is not None:
        param_address = getattr(trace, "_grad_fn_param_refs", {}).get(grad_fn_handle.label)
        if param_address is not None:
            return f"param {param_address}"
    return None


def _backward_grad_value_row(grad_fn_handle: GradFn, call: Any | None) -> str:
    """Return a compact gradient-value row: L2 norm and largest magnitude.

    Parameters
    ----------
    grad_fn_handle:
        GradFn whose captured output gradient is summarized.
    call:
        GradFnCall in unrolled mode; ``None`` reads the latest captured call.

    Returns
    -------
    str
        ``grad norm <x>, absmax <y>``, or ``grad values not captured``.
    """

    calls = [call] if call is not None else list(reversed(list(grad_fn_handle.calls.values())))
    for grad_fn_call in calls:
        tensor = _first_tensor_in_obj(getattr(grad_fn_call, "grad_outputs", None))
        if tensor is None:
            continue
        with torch.no_grad():
            values = tensor.detach().to(torch.float64)
            if values.numel() == 0:
                return "grad values empty"
            norm = float(torch.linalg.vector_norm(values))
            absmax = float(values.abs().max())
        return f"grad norm {norm:.3g}, absmax {absmax:.3g}"
    return "grad values not captured"
